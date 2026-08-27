from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lora_factory.gpu import (
    GpuDevice,
    GpuLeaseManager,
    GpuTaskKind,
    GpuTaskRequest,
    LeaseConflict,
    SelectedGpuScheduler,
    bind_gpu_for_child,
    discover_nvidia_gpus,
    parse_nvidia_smi_inventory,
    sample_telemetry,
)
from lora_factory.runtime import (
    BackendPin,
    ManagedRuntimeManifest,
    evaluate_runtime_compatibility,
    inspect_runtime,
    load_runtime_manifest,
    write_runtime_manifest,
)
from lora_factory.storage.database import Database
from lora_factory.storage.orm import GpuDeviceRow, ProjectRow, RunRow

GPU_A = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU_B = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
GPU_C = "GPU-cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"


def _device(
    uuid: str,
    index: int,
    *,
    free: int = 16_000,
    utilization: float = 0,
    compatible: bool = True,
) -> GpuDevice:
    return GpuDevice(
        uuid=uuid,
        index=index,
        name=f"GPU {index}",
        total_vram_mb=max(24_000, free),
        free_vram_mb=free,
        utilization_percent=utilization,
        capability_major=12,
        capability_minor=0,
        compatible=compatible,
    )


def test_archaudit_discovers_csv_and_re_resolves_uuid_after_index_reorder() -> None:
    output = (
        f"0, {GPU_A}, NVIDIA RTX A, 24564, 20000, 12, 12.0\n"
        f"1, {GPU_B}, NVIDIA RTX B, 16384, 8000, 3, 8.9\n"
    )
    devices = parse_nvidia_smi_inventory(output)
    assert [device.uuid for device in devices] == [GPU_A, GPU_B]
    assert devices[0].bf16_supported

    reordered = (_device(GPU_A, 1), _device(GPU_B, 0))
    binding = bind_gpu_for_child(
        GPU_B,
        selected_uuids=(GPU_B,),
        devices=reordered,
        base_environment={
            "CUDA_VISIBLE_DEVICES": "99",
            "SAFE": "yes",
            "CODEX_API_KEY": "must-not-reach-managed-child",
            "CUDA_PATH_V12_4": "C:\\CUDA\\v12.4",
            "CUDA_PATH_SECRET": "must-not-reach-managed-child",
        },
    )
    assert binding.physical_index == 0
    assert binding.logical_index == 0
    assert binding.environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert "SAFE" not in binding.environment
    assert "CODEX_API_KEY" not in binding.environment
    assert binding.environment["CUDA_PATH_V12_4"] == "C:\\CUDA\\v12.4"
    assert "CUDA_PATH_SECRET" not in binding.environment


def test_archaudit_gpu_binding_does_not_inherit_parent_secret_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LORA_FACTORY_AUDIT_SENTINEL", "must-not-reach-managed-child")
    binding = bind_gpu_for_child(
        GPU_A,
        selected_uuids=(GPU_A,),
        devices=(_device(GPU_A, 0),),
    )

    assert "LORA_FACTORY_AUDIT_SENTINEL" not in binding.environment


def test_archaudit_discovery_uses_argument_array() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...], _timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"0, {GPU_A}, RTX, 24000, 20000, 0, 12.0\n",
            stderr="",
        )

    devices = discover_nvidia_gpus(runner=runner)
    assert devices[0].uuid == GPU_A
    assert calls == [
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_cap",
            "--format=csv,noheader,nounits",
        )
    ]


def test_archaudit_telemetry_filters_to_selected_pool() -> None:
    def runner(argv: tuple[str, ...], _timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(f"{GPU_A}, 20, 4000, 20000, 55, 125.5\n{GPU_C}, 99, 23000, 1000, 80, 400.0\n"),
            stderr="",
        )

    samples = sample_telemetry((GPU_A,), runner=runner)
    assert len(samples) == 1
    assert samples[0].uuid == GPU_A
    assert samples[0].power_watts == 125.5


def test_archaudit_scheduler_never_assigns_unselected_or_leased_gpu() -> None:
    scheduler = SelectedGpuScheduler((GPU_A, GPU_B))
    devices = (
        _device(GPU_A, 0, free=12_000, utilization=70),
        _device(GPU_B, 1, free=18_000, utilization=20),
        _device(GPU_C, 2, free=48_000, utilization=0),
    )
    training = scheduler.assign(
        GpuTaskRequest(task_id="train", kind=GpuTaskKind.TRAIN, estimated_vram_mb=10_000),
        devices,
    )
    assert training.gpu_uuid == GPU_B
    tagging = scheduler.assign(
        GpuTaskRequest(task_id="tag", kind=GpuTaskKind.WD14_TAG, estimated_vram_mb=2_000),
        devices,
        leased_gpu_uuids=(GPU_B,),
    )
    assert tagging.gpu_uuid == GPU_A
    assert GPU_C not in scheduler.shard_items(("a", "b", "c"), devices)


def test_archaudit_three_gpu_shards_follow_uuid_after_index_reorder_and_training_fallback() -> None:
    scheduler = SelectedGpuScheduler((GPU_A, GPU_B, GPU_C))
    devices = (
        _device(GPU_C, 0, free=20_000, utilization=30),
        _device(GPU_B, 1, free=22_000, utilization=20),
        _device(GPU_A, 2, free=2_000, utilization=0),
    )

    shards = scheduler.shard_items(tuple(f"item-{index}" for index in range(7)), devices)
    flattened = {item for items in shards.values() for item in items}
    assert set(shards) == {GPU_A, GPU_B, GPU_C}
    assert flattened == {f"item-{index}" for index in range(7)}

    assignment = scheduler.assign(
        GpuTaskRequest(
            task_id="fallback",
            kind=GpuTaskKind.TRAIN,
            estimated_vram_mb=10_000,
            preferred_gpu_uuid=GPU_A,
        ),
        devices,
    )
    assert assignment.gpu_uuid == GPU_B
    assert assignment.physical_index == 1


def test_archaudit_sqlite_lease_conflict_heartbeat_and_stale_recovery(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.session() as session:
        session.add(ProjectRow(id="project", name="Project", root_path=str(tmp_path)))
        session.add(RunRow(id="run", project_id="project", status="ACTIVE"))
        session.add(
            GpuDeviceRow(
                uuid=GPU_A,
                last_index=0,
                name="RTX",
                capability="12.0",
                total_vram_mb=24_000,
                compatible=True,
            )
        )
    manager = GpuLeaseManager(database, (GPU_A,))
    old = datetime(2026, 1, 1, tzinfo=UTC)
    lease = manager.acquire(
        gpu_uuid=GPU_A,
        task=GpuTaskKind.TRAIN,
        run_id="run",
        worker_id="worker",
        pid=424242,
        estimated_vram_mb=12_000,
        now=old,
    )
    with pytest.raises(LeaseConflict):
        manager.acquire(
            gpu_uuid=GPU_A,
            task=GpuTaskKind.SAMPLE,
            run_id="run",
            worker_id="other",
            pid=12,
            estimated_vram_mb=1,
        )
    assert manager.heartbeat(lease.lease_id, worker_id="worker", now=old).worker_id == "worker"
    recovered = manager.recover_stale(
        heartbeat_timeout=timedelta(seconds=30),
        now=old + timedelta(minutes=1),
        process_alive=lambda pid: pid != 424242,
    )
    assert recovered == (lease.lease_id,)
    assert manager.active() == ()


def test_archaudit_blackwell_requires_arch_and_cuda_128() -> None:
    incompatible = evaluate_runtime_compatibility(
        capability_major=12,
        capability_minor=0,
        torch_version="2.6.0",
        torch_cuda_version="12.6",
        torch_arch_list=("sm_90",),
        tensor_smoke_ok=True,
        bf16_smoke_ok=True,
    )
    assert not incompatible.compatible
    assert any("12.8" in reason for reason in incompatible.reasons)

    compatible = evaluate_runtime_compatibility(
        capability_major=12,
        capability_minor=0,
        torch_version="2.8.0",
        torch_cuda_version="12.8",
        torch_arch_list=("sm_90", "sm_120"),
        tensor_smoke_ok=True,
        bf16_smoke_ok=True,
    )
    assert compatible.compatible


def test_archaudit_doctor_runs_all_probes_in_selected_gpu_environment(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], Path | None, str]] = []

    def runner(
        argv: tuple[str, ...],
        cwd: Path | None,
        environment: dict[str, str],
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, cwd, environment["CUDA_VISIBLE_DEVICES"]))
        script = argv[-1]
        if "torch.cuda" in script:
            payload = {
                "torch_version": "2.8.0",
                "torch_cuda_version": "12.8",
                "cuda_available": True,
                "arch_list": ["sm_120"],
                "tensor_smoke_ok": True,
                "bf16_supported": True,
                "bf16_smoke_ok": True,
                "capability_major": 12,
                "capability_minor": 0,
            }
        elif "onnxruntime" in script:
            payload = {
                "onnxruntime_version": "1.22.0",
                "providers": ["CUDAExecutionProvider"],
                "cuda_provider": True,
                "cuda_transfer_smoke_ok": True,
            }
        else:
            payload = {"module": "sdxl_train_network", "file": "sdxl_train_network.py"}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload) + "\n", stderr="")

    binding = bind_gpu_for_child(
        GPU_A,
        selected_uuids=(GPU_A,),
        devices=(_device(GPU_A, 3),),
        base_environment={},
    )
    report = inspect_runtime(
        python_executable=tmp_path / "python.exe",
        binding=binding,
        sd_scripts_root=tmp_path,
        runner=runner,
    )
    assert report.ready
    assert report.compatibility is not None and report.compatibility.compatible
    assert len(calls) == 3
    assert {visible for _, _, visible in calls} == {"3"}
    assert calls[-1][1] == tmp_path


def test_archaudit_runtime_manifest_round_trip_is_hashed_atomically(tmp_path: Path) -> None:
    lock_hash = hashlib.sha256(b"lock").hexdigest()
    manifest = ManagedRuntimeManifest(
        runtime_id="sdxl-test",
        python_version="3.10.14",
        python_executable=tmp_path / "python.exe",
        sd_scripts_root=tmp_path / "sd-scripts",
        sd_scripts=BackendPin(
            repository="https://github.com/kohya-ss/sd-scripts",
            release="v0.9.1",
            commit_sha="a" * 40,
        ),
        torch_version="2.8.0",
        torch_cuda_version="12.8",
        onnxruntime_version="1.22.0",
        packages_lock_sha256=lock_hash,
    )
    path = tmp_path / "backend-manifest.json"
    digest = write_runtime_manifest(path, manifest)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_runtime_manifest(path) == manifest
