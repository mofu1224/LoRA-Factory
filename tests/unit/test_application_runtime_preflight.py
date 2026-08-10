"""Regression coverage for GUI setup repair and selected-GPU REAL preflight."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lora_factory.application.service import LoRAFactoryController
from lora_factory.application.setup_status import build_setup_checks
from lora_factory.config.models import (
    AppSettings,
    BackendMode,
    PresetKind,
    ProjectConfig,
    TrainingPlan,
)
from lora_factory.core.cancellation import CancellationToken
from lora_factory.core.context import PipelineContext
from lora_factory.core.events import EventBus
from lora_factory.core.stage import PipelineStage
from lora_factory.gpu.models import GpuBinding, GpuDevice
from lora_factory.project.layout import ProjectLayout
from lora_factory.runtime.compatibility_matrix import CompatibilityDecision
from lora_factory.runtime.doctor import ProbeResult, RuntimeDoctorReport
from lora_factory.runtime.validation_record import RuntimeValidationRecord

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_GPU_UUID = "GPU-ffffffff-1111-2222-3333-444444444444"


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )


def _gpu(
    uuid: str = GPU_UUID,
    *,
    index: int = 3,
    compatible: bool = True,
    free_vram_mb: int = 12_000,
) -> GpuDevice:
    return GpuDevice(
        uuid=uuid,
        index=index,
        name="RTX Test",
        total_vram_mb=16_384,
        free_vram_mb=free_vram_mb,
        capability_major=12,
        capability_minor=0,
        compatible=compatible,
    )


def _doctor_report(*, ready: bool = True, gpu_uuid: str = GPU_UUID) -> RuntimeDoctorReport:
    torch = ProbeResult(
        name="torch_cuda",
        ok=True,
        argv=("python.exe", "-c", "probe"),
        return_code=0,
    )
    onnx = ProbeResult(
        name="onnx_cuda",
        ok=ready,
        argv=("python.exe", "-c", "probe"),
        return_code=0 if ready else 1,
        details={"cuda_provider": ready, "cuda_transfer_smoke_ok": ready},
        error="" if ready else "CUDA device transfer failed",
    )
    sd_scripts = ProbeResult(
        name="sd_scripts_import",
        ok=True,
        argv=("python.exe", "-c", "probe"),
        return_code=0,
    )
    compatibility = CompatibilityDecision(
        compatible=True,
        compute_capability="12.0",
        required_arch="sm_120",
        torch_version="2.13.0",
        torch_cuda_version="13.0",
        reasons=("Installed runtime passed capability probes",),
    )
    return RuntimeDoctorReport(
        gpu_uuid=gpu_uuid,
        torch=torch,
        onnx=onnx,
        sd_scripts=sd_scripts,
        compatibility=compatibility,
        ready=ready,
    )


def _plan() -> TrainingPlan:
    return TrainingPlan(
        resolution=768,
        batch_size=1,
        gradient_accumulation=1,
        repeats=1,
        epochs=1,
        estimated_steps=8,
        network_dim=16,
        network_alpha=8,
        unet_lr=1e-4,
        text_encoder_lr=5e-5,
        optimizer="AdamW8bit",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=False,
        training_gpu_uuid=GPU_UUID,
        estimated_disk_mb=1024,
    )


def _context(tmp_path: Path, *, mode: BackendMode) -> PipelineContext:
    base_model = tmp_path / "base.safetensors"
    base_model.write_bytes(b"test")
    source = tmp_path / "source"
    source.mkdir()
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    return PipelineContext(
        run_id="run-1",
        config=ProjectConfig(
            project_id="Runtime preflight",
            lora_name="Runtime preflight",
            preset=PresetKind.CHARACTER,
            trigger_token="runtime_test",  # noqa: S106 - domain token, not a credential.
            base_model=base_model,
            input_paths=(source,),
            selected_gpu_uuids=(GPU_UUID,),
            output_root=tmp_path / "output",
            backend_mode=mode,
        ),
        layout=layout,
        events=EventBus(),
        cancellation=CancellationToken(),
        artifacts={PipelineStage.PLANNING.value: {"plan": _plan().model_dump(mode="json")}},
    )


def test_gui_runtime_repair_runs_deep_probe_without_reinstall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = LoRAFactoryController(_settings(tmp_path))
    monkeypatch.setattr(controller.runtime, "installed", lambda: True)
    monkeypatch.setattr(controller.runtime, "source_matches_manifest", lambda: True)

    def unexpected_install(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an already-complete runtime must not be reinstalled")

    monkeypatch.setattr(
        "lora_factory.application.service.ManagedRuntimeInstaller.install",
        unexpected_install,
    )
    monkeypatch.setattr(
        "lora_factory.application.service.discover_nvidia_gpus",
        lambda: (
            _gpu(
                OTHER_GPU_UUID,
                index=0,
                compatible=False,
                free_vram_mb=15_000,
            ),
            _gpu(),
        ),
    )
    bindings: list[GpuBinding] = []

    def doctor(**kwargs: Any) -> RuntimeDoctorReport:
        bindings.append(kwargs["binding"])
        return _doctor_report()

    records: list[RuntimeValidationRecord] = []
    monkeypatch.setattr("lora_factory.application.service.inspect_runtime", doctor)
    monkeypatch.setattr(controller.runtime, "write_validation_record", records.append)

    controller.repair_setup("Managed Training Runtime")

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.uuid == GPU_UUID
    assert binding.physical_index == 3
    assert binding.environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert len(records) == 1 and records[0].ready
    assert records[0].gpu_uuid == GPU_UUID


def test_gui_runtime_repair_installs_missing_runtime_before_deep_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = LoRAFactoryController(_settings(tmp_path))
    state = {"installed": False, "install_calls": 0}
    monkeypatch.setattr(controller.runtime, "installed", lambda: state["installed"])
    monkeypatch.setattr(controller.runtime, "source_matches_manifest", lambda: state["installed"])

    def install(
        _installer: object,
        _cancellation: CancellationToken,
        progress: Any,
    ) -> None:
        state["install_calls"] += 1
        state["installed"] = True
        progress("complete", 1.0, "installed")

    monkeypatch.setattr("lora_factory.application.service.ManagedRuntimeInstaller.install", install)
    monkeypatch.setattr("lora_factory.application.service.discover_nvidia_gpus", lambda: (_gpu(),))
    monkeypatch.setattr(
        "lora_factory.application.service.inspect_runtime",
        lambda **_kwargs: _doctor_report(),
    )
    records: list[RuntimeValidationRecord] = []
    monkeypatch.setattr(controller.runtime, "write_validation_record", records.append)

    controller.repair_setup("Managed Training Runtime")

    assert state["install_calls"] == 1
    assert records and records[0].ready
    assert (controller.settings.managed_runtime_root / "install-progress.json").is_file()


def test_real_preflight_probes_exact_planned_gpu_and_blocks_not_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = LoRAFactoryController(_settings(tmp_path))
    context = _context(tmp_path, mode=BackendMode.REAL)
    monkeypatch.setattr(controller.runtime, "installed", lambda: True)
    monkeypatch.setattr(controller.runtime, "source_matches_manifest", lambda: True)
    monkeypatch.setattr(
        "lora_factory.application.service.inspect_sdxl_safetensors",
        lambda _path: SimpleNamespace(
            is_sdxl=True,
            model_dump=lambda **_kwargs: {"is_sdxl": True, "sha256": "0" * 64},
        ),
    )
    monkeypatch.setattr(
        "lora_factory.application.service.discover_nvidia_gpus",
        lambda: (_gpu(index=7),),
    )
    bindings: list[GpuBinding] = []

    def doctor(**kwargs: Any) -> RuntimeDoctorReport:
        bindings.append(kwargs["binding"])
        return _doctor_report(ready=False)

    records: list[RuntimeValidationRecord] = []
    monkeypatch.setattr("lora_factory.application.service.inspect_runtime", doctor)
    monkeypatch.setattr(controller.runtime, "write_validation_record", records.append)

    with pytest.raises(RuntimeError, match="CUDA device transfer failed"):
        controller._preflight_stage(context)

    assert bindings[0].uuid == _plan().training_gpu_uuid
    assert bindings[0].physical_index == 7
    assert records and not records[0].ready


def test_fake_preflight_never_discovers_or_probes_nvidia(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = LoRAFactoryController(_settings(tmp_path))
    context = _context(tmp_path, mode=BackendMode.FAKE)
    monkeypatch.setattr(
        "lora_factory.application.service.inspect_sdxl_safetensors",
        lambda _path: SimpleNamespace(
            is_sdxl=True,
            model_dump=lambda **_kwargs: {"is_sdxl": True, "sha256": "0" * 64},
        ),
    )

    def unexpected_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("FAKE preflight must not touch NVIDIA runtime discovery or probes")

    monkeypatch.setattr("lora_factory.application.service.discover_nvidia_gpus", unexpected_call)
    monkeypatch.setattr("lora_factory.application.service.inspect_runtime", unexpected_call)

    output = controller._preflight_stage(context)

    assert output["ready"] is True
    assert output["runtime_validation"] is None


def test_installed_runtime_without_deep_record_remains_gui_repairable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = LoRAFactoryController(_settings(tmp_path))
    controller.runtime.layout.python.parent.mkdir(parents=True, exist_ok=True)
    controller.runtime.layout.python.write_text("managed", encoding="utf-8")
    controller.runtime.write_validation_record(
        RuntimeValidationRecord.after_install(
            str(controller.runtime.manifest["profile_id"]),
            str(controller.runtime.layout.python),
        )
    )
    monkeypatch.setattr(controller.runtime, "installed", lambda: True)
    monkeypatch.setattr(controller.runtime, "source_matches_manifest", lambda: True)

    def runner(argv: tuple[str, ...], _timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    checks = build_setup_checks(
        settings=controller.settings,
        runtime=controller.runtime,
        destinations=(),
        which=lambda command: f"C:/tools/{command}.exe",
        runner=runner,
        gpu_discovery=lambda: (_gpu(),),
    )
    runtime_check = next(item for item in checks if item["name"] == "Managed Training Runtime")

    assert runtime_check["status"] == "needs setup"
    assert runtime_check["repairable"] is True
