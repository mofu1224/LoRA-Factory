"""Regression tests for cached setup validation and throttled progress telemetry."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from lora_factory.application.progress_reporting import PipelineProgressReporter
from lora_factory.application.setup_status import build_setup_checks
from lora_factory.cli import doctor_command
from lora_factory.config.models import (
    AppSettings,
    DestinationConfig,
    DestinationKind,
)
from lora_factory.gpu.models import GpuDevice, GpuTelemetrySample
from lora_factory.runtime.manager import RuntimeManager
from lora_factory.runtime.validation_record import (
    RuntimeValidationCheck,
    RuntimeValidationRecord,
    ValidationStatus,
)

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _manager(tmp_path: Path) -> RuntimeManager:
    return RuntimeManager(tmp_path / "runtime", Path("backend-manifest.json"))


def _ready_record(manager: RuntimeManager) -> RuntimeValidationRecord:
    ok = RuntimeValidationCheck(status=ValidationStatus.OK, detail="passed")
    return RuntimeValidationRecord(
        profile_id=str(manager.manifest["profile_id"]),
        gpu_uuid=GPU_UUID,
        ready=True,
        checks={
            "managed_python": ok,
            "pytorch_gpu": ok,
            "onnx_cuda": ok,
            "sd_scripts": ok,
        },
    )


def test_runtime_validation_record_round_trip_and_profile_guard(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    record = RuntimeValidationRecord.after_install(
        str(manager.manifest["profile_id"]), str(manager.layout.python)
    )
    manager.write_validation_record(record)
    assert manager.read_validation_record() == record
    with pytest.raises(ValueError, match="different compatibility profile"):
        manager.write_validation_record(record.model_copy(update={"profile_id": "wrong"}))


def test_setup_checks_use_login_status_and_cached_deep_validation(
    monkeypatch: Any, tmp_path: Path
) -> None:
    manager = _manager(tmp_path)
    manager.layout.python.parent.mkdir(parents=True)
    manager.layout.python.write_text("managed", encoding="utf-8")
    manager.write_validation_record(_ready_record(manager))
    monkeypatch.setattr(manager, "installed", lambda: True)
    monkeypatch.setattr(manager, "source_matches_manifest", lambda: True)
    destination = tmp_path / "stable-diffusion-webui"
    destination.mkdir()
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
        destinations=[
            DestinationConfig(
                kind=DestinationKind.A1111,
                root=destination,
            )
        ],
        codex_required=True,
    )
    commands: list[tuple[str, ...]] = []

    def which(command: str) -> str | None:
        return {
            "uv": "C:/tools/uv.exe",
            "git": "C:/tools/git.exe",
            "codex": "C:/tools/codex.exe",
            "nvidia-smi": "C:/Windows/nvidia-smi.exe",
        }.get(command)

    def runner(argv: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
        assert timeout == 10.0
        commands.append(argv)
        stdout = "Logged in" if argv[0].endswith("codex.exe") else "555.42"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    gpu = GpuDevice(
        uuid=GPU_UUID,
        index=0,
        name="RTX Test",
        total_vram_mb=16_384,
        free_vram_mb=12_000,
        capability_major=12,
        capability_minor=0,
    )
    checks = build_setup_checks(
        settings=settings,
        runtime=manager,
        destinations=tuple(settings.destinations),
        which=which,
        runner=runner,
        gpu_discovery=lambda: (gpu,),
    )
    by_name = {str(item["name"]): item for item in checks}
    assert by_name["Codex Authentication"]["status"] == "ok"
    assert by_name["Managed Python"]["status"] == "ok"
    assert by_name["NVIDIA Driver"]["detail"] == "555.42"
    assert by_name["NVML"]["status"] == "ok"
    assert by_name["PyTorch GPU Smoke Test"]["status"] == "ok"
    assert by_name["ONNX Runtime CUDA Provider"]["status"] == "ok"
    assert by_name["sd-scripts backend"]["status"] == "ok"
    assert by_name["AUTOMATIC1111 destination"]["status"] == "ok"
    assert by_name["Forge destination"]["status"] == "optional"
    assert by_name["Forge destination"]["required"] is False
    assert commands == [
        ("C:/tools/codex.exe", "login", "status"),
        (
            "C:/Windows/nvidia-smi.exe",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
    ]


def test_progress_reporter_throttles_selected_gpu_telemetry() -> None:
    uuids = (GPU_UUID, "GPU-ffffffff-1111-2222-3333-444444444444")
    samples = [0]
    clock_values = iter((0.0, 1.0, 6.0))

    def sampler(selected: Sequence[str]) -> tuple[GpuTelemetrySample, ...]:
        samples[0] += 1
        return tuple(
            GpuTelemetrySample(
                uuid=uuid,
                utilization_percent=25.0 + index,
                memory_used_mb=4000 + index,
                memory_free_mb=12_000 - index,
                temperature_c=50.0,
                power_watts=120.0,
            )
            for index, uuid in enumerate(selected)
        )

    reporter = PipelineProgressReporter(
        uuids,
        interval_seconds=5.0,
        fake_backend=False,
        sampler=sampler,
        clock=lambda: next(clock_values),
    )
    first = reporter.details(
        current_task="TRAINING",
        images_current=8,
        images_total=8,
        active_gpu_uuid=GPU_UUID,
    )
    second = reporter.details(
        current_task="TRAINING",
        images_current=8,
        images_total=8,
        active_gpu_uuid=GPU_UUID,
    )
    third = reporter.details(
        current_task="SAMPLING",
        images_current=1,
        images_total=3,
        active_gpu_uuid=GPU_UUID,
    )
    assert samples[0] == 2
    assert first["gpus"][0]["task"] == "TRAINING"
    assert first["gpus"][1]["task"] == "Idle"
    assert second["gpus"][0]["utilization_percent"] == 25.0
    assert third["images_current"] == 1
    assert third["images_total"] == 3


def test_doctor_strict_ignores_optional_setup_failures(monkeypatch: Any, tmp_path: Path) -> None:
    class Controller:
        def __init__(self, _settings: AppSettings) -> None:
            self.runtime = object()

        def setup_checks(self) -> tuple[dict[str, object], ...]:
            return (
                {
                    "name": "Optional destination",
                    "status": "error",
                    "detail": "Not installed",
                    "required": False,
                },
            )

    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    monkeypatch.setattr("lora_factory.cli._settings_with_overrides", lambda **_kwargs: settings)
    monkeypatch.setattr("lora_factory.cli.LoRAFactoryController", Controller)
    doctor_command(
        deep=False,
        gpu_uuid=None,
        json_output=False,
        strict=True,
        managed_runtime_root=None,
    )


def test_doctor_deep_strict_refreshes_setup_after_ready_probe(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class ReadyReport:
        ready = True

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"ready": True}

    class Runtime:
        def __init__(self) -> None:
            self.manifest = {"profile_id": "test-profile"}

        def write_validation_record(self, _record: object) -> None:
            return None

    class Controller:
        def __init__(self, _settings: AppSettings) -> None:
            self.runtime = Runtime()
            self.calls = 0

        def setup_checks(self) -> tuple[dict[str, object], ...]:
            self.calls += 1
            return (
                {
                    "name": "Managed Training Runtime",
                    "status": "needs setup" if self.calls == 1 else "ok",
                    "detail": "cached",
                    "required": True,
                },
            )

    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = Controller(settings)
    monkeypatch.setattr("lora_factory.cli._settings_with_overrides", lambda **_kwargs: settings)
    monkeypatch.setattr("lora_factory.cli.LoRAFactoryController", lambda _settings: controller)
    monkeypatch.setattr("lora_factory.cli._run_runtime_doctor", lambda *_args: ReadyReport())
    monkeypatch.setattr(
        "lora_factory.cli.RuntimeValidationRecord.from_doctor_report",
        lambda *_args: object(),
    )
    doctor_command(
        deep=True,
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
        json_output=False,
        strict=True,
        managed_runtime_root=None,
    )
    assert controller.calls == 2
