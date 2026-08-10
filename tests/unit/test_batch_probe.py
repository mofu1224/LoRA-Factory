from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lora_factory.config.models import (
    AdvancedOverrides,
    DatasetStatistics,
    GpuCapability,
    PresetKind,
    ProjectConfig,
)
from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.batch_probe import (
    BatchProbeError,
    BatchProbeRequest,
    FakeBatchProbe,
    TorchCudaBatchProbe,
    descending_batch_candidates,
)
from lora_factory.training.planner import TrainingPlanningError, plan_training
from lora_factory.training.process_manager import ManagedProcessResult
from lora_factory.training.profiles import load_preset_profile

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _gpu(*, free_vram_mb: int = 10_000) -> GpuCapability:
    return GpuCapability(
        uuid=GPU_UUID,
        index=0,
        name="Test GPU",
        total_vram_mb=16_384,
        free_vram_mb=free_vram_mb,
        capability_major=12,
        capability_minor=0,
        bf16_supported=True,
    )


def _project(
    tmp_path: Path,
    *,
    advanced: AdvancedOverrides | None = None,
) -> ProjectConfig:
    return ProjectConfig(
        project_id="batch-probe",
        lora_name="Batch Probe",
        preset=PresetKind.CHARACTER,
        trigger_token="lf_batch_probe",  # noqa: S106 - domain trigger token.
        base_model=tmp_path / "base.safetensors",
        input_paths=(tmp_path / "images",),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        advanced=advanced or AdvancedOverrides(),
    )


def _request(tmp_path: Path) -> BatchProbeRequest:
    return BatchProbeRequest(
        gpu_uuid=GPU_UUID,
        resolution=768,
        network_dim=32,
        precision="bf16",
        candidate_batch_sizes=(4, 2, 1),
        work_directory=tmp_path / "probe",
    )


def test_fake_batch_probe_descends_after_insufficient_live_vram(tmp_path: Path) -> None:
    result = FakeBatchProbe(free_vram_mb=10_000, total_vram_mb=16_384).probe(
        _request(tmp_path), CancellationToken()
    )

    assert result.selected_batch_size == 2
    assert [attempt.batch_size for attempt in result.attempts] == [4, 2]
    assert [attempt.succeeded for attempt in result.attempts] == [False, True]
    assert not result.training_artifacts_created
    assert descending_batch_candidates(7) == (7, 3, 1)


def test_fake_batch_probe_fails_when_even_batch_one_has_no_margin(tmp_path: Path) -> None:
    with pytest.raises(BatchProbeError, match="No candidate batch"):
        FakeBatchProbe(free_vram_mb=7_000, total_vram_mb=8_192).probe(
            _request(tmp_path), CancellationToken()
        )


class _RecordingProcessManager:
    def __init__(self) -> None:
        self.arguments: list[str] = []
        self.environment: dict[str, str] = {}

    def run(self, arguments: list[str], **kwargs: Any) -> ManagedProcessResult:
        self.arguments = list(arguments)
        self.environment = dict(kwargs["environment"])
        stdout_path = Path(kwargs["stdout_path"])
        stderr_path = Path(kwargs["stderr_path"])
        stdout_path.write_text(
            json.dumps({"free_vram_mb": 10_000, "total_vram_mb": 16_384}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return ManagedProcessResult(
            return_code=0,
            cancelled=False,
            timed_out=False,
            duration_seconds=0.01,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


def test_real_provider_uses_argument_array_and_selected_gpu_environment(tmp_path: Path) -> None:
    manager = _RecordingProcessManager()
    binding = GpuBinding(
        uuid=GPU_UUID,
        physical_index=2,
        environment={
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "2",
        },
    )
    backend = TorchCudaBatchProbe(
        python_executable=tmp_path / "managed-python.exe",
        binding=binding,
        selected_gpu_uuids=(GPU_UUID,),
        process_manager=manager,  # type: ignore[arg-type] - structural test double.
    )

    result = backend.probe(_request(tmp_path), CancellationToken())

    assert result.method == "torch_cuda_live_vram"
    assert result.selected_batch_size == 2
    assert manager.arguments[:2] == [str(tmp_path / "managed-python.exe"), "-c"]
    assert manager.environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert "shell" not in manager.arguments
    assert result.command_argv[2] == "<lora-factory bounded CUDA batch probe>"
    assert result.stdout_path is not None and result.stdout_path.is_file()


def test_planner_consumes_probe_result_but_never_changes_explicit_batch(
    tmp_path: Path,
) -> None:
    result = FakeBatchProbe(free_vram_mb=10_000, total_vram_mb=16_384).probe(
        _request(tmp_path), CancellationToken()
    )
    statistics = DatasetStatistics(
        accepted_count=20,
        short_side_p10=700,
        short_side_median=768,
        area_median=600_000,
    )
    profile = load_preset_profile(PresetKind.CHARACTER)
    capability = _gpu()
    plan = plan_training(
        _project(tmp_path),
        statistics,
        (capability,),
        profile,
        batch_probe_result=result,
    )

    assert plan.batch_size == 2
    assert plan.provenance["batch_size"] == "batch_probe:fake-batch-probe/1"
    with pytest.raises(TrainingPlanningError, match="explicit batch"):
        plan_training(
            _project(
                tmp_path,
                advanced=AdvancedOverrides(batch_size=4),
            ),
            statistics,
            (capability,),
            profile,
            batch_probe_result=result,
        )
