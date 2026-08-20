from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lora_factory.config.models import GpuCapability, TrainingPlan
from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.core.exceptions import ErrorClassification
from lora_factory.training.backend import TrainingProgress, TrainingRequest
from lora_factory.training.checkpoints import index_checkpoints, inspect_checkpoint
from lora_factory.training.fake_backend import FakeTrainingBackend
from lora_factory.training.log_parser import parse_progress_line
from lora_factory.training.process_manager import ProcessManager
from lora_factory.training.recovery import next_recovery


def make_plan(**updates: object) -> TrainingPlan:
    payload: dict[str, object] = {
        "resolution": 1024,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "repeats": 5,
        "epochs": 8,
        "estimated_steps": 80,
        "network_dim": 32,
        "network_alpha": 16,
        "unet_lr": 0.0001,
        "text_encoder_lr": 0.00001,
        "optimizer": "AdamW8bit",
        "precision": "bf16",
        "keep_tokens": 2,
        "validation_enabled": True,
        "training_gpu_uuid": "GPU-11111111-abcd",
        "estimated_disk_mb": 512,
    }
    payload.update(updates)
    return TrainingPlan.model_validate(payload)


def make_request(tmp_path: Path, **updates: object) -> TrainingRequest:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "attempt": 1,
        "output_name": "Example",
        "base_model": tmp_path / "base.safetensors",
        "dataset_config": tmp_path / "dataset.toml",
        "run_directory": tmp_path / "run",
        "plan": make_plan(),
        "seed": 123,
    }
    payload.update(updates)
    return TrainingRequest.model_validate(payload)


def test_fake_trainer_emits_valid_comparable_checkpoints_and_state(tmp_path: Path) -> None:
    progress_events = []

    result = FakeTrainingBackend().train(
        make_request(tmp_path),
        CancellationToken(),
        progress_events.append,
    )

    assert len(result.checkpoints) == 4
    assert result.completed_steps == 80
    assert result.state_path.is_file()
    assert result.log_path.is_file()
    assert len(progress_events) == 8
    for checkpoint in result.checkpoints:
        inspected = inspect_checkpoint(checkpoint.path)
        assert inspected.sha256 == checkpoint.sha256
        assert inspected.metadata["ss_network_module"] == "networks.lora"
    assert len(index_checkpoints(result.checkpoints[0].path.parent)) == 4


def test_fake_trainer_cancel_persists_optimizer_state_and_resume_continues(tmp_path: Path) -> None:
    cancellation = CancellationToken()

    def cancel_after_third_epoch(progress: TrainingProgress) -> None:
        if progress.epoch == 3:
            cancellation.cancel()

    with pytest.raises(CancelledError):
        FakeTrainingBackend().train(
            make_request(tmp_path),
            cancellation,
            cancel_after_third_epoch,
        )
    state = tmp_path / "run" / "attempt-001" / "states" / "training-state.json"
    assert state.is_file()

    resumed = FakeTrainingBackend().train(
        make_request(tmp_path, attempt=2, resume_state=state),
        CancellationToken(),
        lambda _progress: None,
    )

    assert resumed.resumed
    assert resumed.completed_steps == 80
    assert resumed.checkpoints
    assert min(item.step for item in resumed.checkpoints) > 30


def test_progress_log_parser_handles_sd_scripts_style_line() -> None:
    parsed = parse_progress_line(
        "epoch 3 step 120/400 loss=0.123 val_loss=0.151",
        default_total_steps=500,
    )

    assert parsed is not None
    assert parsed.epoch == 3
    assert parsed.step == 120
    assert parsed.total_steps == 400
    assert parsed.train_loss == pytest.approx(0.123)
    assert parsed.validation_loss == pytest.approx(0.151)


def test_progress_log_parser_handles_pinned_epoch_validation_tqdm_line() -> None:
    parsed = parse_progress_line(
        "epoch validation steps: 8/8 val_epoch_avg_loss=0.2468 timestep=800",
        default_total_steps=6,
    )

    assert parsed is not None
    assert parsed.step == 8
    assert parsed.total_steps == 8
    assert parsed.train_loss is None
    assert parsed.validation_loss == pytest.approx(0.2468)


def test_progress_log_parser_uses_tqdm_count_instead_of_percentage() -> None:
    parsed = parse_progress_line(
        "steps: 100%|██████████| 6/6 [00:51<00:00, 8.51s/it, avr_loss=0.0438]",
        default_total_steps=6,
    )

    assert parsed is not None
    assert parsed.step == 6
    assert parsed.total_steps == 6
    assert parsed.train_loss == pytest.approx(0.0438)
    assert parsed.validation_loss is None


def test_oom_recovery_uses_only_selected_alternative_then_preserves_effective_batch() -> None:
    plan = make_plan()
    selected = (
        GpuCapability(
            uuid=plan.training_gpu_uuid,
            index=0,
            name="GPU A",
            total_vram_mb=16000,
            free_vram_mb=2000,
            capability_major=12,
            capability_minor=0,
            bf16_supported=True,
        ),
        GpuCapability(
            uuid="GPU-22222222-abcd",
            index=1,
            name="GPU B",
            total_vram_mb=16000,
            free_vram_mb=15500,
            capability_major=12,
            capability_minor=0,
            bf16_supported=True,
        ),
    )

    gpu_fallback = next_recovery(
        classification=ErrorClassification.CUDA_OOM,
        plan=plan,
        selected_gpus=selected,
        locked_fields=frozenset(),
        previous_changes=(),
        attempt=1,
    )
    batch_fallback = next_recovery(
        classification=ErrorClassification.CUDA_OOM,
        plan=gpu_fallback.plan,
        selected_gpus=selected,
        locked_fields=frozenset(),
        previous_changes=("gpu_fallback",),
        attempt=2,
    )

    assert gpu_fallback.plan.training_gpu_uuid == "GPU-22222222-abcd"
    assert batch_fallback.plan.batch_size == 1
    assert batch_fallback.plan.gradient_accumulation == 2


def test_locked_oom_plan_stops_before_quality_changes() -> None:
    plan = make_plan(batch_size=1, gradient_checkpointing=True, optimizer="AdamW")
    decision = next_recovery(
        classification=ErrorClassification.CUDA_OOM,
        plan=plan,
        selected_gpus=(),
        locked_fields=frozenset({"network_dim", "resolution"}),
        previous_changes=(),
        attempt=1,
    )
    assert not decision.recoverable


def test_process_manager_captures_separate_streams_without_shell(tmp_path: Path) -> None:
    lines: list[tuple[str, str]] = []
    result = ProcessManager().run(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); print('warning', file=sys.stderr)",
        ],
        cwd=tmp_path,
        environment={},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        cancellation=CancellationToken(),
        timeout_seconds=10,
        on_line=lambda channel, line: lines.append((channel, line)),
    )

    assert result.return_code == 0
    assert not result.cancelled
    assert ("stdout", "hello") in lines
    assert ("stderr", "warning") in lines
    assert result.stdout_path.read_text(encoding="utf-8").strip() == "hello"
    assert result.stderr_path.read_text(encoding="utf-8").strip() == "warning"


def test_process_manager_streams_carriage_return_progress_lines(tmp_path: Path) -> None:
    lines: list[tuple[str, str]] = []
    result = ProcessManager().run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('steps: 1/2\\rsteps: 2/2\\r'); sys.stdout.flush()",
        ],
        cwd=tmp_path,
        environment={},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        cancellation=CancellationToken(),
        timeout_seconds=10,
        on_line=lambda channel, line: lines.append((channel, line)),
    )

    assert result.return_code == 0
    assert [line.strip() for channel, line in lines if channel == "stdout"] == [
        "steps: 1/2",
        "steps: 2/2",
    ]
