from __future__ import annotations

from pathlib import Path

import pytest

from lora_factory.config.models import TrainingPlan
from lora_factory.training.resume import prepare_training_attempt, training_request_fingerprint

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _plan() -> TrainingPlan:
    return TrainingPlan(
        resolution=768,
        batch_size=1,
        gradient_accumulation=2,
        repeats=2,
        epochs=4,
        estimated_steps=32,
        network_dim=16,
        network_alpha=8,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=False,
        training_gpu_uuid=GPU_UUID,
        estimated_disk_mb=256,
    )


def _fingerprint(dataset: Path, *, seed: int = 42) -> str:
    return training_request_fingerprint(
        run_id="run-1",
        output_name="safe-name",
        base_model_sha256="a" * 64,
        dataset_config=dataset,
        plan=_plan(),
        seed=seed,
        backend_version="fake-trainer/1",
    )


def test_archaudit_resume_uses_matching_state_in_a_fresh_attempt(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("fixture = true\n", encoding="utf-8")
    run = tmp_path / "run"
    fingerprint = _fingerprint(dataset)

    first = prepare_training_attempt(run_directory=run, request_fingerprint=fingerprint)
    state = first.record_path.parent / "states" / "training-state.json"
    state.parent.mkdir()
    state.write_text('{"optimizer_state_preserved": true}', encoding="utf-8")
    checkpoint = first.record_path.parent / "checkpoints" / "kept.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"must not be overwritten")

    second = prepare_training_attempt(run_directory=run, request_fingerprint=fingerprint)

    assert second.attempt == 2
    assert second.resume_state == state
    assert second.resumed_from_attempt == 1
    assert second.record_path.parent != first.record_path.parent
    assert checkpoint.read_bytes() == b"must not be overwritten"


def test_archaudit_resume_rejects_incompatible_state_and_bounds_attempts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("fixture = true\n", encoding="utf-8")
    run = tmp_path / "run"
    first = prepare_training_attempt(
        run_directory=run,
        request_fingerprint=_fingerprint(dataset, seed=1),
    )
    state = first.record_path.parent / "states" / "training-state.json"
    state.parent.mkdir()
    state.write_text("{}", encoding="utf-8")

    incompatible = _fingerprint(dataset, seed=2)
    second = prepare_training_attempt(
        run_directory=run,
        request_fingerprint=incompatible,
    )
    third = prepare_training_attempt(
        run_directory=run,
        request_fingerprint=incompatible,
    )

    assert second.resume_state is None
    assert third.resume_state is None
    with pytest.raises(RuntimeError, match="Maximum training attempts"):
        prepare_training_attempt(
            run_directory=run,
            request_fingerprint=incompatible,
        )
