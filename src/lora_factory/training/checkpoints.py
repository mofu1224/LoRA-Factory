"""Checkpoint integrity and metadata inspection."""

from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

from lora_factory.training.backend import CheckpointArtifact
from lora_factory.util.hashing import sha256_file


def inspect_checkpoint(path: Path) -> CheckpointArtifact:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Checkpoint is missing or empty: {path}")
    try:
        with safe_open(path, framework="numpy") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise ValueError(f"Checkpoint is not valid safetensors: {path}") from exc
    if not keys:
        raise ValueError(f"Checkpoint contains no tensors: {path}")
    epoch = int(metadata.get("ss_epoch", 0))
    step = int(metadata.get("ss_steps", 0))
    if epoch < 1 or step < 1:
        raise ValueError(f"Checkpoint lacks valid epoch/step metadata: {path}")
    train_loss_text = metadata.get("lora_factory_train_loss")
    validation_text = metadata.get("lora_factory_validation_loss")
    return CheckpointArtifact(
        checkpoint_id=f"epoch-{epoch}-step-{step}",
        path=path,
        epoch=epoch,
        step=step,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        train_loss=float(train_loss_text) if train_loss_text is not None else None,
        validation_loss=float(validation_text) if validation_text is not None else None,
        metadata=dict(metadata),
    )


def index_checkpoints(directory: Path) -> tuple[CheckpointArtifact, ...]:
    valid: list[CheckpointArtifact] = []
    for path in sorted(directory.glob("*.safetensors")):
        try:
            valid.append(inspect_checkpoint(path))
        except ValueError:
            continue
    return tuple(valid)
