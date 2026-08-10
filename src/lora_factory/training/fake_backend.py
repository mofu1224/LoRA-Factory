"""Deterministic Fake trainer producing valid resumable safetensors checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from lora_factory.core.cancellation import CancellationToken
from lora_factory.training.backend import (
    CheckpointArtifact,
    ProgressCallback,
    TrainingProgress,
    TrainingRequest,
    TrainingResult,
)
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json, write_json_atomic


class FakeTrainingBackend:
    @property
    def version(self) -> str:
        return "fake-trainer/1"

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult:
        attempt_root = request.run_directory / f"attempt-{request.attempt:03d}"
        checkpoints_root = attempt_root / "checkpoints"
        states_root = attempt_root / "states"
        logs_root = attempt_root / "logs"
        for directory in (attempt_root, checkpoints_root, states_root, logs_root):
            directory.mkdir(parents=True, exist_ok=True)
        state_path = states_root / "training-state.json"
        log_path = logs_root / "training.jsonl"
        start_step = self._resume_step(request.resume_state)
        resumed = start_step > 0
        total_steps = request.plan.estimated_steps
        target_epochs = sorted(
            {
                max(1, math.ceil(request.plan.epochs * fraction))
                for fraction in (0.25, 0.5, 0.75, 1.0)
            }
        )
        checkpoints: list[CheckpointArtifact] = []
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            for epoch in range(1, request.plan.epochs + 1):
                step = max(1, math.ceil(total_steps * epoch / request.plan.epochs))
                if step <= start_step:
                    continue
                cancellation.raise_if_cancelled()
                train_loss = max(0.02, 0.65 * math.exp(-2.2 * step / total_steps))
                validation_loss = (
                    max(0.03, 0.58 * math.exp(-1.7 * step / total_steps) + epoch * 0.002)
                    if request.plan.validation_enabled
                    else None
                )
                entry = {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
                log_handle.write(json.dumps(entry, sort_keys=True) + "\n")
                log_handle.flush()
                write_json_atomic(
                    state_path,
                    {
                        "backend": self.version,
                        "run_id": request.run_id,
                        "attempt": request.attempt,
                        "completed_step": step,
                        "total_steps": total_steps,
                        "epoch": epoch,
                        "optimizer_state_preserved": True,
                        "seed": request.seed,
                    },
                )
                checkpoint_path: Path | None = None
                if epoch in target_epochs:
                    checkpoint_path = (
                        checkpoints_root / f"{request.output_name}-e{epoch:03d}.safetensors"
                    )
                    checkpoint = self._write_checkpoint(
                        request=request,
                        path=checkpoint_path,
                        epoch=epoch,
                        step=step,
                        train_loss=train_loss,
                        validation_loss=validation_loss,
                    )
                    checkpoints.append(checkpoint)
                progress(
                    TrainingProgress(
                        epoch=epoch,
                        step=step,
                        total_steps=total_steps,
                        train_loss=train_loss,
                        validation_loss=validation_loss,
                        checkpoint=checkpoint_path,
                        message=f"Fake training epoch {epoch}/{request.plan.epochs}",
                    )
                )
        finally:
            log_handle.close()

        if not checkpoints:
            existing = sorted(checkpoints_root.glob("*.safetensors"))
            checkpoints = [self._read_checkpoint(path) for path in existing]
        return TrainingResult(
            checkpoints=tuple(checkpoints),
            state_path=state_path,
            log_path=log_path,
            completed_steps=total_steps,
            resumed=resumed,
        )

    @staticmethod
    def _resume_step(state_path: Path | None) -> int:
        if state_path is None:
            return 0
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume state does not exist: {state_path}")
        payload = read_json(state_path)
        if not isinstance(payload, dict) or not payload.get("optimizer_state_preserved"):
            raise ValueError("Resume state is invalid or lacks optimizer state")
        return int(payload.get("completed_step", 0))

    def _write_checkpoint(
        self,
        *,
        request: TrainingRequest,
        path: Path,
        epoch: int,
        step: int,
        train_loss: float,
        validation_loss: float | None,
    ) -> CheckpointArtifact:
        generator = np.random.default_rng(request.seed + epoch)
        tensors = {
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight": generator.normal(
                0, 0.01, (request.plan.network_dim, 8)
            ).astype(np.float32),
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight": generator.normal(
                0, 0.01, (8, request.plan.network_dim)
            ).astype(np.float32),
        }
        metadata = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(request.plan.network_dim),
            "ss_network_alpha": str(request.plan.network_alpha),
            "ss_epoch": str(epoch),
            "ss_steps": str(step),
            "lora_factory_backend": self.version,
            "lora_factory_train_loss": f"{train_loss:.8f}",
        }
        if validation_loss is not None:
            metadata["lora_factory_validation_loss"] = f"{validation_loss:.8f}"
        save_file(tensors, path, metadata=metadata)
        return CheckpointArtifact(
            checkpoint_id=f"attempt-{request.attempt}-epoch-{epoch}",
            path=path,
            epoch=epoch,
            step=step,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            train_loss=train_loss,
            validation_loss=validation_loss,
            metadata=metadata,
        )

    @staticmethod
    def _read_checkpoint(path: Path) -> CheckpointArtifact:
        from lora_factory.training.checkpoints import inspect_checkpoint

        return inspect_checkpoint(path)
