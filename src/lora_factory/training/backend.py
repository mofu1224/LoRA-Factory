"""Training backend contract shared by managed sd-scripts and Fake training."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.config.models import TrainingPlan
from lora_factory.core.cancellation import CancellationToken


class TrainingRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    attempt: int = Field(ge=1, le=3)
    output_name: str
    base_model: Path
    dataset_config: Path
    run_directory: Path
    plan: TrainingPlan
    seed: int = 42
    resume_state: Path | None = None


class CheckpointArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_id: str
    path: Path
    epoch: int = Field(ge=1)
    step: int = Field(ge=1)
    size_bytes: int = Field(gt=0)
    sha256: str
    train_loss: float | None = None
    validation_loss: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class TrainingProgress(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    total_steps: int = Field(ge=1)
    train_loss: float | None = None
    validation_loss: float | None = None
    checkpoint: Path | None = None
    message: str = ""


class TrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoints: tuple[CheckpointArtifact, ...]
    state_path: Path
    log_path: Path
    completed_steps: int = Field(ge=0)
    resumed: bool
    command_argv: tuple[str, ...] = Field(default_factory=tuple)


ProgressCallback = Callable[[TrainingProgress], None]


class TrainingBackend(Protocol):
    @property
    def version(self) -> str: ...

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult: ...
