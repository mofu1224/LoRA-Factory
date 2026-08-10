"""Sampler port shared by real and deterministic fake implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class SampleRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_id: str
    checkpoint_path: Path
    base_model_path: Path
    prompt_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int
    weight: float = Field(ge=0, le=2)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=24, ge=1, le=100)
    cfg_scale: float = Field(default=5.0, ge=0, le=30)
    sampler: str = "euler_a"


class SampleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: SampleRequest
    image_path: Path
    metadata_path: Path
    success: bool
    error: str | None = None


class SamplerBackend(Protocol):
    @property
    def version(self) -> str: ...

    def sample(self, request: SampleRequest, output_directory: Path) -> SampleResult: ...
