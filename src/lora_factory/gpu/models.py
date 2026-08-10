"""Typed GPU discovery, binding, scheduling, and telemetry boundaries."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_GPU_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]{8,}$")


class ImmutableModel(BaseModel):
    """Small immutable Pydantic boundary shared by GPU adapters."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class GpuDevice(ImmutableModel):
    """One physical NVIDIA device as observed in a single discovery snapshot."""

    uuid: str
    index: Annotated[int, Field(ge=0)]
    name: str
    total_vram_mb: Annotated[int, Field(gt=0)]
    free_vram_mb: Annotated[int, Field(ge=0)]
    utilization_percent: Annotated[float, Field(ge=0, le=100)] = 0.0
    capability_major: Annotated[int, Field(ge=0)] = 0
    capability_minor: Annotated[int, Field(ge=0)] = 0
    compatible: bool = True
    compatibility_reason: str = ""

    @field_validator("uuid")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        if not _GPU_UUID.fullmatch(value):
            raise ValueError(f"Invalid NVIDIA GPU UUID: {value!r}")
        return value

    @property
    def capability(self) -> tuple[int, int]:
        return self.capability_major, self.capability_minor

    @property
    def bf16_supported(self) -> bool:
        return self.capability >= (8, 0)

    @model_validator(mode="after")
    def validate_memory_snapshot(self) -> GpuDevice:
        if self.free_vram_mb > self.total_vram_mb:
            raise ValueError("Free VRAM cannot exceed total VRAM")
        return self


class GpuBinding(ImmutableModel):
    """Auditable physical-to-logical mapping applied to one child process."""

    uuid: str
    physical_index: Annotated[int, Field(ge=0)]
    logical_index: Annotated[int, Field(ge=0)] = 0
    environment: dict[str, str]

    @field_validator("uuid")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        if not _GPU_UUID.fullmatch(value):
            raise ValueError(f"Invalid NVIDIA GPU UUID: {value!r}")
        return value

    @model_validator(mode="after")
    def validate_isolation_environment(self) -> GpuBinding:
        if self.logical_index != 0:
            raise ValueError("A LoRA Factory GPU child must see its assigned GPU as logical cuda:0")
        if self.environment.get("CUDA_VISIBLE_DEVICES") != str(self.physical_index):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain only the resolved physical index")
        if self.environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise ValueError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
        return self


class GpuTaskKind(StrEnum):
    IMAGE_NORMALIZE = "IMAGE_NORMALIZE"
    WD14_TAG = "WD14_TAG"
    REFERENCE_EMBED = "REFERENCE_EMBED"
    BATCH_PROBE = "BATCH_PROBE"
    TRAIN = "TRAIN"
    SAMPLE = "SAMPLE"
    GENERATED_TAG = "GENERATED_TAG"
    GENERATED_EMBED = "GENERATED_EMBED"
    EVALUATE = "EVALUATE"
    OPTIONAL_CANDIDATE_TRAIN = "OPTIONAL_CANDIDATE_TRAIN"


class GpuTaskRequest(ImmutableModel):
    task_id: str
    kind: GpuTaskKind
    estimated_vram_mb: Annotated[int, Field(ge=0)]
    priority: int = 0
    preferred_gpu_uuid: str | None = None


class GpuAssignment(ImmutableModel):
    task_id: str
    kind: GpuTaskKind
    gpu_uuid: str
    physical_index: Annotated[int, Field(ge=0)]
    estimated_vram_mb: Annotated[int, Field(ge=0)]


class GpuTelemetrySample(ImmutableModel):
    uuid: str
    utilization_percent: Annotated[float, Field(ge=0, le=100)]
    memory_used_mb: Annotated[int, Field(ge=0)]
    memory_free_mb: Annotated[int, Field(ge=0)]
    temperature_c: Annotated[float, Field(ge=-100, le=200)]
    power_watts: Annotated[float, Field(ge=0)]
