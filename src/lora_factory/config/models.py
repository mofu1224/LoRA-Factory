"""Pydantic models used at application and process boundaries."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$")
WINDOWS_FORBIDDEN = set('<>:"/\\|?*')


class FrozenModel(BaseModel):
    """Immutable boundary model."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class PresetKind(StrEnum):
    CHARACTER = "character"
    STYLE = "style"


class BackendMode(StrEnum):
    FAKE = "fake"
    REAL = "real"


class DestinationKind(StrEnum):
    A1111 = "a1111"
    FORGE = "forge"
    COMFYUI = "comfyui"


class DestinationConfig(FrozenModel):
    kind: DestinationKind
    root: Path
    enabled: bool = True


class ProjectDraft(FrozenModel):
    """Name-only project record created before training inputs are configured."""

    schema_version: int = 1
    status: Literal["draft"] = "draft"
    project_id: str
    lora_name: str

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 80:
            raise ValueError("project_id must contain 1-80 characters")
        if any(char in WINDOWS_FORBIDDEN for char in value) or value.endswith((" ", ".")):
            raise ValueError("project_id contains characters invalid in a Windows directory")
        return value

    @field_validator("lora_name")
    @classmethod
    def validate_lora_name(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 128:
            raise ValueError("LoRA name must contain 1-128 characters")
        if any(char in WINDOWS_FORBIDDEN for char in value) or value.endswith((" ", ".")):
            raise ValueError("LoRA name contains characters invalid in a Windows filename")
        return value


class AdvancedOverrides(FrozenModel):
    resolution: Literal[768, 896, 1024] | None = None
    network_dim: Annotated[int | None, Field(ge=4, le=256)] = None
    network_alpha: Annotated[int | None, Field(ge=1, le=256)] = None
    batch_size: Annotated[int | None, Field(ge=1, le=32)] = None
    gradient_accumulation: Annotated[int | None, Field(ge=1, le=64)] = None
    repeats: Annotated[int | None, Field(ge=1, le=100)] = None
    epochs: Annotated[int | None, Field(ge=1, le=100)] = None
    unet_lr: Annotated[float | None, Field(gt=0, le=0.1)] = None
    text_encoder_lr: Annotated[float | None, Field(gt=0, le=0.1)] = None
    optimizer: str | None = None
    precision: Literal["bf16", "fp16", "fp32"] | None = None
    recursive_import: bool | None = None


class ProjectConfig(FrozenModel):
    schema_version: int = 1
    project_id: str
    lora_name: str
    preset: PresetKind
    trigger_token: str
    base_model: Path
    input_paths: tuple[Path, ...]
    selected_gpu_uuids: tuple[str, ...]
    output_root: Path
    backend_mode: BackendMode = BackendMode.REAL
    quality_mode: bool = False
    allow_without_codex: bool = True
    advanced: AdvancedOverrides = Field(default_factory=AdvancedOverrides)
    locked_fields: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 80:
            raise ValueError("project_id must contain 1-80 characters")
        if any(char in WINDOWS_FORBIDDEN for char in value) or value.endswith((" ", ".")):
            raise ValueError("project_id contains characters invalid in a Windows directory")
        return value

    @field_validator("lora_name")
    @classmethod
    def validate_lora_name(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 128:
            raise ValueError("LoRA name must contain 1-128 characters")
        if any(char in WINDOWS_FORBIDDEN for char in value) or value.endswith((" ", ".")):
            raise ValueError("LoRA name contains characters invalid in a Windows filename")
        return value

    @field_validator("trigger_token")
    @classmethod
    def validate_trigger_token(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Trigger token is required")
        if len(value) > 64:
            raise ValueError("Trigger token must not exceed 64 characters")
        if any(char in value for char in ("\n", "\r", ",", ";", "|")):
            raise ValueError("Trigger token cannot contain newlines or caption separators")
        if value != " ".join(value.split()):
            raise ValueError("Trigger token cannot contain repeated whitespace")
        return value

    @field_validator("selected_gpu_uuids")
    @classmethod
    def validate_gpu_pool(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("At least one CUDA GPU must be selected")
        if len(set(value)) != len(value):
            raise ValueError("Selected GPU UUIDs must be unique")
        invalid = [item for item in value if not GPU_UUID_PATTERN.match(item)]
        if invalid:
            raise ValueError(f"Invalid NVIDIA GPU UUID: {invalid[0]}")
        return value

    @field_validator("input_paths")
    @classmethod
    def validate_inputs(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not value:
            raise ValueError("At least one image file or folder is required")
        return value

    @model_validator(mode="after")
    def validate_locks(self) -> ProjectConfig:
        valid = set(AdvancedOverrides.model_fields)
        unknown = set(self.locked_fields) - valid
        if unknown:
            raise ValueError(f"Unknown locked setting: {sorted(unknown)[0]}")
        for key in self.locked_fields:
            if getattr(self.advanced, key) is None:
                raise ValueError(f"Locked setting {key!r} requires an explicit override")
        return self


class DatasetStatistics(FrozenModel):
    accepted_count: Annotated[int, Field(ge=0)]
    validation_count: Annotated[int, Field(ge=0)] = 0
    short_side_p10: Annotated[int, Field(ge=0)] = 0
    short_side_median: Annotated[int, Field(ge=0)] = 0
    area_median: Annotated[int, Field(ge=0)] = 0
    aspect_ratio_min: Annotated[float, Field(gt=0)] = 1.0
    aspect_ratio_max: Annotated[float, Field(gt=0)] = 1.0
    diversity_score: Annotated[float, Field(ge=0, le=1)] = 0.5


class GpuCapability(FrozenModel):
    uuid: str
    index: Annotated[int, Field(ge=0)]
    name: str
    total_vram_mb: Annotated[int, Field(gt=0)]
    free_vram_mb: Annotated[int, Field(ge=0)]
    capability_major: Annotated[int, Field(ge=0)]
    capability_minor: Annotated[int, Field(ge=0)]
    bf16_supported: bool
    compatible: bool = True
    compatibility_reason: str = ""


class TrainingPlan(FrozenModel):
    resolution: Literal[768, 896, 1024]
    enable_bucket: bool = True
    bucket_no_upscale: bool = True
    random_crop: bool = False
    flip_aug: bool = False
    color_aug: bool = False
    batch_size: Annotated[int, Field(ge=1)]
    gradient_accumulation: Annotated[int, Field(ge=1)]
    repeats: Annotated[int, Field(ge=1)]
    epochs: Annotated[int, Field(ge=1)]
    estimated_steps: Annotated[int, Field(ge=1)]
    save_every_n_epochs: Annotated[int, Field(ge=1)] = 1
    network_dim: Annotated[int, Field(ge=4)]
    network_alpha: Annotated[int, Field(ge=1)]
    unet_lr: Annotated[float, Field(gt=0)]
    text_encoder_lr: Annotated[float, Field(gt=0)]
    optimizer: str
    scheduler: str = "constant"
    precision: Literal["bf16", "fp16", "fp32"]
    no_half_vae: bool = False
    sdpa: bool = True
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = False
    shuffle_caption: bool = True
    keep_tokens: Literal[1, 2]
    validation_enabled: bool
    validation_image_count: Annotated[int, Field(ge=0)] = 0
    validation_seed: int = 42
    training_gpu_uuid: str
    estimated_disk_mb: Annotated[int, Field(ge=1)]
    provenance: dict[str, str] = Field(default_factory=dict)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    schema_version: int = 1
    projects_root: Path
    managed_runtime_root: Path
    codex_runtime_root: Path
    destinations: list[DestinationConfig] = Field(default_factory=list)
    telemetry_interval_seconds: Annotated[float, Field(ge=1, le=60)] = 5.0
    max_sample_images_per_pass: Annotated[int, Field(ge=8, le=512)] = 96
    codex_required: bool = False
    codex_timeout_seconds: Annotated[int, Field(ge=10, le=1800)] = 180


class ResolvedConfig(FrozenModel):
    values: dict[str, Any]
    provenance: dict[str, str]
    locked_fields: frozenset[str]
