"""Versioned Character and Style preset schemas."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lora_factory.config.loader import load_yaml_model
from lora_factory.config.models import PresetKind


class ProfileModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class ClassTokenPolicy(ProfileModel):
    enabled: bool = True
    frequency_threshold: Annotated[float, Field(ge=0, le=1)] = 0.70
    mean_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.55
    ambiguity_margin: Annotated[float, Field(ge=0, le=1)] = 0.10


class InvariantPolicy(ProfileModel):
    frequency_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85
    mean_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.60
    max_conflict_ratio: Annotated[float, Field(ge=0, le=1)] = 0.10
    max_cluster_dominance: Annotated[float, Field(ge=0, le=1)] = 0.35


class CaptionProfile(ProfileModel):
    min_confidence: Annotated[float, Field(ge=0, le=1)]
    max_tags: Annotated[int, Field(ge=1, le=256)]
    forbidden_categories: tuple[str, ...]
    keep_tokens: Literal[1, 2]
    class_token: ClassTokenPolicy | None = None
    invariant: InvariantPolicy | None = None


class QualityGateProfile(ProfileModel):
    hard_minimum: Annotated[int, Field(ge=1)]
    warning_below: Annotated[int, Field(ge=1)]
    max_near_duplicate_dominance: Annotated[float, Field(ge=0, le=1)]
    max_dominant_character_ratio: Annotated[float, Field(ge=0, le=1)]
    min_content_diversity: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_counts(self) -> QualityGateProfile:
        if self.warning_below < self.hard_minimum:
            raise ValueError("warning_below must be at least hard_minimum")
        return self


class ValidationProfile(ProfileModel):
    enabled_at: Annotated[int, Field(ge=1)]
    ratio: Annotated[float, Field(gt=0, lt=1)]
    min_count: Annotated[int, Field(ge=1)]
    max_count: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0)] = 42

    @model_validator(mode="after")
    def validate_range(self) -> ValidationProfile:
        if self.max_count < self.min_count:
            raise ValueError("validation max_count must be at least min_count")
        if self.enabled_at <= self.min_count:
            raise ValueError("validation must leave more images for training than its minimum")
        return self


class TrainingDefaults(ProfileModel):
    resolution_candidates: tuple[Literal[768, 896, 1024], ...]
    network_dim: Annotated[int, Field(ge=4, le=256)]
    network_alpha: Annotated[int, Field(ge=1, le=256)]
    unet_lr: Annotated[float, Field(gt=0, le=0.1)]
    text_encoder_lr: Annotated[float, Field(gt=0, le=0.1)]
    optimizer_preference: tuple[str, ...]
    scheduler: str
    precision_preference: tuple[Literal["bf16", "fp16", "fp32"], ...]
    no_half_vae_for_fp16: bool = True
    sdpa: bool = True
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = False
    shuffle_caption: bool = True
    enable_bucket: bool = True
    bucket_no_upscale: bool = True
    random_crop: bool = False
    flip_aug: bool = False
    color_aug: bool = False
    target_steps_min: Annotated[int, Field(ge=1)]
    target_steps_max: Annotated[int, Field(ge=1)]
    target_checkpoint_count: Annotated[int, Field(ge=4, le=30)] = 8

    @model_validator(mode="after")
    def validate_training_defaults(self) -> TrainingDefaults:
        if not self.resolution_candidates:
            raise ValueError("At least one resolution candidate is required")
        unique_descending = tuple(sorted(set(self.resolution_candidates), reverse=True))
        if unique_descending != self.resolution_candidates:
            raise ValueError("resolution_candidates must be unique and descending")
        if self.network_alpha > self.network_dim:
            raise ValueError("network_alpha must not exceed network_dim")
        if not self.optimizer_preference or not self.precision_preference:
            raise ValueError("Optimizer and precision preference lists cannot be empty")
        if self.target_steps_max < self.target_steps_min:
            raise ValueError("target_steps_max must be at least target_steps_min")
        return self


class EvaluationProfile(ProfileModel):
    weights: dict[str, Annotated[float, Field(ge=0, le=1)]]
    overfit_penalty_weight: Annotated[float, Field(ge=0, le=1)]
    screening_weight: Annotated[float, Field(ge=0, le=2)] = 0.8
    sweep_weights: tuple[Annotated[float, Field(ge=0, le=2)], ...]

    @model_validator(mode="after")
    def validate_weights(self) -> EvaluationProfile:
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Evaluation metric weights must sum to 1.0")
        if not self.sweep_weights:
            raise ValueError("At least one LoRA sweep weight is required")
        return self


class PresetProfile(ProfileModel):
    schema_version: Literal[1] = 1
    preset: PresetKind
    caption: CaptionProfile
    quality_gate: QualityGateProfile
    validation: ValidationProfile
    training: TrainingDefaults
    evaluation: EvaluationProfile


def default_preset_path(kind: PresetKind, *, repository_root: Path | None = None) -> Path:
    root = repository_root
    if root is None:
        bundle_root = (
            getattr(sys, "_MEIPASS", None) if bool(getattr(sys, "frozen", False)) else None
        )
        root = (
            Path(bundle_root).resolve(strict=True)
            if isinstance(bundle_root, str)
            else Path(__file__).resolve().parents[3]
        )
    return root / "presets" / f"{kind.value}.yaml"


def load_preset_profile(kind: PresetKind, *, repository_root: Path | None = None) -> PresetProfile:
    path = default_preset_path(kind, repository_root=repository_root)
    profile = load_yaml_model(path, PresetProfile)
    if profile.preset is not kind:
        raise ValueError(
            f"Preset file {path} declares {profile.preset.value}, expected {kind.value}"
        )
    return profile
