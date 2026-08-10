"""Adaptive SDXL training and deterministic validation-split planner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.config.models import (
    DatasetStatistics,
    GpuCapability,
    PresetKind,
    ProjectConfig,
    TrainingPlan,
)
from lora_factory.training.batch_probe import BatchProbeResult
from lora_factory.training.dataset_toml import sd_scripts_validation_partition
from lora_factory.training.profiles import PresetProfile


class TrainingPlanningError(RuntimeError):
    """Raised when inputs cannot produce a safe executable training plan."""


class ValidationSplitPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    seed: Annotated[int, Field(ge=0)]
    train_asset_ids: tuple[str, ...]
    validation_asset_ids: tuple[str, ...]


def _source(project: ProjectConfig, field_name: str) -> str:
    if field_name in project.locked_fields:
        return "user_lock"
    return "user_override"


def _select_gpu(project: ProjectConfig, capabilities: Sequence[GpuCapability]) -> GpuCapability:
    by_uuid = {gpu.uuid: gpu for gpu in capabilities}
    missing = [uuid for uuid in project.selected_gpu_uuids if uuid not in by_uuid]
    if missing:
        raise TrainingPlanningError(f"Selected GPU is unavailable: {missing[0]}")
    candidates = [by_uuid[uuid] for uuid in project.selected_gpu_uuids if by_uuid[uuid].compatible]
    if not candidates:
        reasons = [by_uuid[uuid].compatibility_reason for uuid in project.selected_gpu_uuids]
        raise TrainingPlanningError(
            "No selected GPU is runtime-compatible: " + "; ".join(filter(None, reasons))
        )
    return max(candidates, key=lambda gpu: (gpu.free_vram_mb, gpu.total_vram_mb, -gpu.index))


def _auto_resolution(
    statistics: DatasetStatistics,
    gpu: GpuCapability,
    network_dim: int,
    candidates: Sequence[int],
) -> Literal[768, 896, 1024]:
    observed_short_side = statistics.short_side_p10 or statistics.short_side_median
    if observed_short_side >= 960:
        image_cap = 1024
    elif observed_short_side >= 832:
        image_cap = 896
    else:
        image_cap = 768
    if gpu.free_vram_mb < 10 * 1024:
        vram_cap = 768
    elif gpu.free_vram_mb < 16 * 1024:
        vram_cap = 896
    else:
        vram_cap = 1024
    if network_dim >= 64 and vram_cap > 768:
        vram_cap = 896
    cap = min(image_cap, vram_cap)
    eligible = [value for value in candidates if value <= cap]
    if not eligible:
        raise TrainingPlanningError("Preset has no resolution compatible with dataset and GPU")
    selected = max(eligible)
    if selected == 768:
        return 768
    if selected == 896:
        return 896
    if selected == 1024:
        return 1024
    raise TrainingPlanningError(f"Unsupported training resolution in profile: {selected}")


def _auto_batch(resolution: int, free_vram_mb: int, network_dim: int) -> int:
    rank_cost = 1 if network_dim <= 32 else 2
    if resolution == 1024:
        batch = 2 if free_vram_mb >= 20 * 1024 else 1
    elif resolution == 896:
        batch = 4 if free_vram_mb >= 24 * 1024 else 2 if free_vram_mb >= 14 * 1024 else 1
    else:
        batch = 4 if free_vram_mb >= 20 * 1024 else 2 if free_vram_mb >= 10 * 1024 else 1
    return max(1, batch // rank_cost)


def _target_steps(preset: PresetKind, statistics: DatasetStatistics, profile: PresetProfile) -> int:
    lower = profile.training.target_steps_min
    span = profile.training.target_steps_max - lower
    size_factor = min(
        1.0,
        statistics.accepted_count / max(profile.quality_gate.warning_below * 2, 1),
    )
    if preset is PresetKind.STYLE:
        quality_factor = statistics.diversity_score
    else:
        quality_factor = 0.75 + 0.25 * statistics.diversity_score
    return lower + round(span * size_factor * quality_factor)


def _epochs_and_repeats(
    *,
    image_count: int,
    target_steps: int,
    effective_batch: int,
    checkpoint_count: int,
    requested_repeats: int | None,
    requested_epochs: int | None,
) -> tuple[int, int]:
    if requested_repeats is not None and requested_epochs is not None:
        return requested_repeats, requested_epochs
    if requested_repeats is not None:
        epochs = math.ceil(target_steps * effective_batch / (image_count * requested_repeats))
        return requested_repeats, max(checkpoint_count, min(100, epochs))
    if requested_epochs is not None:
        repeats = math.ceil(target_steps * effective_batch / (image_count * requested_epochs))
        return max(1, min(100, repeats)), requested_epochs
    desired_epochs = checkpoint_count
    repeats = math.ceil(target_steps * effective_batch / (image_count * desired_epochs))
    repeats = max(1, min(100, repeats))
    epochs = math.ceil(target_steps * effective_batch / (image_count * repeats))
    return repeats, max(checkpoint_count, min(100, epochs))


def validation_count(accepted_count: int, profile: PresetProfile) -> int:
    policy = profile.validation
    if accepted_count < policy.enabled_at:
        return 0
    proposed = round(accepted_count * policy.ratio)
    return min(policy.max_count, max(policy.min_count, proposed))


def plan_validation_split(asset_ids: Sequence[str], profile: PresetProfile) -> ValidationSplitPlan:
    if len(set(asset_ids)) != len(asset_ids):
        raise TrainingPlanningError("Validation split input contains duplicate asset IDs")
    count = validation_count(len(asset_ids), profile)
    training_ids, validation_ids = sd_scripts_validation_partition(
        asset_ids,
        validation_count=count,
        seed=profile.validation.seed,
    )
    return ValidationSplitPlan(
        enabled=bool(validation_ids),
        seed=profile.validation.seed,
        train_asset_ids=training_ids,
        validation_asset_ids=validation_ids,
    )


def plan_training(
    project: ProjectConfig,
    statistics: DatasetStatistics,
    gpu_capabilities: Sequence[GpuCapability],
    profile: PresetProfile,
    *,
    has_class_token: bool = False,
    available_optimizers: Sequence[str] = ("AdamW",),
    batch_probe_result: BatchProbeResult | None = None,
) -> TrainingPlan:
    """Resolve an adaptive, auditable plan while preserving every user override."""

    if profile.preset is not project.preset:
        raise TrainingPlanningError("Selected preset and loaded preset profile differ")
    if statistics.accepted_count < profile.quality_gate.hard_minimum:
        raise TrainingPlanningError(
            f"{project.preset.value.title()} requires at least "
            f"{profile.quality_gate.hard_minimum} accepted images"
        )
    gpu = _select_gpu(project, gpu_capabilities)
    advanced = project.advanced
    provenance: dict[str, str] = {}

    network_dim = advanced.network_dim or profile.training.network_dim
    provenance["network_dim"] = (
        _source(project, "network_dim") if advanced.network_dim is not None else "preset"
    )
    network_alpha = advanced.network_alpha or profile.training.network_alpha
    provenance["network_alpha"] = (
        _source(project, "network_alpha") if advanced.network_alpha is not None else "preset"
    )
    if network_alpha > network_dim:
        raise TrainingPlanningError("network_alpha must not exceed network_dim")

    resolution = advanced.resolution or _auto_resolution(
        statistics, gpu, network_dim, profile.training.resolution_candidates
    )
    provenance["resolution"] = (
        _source(project, "resolution") if advanced.resolution is not None else "dataset_gpu_plan"
    )
    if advanced.batch_size is not None:
        batch_size = advanced.batch_size
        provenance["batch_size"] = _source(project, "batch_size")
        if batch_probe_result is not None and batch_probe_result.selected_batch_size != batch_size:
            raise TrainingPlanningError("Batch probe did not validate the explicit batch size")
    elif batch_probe_result is not None:
        if batch_probe_result.gpu_uuid != gpu.uuid:
            raise TrainingPlanningError("Batch probe result belongs to another training GPU")
        batch_size = batch_probe_result.selected_batch_size
        provenance["batch_size"] = f"batch_probe:{batch_probe_result.provider_version}"
    else:
        batch_size = _auto_batch(resolution, gpu.free_vram_mb, network_dim)
        provenance["batch_size"] = "gpu_plan_pending_probe"
    gradient_accumulation = advanced.gradient_accumulation or max(1, math.ceil(4 / batch_size))
    provenance["gradient_accumulation"] = (
        _source(project, "gradient_accumulation")
        if advanced.gradient_accumulation is not None
        else "effective_batch_plan"
    )

    planned_validation_images = validation_count(statistics.accepted_count, profile)
    if statistics.validation_count > 0:
        if planned_validation_images == 0:
            raise TrainingPlanningError(
                "Validation images were supplied below the preset threshold"
            )
        if not (
            profile.validation.min_count
            <= statistics.validation_count
            <= profile.validation.max_count
        ):
            raise TrainingPlanningError("Validation count is outside the preset limits")
        validation_images = statistics.validation_count
    else:
        validation_images = planned_validation_images
    if validation_images >= statistics.accepted_count:
        raise TrainingPlanningError("Validation split leaves no training images")
    training_image_count = statistics.accepted_count - validation_images
    target_steps = _target_steps(project.preset, statistics, profile)
    repeats, epochs = _epochs_and_repeats(
        image_count=training_image_count,
        target_steps=target_steps,
        effective_batch=batch_size * gradient_accumulation,
        checkpoint_count=profile.training.target_checkpoint_count,
        requested_repeats=advanced.repeats,
        requested_epochs=advanced.epochs,
    )
    provenance["repeats"] = (
        _source(project, "repeats") if advanced.repeats is not None else "dataset_step_plan"
    )
    provenance["epochs"] = (
        _source(project, "epochs") if advanced.epochs is not None else "dataset_step_plan"
    )
    estimated_steps = math.ceil(
        training_image_count * repeats * epochs / (batch_size * gradient_accumulation)
    )

    available = set(available_optimizers)
    if advanced.optimizer is not None:
        if advanced.optimizer not in available:
            raise TrainingPlanningError(
                f"Requested optimizer is unavailable in the managed runtime: {advanced.optimizer}"
            )
        optimizer = advanced.optimizer
        provenance["optimizer"] = _source(project, "optimizer")
    else:
        optimizer = next(
            (item for item in profile.training.optimizer_preference if item in available),
            "",
        )
        if not optimizer:
            raise TrainingPlanningError("No preset optimizer is available in the managed runtime")
        provenance["optimizer"] = "runtime_capability"

    if advanced.precision is not None:
        precision = advanced.precision
        provenance["precision"] = _source(project, "precision")
    else:
        precision = next(
            (
                item
                for item in profile.training.precision_preference
                if item != "bf16" or gpu.bf16_supported
            ),
            "fp32",
        )
        provenance["precision"] = "runtime_capability"
    if precision == "bf16" and not gpu.bf16_supported:
        raise TrainingPlanningError("Selected GPU did not pass bf16 capability checks")

    unet_lr = advanced.unet_lr or profile.training.unet_lr
    text_encoder_lr = advanced.text_encoder_lr or profile.training.text_encoder_lr
    provenance["unet_lr"] = (
        _source(project, "unet_lr") if advanced.unet_lr is not None else "preset"
    )
    provenance["text_encoder_lr"] = (
        _source(project, "text_encoder_lr") if advanced.text_encoder_lr is not None else "preset"
    )
    estimated_disk_mb = max(
        256,
        profile.training.target_checkpoint_count * network_dim * 12
        + statistics.accepted_count * resolution * resolution // 1_000_000,
    )
    keep_tokens: Literal[1, 2] = (
        2
        if project.preset is PresetKind.CHARACTER
        and has_class_token
        and profile.caption.keep_tokens == 2
        else 1
    )
    return TrainingPlan(
        resolution=resolution,
        enable_bucket=profile.training.enable_bucket,
        bucket_no_upscale=profile.training.bucket_no_upscale,
        random_crop=profile.training.random_crop,
        flip_aug=profile.training.flip_aug,
        color_aug=profile.training.color_aug,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        repeats=repeats,
        epochs=epochs,
        estimated_steps=estimated_steps,
        save_every_n_epochs=1,
        network_dim=network_dim,
        network_alpha=network_alpha,
        unet_lr=unet_lr,
        text_encoder_lr=text_encoder_lr,
        optimizer=optimizer,
        scheduler=profile.training.scheduler,
        precision=precision,
        no_half_vae=precision == "fp16" and profile.training.no_half_vae_for_fp16,
        sdpa=profile.training.sdpa,
        gradient_checkpointing=profile.training.gradient_checkpointing,
        cache_latents=profile.training.cache_latents,
        cache_text_encoder_outputs=profile.training.cache_text_encoder_outputs,
        shuffle_caption=profile.training.shuffle_caption,
        keep_tokens=keep_tokens,
        validation_enabled=validation_images > 0,
        validation_image_count=validation_images,
        validation_seed=profile.validation.seed,
        training_gpu_uuid=gpu.uuid,
        estimated_disk_mb=estimated_disk_mb,
        provenance=provenance,
    )
