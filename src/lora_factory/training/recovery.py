"""Bounded OOM/NaN recovery ladder that never changes locked settings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from lora_factory.config.models import GpuCapability, TrainingPlan
from lora_factory.core.exceptions import ErrorClassification


class RecoveryDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recoverable: bool
    plan: TrainingPlan
    change: str | None
    quality_impact: bool = False
    reason: str


def next_recovery(
    *,
    classification: ErrorClassification,
    plan: TrainingPlan,
    selected_gpus: tuple[GpuCapability, ...],
    locked_fields: frozenset[str],
    previous_changes: tuple[str, ...],
    attempt: int,
) -> RecoveryDecision:
    if attempt >= 3:
        return RecoveryDecision(
            recoverable=False,
            plan=plan,
            change=None,
            reason="Maximum automatic training attempts reached.",
        )
    if classification is ErrorClassification.CUDA_OOM:
        alternatives = sorted(
            (
                gpu
                for gpu in selected_gpus
                if gpu.uuid != plan.training_gpu_uuid
                and gpu.compatible
                and gpu.free_vram_mb >= 0.9 * gpu.total_vram_mb
            ),
            key=lambda gpu: gpu.free_vram_mb,
            reverse=True,
        )
        if alternatives and "gpu_fallback" not in previous_changes:
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(update={"training_gpu_uuid": alternatives[0].uuid}),
                change="gpu_fallback",
                reason="Retrying on another selected compatible GPU with sufficient free VRAM.",
            )
        if (
            "batch_size" not in locked_fields
            and "gradient_accumulation" not in locked_fields
            and plan.batch_size > 1
            and plan.gradient_accumulation <= 32
        ):
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(
                    update={
                        "batch_size": max(1, plan.batch_size // 2),
                        "gradient_accumulation": plan.gradient_accumulation * 2,
                    }
                ),
                change="batch_size",
                reason="Reducing batch size while preserving effective batch through accumulation.",
            )
        if not plan.gradient_checkpointing and "gradient_checkpointing" not in locked_fields:
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(update={"gradient_checkpointing": True}),
                change="gradient_checkpointing",
                reason="Enabling gradient checkpointing to lower peak VRAM.",
            )
        if "optimizer" not in locked_fields and plan.optimizer != "AdamW":
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(update={"optimizer": "AdamW"}),
                change="optimizer",
                reason="Falling back to the broadly compatible AdamW optimizer.",
            )
        if "network_dim" not in locked_fields and plan.network_dim > 8:
            new_dim = max(8, plan.network_dim // 2)
            if "network_alpha" not in locked_fields or plan.network_alpha <= new_dim:
                return RecoveryDecision(
                    recoverable=True,
                    plan=plan.model_copy(
                        update={
                            "network_dim": new_dim,
                            "network_alpha": min(plan.network_alpha, new_dim),
                        }
                    ),
                    change="network_dim",
                    quality_impact=True,
                    reason="Reducing network rank as a late OOM fallback; quality may change.",
                )
        if "resolution" not in locked_fields and plan.resolution > 768:
            resolution = 896 if plan.resolution == 1024 else 768
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(update={"resolution": resolution}),
                change="resolution",
                quality_impact=True,
                reason="Reducing resolution as the final OOM fallback; quality may change.",
            )
    if classification is ErrorClassification.NAN_LOSS:
        if "no_half_vae" not in locked_fields and not plan.no_half_vae:
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(update={"no_half_vae": True}),
                change="no_half_vae",
                reason="Enabling full-precision VAE after NaN loss.",
            )
        if "unet_lr" not in locked_fields and "text_encoder_lr" not in locked_fields:
            return RecoveryDecision(
                recoverable=True,
                plan=plan.model_copy(
                    update={
                        "unet_lr": plan.unet_lr * 0.5,
                        "text_encoder_lr": plan.text_encoder_lr * 0.5,
                    }
                ),
                change="learning_rate",
                quality_impact=True,
                reason="Reducing learning rates after persistent NaN loss.",
            )
    return RecoveryDecision(
        recoverable=False,
        plan=plan,
        change=None,
        reason="No safe deterministic recovery remains without violating user locks.",
    )
