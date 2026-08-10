"""Strict structured-output contracts for Runtime Codex tasks."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodexTaskType(StrEnum):
    DATASET_REVIEW = "dataset_review"
    CAPTION_REVIEW = "caption_review"
    TRAINING_PLAN = "training_plan"
    RECOVERY = "recovery"
    FINAL_REVIEW = "final_review"


class DatasetReview(StrictModel):
    approved: bool
    warnings: tuple[str, ...] = ()
    reason_summary: str
    confidence: Annotated[float, Field(ge=0, le=1)]


class CaptionReview(StrictModel):
    approved: bool
    remove_tags: tuple[str, ...] = ()
    restore_tags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    reason_summary: str
    confidence: Annotated[float, Field(ge=0, le=1)]


class ProposedChange(StrictModel):
    field: Literal[
        "batch_size",
        "gradient_accumulation",
        "epochs",
        "repeats",
        "target_steps",
        "network_dim",
        "network_alpha",
        "unet_lr",
        "text_encoder_lr",
        "optimizer",
        "precision",
        "no_half_vae",
        "cache_latents",
        "gradient_checkpointing",
    ]
    value: str | int | float | bool
    reason: str


class TrainingPlanReview(StrictModel):
    approved: bool
    proposed_changes: tuple[ProposedChange, ...] = ()
    warnings: tuple[str, ...] = ()
    reason_summary: str
    confidence: Annotated[float, Field(ge=0, le=1)]


class RecoveryReview(StrictModel):
    classification: str
    recoverable: bool
    proposed_changes: tuple[ProposedChange, ...] = ()
    reason: str
    confidence: Annotated[float, Field(ge=0, le=1)]


class CandidateOpinion(StrictModel):
    checkpoint_id: str
    observation: str


class FinalReview(StrictModel):
    recommended_checkpoint_id: str | None
    candidate_opinions: tuple[CandidateOpinion, ...] = ()
    warnings: tuple[str, ...] = ()
    reason_summary: str
    confidence: Annotated[float, Field(ge=0, le=1)]
    numeric_ranking_overridden: Literal[False] = False


CodexResponse = DatasetReview | CaptionReview | TrainingPlanReview | RecoveryReview | FinalReview

SCHEMA_MODELS: dict[CodexTaskType, type[StrictModel]] = {
    CodexTaskType.DATASET_REVIEW: DatasetReview,
    CodexTaskType.CAPTION_REVIEW: CaptionReview,
    CodexTaskType.TRAINING_PLAN: TrainingPlanReview,
    CodexTaskType.RECOVERY: RecoveryReview,
    CodexTaskType.FINAL_REVIEW: FinalReview,
}


def strict_output_schema(model: type[StrictModel]) -> dict[str, Any]:
    """Return the supported OpenAI Structured Outputs subset of a Pydantic schema."""

    schema = model.model_json_schema()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            # Pydantic emits annotations that are not needed by Codex' strict output
            # contract. Defaults are especially misleading because every property must
            # be required; the model must emit the value and Pydantic applies defaults
            # only after validation of non-Codex fallback paths.
            value.pop("default", None)
            value.pop("title", None)
            properties = value.get("properties")
            if value.get("type") == "object" and isinstance(properties, dict):
                value["required"] = list(properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema
