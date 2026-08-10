"""Deterministic reviews used when Runtime Codex is unavailable."""

from __future__ import annotations

from typing import Any

from lora_factory.codex.schemas import (
    CaptionReview,
    CodexTaskType,
    DatasetReview,
    FinalReview,
    RecoveryReview,
    StrictModel,
    TrainingPlanReview,
)


def deterministic_fallback(task_type: CodexTaskType, payload: dict[str, Any]) -> StrictModel:
    warning = "Runtime Codex was unavailable; deterministic policy was used."
    if task_type is CodexTaskType.DATASET_REVIEW:
        hard_gate_passed = bool(payload.get("hard_gate_passed", False))
        return DatasetReview(
            approved=hard_gate_passed,
            warnings=(warning,),
            reason_summary="Dataset gate result was preserved without model judgment.",
            confidence=1.0,
        )
    if task_type is CodexTaskType.CAPTION_REVIEW:
        return CaptionReview(
            approved=bool(payload.get("caption_qa_passed", False)),
            warnings=(warning,),
            reason_summary="Deterministic caption QA was preserved without tag changes.",
            confidence=1.0,
        )
    if task_type is CodexTaskType.TRAINING_PLAN:
        return TrainingPlanReview(
            approved=bool(payload.get("preflight_passed", False)),
            warnings=(warning,),
            reason_summary="The deterministic training plan was not modified.",
            confidence=1.0,
        )
    if task_type is CodexTaskType.RECOVERY:
        return RecoveryReview(
            classification=str(payload.get("classification", "UNKNOWN")),
            recoverable=bool(payload.get("deterministically_recoverable", False)),
            proposed_changes=(),
            reason=f"{warning} No model-proposed changes were applied.",
            confidence=1.0,
        )
    ranking = payload.get("numeric_ranking")
    recommended = ranking[0] if isinstance(ranking, list) and ranking else None
    return FinalReview(
        recommended_checkpoint_id=str(recommended) if recommended is not None else None,
        warnings=(warning,),
        reason_summary="Numeric ranking was retained without model override.",
        confidence=1.0,
        numeric_ranking_overridden=False,
    )
