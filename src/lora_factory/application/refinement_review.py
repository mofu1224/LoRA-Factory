"""Durable Pydantic boundary for refinement review and same-run approval."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lora_factory.caption.refinement import TriggerCandidate
from lora_factory.config.models import PresetKind, validate_trigger_word


class ReviewModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RefinementReviewItem(ReviewModel):
    asset_id: str
    original_tags: tuple[str, ...]
    baseline_tags: tuple[str, ...] = ()
    proposed_tags: tuple[str, ...]
    effective_tags: tuple[str, ...]
    draft_caption: str
    proposed_caption: str
    reason: str
    confidence: Annotated[float, Field(ge=0, le=1)]
    factory_accepted: bool
    rejection_reason: str | None = None
    added_tags: tuple[str, ...] = ()
    removed_tags: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def compute_tag_diffs(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        original_tags = tuple(value.get("original_tags", ()))
        effective_tags = tuple(value.get("effective_tags", ()))
        original = set(original_tags)
        effective = set(effective_tags)
        return {
            **value,
            "added_tags": tuple(tag for tag in effective_tags if tag not in original),
            "removed_tags": tuple(tag for tag in original_tags if tag not in effective),
        }


class RefinementReviewState(ReviewModel):
    schema_version: Literal[1] = 1
    run_id: str
    upstream_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    requires_refinement_review: bool
    requires_trigger_selection: bool
    preset: PresetKind = PresetKind.CHARACTER
    class_token: str | None = None
    invariants: tuple[str, ...] = ()
    pinned_tag_vocabulary: tuple[str, ...] = ()
    pinned_tag_categories: dict[str, str] = Field(default_factory=dict)
    max_effective_tags: Annotated[int, Field(ge=1, le=500)] = 75
    trigger_word: str = ""
    trigger_candidates: tuple[TriggerCandidate, ...] = ()
    items: tuple[RefinementReviewItem, ...]
    warnings: tuple[str, ...] = ()


class RefinementApprovalItem(ReviewModel):
    asset_id: str
    decision: Literal["accept", "reject", "edit"]
    effective_tags: tuple[str, ...] | None = None
    caption: str | None = None

    @field_validator("caption")
    @classmethod
    def validate_caption(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or "\n" in value or "\r" in value:
            raise ValueError("Edited caption must contain one non-empty line")
        return value


class RefinementApproval(ReviewModel):
    schema_version: Literal[1] = 1
    upstream_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    trigger_word: str
    items: tuple[RefinementApprovalItem, ...] = ()

    @field_validator("trigger_word")
    @classmethod
    def validate_trigger(cls, value: str) -> str:
        return validate_trigger_word(value)


def refinement_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
