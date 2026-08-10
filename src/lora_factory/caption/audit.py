"""Caption QA with per-asset findings and aggregate coverage metrics."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from statistics import mean
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.parser import parse_caption_tags
from lora_factory.config.models import PresetKind
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY, OntologyCategory, TagOntology


class QaSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class CaptionQaIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    code: str
    severity: QaSeverity
    message: str


class CaptionAuditSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    caption_count: Annotated[int, Field(ge=0)]
    tag_count_min: Annotated[int, Field(ge=0)] = 0
    tag_count_max: Annotated[int, Field(ge=0)] = 0
    tag_count_mean: Annotated[float, Field(ge=0)] = 0.0
    clothing_coverage: Annotated[float, Field(ge=0, le=1)] = 0.0
    semantic_content_coverage: Annotated[float, Field(ge=0, le=1)] = 0.0
    class_token_consistency: Annotated[float, Field(ge=0, le=1)] = 1.0
    invariant_tags: tuple[str, ...] = Field(default_factory=tuple)
    issues: tuple[CaptionQaIssue, ...] = Field(default_factory=tuple)


def audit_captions(
    captions: Mapping[str, str],
    trigger_token: str,
    preset: PresetKind,
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    class_token: str | None = None,
    invariant_tags: tuple[str, ...] = (),
) -> CaptionAuditSummary:
    issues: list[CaptionQaIssue] = []
    tag_counts: list[int] = []
    clothing_images = 0
    semantic_images = 0
    class_consistent = 0
    forbidden = {
        OntologyCategory.QUALITY,
        OntologyCategory.RATING,
        OntologyCategory.RESOLUTION,
        OntologyCategory.TEXT_WATERMARK,
        OntologyCategory.CHARACTER,
        OntologyCategory.COPYRIGHT,
    }
    if preset is PresetKind.STYLE:
        forbidden.update({OntologyCategory.ARTIST, OntologyCategory.STYLE})
    expected_class = ontology.display(class_token) if class_token else None
    expected_invariants = {ontology.canonicalize(tag) for tag in invariant_tags}
    for asset_id, caption in sorted(captions.items()):
        try:
            tags = parse_caption_tags(caption)
        except ValueError as exc:
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code="invalid_caption",
                    severity=QaSeverity.ERROR,
                    message=str(exc),
                )
            )
            tags = ()
        tag_counts.append(len(tags))
        if not tags:
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code="empty_caption",
                    severity=QaSeverity.ERROR,
                    message="Caption is empty",
                )
            )
            continue
        if tags[0] != trigger_token:
            code = "trigger_missing" if trigger_token not in tags else "trigger_not_first"
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code=code,
                    severity=QaSeverity.ERROR,
                    message="Trigger token must be the first caption token",
                )
            )
        canonical = [ontology.canonicalize(tag) for tag in tags[1:]]
        duplicate_tags = sorted(tag for tag in set(canonical) if canonical.count(tag) > 1)
        if duplicate_tags:
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code="duplicated_tags",
                    severity=QaSeverity.ERROR,
                    message=f"Duplicated tags: {', '.join(duplicate_tags)}",
                )
            )
        leaked = sorted(tag for tag in canonical if ontology.classify(tag) in forbidden)
        if leaked:
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code="forbidden_tags",
                    severity=QaSeverity.ERROR,
                    message=f"Forbidden/source/quality tags leaked: {', '.join(leaked)}",
                )
            )
        leaked_invariants = sorted(expected_invariants.intersection(canonical))
        if leaked_invariants:
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code="identity_invariant_leak",
                    severity=QaSeverity.ERROR,
                    message=(
                        "Identity invariants selected for trigger binding remain in caption: "
                        f"{', '.join(leaked_invariants)}"
                    ),
                )
            )
        clothing_images += any(
            ontology.classify(tag) is OntologyCategory.CLOTHING for tag in canonical
        )
        semantic_images += any(
            ontology.classify(tag)
            not in {
                OntologyCategory.QUALITY,
                OntologyCategory.RATING,
                OntologyCategory.RESOLUTION,
                OntologyCategory.TEXT_WATERMARK,
            }
            for tag in canonical
        )
        if expected_class is None or (len(tags) > 1 and tags[1] == expected_class):
            class_consistent += 1
        elif expected_class is not None:
            issues.append(
                CaptionQaIssue(
                    asset_id=asset_id,
                    code="class_token_inconsistent",
                    severity=QaSeverity.ERROR,
                    message=f"Expected fixed class token {expected_class!r} in position 2",
                )
            )
    count = len(captions)
    return CaptionAuditSummary(
        passed=not any(issue.severity is QaSeverity.ERROR for issue in issues),
        caption_count=count,
        tag_count_min=min(tag_counts, default=0),
        tag_count_max=max(tag_counts, default=0),
        tag_count_mean=mean(tag_counts) if tag_counts else 0.0,
        clothing_coverage=clothing_images / count if count else 0.0,
        semantic_content_coverage=semantic_images / count if count else 0.0,
        class_token_consistency=class_consistent / count if count else 1.0,
        invariant_tags=tuple(invariant_tags),
        issues=tuple(issues),
    )


caption_qa = audit_captions
