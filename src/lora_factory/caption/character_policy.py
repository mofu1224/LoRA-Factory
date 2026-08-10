"""Character captions: trigger, optional stable class, then variable facts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.invariant import (
    ClassTokenDecision,
    ClassTokenPolicyConfig,
    InvariantAnalysis,
    InvariantPolicyConfig,
    TagInput,
    detect_class_token,
    detect_identity_invariants,
)
from lora_factory.caption.normalizer import NormalizedTag, normalize_tag_scores
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY, OntologyCategory, TagOntology

CHARACTER_FORBIDDEN = (
    OntologyCategory.ARTIST,
    OntologyCategory.CHARACTER,
    OntologyCategory.COPYRIGHT,
    OntologyCategory.STYLE,
    OntologyCategory.QUALITY,
    OntologyCategory.RATING,
    OntologyCategory.RESOLUTION,
    OntologyCategory.TEXT_WATERMARK,
)

TAG_ORDER = tuple(
    category for category in OntologyCategory if category is not OntologyCategory.UNCLASSIFIED
)
TAG_ORDER_INDEX: dict[OntologyCategory, int] = {
    category: index for index, category in enumerate(TAG_ORDER)
}


class CharacterCaptionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_confidence: Annotated[float, Field(ge=0, le=1)] = 0.35
    max_tags: Annotated[int, Field(ge=2, le=256)] = 75
    forbidden_categories: tuple[OntologyCategory, ...] = CHARACTER_FORBIDDEN
    # Accepted from versioned presets; runtime keep_tokens is still derived from class stability.
    keep_tokens: Literal[1, 2] | None = None
    class_token: ClassTokenPolicyConfig = Field(default_factory=ClassTokenPolicyConfig)
    invariant: InvariantPolicyConfig = Field(default_factory=InvariantPolicyConfig)


class CharacterCaptionBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    captions: dict[str, str]
    class_token: ClassTokenDecision
    invariants: InvariantAnalysis
    keep_tokens: Annotated[int, Field(ge=1, le=2)]
    removed_tags: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def _ordered(tags: Iterable[NormalizedTag]) -> list[NormalizedTag]:
    return sorted(
        tags,
        key=lambda item: (
            TAG_ORDER_INDEX.get(item.category, len(TAG_ORDER_INDEX)),
            -item.score,
            item.name,
        ),
    )


def build_character_captions(
    trigger_token: str,
    image_tags: Mapping[str, TagInput],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    config: CharacterCaptionConfig | None = None,
    duplicate_cluster_ids: Mapping[str, str] | None = None,
) -> CharacterCaptionBatch:
    policy = config or CharacterCaptionConfig()
    trigger = trigger_token.strip()
    if not trigger or any(separator in trigger for separator in (",", "\n", "\r", ";", "|")):
        raise ValueError("Trigger token is empty or contains a caption separator")
    class_decision = detect_class_token(
        image_tags,
        ontology=ontology,
        config=policy.class_token,
        cluster_ids=duplicate_cluster_ids,
    )
    invariants = detect_identity_invariants(
        image_tags,
        ontology=ontology,
        config=policy.invariant,
        cluster_ids=duplicate_cluster_ids,
    )
    invariant_set = set(invariants.selected_tags)
    # Contractual source/quality exclusions cannot be weakened by a partial preset.
    forbidden = set(CHARACTER_FORBIDDEN).union(policy.forbidden_categories)
    captions: dict[str, str] = {}
    removed: dict[str, tuple[str, ...]] = {}
    for asset_id in sorted(image_tags):
        normalized = normalize_tag_scores(
            image_tags[asset_id], ontology=ontology, minimum_confidence=policy.min_confidence
        )
        kept: list[NormalizedTag] = []
        removed_for_asset: list[str] = []
        for tag in normalized:
            if tag.category in forbidden or tag.name in invariant_set:
                removed_for_asset.append(tag.name)
                continue
            if class_decision.token is not None and tag.category is OntologyCategory.PERSON_CLASS:
                removed_for_asset.append(tag.name)
                continue
            kept.append(tag)
        fixed = [trigger]
        if class_decision.token is not None:
            fixed.append(ontology.display(class_decision.token))
        room = max(0, policy.max_tags - len(fixed))
        variable = [ontology.display(item.name) for item in _ordered(kept)[:room]]
        captions[asset_id] = ", ".join((*fixed, *variable))
        removed[asset_id] = tuple(sorted(set(removed_for_asset)))
    return CharacterCaptionBatch(
        captions=captions,
        class_token=class_decision,
        invariants=invariants,
        keep_tokens=class_decision.keep_tokens,
        removed_tags=removed,
        warnings=invariants.warnings,
    )


class CharacterCaptionPolicy:
    def __init__(
        self,
        config: CharacterCaptionConfig | None = None,
        ontology: TagOntology = DEFAULT_ONTOLOGY,
    ) -> None:
        self.config = config or CharacterCaptionConfig()
        self.ontology = ontology

    def build(
        self,
        trigger_token: str,
        image_tags: Mapping[str, TagInput],
        *,
        duplicate_cluster_ids: Mapping[str, str] | None = None,
    ) -> CharacterCaptionBatch:
        return build_character_captions(
            trigger_token,
            image_tags,
            ontology=self.ontology,
            config=self.config,
            duplicate_cluster_ids=duplicate_cluster_ids,
        )
