"""Style captions retain semantic content and remove names/style labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.character_policy import _ordered
from lora_factory.caption.invariant import TagInput
from lora_factory.caption.normalizer import normalize_tag_scores
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY, OntologyCategory, TagOntology

STYLE_FORBIDDEN = (
    OntologyCategory.ARTIST,
    OntologyCategory.STYLE,
    OntologyCategory.COPYRIGHT,
    OntologyCategory.CHARACTER,
    OntologyCategory.QUALITY,
    OntologyCategory.RATING,
    OntologyCategory.RESOLUTION,
    OntologyCategory.TEXT_WATERMARK,
)


class StyleCaptionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_confidence: Annotated[float, Field(ge=0, le=1)] = 0.35
    max_tags: Annotated[int, Field(ge=1, le=256)] = 75
    keep_tokens: Annotated[int, Field(ge=1, le=1)] = 1
    forbidden_categories: tuple[OntologyCategory, ...] = STYLE_FORBIDDEN
    # Explicit None values keep Character/Style preset schemas structurally compatible.
    class_token: None = None
    invariant: None = None


class StyleCaptionBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    captions: dict[str, str]
    keep_tokens: Annotated[int, Field(ge=1, le=1)] = 1
    removed_tags: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def build_style_captions(
    trigger_token: str,
    image_tags: Mapping[str, TagInput],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    config: StyleCaptionConfig | None = None,
) -> StyleCaptionBatch:
    policy = config or StyleCaptionConfig()
    trigger = trigger_token.strip()
    if not trigger or any(separator in trigger for separator in (",", "\n", "\r", ";", "|")):
        raise ValueError("Trigger token is empty or contains a caption separator")
    # Contractual name/style/quality exclusions cannot be weakened by a partial preset.
    forbidden = set(STYLE_FORBIDDEN).union(policy.forbidden_categories)
    captions: dict[str, str] = {}
    removed: dict[str, tuple[str, ...]] = {}
    warnings: list[str] = []
    for asset_id in sorted(image_tags):
        normalized = normalize_tag_scores(
            image_tags[asset_id], ontology=ontology, minimum_confidence=policy.min_confidence
        )
        semantic = [tag for tag in normalized if tag.category not in forbidden]
        removed_tags = [tag.name for tag in normalized if tag.category in forbidden]
        room = max(0, policy.max_tags - 1)
        caption_tags = [
            trigger,
            *(ontology.display(item.name) for item in _ordered(semantic)[:room]),
        ]
        captions[asset_id] = ", ".join(caption_tags)
        removed[asset_id] = tuple(sorted(set(removed_tags)))
        if not semantic:
            warnings.append(f"{asset_id}: no semantic content tag passed the threshold")
    return StyleCaptionBatch(
        captions=captions,
        keep_tokens=1,
        removed_tags=removed,
        warnings=tuple(warnings),
    )


class StyleCaptionPolicy:
    def __init__(
        self,
        config: StyleCaptionConfig | None = None,
        ontology: TagOntology = DEFAULT_ONTOLOGY,
    ) -> None:
        self.config = config or StyleCaptionConfig()
        self.ontology = ontology

    def build(self, trigger_token: str, image_tags: Mapping[str, TagInput]) -> StyleCaptionBatch:
        return build_style_captions(
            trigger_token, image_tags, ontology=self.ontology, config=self.config
        )
