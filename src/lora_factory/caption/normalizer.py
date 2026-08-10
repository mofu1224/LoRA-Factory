"""Canonical tag normalization, thresholding, and duplicate resolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.parser import parse_tag_scores
from lora_factory.caption.tagger import TagScore
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY, OntologyCategory, TagOntology


class NormalizedTag(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    score: Annotated[float, Field(ge=0, le=1)]
    category: OntologyCategory
    model_category: str = "general"


def normalize_tag_scores(
    tags: Mapping[str, float] | Iterable[TagScore],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    minimum_confidence: float = 0.35,
    selected_only: bool = False,
) -> tuple[NormalizedTag, ...]:
    best: dict[str, NormalizedTag] = {}
    for raw in parse_tag_scores(tags):
        if raw.score < minimum_confidence or (selected_only and not raw.selected):
            continue
        canonical = ontology.canonicalize(raw.name)
        if not canonical:
            continue
        candidate = NormalizedTag(
            name=canonical,
            score=raw.score,
            category=ontology.classify(canonical, model_category=raw.model_category),
            model_category=raw.model_category,
        )
        previous = best.get(canonical)
        if previous is None or candidate.score > previous.score:
            best[canonical] = candidate
    return tuple(sorted(best.values(), key=lambda item: (-item.score, item.name)))


def normalized_score_map(
    tags: Mapping[str, float] | Iterable[TagScore],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
) -> dict[str, float]:
    return {
        item.name: item.score
        for item in normalize_tag_scores(tags, ontology=ontology, minimum_confidence=0.0)
    }
