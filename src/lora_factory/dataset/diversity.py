"""Ontology-aware content-diversity metrics for Character and Style review."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.tagger import TagScore
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY, OntologyCategory, TagOntology


class CategoryDiversity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: OntologyCategory
    unique_tags: Annotated[int, Field(ge=0)]
    coverage: Annotated[float, Field(ge=0, le=1)]
    normalized_entropy: Annotated[float, Field(ge=0, le=1)]
    dominant_tag: str | None = None
    dominant_ratio: Annotated[float, Field(ge=0, le=1)] = 0.0


class DiversityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_count: Annotated[int, Field(ge=0)]
    categories: dict[OntologyCategory, CategoryDiversity]
    content_diversity: Annotated[float, Field(ge=0, le=1)]
    dominant_character_ratio: Annotated[float, Field(ge=0, le=1)]
    dominant_copyright_ratio: Annotated[float, Field(ge=0, le=1)]
    near_duplicate_cluster_dominance: Annotated[float, Field(ge=0, le=1)]
    style_consistency: Annotated[float, Field(ge=0, le=1)]
    warnings: tuple[str, ...] = Field(default_factory=tuple)


CONTENT_CATEGORIES = (
    OntologyCategory.PERSON_CLASS,
    OntologyCategory.APPEARANCE,
    OntologyCategory.CLOTHING,
    OntologyCategory.POSE,
    OntologyCategory.COMPOSITION,
    OntologyCategory.BACKGROUND,
    OntologyCategory.OBJECT,
)


def _tag_entries(
    tags: Mapping[str, float] | Iterable[str] | Iterable[TagScore],
) -> Iterable[tuple[str, str | None]]:
    if isinstance(tags, Mapping):
        return ((tag, None) for tag in tags)
    return (
        (tag.name, tag.model_category) if isinstance(tag, TagScore) else (tag, None) for tag in tags
    )


def _category_metric(
    category: OntologyCategory,
    per_image: Mapping[str, set[str]],
    total: int,
) -> CategoryDiversity:
    counts = Counter(tag for tags in per_image.values() for tag in tags)
    covered = sum(bool(tags) for tags in per_image.values())
    occurrences = sum(counts.values())
    if occurrences == 0:
        entropy = 0.0
        dominant_tag = None
        dominant_ratio = 0.0
    else:
        probabilities = [count / occurrences for count in counts.values()]
        raw_entropy = -sum(value * math.log(value) for value in probabilities)
        entropy = raw_entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        dominant_tag, dominant_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        dominant_ratio = dominant_count / total if total else 0.0
    return CategoryDiversity(
        category=category,
        unique_tags=len(counts),
        coverage=covered / total if total else 0.0,
        normalized_entropy=max(0.0, min(1.0, entropy)),
        dominant_tag=dominant_tag,
        dominant_ratio=max(0.0, min(1.0, dominant_ratio)),
    )


def analyze_diversity(
    image_tags: Mapping[
        str,
        Mapping[str, float] | Iterable[str] | Iterable[TagScore],
    ],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    duplicate_cluster_ids: Mapping[str, str] | None = None,
    style_similarity_scores: Iterable[float] | None = None,
) -> DiversityReport:
    image_count = len(image_tags)
    categorized: dict[OntologyCategory, dict[str, set[str]]] = {
        category: {asset_id: set() for asset_id in image_tags} for category in OntologyCategory
    }
    for asset_id, tags in image_tags.items():
        for raw_tag, model_category in _tag_entries(tags):
            canonical = ontology.canonicalize(raw_tag)
            category = ontology.classify(canonical, model_category=model_category)
            categorized[category][asset_id].add(canonical)
    metrics = {
        category: _category_metric(category, values, image_count)
        for category, values in categorized.items()
    }
    available_content = [
        metrics[category].normalized_entropy
        for category in CONTENT_CATEGORIES
        if metrics[category].coverage > 0
    ]
    content_diversity = (
        sum(available_content) / len(available_content) if available_content else 0.0
    )
    if duplicate_cluster_ids and image_count:
        cluster_counts = Counter(
            duplicate_cluster_ids.get(asset_id, asset_id) for asset_id in image_tags
        )
        duplicate_dominance = max(cluster_counts.values(), default=0) / image_count
    else:
        duplicate_dominance = 0.0
    similarities = tuple(style_similarity_scores or ())
    style_consistency = (
        max(0.0, min(1.0, sum(similarities) / len(similarities))) if similarities else 1.0
    )
    character_ratio = metrics[OntologyCategory.CHARACTER].dominant_ratio
    copyright_ratio = metrics[OntologyCategory.COPYRIGHT].dominant_ratio
    warnings: list[str] = []
    if content_diversity < 0.25:
        warnings.append("Content diversity is low; style and subject may become entangled")
    if character_ratio > 0.75:
        warnings.append("One character dominates; consider Character or Character+Style")
    if duplicate_dominance > 0.35:
        warnings.append("A near-duplicate cluster dominates the dataset")
    if style_consistency < 0.25:
        warnings.append("Style signals are inconsistent")
    return DiversityReport(
        image_count=image_count,
        categories=metrics,
        content_diversity=content_diversity,
        dominant_character_ratio=character_ratio,
        dominant_copyright_ratio=copyright_ratio,
        near_duplicate_cluster_dominance=duplicate_dominance,
        style_consistency=style_consistency,
        warnings=tuple(warnings),
    )


compute_diversity = analyze_diversity
