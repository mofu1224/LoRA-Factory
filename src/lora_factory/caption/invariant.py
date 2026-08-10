"""Conservative stable class-token and identity-invariant inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.normalizer import normalized_score_map
from lora_factory.caption.tagger import TagScore
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY, OntologyCategory, TagOntology

TagInput = Mapping[str, float] | Iterable[TagScore]


class ClassTokenPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    frequency_threshold: Annotated[float, Field(ge=0, le=1)] = 0.60
    mean_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.45
    ambiguity_margin: Annotated[float, Field(ge=0, le=1)] = 0.15
    presence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.35


class ClassTokenDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str | None
    stable: bool
    frequency: Annotated[float, Field(ge=0, le=1)] = 0.0
    mean_confidence: Annotated[float, Field(ge=0, le=1)] = 0.0
    runner_up_frequency: Annotated[float, Field(ge=0, le=1)] = 0.0
    keep_tokens: Annotated[int, Field(ge=1, le=2)]
    reason: str


class InvariantPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85
    mean_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.55
    presence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.35
    max_conflict_ratio: Annotated[float, Field(ge=0, le=1)] = 0.10
    max_cluster_dominance: Annotated[float, Field(ge=0, le=1)] = 0.50


class InvariantCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str
    category: OntologyCategory
    raw_frequency: Annotated[float, Field(ge=0, le=1)]
    deduplicated_frequency: Annotated[float, Field(ge=0, le=1)]
    mean_confidence: Annotated[float, Field(ge=0, le=1)]
    conflict_ratio: Annotated[float, Field(ge=0, le=1)]
    cluster_dominance: Annotated[float, Field(ge=0, le=1)]
    selected: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)


class InvariantAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_tags: tuple[str, ...]
    candidates: tuple[InvariantCandidate, ...]
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def _score_maps(
    image_tags: Mapping[str, TagInput], ontology: TagOntology
) -> dict[str, dict[str, float]]:
    return {
        asset_id: normalized_score_map(tags, ontology=ontology)
        for asset_id, tags in image_tags.items()
    }


def _cluster_groups(
    asset_ids: Iterable[str], cluster_ids: Mapping[str, str] | None
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for asset_id in asset_ids:
        groups[cluster_ids.get(asset_id, asset_id) if cluster_ids else asset_id].append(asset_id)
    return {key: tuple(values) for key, values in groups.items()}


def detect_class_token(
    image_tags: Mapping[str, TagInput],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    config: ClassTokenPolicyConfig | None = None,
    cluster_ids: Mapping[str, str] | None = None,
) -> ClassTokenDecision:
    policy = config or ClassTokenPolicyConfig()
    if not policy.enabled:
        return ClassTokenDecision(
            token=None, stable=False, keep_tokens=1, reason="Class-token detection is disabled"
        )
    scores = _score_maps(image_tags, ontology)
    if not scores:
        return ClassTokenDecision(
            token=None, stable=False, keep_tokens=1, reason="Dataset contains no tag results"
        )
    groups = _cluster_groups(scores, cluster_ids)
    class_tags = sorted(
        {
            tag
            for score_map in scores.values()
            for tag in score_map
            if ontology.classify(tag) is OntologyCategory.PERSON_CLASS
        }
    )
    statistics: list[tuple[str, float, float]] = []
    for tag in class_tags:
        cluster_scores = [
            max((scores[asset_id].get(tag, 0.0) for asset_id in members), default=0.0)
            for members in groups.values()
        ]
        frequency = sum(value >= policy.presence_threshold for value in cluster_scores) / len(
            groups
        )
        mean = sum(cluster_scores) / len(groups)
        statistics.append((tag, frequency, mean))
    statistics.sort(key=lambda item: (-item[1], -item[2], item[0]))
    if not statistics:
        return ClassTokenDecision(
            token=None,
            stable=False,
            keep_tokens=1,
            reason="No person class was detected; class token was not invented",
        )
    tag, frequency, mean = statistics[0]
    runner_up = statistics[1][1] if len(statistics) > 1 else 0.0
    stable = (
        frequency >= policy.frequency_threshold
        and mean >= policy.mean_confidence_threshold
        and frequency - runner_up >= policy.ambiguity_margin
    )
    if stable:
        reason = (
            f"{tag} is stable after duplicate-cluster deweighting "
            f"(frequency={frequency:.3f}, mean={mean:.3f})"
        )
    else:
        reason = (
            "Person class is mixed or uncertain; no fixed class token "
            f"(best={tag}, frequency={frequency:.3f}, mean={mean:.3f}, "
            f"runner_up={runner_up:.3f})"
        )
    return ClassTokenDecision(
        token=tag if stable else None,
        stable=stable,
        frequency=frequency,
        mean_confidence=mean,
        runner_up_frequency=runner_up,
        keep_tokens=2 if stable else 1,
        reason=reason,
    )


IDENTITY_CATEGORIES = frozenset(
    {
        OntologyCategory.APPEARANCE,
        OntologyCategory.SPECIES,
        OntologyCategory.PERMANENT_ACCESSORY,
    }
)


def detect_identity_invariants(
    image_tags: Mapping[str, TagInput],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    config: InvariantPolicyConfig | None = None,
    cluster_ids: Mapping[str, str] | None = None,
) -> InvariantAnalysis:
    policy = config or InvariantPolicyConfig()
    scores = _score_maps(image_tags, ontology)
    if not scores:
        return InvariantAnalysis(selected_tags=(), candidates=(), warnings=("No tag results",))
    groups = _cluster_groups(scores, cluster_ids)
    eligible = sorted(
        {
            tag
            for values in scores.values()
            for tag in values
            if ontology.classify(tag) in IDENTITY_CATEGORIES
        }
    )
    candidates: list[InvariantCandidate] = []
    warnings: list[str] = []
    for tag in eligible:
        category = ontology.classify(tag)
        raw_hits = [
            asset_id
            for asset_id, values in scores.items()
            if values.get(tag, 0.0) >= policy.presence_threshold
        ]
        raw_frequency = len(raw_hits) / len(scores)
        group_values = {
            group_id: max((scores[item].get(tag, 0.0) for item in members), default=0.0)
            for group_id, members in groups.items()
        }
        deduplicated_frequency = sum(
            value >= policy.presence_threshold for value in group_values.values()
        ) / len(groups)
        mean_confidence = sum(group_values.values()) / len(groups)
        conflict_tags = ontology.conflicts_for(tag)
        conflict_ratio = (
            sum(
                any(
                    values.get(conflict, 0.0) >= policy.presence_threshold
                    for conflict in conflict_tags
                )
                for values in scores.values()
            )
            / len(scores)
            if conflict_tags
            else 0.0
        )
        hit_clusters = Counter(
            cluster_ids.get(asset_id, asset_id) if cluster_ids else asset_id
            for asset_id in raw_hits
        )
        cluster_dominance = (
            max(hit_clusters.values(), default=0) / len(raw_hits) if raw_hits else 0.0
        )
        reasons: list[str] = []
        if raw_frequency < policy.frequency_threshold:
            reasons.append("raw frequency below threshold")
        if deduplicated_frequency < policy.frequency_threshold:
            reasons.append("frequency falls below threshold after near-duplicate deweighting")
        if mean_confidence < policy.mean_confidence_threshold:
            reasons.append("mean confidence below threshold")
        if conflict_ratio > policy.max_conflict_ratio:
            reasons.append("contradictory tags are too frequent")
        if cluster_ids and cluster_dominance > policy.max_cluster_dominance:
            reasons.append("one near-duplicate cluster supplies too much evidence")
        selected = not reasons
        candidate = InvariantCandidate(
            tag=tag,
            category=category,
            raw_frequency=raw_frequency,
            deduplicated_frequency=deduplicated_frequency,
            mean_confidence=mean_confidence,
            conflict_ratio=conflict_ratio,
            cluster_dominance=cluster_dominance,
            selected=selected,
            reasons=tuple(reasons),
        )
        candidates.append(candidate)
        if not selected and raw_frequency >= policy.frequency_threshold:
            warnings.append(
                f"Invariant candidate {tag!r} retained in captions: {', '.join(reasons)}"
            )
    selected_tags = tuple(item.tag for item in candidates if item.selected)
    return InvariantAnalysis(
        selected_tags=selected_tags,
        candidates=tuple(candidates),
        warnings=tuple(warnings),
    )


detect_invariants = detect_identity_invariants
