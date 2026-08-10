"""Transparent weighted ranking with technical hard gates and overfit penalty."""

from __future__ import annotations

from collections.abc import Mapping

from lora_factory.config.models import PresetKind
from lora_factory.evaluation.models import CandidateMetrics, RankedCandidate

CHARACTER_WEIGHTS = {
    "identity": 0.35,
    "prompt_compliance": 0.20,
    "flexibility": 0.15,
    "consistency": 0.10,
    "validation": 0.10,
    "technical": 0.05,
    "visual_heuristic": 0.05,
}
STYLE_WEIGHTS = {
    "style_similarity": 0.30,
    "prompt_compliance": 0.20,
    "flexibility": 0.20,
    "consistency": 0.10,
    "validation": 0.10,
    "technical": 0.05,
    "visual_heuristic": 0.05,
}


def score_candidate(
    candidate: CandidateMetrics,
    preset: PresetKind,
    *,
    weights: Mapping[str, float] | None = None,
    overfit_penalty_weight: float = 0.2,
) -> RankedCandidate:
    configured = dict(
        weights or (CHARACTER_WEIGHTS if preset is PresetKind.CHARACTER else STYLE_WEIGHTS)
    )
    aliases = {
        "content_compliance": "prompt_compliance",
        "content_flexibility": "flexibility",
    }
    score = sum(
        float(getattr(candidate, aliases.get(key, key))) * weight
        for key, weight in configured.items()
    )
    score = max(0.0, min(1.0, score - candidate.overfit_risk * overfit_penalty_weight))
    warnings: list[str] = []
    gate_passed = not candidate.corrupted and candidate.generation_failure_rate <= 0.25
    if candidate.corrupted:
        warnings.append("Checkpoint is corrupted or unreadable.")
    if candidate.generation_failure_rate > 0.25:
        warnings.append("Generation failure rate exceeds the hard-gate threshold.")
    if candidate.overfit_risk >= 0.7:
        warnings.append("High memorization/overfit risk detected.")
    if not gate_passed:
        score = 0.0
    return RankedCandidate(
        checkpoint_id=candidate.checkpoint_id,
        score=score,
        recommended_weight=candidate.recommended_weight,
        hard_gate_passed=gate_passed,
        warnings=tuple(warnings),
        evidence=candidate.evidence,
    )


def rank_candidates(
    candidates: list[CandidateMetrics],
    preset: PresetKind,
    *,
    weights: Mapping[str, float] | None = None,
    overfit_penalty_weight: float = 0.2,
) -> tuple[RankedCandidate, ...]:
    ranked = [
        score_candidate(
            candidate,
            preset,
            weights=weights,
            overfit_penalty_weight=overfit_penalty_weight,
        )
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: (-int(item.hard_gate_passed), -item.score, item.checkpoint_id))
    return tuple(ranked)
