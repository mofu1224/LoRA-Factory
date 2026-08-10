"""Bounded normalized evaluation data."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_id: str
    identity: float = Field(default=0, ge=0, le=1)
    style_similarity: float = Field(default=0, ge=0, le=1)
    prompt_compliance: float = Field(ge=0, le=1)
    flexibility: float = Field(ge=0, le=1)
    consistency: float = Field(ge=0, le=1)
    validation: float = Field(ge=0, le=1)
    technical: float = Field(ge=0, le=1)
    visual_heuristic: float = Field(ge=0, le=1)
    overfit_risk: float = Field(default=0, ge=0, le=1)
    generation_failure_rate: float = Field(default=0, ge=0, le=1)
    corrupted: bool = False
    recommended_weight: float = Field(default=0.8, ge=0, le=2)
    evidence: dict[str, str | float | int | bool] = Field(default_factory=dict)


class RankedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_id: str
    score: float = Field(ge=0, le=1)
    recommended_weight: float = Field(ge=0, le=2)
    hard_gate_passed: bool
    warnings: tuple[str, ...] = ()
    evidence: dict[str, str | float | int | bool] = Field(default_factory=dict)
