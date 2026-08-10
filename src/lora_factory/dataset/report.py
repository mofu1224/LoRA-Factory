"""Dataset-level summaries consumed by review UI and planning services."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from statistics import median
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.dataset.quality import QualityAssessment, QualityDisposition


class DistributionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum: Annotated[float, Field(ge=0)] = 0.0
    median: Annotated[float, Field(ge=0)] = 0.0
    maximum: Annotated[float, Field(ge=0)] = 0.0


class DatasetAnalysisReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_count: Annotated[int, Field(ge=0)]
    accepted_count: Annotated[int, Field(ge=0)]
    warning_count: Annotated[int, Field(ge=0)]
    rejected_count: Annotated[int, Field(ge=0)]
    short_side: DistributionSummary
    area: DistributionSummary
    aspect_ratio: DistributionSummary
    reason_counts: dict[str, int]


def _distribution(values: list[float]) -> DistributionSummary:
    return DistributionSummary(
        minimum=min(values, default=0.0),
        median=median(values) if values else 0.0,
        maximum=max(values, default=0.0),
    )


def build_dataset_report(assessments: Iterable[QualityAssessment]) -> DatasetAnalysisReport:
    items = tuple(assessments)
    usable = [item for item in items if item.accepted]
    return DatasetAnalysisReport(
        total_count=len(items),
        accepted_count=len(usable),
        warning_count=sum(item.disposition is QualityDisposition.WARN for item in items),
        rejected_count=sum(item.disposition is QualityDisposition.REJECT for item in items),
        short_side=_distribution([float(item.metrics.short_side) for item in usable]),
        area=_distribution([float(item.metrics.area) for item in usable]),
        aspect_ratio=_distribution([item.metrics.aspect_ratio for item in usable]),
        reason_counts=dict(
            sorted(Counter(reason.code for item in items for reason in item.reasons).items())
        ),
    )
