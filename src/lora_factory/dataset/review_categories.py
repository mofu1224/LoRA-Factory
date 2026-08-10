"""Explainable multi-category labels for the Dataset Review surface."""

from __future__ import annotations

from enum import StrEnum

from lora_factory.dataset.quality import QualityDisposition


class ReviewCategory(StrEnum):
    ACCEPTED = "Accepted"
    WARNING = "Warning"
    REJECTED = "Rejected"
    DUPLICATE = "Duplicate"
    VALIDATION = "Validation"


def review_categories(
    *,
    included: bool,
    disposition: QualityDisposition,
    duplicate: bool,
    validation: bool,
    embedding_outlier: bool = False,
) -> tuple[ReviewCategory, ...]:
    """Return stable labels without hiding overlapping review signals."""

    categories: list[ReviewCategory] = [
        ReviewCategory.ACCEPTED if included else ReviewCategory.REJECTED
    ]
    if disposition is QualityDisposition.WARN or embedding_outlier:
        categories.append(ReviewCategory.WARNING)
    if duplicate:
        categories.append(ReviewCategory.DUPLICATE)
    if included and validation:
        categories.append(ReviewCategory.VALIDATION)
    return tuple(categories)


def primary_review_category(categories: tuple[ReviewCategory, ...]) -> ReviewCategory:
    """Choose a backward-compatible primary label while retaining every category."""

    for category in (
        ReviewCategory.REJECTED,
        ReviewCategory.VALIDATION,
        ReviewCategory.DUPLICATE,
        ReviewCategory.WARNING,
        ReviewCategory.ACCEPTED,
    ):
        if category in categories:
            return category
    raise ValueError("Dataset review item has no category")
