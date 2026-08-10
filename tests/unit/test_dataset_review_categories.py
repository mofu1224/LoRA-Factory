from __future__ import annotations

from lora_factory.dataset.quality import QualityDisposition
from lora_factory.dataset.review_categories import (
    ReviewCategory,
    primary_review_category,
    review_categories,
)


def test_review_categories_preserve_overlapping_warning_duplicate_and_validation() -> None:
    categories = review_categories(
        included=True,
        disposition=QualityDisposition.WARN,
        duplicate=True,
        validation=True,
    )

    assert categories == (
        ReviewCategory.ACCEPTED,
        ReviewCategory.WARNING,
        ReviewCategory.DUPLICATE,
        ReviewCategory.VALIDATION,
    )
    assert primary_review_category(categories) is ReviewCategory.VALIDATION


def test_rejected_duplicate_remains_rejected_and_never_becomes_validation() -> None:
    categories = review_categories(
        included=False,
        disposition=QualityDisposition.WARN,
        duplicate=True,
        validation=True,
        embedding_outlier=True,
    )

    assert categories == (
        ReviewCategory.REJECTED,
        ReviewCategory.WARNING,
        ReviewCategory.DUPLICATE,
    )
    assert primary_review_category(categories) is ReviewCategory.REJECTED
