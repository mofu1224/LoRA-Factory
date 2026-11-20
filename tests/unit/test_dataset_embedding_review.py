from __future__ import annotations

import numpy as np
import pytest

from lora_factory.dataset.clustering import MAX_PAIRWISE_COMPARISONS
from lora_factory.dataset.embedding_review import (
    analyze_dataset_embeddings,
    combine_duplicate_cluster_ids,
)


def test_dataset_embedding_review_reports_centroid_duplicates_and_conservative_outlier() -> None:
    asset_ids = ("near-a", "near-b", "ordinary", "outlier")
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99999, 0.001, 0.0],
            [0.9, 0.3, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    report = analyze_dataset_embeddings(asset_ids, embeddings)

    assert report.near_duplicate_clusters == (("near-a", "near-b"),)
    assert report.outlier_asset_ids == ("outlier",)
    assert report.centroid_similarity_by_asset["near-a"] > 0.9
    assert report.centroid_similarity_by_asset["outlier"] == 0.0
    assert len(report.warnings) == 2


def test_dataset_embedding_review_does_not_flag_moderate_variation_as_outlier() -> None:
    report = analyze_dataset_embeddings(
        ("a", "b", "c"),
        np.asarray(
            [[1.0, 0.0], [0.95, 0.2], [0.85, 0.35]],
            dtype=np.float32,
        ),
    )

    assert report.outlier_asset_ids == ()
    assert all(0 <= score <= 1 for score in report.centroid_similarity_by_asset.values())


def test_dataset_embedding_review_rejects_misaligned_matrix() -> None:
    with pytest.raises(ValueError, match="align"):
        analyze_dataset_embeddings(("a", "b"), np.ones((1, 3), dtype=np.float32))


def test_dataset_embedding_review_rejects_unbounded_pairwise_work() -> None:
    item_count = next(
        count for count in range(1, 100_000) if count * (count - 1) // 2 > MAX_PAIRWISE_COMPARISONS
    )

    with pytest.raises(ValueError, match="pairwise safety limit"):
        analyze_dataset_embeddings(
            tuple(f"asset-{index}" for index in range(item_count)),
            np.ones((item_count, 1), dtype=np.float32),
        )


def test_duplicate_cluster_merge_connects_phash_and_embedding_relationships() -> None:
    merged = combine_duplicate_cluster_ids(
        ("a", "b", "c", "single"),
        {"a": "phash", "b": "phash"},
        (("b", "c"),),
    )

    assert merged["a"] == merged["b"] == merged["c"]
    assert "single" not in merged
