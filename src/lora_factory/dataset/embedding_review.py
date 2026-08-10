"""Conservative learned-embedding signals for dataset review."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lora_factory.dataset.clustering import UnionFind, cosine_similarity_clusters


class DatasetEmbeddingSignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_count: Annotated[int, Field(ge=0)]
    centroid_similarity_by_asset: dict[str, Annotated[float, Field(ge=0, le=1)]]
    near_duplicate_threshold: Annotated[float, Field(ge=0, le=1)]
    near_duplicate_clusters: tuple[tuple[str, ...], ...]
    outlier_threshold: Annotated[float, Field(ge=0, le=1)]
    outlier_asset_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def combine_duplicate_cluster_ids(
    asset_ids: list[str] | tuple[str, ...],
    existing_cluster_ids: Mapping[str, str],
    embedding_clusters: tuple[tuple[str, ...], ...],
) -> dict[str, str]:
    """Merge pHash and embedding relationships into deterministic connected components."""

    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("Duplicate cluster asset IDs must be unique")
    index_by_id = {asset_id: index for index, asset_id in enumerate(asset_ids)}
    union = UnionFind(len(asset_ids))
    existing_groups: dict[str, list[str]] = {}
    for asset_id in asset_ids:
        cluster_id = existing_cluster_ids.get(asset_id)
        if cluster_id is not None:
            existing_groups.setdefault(cluster_id, []).append(asset_id)
    groups: list[Sequence[str]] = [*existing_groups.values(), *embedding_clusters]
    for group in groups:
        members = [asset_id for asset_id in group if asset_id in index_by_id]
        for member in members[1:]:
            union.union(index_by_id[members[0]], index_by_id[member])
    result: dict[str, str] = {}
    for indexes in union.components():
        if len(indexes) < 2:
            continue
        component_members = tuple(sorted(asset_ids[index] for index in indexes))
        digest = hashlib.sha256("\0".join(component_members).encode()).hexdigest()[:16]
        for asset_id in component_members:
            result[asset_id] = f"combined-{digest}"
    return result


def analyze_dataset_embeddings(
    asset_ids: list[str] | tuple[str, ...],
    embeddings: np.ndarray,
    *,
    near_duplicate_threshold: float = 0.995,
) -> DatasetEmbeddingSignals:
    """Return advisory CLIP signals without automatically rejecting any source image."""

    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("Embedding review asset IDs must be unique")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(asset_ids) or not asset_ids:
        raise ValueError("Embedding review matrix must align with a non-empty asset ID list")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding review vectors must be non-zero")
    normalized = matrix / norms
    centroid = normalized.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0:
        raise ValueError("Embedding review centroid must be non-zero")
    similarities = np.clip(normalized @ (centroid / centroid_norm), 0.0, 1.0)
    similarity_by_asset = {
        asset_id: float(similarity)
        for asset_id, similarity in zip(asset_ids, similarities, strict=True)
    }
    clusters = cosine_similarity_clusters(
        asset_ids,
        normalized,
        minimum_similarity=near_duplicate_threshold,
    )
    median = float(np.median(similarities))
    median_absolute_deviation = float(np.median(np.abs(similarities - median)))
    # Generic CLIP is advisory: require both a robust 3.5-MAD gap and a 0.10 absolute gap,
    # and never call a score >= 0.75 an outlier.
    outlier_threshold = min(
        0.75,
        max(0.0, median - max(0.10, 3.5 * median_absolute_deviation)),
    )
    outliers = tuple(
        asset_id
        for asset_id, similarity in similarity_by_asset.items()
        if similarity < outlier_threshold
    )
    warnings: list[str] = []
    if clusters:
        warnings.append(
            f"{len(clusters)} learned-embedding near-duplicate cluster(s) require manual review"
        )
    if outliers:
        warnings.append(
            f"{len(outliers)} conservative learned-embedding outlier(s) require manual review"
        )
    return DatasetEmbeddingSignals(
        image_count=len(asset_ids),
        centroid_similarity_by_asset=similarity_by_asset,
        near_duplicate_threshold=near_duplicate_threshold,
        near_duplicate_clusters=clusters,
        outlier_threshold=outlier_threshold,
        outlier_asset_ids=outliers,
        warnings=tuple(warnings),
    )
