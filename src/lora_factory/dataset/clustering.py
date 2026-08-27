"""Small deterministic clustering primitives used by deduplication and diversity analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

MAX_PAIRWISE_COMPARISONS = 2_000_000


def require_pairwise_budget(
    item_count: int,
    *,
    operation: str,
    max_pairwise_comparisons: int = MAX_PAIRWISE_COMPARISONS,
) -> None:
    """Reject inputs whose all-pairs work would exceed the local safety budget."""

    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if max_pairwise_comparisons < 1:
        raise ValueError("max_pairwise_comparisons must be positive")
    if max_pairwise_comparisons > MAX_PAIRWISE_COMPARISONS:
        raise ValueError(
            "max_pairwise_comparisons cannot exceed the global safety limit of "
            f"{MAX_PAIRWISE_COMPARISONS} comparisons"
        )
    comparisons = item_count * max(0, item_count - 1) // 2
    if comparisons > max_pairwise_comparisons:
        raise ValueError(
            f"{operation} exceeds the pairwise safety limit of "
            f"{max_pairwise_comparisons} comparisons"
        )


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1

    def components(self) -> tuple[tuple[int, ...], ...]:
        groups: dict[int, list[int]] = {}
        for item in range(len(self._parent)):
            groups.setdefault(self.find(item), []).append(item)
        return tuple(tuple(group) for group in sorted(groups.values(), key=lambda value: value[0]))


def cluster_by_pairwise[T](
    items: Sequence[T],
    related: Callable[[T, T], bool],
    *,
    max_pairwise_comparisons: int = MAX_PAIRWISE_COMPARISONS,
) -> tuple[tuple[T, ...], ...]:
    """Connected components under a symmetric pairwise relation."""

    require_pairwise_budget(
        len(items),
        operation="Pairwise clustering",
        max_pairwise_comparisons=max_pairwise_comparisons,
    )
    union = UnionFind(len(items))
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if related(items[left], items[right]):
                union.union(left, right)
    return tuple(tuple(items[index] for index in group) for group in union.components())


def cosine_similarity_clusters(
    asset_ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    minimum_similarity: float = 0.985,
) -> tuple[tuple[str, ...], ...]:
    """Cluster normalized embedding neighbors without requiring scikit-learn."""

    require_pairwise_budget(len(asset_ids), operation="Embedding clustering")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(asset_ids):
        raise ValueError("embeddings must be a 2D matrix aligned with asset_ids")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must be non-zero")
    normalized = matrix / norms
    union = UnionFind(len(asset_ids))
    block_size = 512
    for block_start in range(0, len(asset_ids), block_size):
        block_end = min(len(asset_ids), block_start + block_size)
        similarities = normalized[block_start:block_end] @ normalized.T
        for offset, left in enumerate(range(block_start, block_end)):
            for right in range(left + 1, len(asset_ids)):
                if float(similarities[offset, right]) >= minimum_similarity:
                    union.union(left, right)
    return tuple(
        tuple(asset_ids[index] for index in group) for group in union.components() if len(group) > 1
    )
