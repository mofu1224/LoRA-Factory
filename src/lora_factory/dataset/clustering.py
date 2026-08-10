"""Small deterministic clustering primitives used by deduplication and diversity analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


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
    items: Sequence[T], related: Callable[[T, T], bool]
) -> tuple[tuple[T, ...], ...]:
    """Connected components under a symmetric pairwise relation."""

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

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(asset_ids):
        raise ValueError("embeddings must be a 2D matrix aligned with asset_ids")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must be non-zero")
    normalized = matrix / norms
    similarities = normalized @ normalized.T
    union = UnionFind(len(asset_ids))
    for left in range(len(asset_ids)):
        for right in range(left + 1, len(asset_ids)):
            if float(similarities[left, right]) >= minimum_similarity:
                union.union(left, right)
    return tuple(
        tuple(asset_ids[index] for index in group) for group in union.components() if len(group) > 1
    )
