"""Deterministic preset-aware holdout splitting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.config.models import PresetKind


class ValidationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled_at: Annotated[int, Field(ge=1)]
    ratio: Annotated[float, Field(gt=0, lt=1)] = 0.10
    min_count: Annotated[int, Field(ge=1)]
    max_count: Annotated[int, Field(ge=1)]
    seed: int = 42

    @classmethod
    def for_preset(cls, preset: PresetKind, *, seed: int = 42) -> ValidationPolicy:
        if preset is PresetKind.CHARACTER:
            return cls(enabled_at=24, min_count=2, max_count=8, seed=seed)
        return cls(enabled_at=40, min_count=4, max_count=16, seed=seed)


class DatasetSplit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    training: tuple[str, ...]
    validation: tuple[str, ...]
    validation_enabled: bool
    seed: int
    reason: str


def validation_count(total: int, policy: ValidationPolicy) -> int:
    if total < policy.enabled_at:
        return 0
    rounded_half_up = math.floor(total * policy.ratio + 0.5)
    return min(policy.max_count, max(policy.min_count, rounded_half_up))


def _rank(seed: int, key: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{key}".encode()).digest()
    return digest, key


def deterministic_validation_split(
    asset_ids: Iterable[str],
    preset: PresetKind,
    *,
    seed: int = 42,
    cluster_ids: Mapping[str, str] | None = None,
    policy: ValidationPolicy | None = None,
) -> DatasetSplit:
    """Split stable IDs, keeping supplied duplicate clusters on the same side."""

    unique_ids = tuple(sorted(set(asset_ids)))
    limits = policy or ValidationPolicy.for_preset(preset, seed=seed)
    wanted = validation_count(len(unique_ids), limits)
    if wanted == 0:
        return DatasetSplit(
            training=unique_ids,
            validation=(),
            validation_enabled=False,
            seed=limits.seed,
            reason=f"Validation requires at least {limits.enabled_at} accepted images",
        )

    groups: dict[str, list[str]] = {}
    for asset_id in unique_ids:
        group_id = cluster_ids.get(asset_id, asset_id) if cluster_ids else asset_id
        groups.setdefault(group_id, []).append(asset_id)
    ranked_groups = sorted(groups.items(), key=lambda item: _rank(limits.seed, item[0]))
    selected: set[str] = set()
    for _, members in ranked_groups:
        if len(selected) >= wanted:
            break
        # Prefer the closer side of the target; a duplicate group is never split.
        current_delta = abs(wanted - len(selected))
        new_delta = abs(wanted - (len(selected) + len(members)))
        if not selected or new_delta <= current_delta:
            selected.update(members)
    if not selected:
        selected.update(ranked_groups[0][1])

    validation = tuple(sorted(selected))
    training = tuple(asset_id for asset_id in unique_ids if asset_id not in selected)
    return DatasetSplit(
        training=training,
        validation=validation,
        validation_enabled=True,
        seed=limits.seed,
        reason=(
            f"Deterministic {limits.ratio:.0%} holdout; target {wanted}, actual "
            f"{len(validation)} with duplicate clusters kept intact"
        ),
    )


make_validation_split = deterministic_validation_split
