"""Exact and perceptual duplicate detection with explainable representative selection."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import imagehash
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from lora_factory.dataset.clustering import UnionFind
from lora_factory.dataset.quality import QualityAssessment, assess_image
from lora_factory.util.hashing import sha256_file


class DuplicateKind(StrEnum):
    EXACT = "exact"
    NEAR = "near"


class DuplicateCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    path: Path
    sha256: str | None = None
    area: Annotated[int, Field(ge=0)] | None = None
    sharpness: Annotated[float, Field(ge=0)] | None = None
    compression_blockiness: Annotated[float, Field(ge=0)] | None = None
    alpha_fraction: Annotated[float, Field(ge=0, le=1)] | None = None
    crop_completeness: Annotated[float, Field(ge=0, le=1)] = 1.0
    watermark_probability: Annotated[float, Field(ge=0, le=1)] = 0.0

    @classmethod
    def from_assessment(
        cls,
        asset_id: str,
        assessment: QualityAssessment,
        *,
        sha256: str | None = None,
        crop_completeness: float = 1.0,
        watermark_probability: float = 0.0,
    ) -> DuplicateCandidate:
        return cls(
            asset_id=asset_id,
            path=assessment.path,
            sha256=sha256,
            area=assessment.metrics.area,
            sharpness=assessment.metrics.sharpness,
            compression_blockiness=assessment.metrics.compression_blockiness,
            alpha_fraction=assessment.metrics.alpha_fraction,
            crop_completeness=crop_completeness,
            watermark_probability=watermark_probability,
        )


class RepresentativeScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    total: Annotated[float, Field(ge=0, le=1)]
    resolution: Annotated[float, Field(ge=0, le=1)]
    sharpness: Annotated[float, Field(ge=0, le=1)]
    compression_quality: Annotated[float, Field(ge=0, le=1)]
    alpha_quality: Annotated[float, Field(ge=0, le=1)]
    crop_completeness: Annotated[float, Field(ge=0, le=1)]
    watermark_quality: Annotated[float, Field(ge=0, le=1)]


class DuplicateCluster(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str
    kind: DuplicateKind
    members: tuple[str, ...]
    representative_id: str
    rejected_exact_ids: tuple[str, ...] = Field(default_factory=tuple)
    scores: tuple[RepresentativeScore, ...]
    rationale: str


class DuplicateReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    clusters: tuple[DuplicateCluster, ...] = Field(default_factory=tuple)
    perceptual_hashes: dict[str, str] = Field(default_factory=dict)
    exact_hashes: dict[str, str] = Field(default_factory=dict)
    largest_cluster_ratio: Annotated[float, Field(ge=0, le=1)] = 0.0


def _perceptual_hash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as opened:
        opened.seek(0)
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return imagehash.phash(image)


def _hydrate(candidate: DuplicateCandidate) -> DuplicateCandidate:
    if (
        all(
            value is not None
            for value in (
                candidate.area,
                candidate.sharpness,
                candidate.compression_blockiness,
                candidate.alpha_fraction,
            )
        )
        and candidate.sha256 is not None
    ):
        return candidate
    assessment = assess_image(candidate.path)
    return candidate.model_copy(
        update={
            "sha256": candidate.sha256 or sha256_file(candidate.path),
            "area": candidate.area if candidate.area is not None else assessment.metrics.area,
            "sharpness": candidate.sharpness
            if candidate.sharpness is not None
            else assessment.metrics.sharpness,
            "compression_blockiness": candidate.compression_blockiness
            if candidate.compression_blockiness is not None
            else assessment.metrics.compression_blockiness,
            "alpha_fraction": candidate.alpha_fraction
            if candidate.alpha_fraction is not None
            else assessment.metrics.alpha_fraction,
        }
    )


def _normalize_log(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return max(
        0.0, min(1.0, (math.log1p(value) - math.log1p(low)) / (math.log1p(high) - math.log1p(low)))
    )


def _score_cluster(candidates: list[DuplicateCandidate]) -> tuple[RepresentativeScore, ...]:
    areas = [float(item.area or 0) for item in candidates]
    sharpness_values = [float(item.sharpness or 0) for item in candidates]
    scores: list[RepresentativeScore] = []
    for item in candidates:
        resolution = _normalize_log(float(item.area or 0), min(areas), max(areas))
        sharpness = _normalize_log(
            float(item.sharpness or 0), min(sharpness_values), max(sharpness_values)
        )
        compression_quality = max(
            0.0, min(1.0, 1.0 - float(item.compression_blockiness or 0) / 3.0)
        )
        alpha_quality = 1.0 - float(item.alpha_fraction or 0)
        watermark_quality = 1.0 - item.watermark_probability
        total = (
            0.35 * resolution
            + 0.25 * sharpness
            + 0.15 * compression_quality
            + 0.10 * alpha_quality
            + 0.10 * item.crop_completeness
            + 0.05 * watermark_quality
        )
        scores.append(
            RepresentativeScore(
                asset_id=item.asset_id,
                total=max(0.0, min(1.0, total)),
                resolution=resolution,
                sharpness=sharpness,
                compression_quality=compression_quality,
                alpha_quality=alpha_quality,
                crop_completeness=item.crop_completeness,
                watermark_quality=watermark_quality,
            )
        )
    return tuple(sorted(scores, key=lambda score: (-score.total, score.asset_id)))


def detect_duplicates(
    candidates: Iterable[DuplicateCandidate],
    *,
    phash_max_distance: Annotated[int, Field(ge=0, le=64)] = 4,
) -> DuplicateReport:
    """Return connected pHash clusters; exact duplicates are always connected."""

    hydrated = sorted((_hydrate(item) for item in candidates), key=lambda item: item.asset_id)
    if len({item.asset_id for item in hydrated}) != len(hydrated):
        raise ValueError("Duplicate candidate asset_id values must be unique")
    hashes = {item.asset_id: str(_perceptual_hash(item.path)) for item in hydrated}
    exact = {item.asset_id: str(item.sha256) for item in hydrated}
    parsed_hashes = {key: imagehash.hex_to_hash(value) for key, value in hashes.items()}
    union = UnionFind(len(hydrated))
    for left, left_item in enumerate(hydrated):
        for right in range(left + 1, len(hydrated)):
            right_item = hydrated[right]
            if left_item.sha256 == right_item.sha256 or (
                parsed_hashes[left_item.asset_id] - parsed_hashes[right_item.asset_id]
                <= phash_max_distance
            ):
                union.union(left, right)

    clusters: list[DuplicateCluster] = []
    for indexes in union.components():
        if len(indexes) < 2:
            continue
        group = [hydrated[index] for index in indexes]
        scores = _score_cluster(group)
        representative = scores[0].asset_id
        score_by_id = {score.asset_id: score for score in scores}
        exact_groups: dict[str, list[str]] = {}
        for item in group:
            exact_groups.setdefault(str(item.sha256), []).append(item.asset_id)
        rejected_exact: list[str] = []
        for exact_members in exact_groups.values():
            if len(exact_members) < 2:
                continue
            exact_representative = min(
                exact_members,
                key=lambda asset_id: (-score_by_id[asset_id].total, asset_id),
            )
            rejected_exact.extend(
                asset_id for asset_id in exact_members if asset_id != exact_representative
            )
        kind = (
            DuplicateKind.EXACT if len({item.sha256 for item in group}) == 1 else DuplicateKind.NEAR
        )
        members = tuple(sorted(item.asset_id for item in group))
        cluster_id = hashlib.sha256("\0".join(members).encode()).hexdigest()[:16]
        clusters.append(
            DuplicateCluster(
                cluster_id=cluster_id,
                kind=kind,
                members=members,
                representative_id=representative,
                rejected_exact_ids=tuple(sorted(rejected_exact)),
                scores=scores,
                rationale=(
                    "Representative maximizes weighted resolution, sharpness, compression, alpha, "
                    "crop completeness, and watermark quality"
                ),
            )
        )
    clusters.sort(key=lambda cluster: cluster.cluster_id)
    largest = max((len(cluster.members) for cluster in clusters), default=0)
    ratio = largest / len(hydrated) if hydrated else 0.0
    return DuplicateReport(
        clusters=tuple(clusters),
        perceptual_hashes=hashes,
        exact_hashes=exact,
        largest_cluster_ratio=ratio,
    )


class DuplicateDetector:
    def __init__(self, *, phash_max_distance: int = 4) -> None:
        self.phash_max_distance = phash_max_distance

    def analyze(self, candidates: Iterable[DuplicateCandidate]) -> DuplicateReport:
        return detect_duplicates(candidates, phash_max_distance=self.phash_max_distance)
