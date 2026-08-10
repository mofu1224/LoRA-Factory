"""Explainable image-set signals used alongside tags and validation loss."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from re import findall

import numpy as np
from numpy.typing import NDArray
from PIL import Image

type FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ImageTechnicalMetrics:
    valid: bool
    mean_luminance: float
    standard_deviation: float
    black_or_blank: bool


@dataclass(frozen=True, slots=True)
class SampleObservation:
    path: Path
    prompt_id: str
    prompt: str
    seed: int
    weight: float
    success: bool
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateVisualSignals:
    reference_similarity: float
    prompt_compliance: float
    flexibility: float
    consistency: float
    technical: float
    visual_heuristic: float
    generation_failure_rate: float
    memorization_risk: float
    recommended_weight: float
    evidence: dict[str, str | float | int | bool]


def inspect_generated_image(path: Path) -> ImageTechnicalMetrics:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return ImageTechnicalMetrics(False, 0.0, 0.0, True)
    mean = float(array.mean())
    deviation = float(array.std())
    blank = deviation < 0.003 or mean < 0.003 or mean > 0.997
    return ImageTechnicalMetrics(True, mean, deviation, blank)


def normalize_values(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return dict.fromkeys(values, 0.5)
    span = maximum - minimum
    return {key: (value - minimum) / span for key, value in values.items()}


def image_descriptor(path: Path) -> FloatArray:
    """Return a compact color/texture/layout descriptor with no model dependency."""

    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot describe generated image: {path}") from exc
    channels = [
        np.histogram(rgb[:, :, index], bins=16, range=(0.0, 1.0))[0].astype(np.float32)
        for index in range(3)
    ]
    gray = rgb.mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1)).mean(axis=1)
    vertical = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    layout = np.asarray(
        Image.fromarray(np.asarray(gray * 255, dtype=np.uint8)).resize((8, 8)),
        dtype=np.float32,
    ).reshape(-1)
    statistics = np.concatenate(
        (
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            np.asarray([gray.mean(), gray.std()], dtype=np.float32),
        )
    ).astype(np.float32)
    descriptor = np.concatenate((*channels, horizontal, vertical, layout, statistics)).astype(
        np.float32
    )
    norm = float(np.linalg.norm(descriptor))
    return descriptor / norm if norm > 0 else descriptor


def cosine_similarity(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape:
        raise ValueError("Image descriptors must have identical shapes")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denominator)))


def _mean_pairwise(descriptors: list[FloatArray]) -> float:
    if len(descriptors) < 2:
        return 1.0 if descriptors else 0.0
    values = [
        cosine_similarity(descriptors[left], descriptors[right])
        for left in range(len(descriptors))
        for right in range(left + 1, len(descriptors))
    ]
    return float(np.mean(values))


def _prompt_terms(prompt: str, trigger_token: str) -> set[str]:
    trigger = "_".join(findall(r"[a-z0-9]+", trigger_token.casefold()))
    ignored = {"high_quality", "best_quality", "masterpiece", trigger, ""}
    terms = {"_".join(findall(r"[a-z0-9]+", segment.casefold())) for segment in prompt.split(",")}
    return terms - ignored


def prompt_tag_compliance(prompt: str, tags: tuple[str, ...], trigger_token: str) -> float:
    expected = _prompt_terms(prompt, trigger_token)
    if not expected:
        return 1.0
    observed = {"_".join(findall(r"[a-z0-9]+", tag.casefold())) for tag in tags}
    matched = sum(
        any(term == tag or term in tag or tag in term for tag in observed if tag)
        for term in expected
    )
    return matched / len(expected)


def aggregate_candidate_signals(
    *,
    reference_paths: list[Path],
    observations: list[SampleObservation],
    trigger_token: str,
    invariant_tags: tuple[str, ...] = (),
    learned_embeddings: Mapping[Path, FloatArray] | None = None,
    embedding_model_id: str = "",
    embedding_revision: str = "",
) -> CandidateVisualSignals:
    """Aggregate actual pixels/tags into bounded, auditable checkpoint signals."""

    reference_descriptors = [image_descriptor(path) for path in reference_paths]
    usable: list[tuple[SampleObservation, FloatArray, ImageTechnicalMetrics]] = []
    failed = 0
    for observation in observations:
        technical = inspect_generated_image(observation.path) if observation.success else None
        if technical is None or not technical.valid or technical.black_or_blank:
            failed += 1
            continue
        usable.append((observation, image_descriptor(observation.path), technical))
    total = len(observations)
    failure_rate = failed / total if total else 1.0

    descriptors = [item[1] for item in usable]
    pixel_reference_similarity = 0.0
    pixel_nearest = 0.0
    if reference_descriptors and descriptors:
        reference_centroid = np.mean(np.stack(reference_descriptors), axis=0).astype(np.float32)
        pixel_reference_similarity = float(
            np.mean([cosine_similarity(item, reference_centroid) for item in descriptors])
        )
        pixel_nearest = max(
            cosine_similarity(candidate, reference)
            for candidate in descriptors
            for reference in reference_descriptors
        )

    semantic_used = learned_embeddings is not None
    semantic_reference_similarity = 0.0
    semantic_nearest = 0.0
    semantic_descriptors: list[FloatArray] = []
    if learned_embeddings is not None:
        if not embedding_model_id or not embedding_revision:
            raise ValueError("Learned embedding model identity and revision are required")
        normalized_embeddings = {
            path.resolve(strict=False): np.asarray(value, dtype=np.float32)
            for path, value in learned_embeddings.items()
        }
        required_paths = [
            *(path.resolve(strict=False) for path in reference_paths),
            *(observation.path.resolve(strict=False) for observation, _descriptor, _tech in usable),
        ]
        missing = [path for path in required_paths if path not in normalized_embeddings]
        if missing:
            raise ValueError(f"Learned embedding is missing for image: {missing[0]}")
        reference_semantic = [
            normalized_embeddings[path.resolve(strict=False)] for path in reference_paths
        ]
        semantic_descriptors = [
            normalized_embeddings[observation.path.resolve(strict=False)]
            for observation, _descriptor, _technical in usable
        ]
        dimensions = {len(value) for value in [*reference_semantic, *semantic_descriptors]}
        if len(dimensions) != 1:
            raise ValueError("Learned embedding vectors must have one shared dimension")
        if reference_semantic and semantic_descriptors:
            semantic_centroid = np.mean(np.stack(reference_semantic), axis=0).astype(np.float32)
            semantic_reference_similarity = float(
                np.mean(
                    [cosine_similarity(item, semantic_centroid) for item in semantic_descriptors]
                )
            )
            semantic_nearest = max(
                cosine_similarity(candidate, reference)
                for candidate in semantic_descriptors
                for reference in reference_semantic
            )

    if semantic_used:
        # Generic CLIP is supporting evidence, never the sole identity/style score.
        reference_similarity = (
            0.5 * pixel_reference_similarity + 0.5 * semantic_reference_similarity
        )
        nearest = 0.5 * pixel_nearest + 0.5 * semantic_nearest
    else:
        reference_similarity = pixel_reference_similarity
        nearest = pixel_nearest

    invariant_normalized = {
        "_".join(findall(r"[a-z0-9]+", tag.casefold())) for tag in invariant_tags
    }
    if invariant_normalized and usable:
        invariant_score = float(
            np.mean(
                [
                    len(
                        invariant_normalized
                        & {
                            "_".join(findall(r"[a-z0-9]+", tag.casefold()))
                            for tag in observation.tags
                        }
                    )
                    / len(invariant_normalized)
                    for observation, _descriptor, _technical in usable
                ]
            )
        )
        reference_similarity = 0.75 * reference_similarity + 0.25 * invariant_score
    else:
        invariant_score = 0.0

    compliance_values = [
        prompt_tag_compliance(observation.prompt, observation.tags, trigger_token)
        for observation, _descriptor, _technical in usable
    ]
    prompt_compliance = float(np.mean(compliance_values)) if compliance_values else 0.0

    by_prompt: dict[str, list[FloatArray]] = {}
    for observation, descriptor, _technical in usable:
        by_prompt.setdefault(observation.prompt_id, []).append(descriptor)
    consistency_values = [_mean_pairwise(values) for values in by_prompt.values()]
    pixel_consistency = float(np.mean(consistency_values)) if consistency_values else 0.0
    prompt_centroids = [
        np.mean(np.stack(values), axis=0).astype(np.float32) for values in by_prompt.values()
    ]
    pixel_prompt_similarity = _mean_pairwise(prompt_centroids)
    semantic_consistency = 0.0
    semantic_prompt_similarity = 0.0
    if semantic_used:
        semantic_by_prompt: dict[str, list[FloatArray]] = {}
        for (observation, _descriptor, _technical), descriptor in zip(
            usable, semantic_descriptors, strict=True
        ):
            semantic_by_prompt.setdefault(observation.prompt_id, []).append(descriptor)
        semantic_consistency_values = [
            _mean_pairwise(values) for values in semantic_by_prompt.values()
        ]
        semantic_consistency = (
            float(np.mean(semantic_consistency_values)) if semantic_consistency_values else 0.0
        )
        semantic_prompt_centroids = [
            np.mean(np.stack(values), axis=0).astype(np.float32)
            for values in semantic_by_prompt.values()
        ]
        semantic_prompt_similarity = _mean_pairwise(semantic_prompt_centroids)
        consistency = 0.5 * pixel_consistency + 0.5 * semantic_consistency
        prompt_similarity = 0.5 * pixel_prompt_similarity + 0.5 * semantic_prompt_similarity
    else:
        consistency = pixel_consistency
        prompt_similarity = pixel_prompt_similarity
    flexibility = min(1.0, max(0.0, (1.0 - prompt_similarity) * 5.0))

    health_values = [
        min(1.0, max(0.0, technical.standard_deviation / 0.18))
        * (1.0 - min(1.0, abs(technical.mean_luminance - 0.5)))
        for _observation, _descriptor, technical in usable
    ]
    visual_heuristic = float(np.mean(health_values)) if health_values else 0.0
    technical_score = max(0.0, 1.0 - failure_rate)
    memorization_risk = min(1.0, max(0.0, (nearest - 0.985) / 0.015))

    weight_scores: dict[float, list[float]] = {}
    for index, (observation, _descriptor, _technical) in enumerate(usable):
        weight_scores.setdefault(observation.weight, []).append(
            0.5 * health_values[index] + 0.5 * compliance_values[index]
        )
    recommended_weight = 0.8
    if weight_scores:
        recommended_weight = max(
            weight_scores,
            key=lambda weight: (
                float(np.mean(weight_scores[weight])),
                -abs(weight - 0.8),
                -weight,
            ),
        )

    return CandidateVisualSignals(
        reference_similarity=max(0.0, min(1.0, reference_similarity)),
        prompt_compliance=max(0.0, min(1.0, prompt_compliance)),
        flexibility=flexibility,
        consistency=max(0.0, min(1.0, consistency)),
        technical=technical_score,
        visual_heuristic=max(0.0, min(1.0, visual_heuristic)),
        generation_failure_rate=failure_rate,
        memorization_risk=memorization_risk,
        recommended_weight=recommended_weight,
        evidence={
            "sample_count": total,
            "usable_sample_count": len(usable),
            "reference_count": len(reference_paths),
            "reference_similarity": reference_similarity,
            "pixel_reference_similarity": pixel_reference_similarity,
            "semantic_embedding_used": semantic_used,
            "semantic_embedding_model": embedding_model_id,
            "semantic_embedding_revision": embedding_revision,
            "semantic_reference_similarity": semantic_reference_similarity,
            "prompt_tag_compliance": prompt_compliance,
            "cross_seed_consistency": consistency,
            "pixel_cross_seed_consistency": pixel_consistency,
            "semantic_cross_seed_consistency": semantic_consistency,
            "prompt_flexibility": flexibility,
            "nearest_reference_similarity": nearest,
            "pixel_nearest_reference_similarity": pixel_nearest,
            "semantic_nearest_reference_similarity": semantic_nearest,
            "invariant_tag_preservation": invariant_score,
            "pixel_health": visual_heuristic,
        },
    )
