"""Explainable per-image quality checks and preset-aware dataset gates."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from lora_factory.config.models import PresetKind
from lora_factory.dataset.image_safety import (
    ImageSafetyError,
    ImageSafetyLimits,
    validate_image_header,
)


class QualityDisposition(StrEnum):
    ACCEPT = "accept"
    WARN = "warn"
    REJECT = "reject"


class ReasonSeverity(StrEnum):
    WARNING = "warning"
    REJECT = "reject"


class QualityReason(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: ReasonSeverity
    message: str
    value: float | int | str | None = None
    threshold: float | int | None = None


class ImageQualityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file_size_bytes: Annotated[int, Field(ge=0)]
    width: Annotated[int, Field(ge=0)] = 0
    height: Annotated[int, Field(ge=0)] = 0
    area: Annotated[int, Field(ge=0)] = 0
    short_side: Annotated[int, Field(ge=0)] = 0
    aspect_ratio: Annotated[float, Field(ge=0)] = 0.0
    luminance_mean: Annotated[float, Field(ge=0, le=255)] = 0.0
    luminance_stddev: Annotated[float, Field(ge=0)] = 0.0
    luminance_range: Annotated[float, Field(ge=0, le=255)] = 0.0
    sharpness: Annotated[float, Field(ge=0)] = 0.0
    compression_blockiness: Annotated[float, Field(ge=0)] = 0.0
    alpha_fraction: Annotated[float, Field(ge=0, le=1)] = 0.0


class QualityAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    disposition: QualityDisposition
    metrics: ImageQualityMetrics
    reasons: tuple[QualityReason, ...] = Field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.disposition is not QualityDisposition.REJECT


class QualityThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_dimension: Annotated[int, Field(ge=1024)] = 32_768
    maximum_pixels: Annotated[int, Field(ge=1)] = 100_000_000
    maximum_file_size_bytes: Annotated[int, Field(ge=1)] = 512 * 1024 * 1024
    minimum_short_side: Annotated[int, Field(ge=1)] = 512
    reject_short_side_below: Annotated[int, Field(ge=1)] = 64
    warn_aspect_ratio_above: Annotated[float, Field(gt=1)] = 3.0
    reject_aspect_ratio_above: Annotated[float, Field(gt=1)] = 8.0
    near_blank_stddev: Annotated[float, Field(ge=0)] = 2.0
    near_blank_range: Annotated[float, Field(ge=0)] = 8.0
    black_mean: Annotated[float, Field(ge=0, le=255)] = 1.0
    white_mean: Annotated[float, Field(ge=0, le=255)] = 254.0
    blur_sharpness_below: Annotated[float, Field(ge=0)] = 35.0
    compression_blockiness_above: Annotated[float, Field(ge=0)] = 1.8
    alpha_fraction_warning: Annotated[float, Field(ge=0, le=1)] = 0.05


def _sharpness(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - (4.0 * center)
    return float(np.var(laplacian))


def _blockiness(gray: np.ndarray) -> float:
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    boundary_values = np.concatenate((vertical[:, 7::8].ravel(), horizontal[7::8, :].ravel()))
    interior_values = np.concatenate(
        (
            np.delete(vertical, np.s_[7::8], axis=1).ravel(),
            np.delete(horizontal, np.s_[7::8], axis=0).ravel(),
        )
    )
    if boundary_values.size == 0 or interior_values.size == 0:
        return 0.0
    return float(np.mean(boundary_values) / (np.mean(interior_values) + 1e-6))


def _alpha_coverage(image: Image.Image) -> float:
    if image.mode not in {"RGBA", "LA", "PA"} and "transparency" not in image.info:
        return 0.0
    alpha = image.convert("RGBA").getchannel("A")
    histogram = alpha.histogram()
    return 1.0 - (histogram[255] / (alpha.width * alpha.height))


def assess_image(
    path: Path,
    *,
    thresholds: QualityThresholds | None = None,
) -> QualityAssessment:
    limits = thresholds or QualityThresholds()
    path = path.resolve(strict=False)
    size = path.stat().st_size if path.is_file() else 0
    empty_metrics = ImageQualityMetrics(file_size_bytes=size)
    if not path.is_file():
        reason = QualityReason(
            code="unreadable",
            severity=ReasonSeverity.REJECT,
            message=f"Image file is missing: {path}",
        )
        return QualityAssessment(
            path=path,
            disposition=QualityDisposition.REJECT,
            metrics=empty_metrics,
            reasons=(reason,),
        )
    if size == 0:
        reason = QualityReason(
            code="zero_size",
            severity=ReasonSeverity.REJECT,
            message="Image file contains zero bytes",
            value=0,
        )

    try:
        validate_image_header(
            path,
            limits=ImageSafetyLimits(
                max_dimension=limits.maximum_dimension,
                max_pixels=limits.maximum_pixels,
                max_file_size_bytes=limits.maximum_file_size_bytes,
            ),
        )
    except ImageSafetyError as exc:
        reason = QualityReason(
            code="unsafe_image",
            severity=ReasonSeverity.REJECT,
            message=str(exc),
        )
        return QualityAssessment(
            path=path,
            disposition=QualityDisposition.REJECT,
            metrics=empty_metrics,
            reasons=(reason,),
        )
        return QualityAssessment(
            path=path,
            disposition=QualityDisposition.REJECT,
            metrics=empty_metrics,
            reasons=(reason,),
        )

    try:
        with Image.open(path) as opened:
            if (
                bool(getattr(opened, "is_animated", False))
                or int(getattr(opened, "n_frames", 1)) > 1
            ):
                reason = QualityReason(
                    code="animated_image",
                    severity=ReasonSeverity.REJECT,
                    message="Animated WEBP/images are outside the still-image dataset contract",
                )
                return QualityAssessment(
                    path=path,
                    disposition=QualityDisposition.REJECT,
                    metrics=empty_metrics,
                    reasons=(reason,),
                )
            opened.load()
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            alpha_fraction = _alpha_coverage(image)
            grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        reason = QualityReason(
            code="unreadable",
            severity=ReasonSeverity.REJECT,
            message=f"Pillow cannot decode the image: {exc}",
        )
        return QualityAssessment(
            path=path,
            disposition=QualityDisposition.REJECT,
            metrics=empty_metrics,
            reasons=(reason,),
        )

    low = float(np.min(grayscale))
    high = float(np.max(grayscale))
    mean = float(np.mean(grayscale))
    stddev = float(np.std(grayscale))
    ratio = max(width / height, height / width) if width and height else 0.0
    metrics = ImageQualityMetrics(
        file_size_bytes=size,
        width=width,
        height=height,
        area=width * height,
        short_side=min(width, height),
        aspect_ratio=ratio,
        luminance_mean=mean,
        luminance_stddev=stddev,
        luminance_range=high - low,
        sharpness=_sharpness(grayscale),
        compression_blockiness=_blockiness(grayscale),
        alpha_fraction=alpha_fraction,
    )
    reasons: list[QualityReason] = []

    def add(
        code: str,
        severity: ReasonSeverity,
        message: str,
        value: float | int,
        threshold: float | int,
    ) -> None:
        reasons.append(
            QualityReason(
                code=code,
                severity=severity,
                message=message,
                value=value,
                threshold=threshold,
            )
        )

    if (
        width <= 0
        or height <= 0
        or max(width, height) > limits.maximum_dimension
        or width * height > limits.maximum_pixels
    ):
        add(
            "impossible_dimensions",
            ReasonSeverity.REJECT,
            "Image dimensions are outside supported safety limits",
            max(width, height),
            limits.maximum_dimension,
        )
    if metrics.short_side < limits.reject_short_side_below:
        add(
            "extremely_low_resolution",
            ReasonSeverity.REJECT,
            "Image is too small for useful bucket training",
            metrics.short_side,
            limits.reject_short_side_below,
        )
    elif metrics.short_side < limits.minimum_short_side:
        add(
            "low_resolution",
            ReasonSeverity.WARNING,
            "Short side is below the preferred training range",
            metrics.short_side,
            limits.minimum_short_side,
        )
    if ratio > limits.reject_aspect_ratio_above:
        add(
            "extreme_aspect_ratio",
            ReasonSeverity.REJECT,
            "Aspect ratio is too extreme for the configured bucket policy",
            ratio,
            limits.reject_aspect_ratio_above,
        )
    elif ratio > limits.warn_aspect_ratio_above:
        add(
            "aspect_ratio_outlier",
            ReasonSeverity.WARNING,
            "Aspect ratio is unusual and should be reviewed",
            ratio,
            limits.warn_aspect_ratio_above,
        )
    if mean <= limits.black_mean:
        add(
            "all_black",
            ReasonSeverity.REJECT,
            "Image is effectively all black",
            mean,
            limits.black_mean,
        )
    elif mean >= limits.white_mean:
        add(
            "all_white",
            ReasonSeverity.REJECT,
            "Image is effectively all white",
            mean,
            limits.white_mean,
        )
    elif stddev < limits.near_blank_stddev or metrics.luminance_range < limits.near_blank_range:
        add(
            "near_empty",
            ReasonSeverity.REJECT,
            "Image has almost no luminance variation",
            stddev,
            limits.near_blank_stddev,
        )
    if metrics.sharpness < limits.blur_sharpness_below:
        add(
            "blur",
            ReasonSeverity.WARNING,
            "Sharpness heuristic is below the review threshold",
            metrics.sharpness,
            limits.blur_sharpness_below,
        )
    if metrics.compression_blockiness > limits.compression_blockiness_above:
        add(
            "compression_artifacts",
            ReasonSeverity.WARNING,
            "8-pixel block boundary heuristic suggests compression artifacts",
            metrics.compression_blockiness,
            limits.compression_blockiness_above,
        )
    if alpha_fraction >= limits.alpha_fraction_warning:
        add(
            "significant_alpha",
            ReasonSeverity.WARNING,
            "Transparency is significant and neutral compositing should be reviewed",
            alpha_fraction,
            limits.alpha_fraction_warning,
        )

    disposition = (
        QualityDisposition.REJECT
        if any(reason.severity is ReasonSeverity.REJECT for reason in reasons)
        else QualityDisposition.WARN
        if reasons
        else QualityDisposition.ACCEPT
    )
    return QualityAssessment(
        path=path, disposition=disposition, metrics=metrics, reasons=tuple(reasons)
    )


class TagSignalThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text_watermark: Annotated[float, Field(ge=0, le=1)] = 0.65
    multi_subject: Annotated[float, Field(ge=0, le=1)] = 0.70
    reject_watermark: bool = False
    reject_multi_subject: bool = False


def add_tagger_quality_signals(
    assessment: QualityAssessment,
    tag_scores: Mapping[str, float],
    *,
    thresholds: TagSignalThresholds | None = None,
) -> QualityAssessment:
    """Attach explainable WD14 watermark/text and multi-subject review signals."""

    limits = thresholds or TagSignalThresholds()
    reasons = list(assessment.reasons)
    canonical = {tag.strip().lower().replace(" ", "_"): score for tag, score in tag_scores.items()}
    watermark_tags = ("text", "watermark", "signature", "logo")
    detected_watermark = max(
        ((tag, canonical.get(tag, 0.0)) for tag in watermark_tags),
        key=lambda item: item[1],
    )
    if detected_watermark[1] >= limits.text_watermark:
        reasons.append(
            QualityReason(
                code="text_watermark",
                severity=ReasonSeverity.REJECT
                if limits.reject_watermark
                else ReasonSeverity.WARNING,
                message=f"WD14 detected {detected_watermark[0]!r}; review for text leakage",
                value=detected_watermark[1],
                threshold=limits.text_watermark,
            )
        )
    multi_subject_tags = ("2girls", "2boys", "multiple_girls", "multiple_boys", "group")
    detected_multi = max(
        ((tag, canonical.get(tag, 0.0)) for tag in multi_subject_tags),
        key=lambda item: item[1],
    )
    if detected_multi[1] >= limits.multi_subject:
        reasons.append(
            QualityReason(
                code="multi_subject",
                severity=ReasonSeverity.REJECT
                if limits.reject_multi_subject
                else ReasonSeverity.WARNING,
                message=f"WD14 detected multi-subject tag {detected_multi[0]!r}",
                value=detected_multi[1],
                threshold=limits.multi_subject,
            )
        )
    disposition = (
        QualityDisposition.REJECT
        if any(reason.severity is ReasonSeverity.REJECT for reason in reasons)
        else QualityDisposition.WARN
        if reasons
        else QualityDisposition.ACCEPT
    )
    return assessment.model_copy(update={"reasons": tuple(reasons), "disposition": disposition})


class DatasetGateThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_minimum: Annotated[int, Field(ge=1)]
    warning_below: Annotated[int, Field(ge=1)]
    max_near_duplicate_dominance: Annotated[float, Field(ge=0, le=1)] = 0.35
    min_content_diversity: Annotated[float, Field(ge=0, le=1)] = 0.25
    max_dominant_character_ratio: Annotated[float, Field(ge=0, le=1)] = 0.75
    min_style_consistency: Annotated[float, Field(ge=0, le=1)] = 0.25

    @classmethod
    def for_preset(cls, preset: PresetKind) -> DatasetGateThresholds:
        if preset is PresetKind.CHARACTER:
            return cls(hard_minimum=8, warning_below=20)
        return cls(hard_minimum=16, warning_below=30)


class DatasetGateIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: ReasonSeverity
    message: str


class DatasetGateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    accepted_count: Annotated[int, Field(ge=0)]
    rejected_count: Annotated[int, Field(ge=0)]
    issues: tuple[DatasetGateIssue, ...] = Field(default_factory=tuple)


def evaluate_quality_gate(
    assessments: list[QualityAssessment] | tuple[QualityAssessment, ...],
    preset: PresetKind,
    *,
    thresholds: DatasetGateThresholds | None = None,
    near_duplicate_dominance: float = 0.0,
    content_diversity: float = 1.0,
    dominant_character_ratio: float = 0.0,
    style_consistency: float = 1.0,
) -> DatasetGateResult:
    limits = thresholds or DatasetGateThresholds.for_preset(preset)
    accepted = sum(item.accepted for item in assessments)
    rejected = len(assessments) - accepted
    issues: list[DatasetGateIssue] = []

    def issue(code: str, severity: ReasonSeverity, message: str) -> None:
        issues.append(DatasetGateIssue(code=code, severity=severity, message=message))

    if accepted < limits.hard_minimum:
        issue(
            "accepted_below_hard_minimum",
            ReasonSeverity.REJECT,
            f"Accepted images {accepted} are below the {preset.value} minimum "
            f"of {limits.hard_minimum}",
        )
    elif accepted < limits.warning_below:
        issue(
            "accepted_below_guidance",
            ReasonSeverity.WARNING,
            f"Accepted images {accepted} are below the guidance value {limits.warning_below}",
        )
    if near_duplicate_dominance > limits.max_near_duplicate_dominance:
        issue(
            "near_duplicate_dominance",
            ReasonSeverity.WARNING,
            "A near-duplicate cluster dominates the accepted dataset",
        )
    if preset is PresetKind.STYLE:
        if content_diversity < limits.min_content_diversity:
            issue(
                "low_style_content_diversity",
                ReasonSeverity.REJECT,
                "Style dataset content diversity is too low to separate style from subject",
            )
        if dominant_character_ratio > limits.max_dominant_character_ratio:
            issue(
                "single_character_dominance",
                ReasonSeverity.WARNING,
                "Dataset is closer to Character or Character+Style than a broad Style dataset",
            )
        if style_consistency < limits.min_style_consistency:
            issue(
                "style_inconsistency",
                ReasonSeverity.WARNING,
                "Style signals are inconsistent across the accepted dataset",
            )

    return DatasetGateResult(
        passed=not any(item.severity is ReasonSeverity.REJECT for item in issues),
        accepted_count=accepted,
        rejected_count=rejected,
        issues=tuple(issues),
    )
