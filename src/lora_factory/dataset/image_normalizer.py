"""Lossless-layout working-copy normalization; Raw sources are only ever read."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from lora_factory.dataset.image_safety import ImageSafetyError, validate_image_header
from lora_factory.util.hashing import sha256_file

RgbColor = tuple[
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
]


class ImageNormalizationError(RuntimeError):
    """The source cannot safely become a static normalized working image."""


class AnimatedImageError(ImageNormalizationError):
    """Animated images are outside the v1 still-image contract."""


class NeutralBackgroundPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    color: RgbColor = (127, 127, 127)
    warn_alpha_fraction: Annotated[float, Field(ge=0, le=1)] = 0.05


class NormalizationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: Path
    output_path: Path
    output_sha256: str
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    exif_transposed: bool
    icc_profile_present: bool
    icc_converted_to_srgb: bool
    assumed_srgb: bool
    alpha_present: bool
    alpha_fraction: Annotated[float, Field(ge=0, le=1)]
    alpha_background: RgbColor | None = None
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def _orientation(image: Image.Image) -> int | None:
    try:
        value = image.getexif().get(274)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return int(value) if value is not None else None


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info


def _alpha_fraction(alpha: Image.Image) -> float:
    histogram = alpha.histogram()
    pixels = alpha.width * alpha.height
    return 0.0 if pixels == 0 else 1.0 - (histogram[255] / pixels)


def _convert_rgb_to_srgb(image: Image.Image, icc_bytes: bytes) -> Image.Image:
    source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
    target_profile = ImageCms.createProfile("sRGB")
    converted = ImageCms.profileToProfile(
        image.convert("RGB"),
        source_profile,
        target_profile,
        outputMode="RGB",
    )
    if converted is None:
        raise OSError("Pillow ImageCms returned no converted image")
    return converted


def normalize_image(
    source: Path,
    destination: Path,
    *,
    background: NeutralBackgroundPolicy | None = None,
) -> NormalizationResult:
    """Decode and normalize a still image to an atomic RGB PNG working copy.

    Dimensions and composition are preserved: this function never crops, resizes, or upscales.
    """

    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("Working output must not overwrite its Raw source")
    policy = background or NeutralBackgroundPolicy()
    warnings: list[str] = []

    try:
        validate_image_header(source)
        with Image.open(source) as opened:
            if (
                bool(getattr(opened, "is_animated", False))
                or int(getattr(opened, "n_frames", 1)) > 1
            ):
                raise AnimatedImageError(f"Animated image is not supported: {source}")
            opened.load()
            original_orientation = _orientation(opened)
            transposed = ImageOps.exif_transpose(opened)
            exif_transposed = original_orientation not in {None, 1}
            icc_value = opened.info.get("icc_profile")
            icc_bytes = bytes(icc_value) if isinstance(icc_value, bytes) else None
            alpha_present = _has_alpha(transposed)
            alpha = transposed.convert("RGBA").getchannel("A") if alpha_present else None

            if icc_bytes:
                try:
                    rgb = _convert_rgb_to_srgb(transposed, icc_bytes)
                    icc_converted = True
                except (OSError, TypeError, ValueError) as exc:
                    rgb = transposed.convert("RGB")
                    icc_converted = False
                    warnings.append(f"Invalid ICC profile; pixels treated as sRGB: {exc}")
            else:
                rgb = transposed.convert("RGB")
                icc_converted = False

            fraction = _alpha_fraction(alpha) if alpha is not None else 0.0
            if alpha is not None:
                foreground = rgb.convert("RGBA")
                foreground.putalpha(alpha)
                neutral = Image.new("RGBA", foreground.size, (*policy.color, 255))
                normalized = Image.alpha_composite(neutral, foreground).convert("RGB")
                if fraction >= policy.warn_alpha_fraction:
                    warnings.append(
                        f"Alpha affects {fraction:.1%} of pixels; neutral composite was applied"
                    )
            else:
                normalized = rgb
            width, height = normalized.size

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            try:
                srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
                normalized.save(
                    temporary,
                    format="PNG",
                    optimize=False,
                    compress_level=6,
                    icc_profile=srgb_profile,
                )
                with temporary.open("r+b") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    except AnimatedImageError:
        raise
    except (ImageSafetyError, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageNormalizationError(f"Cannot normalize {source}: {exc}") from exc

    return NormalizationResult(
        source_path=source,
        output_path=destination,
        output_sha256=sha256_file(destination),
        width=width,
        height=height,
        exif_transposed=exif_transposed,
        icc_profile_present=icc_bytes is not None,
        icc_converted_to_srgb=icc_converted,
        assumed_srgb=icc_bytes is None or not icc_converted,
        alpha_present=alpha_present,
        alpha_fraction=fraction,
        alpha_background=policy.color if alpha_present else None,
        warnings=tuple(warnings),
    )


class ImageNormalizer:
    def __init__(self, background: NeutralBackgroundPolicy | None = None) -> None:
        self.background = background or NeutralBackgroundPolicy()

    def normalize(self, source: Path, destination: Path) -> NormalizationResult:
        return normalize_image(source, destination, background=self.background)
