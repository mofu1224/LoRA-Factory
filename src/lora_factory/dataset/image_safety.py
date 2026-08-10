"""Bounded image-input checks performed before copying or full pixel decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field


class ImageSafetyError(ValueError):
    """An image exceeds the public input-safety contract."""


class ImageSafetyLimits(BaseModel):
    """Conservative limits that keep accidental or hostile inputs bounded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_enumerated_entries: Annotated[int, Field(ge=1)] = 100_000
    max_image_files: Annotated[int, Field(ge=1)] = 10_000
    max_file_size_bytes: Annotated[int, Field(ge=1)] = 512 * 1024 * 1024
    max_total_size_bytes: Annotated[int, Field(ge=1)] = 50 * 1024 * 1024 * 1024
    max_dimension: Annotated[int, Field(ge=1)] = 32_768
    max_pixels: Annotated[int, Field(ge=1)] = 100_000_000


def validate_image_header(
    path: Path,
    *,
    limits: ImageSafetyLimits | None = None,
) -> tuple[int, int]:
    """Validate file size and header dimensions without decoding the full image."""

    active = limits or ImageSafetyLimits()
    size = path.stat().st_size
    if size <= 0:
        raise ImageSafetyError("Image file contains zero bytes")
    if size > active.max_file_size_bytes:
        raise ImageSafetyError(
            f"Image file is {size} bytes; limit is {active.max_file_size_bytes} bytes"
        )
    try:
        with Image.open(path) as opened:
            width, height = opened.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageSafetyError(f"Pillow cannot read the image header: {exc}") from exc
    pixels = width * height
    if width <= 0 or height <= 0:
        raise ImageSafetyError(f"Image dimensions must be positive: {width}x{height}")
    if max(width, height) > active.max_dimension:
        raise ImageSafetyError(
            f"Image dimension {width}x{height} exceeds the {active.max_dimension}-pixel limit"
        )
    if pixels > active.max_pixels:
        raise ImageSafetyError(f"Image has {pixels} pixels; limit is {active.max_pixels} pixels")
    return width, height
