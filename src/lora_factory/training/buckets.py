"""Display the no-upscale bucket selected by the pinned sd-scripts policy."""

from __future__ import annotations

import math
from typing import Literal


def assigned_no_upscale_bucket(
    width: int,
    height: int,
    *,
    max_resolution: Literal[768, 896, 1024],
    resolution_step: int = 64,
) -> tuple[int, int] | None:
    """Mirror sd-scripts v0.11.1 ``BucketManager.select_bucket`` for display.

    The application never resizes the immutable Raw image.  The returned size is
    the runtime bucket/crop target for the derived working copy.  Inputs smaller
    than one bucket step have no valid no-upscale bucket and return ``None``.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if resolution_step <= 0:
        raise ValueError("Bucket resolution step must be positive")
    if width < resolution_step or height < resolution_step:
        return None

    max_area = max_resolution * max_resolution
    aspect_ratio = width / height

    def round_to_step(value: float) -> int:
        rounded = int(value + 0.5)
        return rounded - rounded % resolution_step

    if width * height > max_area:
        resized_width = math.sqrt(max_area * aspect_ratio)
        resized_height = max_area / resized_width

        width_rounded = round_to_step(resized_width)
        height_from_width = round_to_step(width_rounded / aspect_ratio)
        width_candidate_valid = min(width_rounded, height_from_width) >= resolution_step
        width_error = (
            abs(width_rounded / height_from_width - aspect_ratio)
            if width_candidate_valid
            else math.inf
        )

        height_rounded = round_to_step(resized_height)
        width_from_height = round_to_step(height_rounded * aspect_ratio)
        height_candidate_valid = min(width_from_height, height_rounded) >= resolution_step
        height_error = (
            abs(width_from_height / height_rounded - aspect_ratio)
            if height_candidate_valid
            else math.inf
        )

        if not width_candidate_valid and not height_candidate_valid:
            return None
        if width_error < height_error:
            resized = (width_rounded, int(width_rounded / aspect_ratio + 0.5))
        else:
            resized = (int(height_rounded * aspect_ratio + 0.5), height_rounded)
    else:
        resized = (width, height)

    bucket = (
        resized[0] - resized[0] % resolution_step,
        resized[1] - resized[1] % resolution_step,
    )
    return bucket if min(bucket) >= resolution_step else None


def format_assigned_bucket(
    width: int,
    height: int,
    *,
    max_resolution: Literal[768, 896, 1024],
) -> str:
    bucket = assigned_no_upscale_bucket(
        width,
        height,
        max_resolution=max_resolution,
    )
    return "Unavailable (<64 px)" if bucket is None else f"{bucket[0]}x{bucket[1]}"
