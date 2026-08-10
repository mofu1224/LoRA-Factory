from __future__ import annotations

import pytest

from lora_factory.training.buckets import (
    assigned_no_upscale_bucket,
    format_assigned_bucket,
)


@pytest.mark.parametrize(
    ("size", "resolution", "expected"),
    [
        ((1024, 1024), 1024, (1024, 1024)),
        ((1920, 1080), 1024, (1344, 768)),
        ((1080, 1920), 1024, (768, 1344)),
        ((1600, 1200), 1024, (1088, 832)),
        ((2048, 1365), 1024, (1216, 832)),
        ((1920, 1080), 768, (1024, 576)),
    ],
)
def test_no_upscale_bucket_matches_pinned_policy(
    size: tuple[int, int],
    resolution: int,
    expected: tuple[int, int],
) -> None:
    assert (
        assigned_no_upscale_bucket(
            *size,
            max_resolution=resolution,  # type: ignore[arg-type]
        )
        == expected
    )


def test_tiny_image_has_no_valid_no_upscale_bucket() -> None:
    assert assigned_no_upscale_bucket(48, 128, max_resolution=768) is None
    assert format_assigned_bucket(48, 128, max_resolution=768) == "Unavailable (<64 px)"


@pytest.mark.parametrize("size", [(10_000, 64), (64, 10_000)])
def test_extreme_aspect_with_zero_rounded_side_is_unavailable(
    size: tuple[int, int],
) -> None:
    assert assigned_no_upscale_bucket(*size, max_resolution=768) is None
    assert format_assigned_bucket(*size, max_resolution=768) == "Unavailable (<64 px)"
