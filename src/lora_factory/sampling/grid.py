"""Comparison-grid rendering for completion artifacts."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def render_grid(
    images: list[tuple[Path, str]],
    output_path: Path,
    *,
    columns: int = 3,
    cell_size: tuple[int, int] = (384, 384),
) -> Path:
    if not images:
        raise ValueError("At least one image is required for a comparison grid")
    if columns < 1:
        raise ValueError("Grid columns must be positive")
    label_height = 44
    rows = math.ceil(len(images) / columns)
    canvas = Image.new(
        "RGB",
        (columns * cell_size[0], rows * (cell_size[1] + label_height)),
        color=(28, 31, 38),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(images):
        with Image.open(path) as source:
            thumbnail = ImageOps.fit(source.convert("RGB"), cell_size)
        column = index % columns
        row = index // columns
        x = column * cell_size[0]
        y = row * (cell_size[1] + label_height)
        canvas.paste(thumbnail, (x, y))
        draw.text((x + 10, y + cell_size[1] + 12), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path
