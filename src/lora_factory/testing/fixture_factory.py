"""Safe, deterministic image and SDXL-checkpoint fixture generation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from safetensors.numpy import save_file

from lora_factory.model.inspector import inspect_sdxl_safetensors
from lora_factory.util.hashing import sha256_file


class GeneratedFakeFixture(BaseModel):
    """Paths and immutable source hashes for one generated Fake E2E fixture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path
    base_model: Path
    image_directory: Path
    image_paths: tuple[Path, ...]
    source_sha256: dict[str, str]
    image_count: Annotated[int, Field(ge=8)]


_FILENAMES = (
    "日本語.png",
    "Screenshot 01.png",
    "test (4).jpg",
    "1.webp",
    "very-long-unicode-name-これは長いUnicodeファイル名の安全性を確認するための画像です-001.png",
    "extreme-aspect-warning.png",
    "low-resolution-warning.png",
    "emoji-🌙-fixture.png",
    "étoile bleue.png",
    "한글 이미지.png",
    "空 白 を 含む 名前.png",
    "portrait_縦長.png",
    "landscape_横長.png",
    "square.sample.png",
    "brackets [safe].png",
    "fullwidth＿name.png",  # noqa: RUF001 - intentional Unicode filename coverage.
    "mix-日本語-English-한글.png",
    "0123456789.png",
    "scene-indoor.png",
    "scene-outdoor.png",
)

_SIZES = (
    (1024, 1024),
    (1920, 1080),
    (1080, 1920),
    (1600, 1200),
    (2048, 1365),
    (2048, 512),
    (480, 360),
    (1024, 768),
    (768, 1024),
    (1344, 768),
    (768, 1344),
)


def _fixture_name(index: int) -> str:
    if index < len(_FILENAMES):
        return _FILENAMES[index]
    return f"generated-scene-{index + 1:03d}.png"


def _write_image(path: Path, *, index: int, size: tuple[int, int]) -> None:
    """Write a sharp, non-blank, perceptually distinct RGB image."""

    width, height = size
    rng = np.random.default_rng(20_260_810 + index)
    x = np.arange(width, dtype=np.uint32)[None, :]
    y = np.arange(height, dtype=np.uint32)[:, None]
    noise = rng.integers(0, 32, size=(height, width), dtype=np.uint16)
    red = (x * (3 + index % 7) + y * (1 + index % 5) + noise) % 256
    green = (x * (1 + index % 11) + y * (5 + index % 3) + noise * 3) % 256
    blue = (x * (7 + index % 5) + y * (2 + index % 9) + noise * 5) % 256
    pixels = np.stack(
        (
            np.broadcast_to(red, (height, width)),
            np.broadcast_to(green, (height, width)),
            np.broadcast_to(blue, (height, width)),
        ),
        axis=2,
    ).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    margin = 32 + index * 3
    draw.rectangle(
        (margin, margin, width - margin - 1, height - margin - 1),
        outline=((index * 47) % 256, (index * 83) % 256, (index * 131) % 256),
        width=8,
    )
    draw.ellipse(
        (
            width // 4,
            height // 5,
            min(width - 1, width * 3 // 4),
            min(height - 1, height * 4 // 5),
        ),
        outline=((255 - index * 13) % 256, (64 + index * 17) % 256, 192),
        width=6,
    )
    suffix = path.suffix.casefold()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=95, subsampling=0)
    elif suffix == ".webp":
        image.save(path, format="WEBP", quality=95, method=4)
    else:
        image.save(path, format="PNG", compress_level=6)


def _write_tiny_sdxl(path: Path) -> None:
    tensors = {
        "model.diffusion_model.input_blocks.0.0.weight": np.asarray([0.25], dtype=np.float16),
        "conditioner.embedders.1.model.text_projection": np.asarray([0.5], dtype=np.float16),
        "first_stage_model.encoder.conv_in.weight": np.asarray([0.75], dtype=np.float16),
    }
    save_file(
        tensors,
        path,
        metadata={
            "modelspec.architecture": "stable-diffusion-xl-v1-base",
            "modelspec.title": "LoRA Factory tiny fake SDXL fixture",
            "lora_factory_fixture": "true",
        },
    )
    inspection = inspect_sdxl_safetensors(path)
    if not inspection.is_sdxl:
        raise RuntimeError("Generated base-model fixture did not pass SDXL structural inspection")


def generate_fake_fixture(root: Path, *, image_count: int = 18) -> GeneratedFakeFixture:
    """Create a fresh fixture without overwriting or mutating any existing path."""

    if image_count < 8:
        raise ValueError("Fake E2E requires at least eight source images")
    resolved = root.resolve(strict=False)
    if resolved.exists():
        raise FileExistsError(f"Fixture destination already exists; refusing overwrite: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    image_directory = resolved / "source images"
    image_directory.mkdir()
    base_model = resolved / "tiny-sdxl-fixture.safetensors"
    _write_tiny_sdxl(base_model)

    image_paths: list[Path] = []
    for index in range(image_count):
        destination = image_directory / _fixture_name(index)
        _write_image(destination, index=index, size=_SIZES[index % len(_SIZES)])
        image_paths.append(destination)

    hashes = {path.name: sha256_file(path) for path in image_paths}
    if len(set(hashes.values())) != len(hashes):
        raise RuntimeError("Generated image fixture unexpectedly contains exact duplicates")
    return GeneratedFakeFixture(
        root=resolved,
        base_model=base_model,
        image_directory=image_directory,
        image_paths=tuple(image_paths),
        source_sha256=hashes,
        image_count=image_count,
    )
