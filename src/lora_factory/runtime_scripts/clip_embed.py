"""Standalone pinned CLIP vision embedding entrypoint for the managed runtime."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

_IMAGE_SIZE = 224


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser.parse_args()


def _prepare(path: Path) -> Any:
    import torch  # type: ignore[import-not-found]

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, image).convert("RGB")
    width, height = rgb.size
    scale = _IMAGE_SIZE / min(width, height)
    resized = rgb.resize(
        (max(_IMAGE_SIZE, round(width * scale)), max(_IMAGE_SIZE, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - _IMAGE_SIZE) // 2
    top = (resized.height - _IMAGE_SIZE) // 2
    cropped = resized.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    return (tensor - mean) / std


def _load_paths(path: Path) -> tuple[Path, ...]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    values = payload.get("paths") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("CLIP embedding input must contain a non-empty paths list")
    paths = tuple(Path(value).resolve(strict=True) for value in values if isinstance(value, str))
    if len(paths) != len(values) or len(set(paths)) != len(paths):
        raise ValueError("CLIP embedding paths must be unique strings")
    return paths


def _encode(
    model: Any,
    paths: tuple[Path, ...],
    batch_size: int,
    *,
    torch_module: Any | None = None,
    normalize_fn: Callable[..., Any] | None = None,
    prepare_fn: Callable[[Path], Any] = _prepare,
) -> list[list[float]]:
    if torch_module is None:
        import torch

        runtime_torch: Any = torch
    else:
        runtime_torch = torch_module
    if normalize_fn is None:
        import torch.nn.functional as functional  # type: ignore[import-not-found]

        normalize_fn = functional.normalize
    embeddings: list[list[float]] = []
    cursor = 0
    safe_batch_size = min(batch_size, len(paths))
    while cursor < len(paths):
        end = min(len(paths), cursor + safe_batch_size)
        pixels: Any | None = None
        projected: Any | None = None
        normalized: Any | None = None
        try:
            pixels = runtime_torch.stack([prepare_fn(path) for path in paths[cursor:end]]).to(
                device="cuda:0", dtype=runtime_torch.float16
            )
            with runtime_torch.inference_mode():
                projected = model(pixel_values=pixels).image_embeds
                normalized = normalize_fn(projected.float(), dim=1).cpu()
        except runtime_torch.OutOfMemoryError as error:
            # Release every live CUDA tensor before asking the allocator to retry
            # a smaller batch.  Rebinding on the next RHS is too late because the
            # old local would otherwise remain referenced during that allocation.
            error.__traceback__ = None
            del normalized, projected, pixels
            if safe_batch_size == 1:
                raise
            safe_batch_size = max(1, safe_batch_size // 2)
            runtime_torch.cuda.empty_cache()
            continue
        embeddings.extend(normalized.tolist())
        del normalized, projected, pixels
        cursor = end
    return embeddings


def main() -> int:
    import torch
    from transformers import CLIPVisionModelWithProjection  # type: ignore[import-not-found]

    args = _arguments()
    if args.batch_size < 1:
        raise ValueError("CLIP embedding batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot initialize the isolated CUDA device for CLIP")
    model_directory = args.model_directory.resolve(strict=True)
    required = (model_directory / "config.json", model_directory / "model.safetensors")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Pinned CLIP model directory is incomplete")
    paths = _load_paths(args.input)
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_directory,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to("cuda:0")
    model.eval()
    values = _encode(model, paths, args.batch_size)
    dimension = len(values[0])
    output: dict[str, Any] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "dimension": dimension,
        "device": "cuda:0",
        "items": [
            {
                "source_path": str(path),
                "source_sha256": _sha256(path),
                "values": embedding,
            }
            for path, embedding in zip(paths, values, strict=True)
        ],
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"images": len(paths), "dimension": dimension, "device": "cuda:0"}))
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
