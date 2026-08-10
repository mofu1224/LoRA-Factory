"""Standalone WD14 ONNX inference entrypoint for the managed runtime."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--general-threshold", type=float, required=True)
    parser.add_argument("--character-threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser.parse_args()


def _tag_rows(path: Path) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("name") or row.get("tag") or "").strip()
            category_id = (row.get("category") or "0").strip()
            if not name:
                raise ValueError("WD14 CSV contains an unnamed tag")
            category = (
                "rating" if category_id == "9" else "character" if category_id == "4" else "general"
            )
            values.append((name, category))
    if not values:
        raise ValueError("WD14 CSV is empty")
    return tuple(values)


def _layout(input_shape: list[Any]) -> tuple[str, int, int]:
    fallback = 448
    if len(input_shape) == 4 and input_shape[-1] == 3:
        return (
            "nhwc",
            int(input_shape[2]) if isinstance(input_shape[2], int) else fallback,
            int(input_shape[1]) if isinstance(input_shape[1], int) else fallback,
        )
    if len(input_shape) == 4 and input_shape[1] == 3:
        return (
            "nchw",
            int(input_shape[3]) if isinstance(input_shape[3], int) else fallback,
            int(input_shape[2]) if isinstance(input_shape[2], int) else fallback,
        )
    return "nhwc", fallback, fallback


def _prepare(path: Path, layout: str, width: int, height: int) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, image).convert("RGB")
    side = max(rgb.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    array = np.asarray(square.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)[
        :, :, ::-1
    ]
    if layout == "nchw":
        array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array)


def _is_out_of_memory(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda_error_out_of_memory",
            "cuda failure 2",
            "failed to allocate memory",
        )
    )


def _run_batched(
    paths: Sequence[Path],
    *,
    initial_batch_size: int,
    prepare: Callable[[Path], np.ndarray],
    infer: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int]:
    if not paths:
        raise ValueError("WD14 inference requires at least one image")
    if initial_batch_size < 1:
        raise ValueError("WD14 batch size must be positive")
    successful_batch_size = min(initial_batch_size, len(paths))
    cursor = 0
    outputs: list[np.ndarray] = []
    while cursor < len(paths):
        end = min(len(paths), cursor + successful_batch_size)
        batch = np.stack([prepare(path) for path in paths[cursor:end]], axis=0)
        try:
            probabilities = np.asarray(infer(batch), dtype=np.float32)
        except Exception as exc:
            if successful_batch_size <= 1 or not _is_out_of_memory(exc):
                raise
            successful_batch_size = max(1, successful_batch_size // 2)
            continue
        if probabilities.ndim == 1:
            probabilities = probabilities[None, :]
        if probabilities.ndim != 2 or probabilities.shape[0] != end - cursor:
            raise RuntimeError(
                "Unexpected WD14 batch output shape: "
                f"{probabilities.shape} for {end - cursor} images"
            )
        outputs.append(probabilities)
        cursor = end
    return np.concatenate(outputs, axis=0), successful_batch_size


def main() -> int:
    args = _arguments()
    import onnxruntime as ort  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    if args.batch_size < 1:
        raise ValueError("WD14 batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot initialize the isolated CUDA device")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    paths = [Path(value).resolve(strict=True) for value in payload["paths"]]
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {providers}")
    session = ort.InferenceSession(
        str(args.model.resolve(strict=True)),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    layout, width, height = _layout(list(model_input.shape))
    rows = _tag_rows(args.tags.resolve(strict=True))
    probabilities, successful_batch_size = _run_batched(
        paths,
        initial_batch_size=args.batch_size,
        prepare=lambda path: _prepare(path, layout, width, height),
        infer=lambda batch: np.asarray(
            session.run(None, {model_input.name: batch})[0], dtype=np.float32
        ),
    )
    if probabilities.shape != (len(paths), len(rows)):
        raise RuntimeError(f"Unexpected WD14 output shape: {probabilities.shape}")
    output: list[dict[str, Any]] = []
    for image_index in range(len(paths)):
        tags: list[dict[str, Any]] = []
        for tag_index, (name, category) in enumerate(rows):
            score = max(0.0, min(1.0, float(probabilities[image_index, tag_index])))
            threshold = (
                args.character_threshold
                if category == "character"
                else 0.0
                if category == "rating"
                else args.general_threshold
            )
            tags.append(
                {
                    "name": name,
                    "score": score,
                    "model_category": category,
                    "selected": score > threshold,
                }
            )
        output.append({"tags": tags})
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "items": output,
                "successful_batch_size": successful_batch_size,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "images": len(paths),
                "provider": session.get_providers()[0],
                "successful_batch_size": successful_batch_size,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
