"""Production WD14-compatible ONNX adapter with lazy optional dependencies."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lora_factory.caption.tagger import ImageTagResult, TagScore
from lora_factory.util.hashing import sha256_file


class WD14BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    revision: str
    model_filename: str = "model.onnx"
    tags_filename: str = "selected_tags.csv"
    model_path: Path | None = None
    tags_path: Path | None = None
    general_threshold: Annotated[float, Field(ge=0, le=1)] = 0.35
    character_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85
    rating_threshold: Annotated[float, Field(ge=0, le=1)] = 0.0
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    input_size_fallback: Annotated[int, Field(ge=64, le=2048)] = 448
    initial_batch_size: Annotated[int, Field(ge=1, le=256)] = 8
    bgr_input: bool = True

    @model_validator(mode="after")
    def validate_local_pair(self) -> WD14BackendConfig:
        if (self.model_path is None) != (self.tags_path is None):
            raise ValueError("model_path and tags_path must be supplied together")
        if not self.revision.strip():
            raise ValueError("A pinned WD14 revision is required")
        return self


class WD14OnnxTagger:
    """Execute a pinned WD14 model without importing ONNX/Hugging Face at module import time."""

    def __init__(self, config: WD14BackendConfig) -> None:
        self.config = config
        self._session: Any | None = None
        self._rows: tuple[tuple[str, str], ...] | None = None
        self._input_name: str | None = None
        self._input_shape: tuple[Any, ...] | None = None
        self._successful_batch_size: int | None = None

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def revision(self) -> str:
        return self.config.revision

    def _resolve_files(self) -> tuple[Path, Path]:
        if self.config.model_path is not None and self.config.tags_path is not None:
            return self.config.model_path.resolve(strict=True), self.config.tags_path.resolve(
                strict=True
            )
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Real WD14 download requires the 'tagger' optional dependency (huggingface-hub)"
            ) from exc
        model = hf_hub_download(
            repo_id=self.config.model_id,
            filename=self.config.model_filename,
            revision=self.config.revision,
        )
        tags = hf_hub_download(
            repo_id=self.config.model_id,
            filename=self.config.tags_filename,
            revision=self.config.revision,
        )
        return Path(model), Path(tags)

    @staticmethod
    def _read_rows(path: Path) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("name") or row.get("tag") or "").strip()
                raw_category = (row.get("category") or "0").strip()
                if not name:
                    raise ValueError(f"WD14 tag CSV contains a row without a name: {path}")
                category = (
                    "rating"
                    if raw_category == "9"
                    else "character"
                    if raw_category == "4"
                    else "general"
                )
                rows.append((name, category))
        if not rows:
            raise ValueError(f"WD14 tag CSV is empty: {path}")
        return tuple(rows)

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Real WD14 inference requires the 'tagger' optional dependency (onnxruntime-gpu)"
            ) from exc
        model_path, tags_path = self._resolve_files()
        available = set(ort.get_available_providers())
        requested = [provider for provider in self.config.providers if provider in available]
        if not requested:
            raise RuntimeError(
                f"None of the configured ONNX providers are available: {self.config.providers}; "
                f"available={sorted(available)}"
            )
        self._session = ort.InferenceSession(str(model_path), providers=requested)
        model_input = self._session.get_inputs()[0]
        self._input_name = str(model_input.name)
        self._input_shape = tuple(model_input.shape)
        self._rows = self._read_rows(tags_path)

    def _layout_and_size(self) -> tuple[str, int, int]:
        shape = self._input_shape or ()
        fallback = self.config.input_size_fallback
        if len(shape) == 4 and shape[-1] == 3:
            height = shape[1] if isinstance(shape[1], int) else fallback
            width = shape[2] if isinstance(shape[2], int) else fallback
            return "nhwc", int(width), int(height)
        if len(shape) == 4 and shape[1] == 3:
            height = shape[2] if isinstance(shape[2], int) else fallback
            width = shape[3] if isinstance(shape[3], int) else fallback
            return "nchw", int(width), int(height)
        return "nhwc", fallback, fallback

    def _prepare(self, path: Path) -> np.ndarray:
        layout, width, height = self._layout_and_size()
        if width != height:
            raise RuntimeError(f"WD14 input is unexpectedly non-square: {width}x{height}")
        with Image.open(path) as opened:
            if (
                bool(getattr(opened, "is_animated", False))
                or int(getattr(opened, "n_frames", 1)) > 1
            ):
                raise ValueError(f"Animated image cannot be tagged: {path}")
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, image).convert("RGB")
        side = max(rgb.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
        resized = square.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32)
        if self.config.bgr_input:
            array = array[:, :, ::-1]
        if layout == "nchw":
            array = np.transpose(array, (2, 0, 1))
        return np.ascontiguousarray(array)

    def _threshold(self, category: str) -> float:
        if category == "character":
            return self.config.character_threshold
        if category == "rating":
            return self.config.rating_threshold
        return self.config.general_threshold

    @property
    def successful_batch_size(self) -> int | None:
        """Last OOM-safe batch size, suitable for a GPU UUID scoped external cache."""

        return self._successful_batch_size

    @staticmethod
    def _is_out_of_memory(exc: Exception) -> bool:
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

    def _run_loaded_batch(
        self,
        paths: Sequence[Path],
        asset_ids: Sequence[str] | None,
    ) -> tuple[ImageTagResult, ...]:
        assert self._session is not None
        assert self._input_name is not None
        assert self._rows is not None
        batch = np.stack([self._prepare(path) for path in paths], axis=0)
        outputs = self._session.run(None, {self._input_name: batch})
        if not outputs:
            raise RuntimeError("WD14 ONNX session returned no outputs")
        probabilities = np.asarray(outputs[0], dtype=np.float32)
        if probabilities.ndim == 1:
            probabilities = probabilities[None, :]
        if probabilities.shape != (len(paths), len(self._rows)):
            raise RuntimeError(
                f"WD14 output shape {probabilities.shape} does not match "
                f"batch/tags {(len(paths), len(self._rows))}"
            )
        results: list[ImageTagResult] = []
        for index, path in enumerate(paths):
            digest = sha256_file(path)
            tags = tuple(
                TagScore(
                    name=name,
                    score=max(0.0, min(1.0, float(probabilities[index, tag_index]))),
                    model_category=category,
                    selected=float(probabilities[index, tag_index]) > self._threshold(category),
                )
                for tag_index, (name, category) in enumerate(self._rows)
            )
            results.append(
                ImageTagResult(
                    asset_id=asset_ids[index] if asset_ids is not None else digest,
                    source_path=path,
                    source_sha256=digest,
                    model_id=self.model_id,
                    revision=self.revision,
                    tags=tags,
                )
            )
        return tuple(results)

    def tag(self, path: Path, *, asset_id: str | None = None) -> ImageTagResult:
        return self.tag_many((path,), asset_ids=(asset_id,) if asset_id is not None else None)[0]

    def tag_many(
        self,
        paths: Sequence[Path],
        *,
        asset_ids: Sequence[str] | None = None,
    ) -> tuple[ImageTagResult, ...]:
        if asset_ids is not None and len(asset_ids) != len(paths):
            raise ValueError("asset_ids must align with paths")
        if not paths:
            return ()
        self._ensure_loaded()
        resolved = tuple(path.resolve(strict=True) for path in paths)
        results: list[ImageTagResult] = []
        cursor = 0
        batch_size = min(
            len(resolved), self._successful_batch_size or self.config.initial_batch_size
        )
        while cursor < len(resolved):
            end = min(len(resolved), cursor + batch_size)
            batch_ids = asset_ids[cursor:end] if asset_ids is not None else None
            try:
                chunk = self._run_loaded_batch(resolved[cursor:end], batch_ids)
            except Exception as exc:
                if batch_size <= 1 or not self._is_out_of_memory(exc):
                    raise
                batch_size = max(1, batch_size // 2)
                continue
            results.extend(chunk)
            cursor = end
            self._successful_batch_size = batch_size
        return tuple(results)


# Compatibility-friendly explicit production name.
WD14Backend = WD14OnnxTagger
