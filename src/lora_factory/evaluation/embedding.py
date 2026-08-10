"""Pinned learned image embeddings isolated inside the managed CUDA runtime."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Protocol

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.process_manager import ProcessManager
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import write_json_atomic


class EmbeddingRequest(BaseModel):
    """Validated input sent to the managed-runtime helper."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paths: tuple[Path, ...] = Field(min_length=1)


class ImageEmbedding(BaseModel):
    """One normalized learned image vector returned across the process boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def require_finite_nonzero_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("Embedding values must all be finite")
        norm = math.sqrt(sum(item * item for item in value))
        if norm <= 0:
            raise ValueError("Embedding vector must be non-zero")
        if not math.isclose(norm, 1.0, rel_tol=2e-3, abs_tol=2e-3):
            raise ValueError(f"Embedding vector must be L2-normalized, got norm={norm:.6f}")
        return value


class ImageEmbeddingBatch(BaseModel):
    """Auditable model identity and aligned embedding results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$|^fake-[0-9]+$")
    dimension: Annotated[int, Field(gt=0)]
    device: str = Field(min_length=1)
    items: tuple[ImageEmbedding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_aligned_dimension(self) -> ImageEmbeddingBatch:
        if any(len(item.values) != self.dimension for item in self.items):
            raise ValueError("Embedding item dimensions do not match the declared dimension")
        paths = [item.source_path.resolve(strict=False) for item in self.items]
        if len(set(paths)) != len(paths):
            raise ValueError("Embedding output contains duplicate source paths")
        return self

    def vectors_by_path(self) -> dict[Path, np.ndarray]:
        return {
            item.source_path.resolve(strict=False): np.asarray(item.values, dtype=np.float32)
            for item in self.items
        }


class ImageEmbeddingBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def embed_many(self, paths: Sequence[Path]) -> ImageEmbeddingBatch: ...


class FakeImageEmbeddingBackend:
    """Deterministic in-process test double; it is always labeled as non-production."""

    model_id = "lora-factory/fake-image-embedding"
    revision = "fake-1"

    @staticmethod
    def _vector(path: Path) -> tuple[float, ...]:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB").resize((8, 8))
            values = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        values -= float(values.mean())
        norm = float(np.linalg.norm(values))
        if norm == 0:
            values = np.zeros_like(values)
            values[0] = 1.0
        else:
            values /= norm
        return tuple(float(item) for item in values)

    def embed_many(self, paths: Sequence[Path]) -> ImageEmbeddingBatch:
        if not paths:
            raise ValueError("At least one image is required for embedding")
        resolved = tuple(path.resolve(strict=True) for path in paths)
        items = tuple(
            ImageEmbedding(
                source_path=path,
                source_sha256=sha256_file(path),
                values=self._vector(path),
            )
            for path in resolved
        )
        return ImageEmbeddingBatch(
            model_id=self.model_id,
            revision=self.revision,
            dimension=len(items[0].values),
            device="fake",
            items=items,
        )


class ManagedClipImageEmbeddingBackend:
    """Run pinned CLIP vision projection inference in an isolated CUDA child process."""

    def __init__(
        self,
        *,
        python_executable: Path,
        helper_script: Path,
        model_directory: Path,
        model_id: str,
        revision: str,
        config_sha256: str,
        model_sha256: str,
        binding: GpuBinding,
        work_directory: Path,
        cancellation: CancellationToken,
        batch_size: int = 8,
        timeout_seconds: int = 1800,
    ) -> None:
        if batch_size < 1:
            raise ValueError("CLIP embedding batch_size must be positive")
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("CLIP embedding revision must be a full lowercase commit SHA")
        self.python_executable = python_executable.resolve(strict=True)
        self.helper_script = helper_script.resolve(strict=True)
        self.model_directory = model_directory.resolve(strict=True)
        self.model_id = model_id
        self.revision = revision
        self.config_sha256 = config_sha256
        self.model_sha256 = model_sha256
        self.binding = binding
        self.work_directory = work_directory.resolve(strict=False)
        self.cancellation = cancellation
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def _verify_model(self) -> None:
        expected = {
            "config.json": self.config_sha256,
            "model.safetensors": self.model_sha256,
        }
        for filename, expected_sha256 in expected.items():
            path = self.model_directory / filename
            if not path.is_file():
                raise RuntimeError(
                    "Pinned CLIP embedding artifact is missing: "
                    f"{path}; repair Managed Training Runtime"
                )
            actual = sha256_file(path)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"Pinned CLIP embedding artifact failed SHA-256 verification: {filename}"
                )

    def embed_many(self, paths: Sequence[Path]) -> ImageEmbeddingBatch:
        if not paths:
            raise ValueError("At least one image is required for embedding")
        resolved = tuple(path.resolve(strict=True) for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Embedding input paths must be unique")
        self._verify_model()
        self.work_directory.mkdir(parents=True, exist_ok=True)
        input_path = self.work_directory / "input.json"
        output_path = self.work_directory / "output.json"
        output_path.unlink(missing_ok=True)
        request = EmbeddingRequest(paths=resolved)
        write_json_atomic(input_path, request.model_dump(mode="json"))
        environment = {
            **self.binding.environment,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        torch_library = (
            self.python_executable.parent.parent / "Lib" / "site-packages" / "torch" / "lib"
        )
        if os.name == "nt" and torch_library.is_dir():
            environment["PATH"] = os.pathsep.join((str(torch_library), os.environ.get("PATH", "")))
        result = ProcessManager().run(
            [
                str(self.python_executable),
                str(self.helper_script),
                "--model-directory",
                str(self.model_directory),
                "--model-id",
                self.model_id,
                "--revision",
                self.revision,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--batch-size",
                str(self.batch_size),
            ],
            cwd=self.helper_script.parent,
            environment=environment,
            stdout_path=self.work_directory / "stdout.log",
            stderr_path=self.work_directory / "stderr.log",
            cancellation=self.cancellation,
            timeout_seconds=self.timeout_seconds,
            on_line=lambda _channel, _line: None,
        )
        if result.cancelled:
            self.cancellation.raise_if_cancelled()
        if result.timed_out:
            raise TimeoutError("Managed CLIP image embedding timed out")
        if result.return_code != 0 or not output_path.is_file():
            stderr = result.stderr_path.read_text(encoding="utf-8", errors="replace")
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no diagnostics"
            raise RuntimeError(
                f"Managed CLIP image embedding failed (exit {result.return_code}): {detail}"
            )
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            batch = ImageEmbeddingBatch.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("Managed CLIP returned malformed embedding JSON") from exc
        if batch.model_id != self.model_id or batch.revision != self.revision:
            raise RuntimeError(
                "Managed CLIP output model identity does not match the pinned request"
            )
        returned = tuple(item.source_path.resolve(strict=False) for item in batch.items)
        if returned != resolved:
            raise RuntimeError("Managed CLIP output paths are not aligned with the request")
        for path, item in zip(resolved, batch.items, strict=True):
            if item.source_sha256 != sha256_file(path):
                raise RuntimeError(f"Managed CLIP source hash changed during inference: {path}")
        if batch.device != "cuda:0":
            raise RuntimeError(f"Managed CLIP did not use isolated CUDA device: {batch.device}")
        return batch
