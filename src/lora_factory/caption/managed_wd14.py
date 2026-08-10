"""Managed-runtime WD14 adapter using one isolated CUDA child process."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from lora_factory.caption.tagger import ImageTagResult, TagScore
from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.process_manager import ProcessManager
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json, write_json_atomic


class ManagedWD14Tagger:
    """Run the pinned model inside the versioned training runtime."""

    model_id = "SmilingWolf/wd-eva02-large-tagger-v3"

    def __init__(
        self,
        *,
        python_executable: Path,
        helper_script: Path,
        model_path: Path,
        tags_path: Path,
        revision: str,
        binding: GpuBinding,
        work_directory: Path,
        cancellation: CancellationToken,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        initial_batch_size: int = 8,
        timeout_seconds: int = 900,
    ) -> None:
        if not 1 <= initial_batch_size <= 256:
            raise ValueError("WD14 initial batch size must be between 1 and 256")
        self.python_executable = python_executable.resolve(strict=True)
        self.helper_script = helper_script.resolve(strict=True)
        self.model_path = model_path.resolve(strict=True)
        self.tags_path = tags_path.resolve(strict=True)
        self.revision = revision
        self.binding = binding
        self.work_directory = work_directory.resolve(strict=False)
        self.cancellation = cancellation
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.initial_batch_size = initial_batch_size
        self.timeout_seconds = timeout_seconds
        self._successful_batch_size: int | None = None

    @property
    def successful_batch_size(self) -> int | None:
        return self._successful_batch_size

    @property
    def _batch_cache_path(self) -> Path:
        safe_uuid = self.binding.uuid.replace("-", "_")
        return self.model_path.parent / "batch-cache" / f"{safe_uuid}.json"

    def _cached_batch_size(self) -> int:
        path = self._batch_cache_path
        if not path.is_file():
            return self.initial_batch_size
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            return self.initial_batch_size
        if not isinstance(payload, dict) or payload.get("revision") != self.revision:
            return self.initial_batch_size
        value = payload.get("successful_batch_size")
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 256:
            return value
        return self.initial_batch_size

    def _record_batch_size(self, value: int) -> None:
        if not 1 <= value <= 256:
            raise ValueError("Managed WD14 returned an invalid successful batch size")
        self._successful_batch_size = value
        write_json_atomic(
            self._batch_cache_path,
            {
                "gpu_uuid": self.binding.uuid,
                "revision": self.revision,
                "successful_batch_size": value,
            },
        )

    def tag(self, path: Path, *, asset_id: str | None = None) -> ImageTagResult:
        return self.tag_many((path,), asset_ids=(asset_id,) if asset_id else None)[0]

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
        self.work_directory.mkdir(parents=True, exist_ok=True)
        resolved = tuple(path.resolve(strict=True) for path in paths)
        identifiers = tuple(
            asset_ids[index] if asset_ids is not None else sha256_file(path)
            for index, path in enumerate(resolved)
        )
        input_path = self.work_directory / "input.json"
        output_path = self.work_directory / "output.json"
        output_path.unlink(missing_ok=True)
        write_json_atomic(
            input_path,
            {
                "paths": [str(path) for path in resolved],
                "asset_ids": list(identifiers),
            },
        )
        environment = dict(self.binding.environment)
        torch_library = (
            self.python_executable.parent.parent / "Lib" / "site-packages" / "torch" / "lib"
        )
        if os.name == "nt" and torch_library.is_dir():
            environment["PATH"] = os.pathsep.join(
                (str(torch_library), environment.get("PATH", os.environ.get("PATH", "")))
            )
        result = ProcessManager().run(
            [
                str(self.python_executable),
                str(self.helper_script),
                "--model",
                str(self.model_path),
                "--tags",
                str(self.tags_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--general-threshold",
                str(self.general_threshold),
                "--character-threshold",
                str(self.character_threshold),
                "--batch-size",
                str(self._cached_batch_size()),
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
            raise TimeoutError("Managed WD14 inference timed out")
        if result.return_code != 0 or not output_path.is_file():
            stderr = result.stderr_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                "Managed WD14 inference failed: "
                + (
                    stderr.strip().splitlines()[-1]
                    if stderr.strip()
                    else f"exit {result.return_code}"
                )
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("items")
            successful_batch_size = payload.get("successful_batch_size")
        else:
            # Compatibility with a helper from an incomplete interrupted installation.
            items = payload
            successful_batch_size = self.initial_batch_size
        if not isinstance(successful_batch_size, int) or isinstance(successful_batch_size, bool):
            raise RuntimeError("Managed WD14 returned an invalid batch-size record")
        self._record_batch_size(successful_batch_size)
        if not isinstance(items, list) or len(items) != len(resolved):
            raise RuntimeError("Managed WD14 returned an invalid result count")
        results: list[ImageTagResult] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("tags"), list):
                raise RuntimeError("Managed WD14 returned malformed result JSON")
            results.append(
                ImageTagResult(
                    asset_id=identifiers[index],
                    source_path=resolved[index],
                    source_sha256=sha256_file(resolved[index]),
                    model_id=self.model_id,
                    revision=self.revision,
                    tags=tuple(TagScore.model_validate(tag) for tag in item["tags"]),
                )
            )
        return tuple(results)
