"""Artifact validation and reproducibility manifest helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from safetensors import safe_open

from lora_factory.util.hashing import sha256_file

_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{8,}")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_PATH_IN_TEXT = re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]|\\\\)[^\r\n\t\"']+")
_POSIX_PATH_IN_TEXT = re.compile(r"(?<![A-Za-z0-9:])/(?:home|Users|tmp|var/tmp|mnt)/[^\r\n\t\"']+")
_OMITTED_PUBLIC_KEYS = frozenset(
    {
        "codex_runtime_root",
        "command_argv",
        "configured_root",
        "input_paths",
        "managed_runtime_root",
        "original_filename",
        "original_filenames",
        "original_path",
        "original_paths",
        "output_root",
        "projects_root",
        "training_command_arguments",
        "training_dataset_config",
    }
)


class PublicMetadataSanitizer:
    """Remove machine-local identifiers while retaining portable evidence."""

    def __init__(self) -> None:
        self._gpu_aliases: dict[str, str] = {}

    def sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self.sanitize(item)
                for key, item in value.items()
                if str(key) not in _OMITTED_PUBLIC_KEYS
            }
        if isinstance(value, list | tuple):
            return [self.sanitize(item) for item in value]
        if isinstance(value, Path):
            return value.name
        if isinstance(value, str):
            sanitized = _GPU_UUID.sub(self._gpu_alias, value)
            if self._is_absolute_path(sanitized):
                normalized = sanitized.replace("\\", "/").rstrip("/")
                return normalized.rsplit("/", 1)[-1] or "<local-path>"
            sanitized = _WINDOWS_PATH_IN_TEXT.sub("<local-path>", sanitized)
            return _POSIX_PATH_IN_TEXT.sub("<local-path>", sanitized)
        return value

    def _gpu_alias(self, match: re.Match[str]) -> str:
        uuid = match.group(0)
        return self._gpu_aliases.setdefault(uuid, f"gpu-{len(self._gpu_aliases) + 1}")

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        return bool(
            _WINDOWS_ABSOLUTE_PATH.match(value)
            or value.startswith("\\\\")
            or (value.startswith("/") and not value.startswith("//"))
        )


def sanitize_public_metadata(value: Any) -> Any:
    """Sanitize one public metadata document with consistent GPU aliases."""

    return PublicMetadataSanitizer().sanitize(value)


def inspect_safetensors(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty safetensors file: {path}")
    try:
        with safe_open(path, framework="numpy") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise ValueError(f"Invalid safetensors checkpoint: {path}") from exc
    if not keys:
        raise ValueError(f"Safetensors checkpoint contains no tensors: {path}")
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_count": len(keys),
        "key_sample": keys[:20],
        "metadata": metadata,
    }


def artifact_record(path: Path) -> dict[str, str | int]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
