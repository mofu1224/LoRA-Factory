"""Atomic caption writer constrained to the derived captions directory."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.util.hashing import sha256_bytes, sha256_file

SAFE_ASSET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


class CaptionWriteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    paths: dict[str, Path]
    sha256: dict[str, str]
    unchanged: tuple[str, ...] = Field(default_factory=tuple)


class CaptionWriter:
    def __init__(self, captions_root: Path, *, raw_root: Path | None = None) -> None:
        self.captions_root = captions_root.resolve(strict=False)
        self.raw_root = raw_root.resolve(strict=False) if raw_root is not None else None
        if self.raw_root is not None and (
            self.captions_root == self.raw_root or self.captions_root.is_relative_to(self.raw_root)
        ):
            raise ValueError("Caption output must be outside the immutable Raw store")

    def write(
        self,
        captions: Mapping[str, str],
        *,
        raw_paths: Mapping[str, Path] | None = None,
    ) -> CaptionWriteResult:
        before = {asset_id: sha256_file(path) for asset_id, path in (raw_paths or {}).items()}
        self.captions_root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        unchanged: list[str] = []
        for asset_id, caption in sorted(captions.items()):
            if not SAFE_ASSET_ID.fullmatch(asset_id) or asset_id in {".", ".."}:
                raise ValueError(f"Unsafe asset_id for caption filename: {asset_id!r}")
            if "\n" in caption or "\r" in caption or not caption.strip():
                raise ValueError(f"Caption for {asset_id!r} must be one non-empty line")
            destination = (self.captions_root / f"{asset_id}.txt").resolve(strict=False)
            if not destination.is_relative_to(self.captions_root):
                raise ValueError("Caption destination escapes the caption store")
            payload = f"{caption.strip()}\n".encode()
            expected = sha256_bytes(payload)
            if destination.is_file() and sha256_file(destination) == expected:
                unchanged.append(asset_id)
            else:
                temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(destination)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            paths[asset_id] = destination
            hashes[asset_id] = expected
        for asset_id, expected in before.items():
            path = (raw_paths or {})[asset_id]
            if sha256_file(path) != expected:
                raise RuntimeError(f"Immutable Raw asset changed while writing captions: {path}")
        return CaptionWriteResult(paths=paths, sha256=hashes, unchanged=tuple(unchanged))


def write_captions(
    captions_root: Path,
    captions: Mapping[str, str],
    *,
    raw_root: Path | None = None,
) -> CaptionWriteResult:
    return CaptionWriter(captions_root, raw_root=raw_root).write(captions)
