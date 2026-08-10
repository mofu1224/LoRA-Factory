"""Non-overwriting, checksum-verified copies to common LoRA directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from lora_factory.config.models import DestinationKind
from lora_factory.util.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class CopyResult:
    source: Path
    destination: Path
    sha256: str
    versioned: bool


def resolve_lora_directory(root: Path, kind: DestinationKind) -> Path:
    root = root.resolve(strict=False)
    if kind in {DestinationKind.A1111, DestinationKind.FORGE}:
        if root.name.casefold() == "lora":
            return root
        return root / "models" / "Lora"
    if root.name.casefold() == "loras":
        return root
    return root / "models" / "loras"


def copy_without_overwrite(
    source: Path,
    *,
    configured_root: Path,
    kind: DestinationKind,
) -> CopyResult:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_directory = resolve_lora_directory(configured_root, kind)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / source.name
    versioned = False
    counter = 2
    while destination.exists():
        versioned = True
        destination = destination_directory / f"{source.stem}_v{counter}{source.suffix}"
        counter += 1
    temporary = destination.with_name(f".{destination.name}.copying")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise OSError("Destination checksum verification failed")
    temporary.replace(destination)
    return CopyResult(
        source=source,
        destination=destination,
        sha256=sha256_file(destination),
        versioned=versioned,
    )
