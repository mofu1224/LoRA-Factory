"""Deterministic Unicode-safe image input discovery."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.dataset.image_safety import ImageSafetyLimits

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


class ScanIssue(BaseModel):
    """A source that was not included and the actionable reason why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    code: Literal[
        "missing",
        "not_a_file_or_directory",
        "unsupported_extension",
        "unreadable",
        "too_many_entries",
        "too_many_images",
        "file_too_large",
        "total_size_exceeded",
    ]
    message: str


class ScanResult(BaseModel):
    """Stable scan result. Paths are absolute but are never modified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[Path, ...] = Field(default_factory=tuple)
    issues: tuple[ScanIssue, ...] = Field(default_factory=tuple)


def _path_sort_key(path: Path) -> tuple[str, str]:
    text = os.path.normcase(str(path))
    return (text.casefold(), text)


def _directory_files(directory: Path, *, recursive: bool) -> Iterable[Path]:
    if not recursive:
        yield from (entry for entry in directory.iterdir() if entry.is_file())
        return

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directories, filenames in os.walk(
        directory, followlinks=False, onerror=raise_walk_error
    ):
        directories.sort(key=str.casefold)
        for filename in sorted(filenames, key=str.casefold):
            candidate = Path(current) / filename
            if candidate.is_file():
                yield candidate


def scan_image_inputs(
    inputs: Iterable[Path | str],
    *,
    recursive: bool = False,
    limits: ImageSafetyLimits | None = None,
) -> ScanResult:
    """Discover supported images without decoding or mutating any source.

    Directories are shallow by default. Explicit files always remain eligible regardless of
    the recursive option. Duplicate paths are collapsed using Windows-compatible case folding.
    """

    active_limits = limits or ImageSafetyLimits()
    discovered: dict[str, Path] = {}
    issues: list[ScanIssue] = []
    enumerated_entries = 0
    total_size_bytes = 0
    stop_scanning = False

    for raw_input in inputs:
        source = Path(raw_input).expanduser().resolve(strict=False)
        if not source.exists():
            issues.append(
                ScanIssue(path=source, code="missing", message=f"Input does not exist: {source}")
            )
            continue
        if stop_scanning:
            break
        if source.is_dir():
            candidates = _directory_files(source, recursive=recursive)
        elif source.is_file():
            candidates = (source,)
        else:
            issues.append(
                ScanIssue(
                    path=source,
                    code="not_a_file_or_directory",
                    message=f"Input is neither a regular file nor a directory: {source}",
                )
            )
            continue

        candidate_iterator = iter(candidates)
        while True:
            try:
                candidate = next(candidate_iterator)
            except StopIteration:
                break
            except OSError as exc:
                issues.append(
                    ScanIssue(
                        path=source,
                        code="unreadable",
                        message=f"Cannot enumerate input directory {source}: {exc}",
                    )
                )
                break
            enumerated_entries += 1
            if enumerated_entries > active_limits.max_enumerated_entries:
                issues.append(
                    ScanIssue(
                        path=source,
                        code="too_many_entries",
                        message=(
                            "Input enumeration exceeded the safety limit of "
                            f"{active_limits.max_enumerated_entries} entries"
                        ),
                    )
                )
                stop_scanning = True
                break
            suffix = candidate.suffix.lower()
            resolved = candidate.resolve(strict=False)
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                if candidate == source and source.is_file():
                    issues.append(
                        ScanIssue(
                            path=resolved,
                            code="unsupported_extension",
                            message=f"Unsupported image extension {candidate.suffix!r}",
                        )
                    )
                continue
            key = os.path.normcase(str(resolved)).casefold()
            if key in discovered:
                continue
            try:
                file_size = resolved.stat().st_size
            except OSError as exc:
                issues.append(
                    ScanIssue(
                        path=resolved,
                        code="unreadable",
                        message=f"Cannot read image file metadata {resolved}: {exc}",
                    )
                )
                continue
            if file_size > active_limits.max_file_size_bytes:
                issues.append(
                    ScanIssue(
                        path=resolved,
                        code="file_too_large",
                        message=(
                            f"Image file is {file_size} bytes; limit is "
                            f"{active_limits.max_file_size_bytes} bytes"
                        ),
                    )
                )
                continue
            if len(discovered) >= active_limits.max_image_files:
                issues.append(
                    ScanIssue(
                        path=source,
                        code="too_many_images",
                        message=(
                            "Image count exceeded the safety limit of "
                            f"{active_limits.max_image_files} files"
                        ),
                    )
                )
                stop_scanning = True
                break
            if total_size_bytes + file_size > active_limits.max_total_size_bytes:
                issues.append(
                    ScanIssue(
                        path=source,
                        code="total_size_exceeded",
                        message=(
                            "Combined image size exceeded the safety limit of "
                            f"{active_limits.max_total_size_bytes} bytes"
                        ),
                    )
                )
                stop_scanning = True
                break
            discovered[key] = resolved
            total_size_bytes += file_size

    return ScanResult(
        files=tuple(sorted(discovered.values(), key=_path_sort_key)),
        issues=tuple(sorted(issues, key=lambda issue: _path_sort_key(issue.path))),
    )


class ImageScanner:
    """Object adapter used by application services."""

    def __init__(self, limits: ImageSafetyLimits | None = None) -> None:
        self.limits = limits or ImageSafetyLimits()

    def scan(self, inputs: Iterable[Path | str], *, recursive: bool = False) -> ScanResult:
        return scan_image_inputs(inputs, recursive=recursive, limits=self.limits)
