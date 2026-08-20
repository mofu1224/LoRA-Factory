"""Filesystem boundary checks for immutable Raw objects."""

from __future__ import annotations

from pathlib import Path

from lora_factory.project.layout import ProjectLayout


class RawStoreBoundaryError(OSError):
    """Raised before Raw-store I/O when a path can alias data outside the store."""


def validate_raw_object_path(layout: ProjectLayout, stored_filename: str) -> Path:
    """Return the lexical Raw path after rejecting aliases and containment escapes."""

    project_root = layout.root.resolve(strict=False)
    raw_root = layout.raw
    if raw_root.is_symlink() or raw_root.is_junction():
        raise RawStoreBoundaryError(
            f"Raw store must not be a symbolic link or junction: {raw_root}"
        )
    resolved_raw_root = raw_root.resolve(strict=False)
    if not resolved_raw_root.is_relative_to(project_root):
        raise RawStoreBoundaryError(f"Raw store escapes the project root: {raw_root}")

    candidate = raw_root / stored_filename
    if candidate.is_symlink() or candidate.is_junction():
        raise RawStoreBoundaryError(
            f"Raw object must not be a symbolic link or junction: {candidate}"
        )
    if not candidate.resolve(strict=False).is_relative_to(resolved_raw_root):
        raise RawStoreBoundaryError(f"Raw object escapes the store: {candidate}")
    try:
        metadata = candidate.stat(follow_symlinks=False)
    except FileNotFoundError:
        return candidate
    if metadata.st_nlink != 1:
        raise RawStoreBoundaryError(
            f"Raw object must have exactly one filesystem link: {candidate}"
        )
    return candidate
