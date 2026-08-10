"""Cross-field validation helpers used by both GUI and CLI."""

from __future__ import annotations

from pathlib import Path

# A deliberately conservative subset of common Danbooru tags.  A match is a
# warning, never a validation error: users remain the source of truth for the
# trigger token, while the beginner-facing UI can flag an obvious collision.
KNOWN_DANBOORU_TRIGGER_COLLISIONS = frozenset(
    {
        "1boy",
        "1girl",
        "2boys",
        "2girls",
        "animal",
        "best_quality",
        "furry",
        "group",
        "highres",
        "indoors",
        "lowres",
        "masterpiece",
        "multiple_boys",
        "multiple_girls",
        "outdoors",
        "person",
        "portrait",
        "signature",
        "solo",
        "text",
        "watermark",
    }
)


def trigger_token_collision_warning(token: str) -> str | None:
    """Return a non-blocking warning for an obvious existing tag collision."""

    canonical = "_".join(token.strip().casefold().split())
    if canonical not in KNOWN_DANBOORU_TRIGGER_COLLISIONS:
        return None
    return (
        f"Warning: {token.strip()!r} is also a common Danbooru tag. "
        "You may keep it, but a more unique trigger usually reduces accidental activation."
    )


def ensure_descendant(path: Path, root: Path) -> Path:
    """Resolve ``path`` and reject traversal outside ``root``."""

    resolved_root = root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes configured root: {path}")
    return resolved


def validate_existing_file(path: Path, *, suffixes: set[str] | None = None) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Expected a file: {path}")
    if suffixes is not None and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported file type: {resolved.suffix}")
    return resolved
