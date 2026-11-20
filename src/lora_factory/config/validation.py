"""Cross-field validation helpers used by both GUI and CLI."""

from __future__ import annotations

from pathlib import Path

# A deliberately conservative subset of common Danbooru tags that cannot be
# used as Trigger Words because they would activate unrelated visual concepts.
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


def validate_trigger_word_collision(token: str) -> None:
    """Reject an obvious existing-tag collision at every Trigger Word boundary."""

    canonical = "_".join(token.strip().casefold().split())
    if canonical not in KNOWN_DANBOORU_TRIGGER_COLLISIONS:
        return
    raise ValueError(
        f"Trigger Word {token.strip()!r} collides with common Danbooru tag {canonical!r}; "
        "choose a unique token"
    )


def trigger_token_collision_warning(token: str) -> str | None:
    """Return the blocking collision message for immediate GUI feedback."""

    try:
        validate_trigger_word_collision(token)
    except ValueError as exc:
        return str(exc)
    return None


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
