"""Conservative secret and home-path redaction for process logs."""

from __future__ import annotations

import re
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|authorization|password)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/-]+"),
)


def redact_text(value: str, *, home: Path | None = None) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}=[REDACTED]" if match.lastindex == 2 else "[REDACTED]",
            redacted,
        )
    if home is not None:
        redacted = redacted.replace(str(home), "%USERPROFILE%")
    return redacted
