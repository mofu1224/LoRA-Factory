"""Stable versioned fingerprints for idempotent stage reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(item) for item in value)
    return value


def stage_fingerprint(
    *,
    stage: str,
    stage_version: str,
    inputs: Any,
    config: Any,
    backends: Any,
) -> str:
    payload = {
        "stage": stage,
        "stage_version": stage_version,
        "inputs": _normalize(inputs),
        "config": _normalize(config),
        "backends": _normalize(backends),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
