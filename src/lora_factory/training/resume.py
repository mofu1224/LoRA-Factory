"""Filesystem-backed training attempt and optimizer-state resume selection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.config.models import TrainingPlan
from lora_factory.util.hashing import sha256_bytes, sha256_file
from lora_factory.util.json import write_json_atomic

_ATTEMPT_DIRECTORY = re.compile(r"^attempt-(?P<number>[0-9]{3})$")
_REQUEST_RECORD = "training-request.json"
_MAX_ATTEMPTS = 3


class TrainingAttemptSelection(BaseModel):
    """A new non-overwriting attempt and an optional compatible prior state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int = Field(ge=1, le=_MAX_ATTEMPTS)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume_state: Path | None = None
    resumed_from_attempt: int | None = Field(default=None, ge=1, le=_MAX_ATTEMPTS)
    record_path: Path


def training_request_fingerprint(
    *,
    run_id: str,
    output_name: str,
    base_model_sha256: str,
    dataset_config: Path,
    plan: TrainingPlan,
    seed: int,
    backend_version: str,
) -> str:
    """Fingerprint every input that must remain compatible with optimizer state."""

    if not re.fullmatch(r"[0-9a-fA-F]{64}", base_model_sha256):
        raise ValueError("base_model_sha256 must contain exactly 64 hexadecimal characters")
    if not dataset_config.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {dataset_config}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "output_name": output_name,
        "base_model_sha256": base_model_sha256.casefold(),
        "dataset_config_sha256": sha256_file(dataset_config),
        "plan": plan.model_dump(mode="json"),
        "seed": seed,
        "backend_version": backend_version,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def prepare_training_attempt(
    *,
    run_directory: Path,
    request_fingerprint: str,
) -> TrainingAttemptSelection:
    """Reserve a fresh attempt and locate a matching prior optimizer state.

    Attempts are never reused, so an interrupted or successful checkpoint cannot be
    overwritten.  A state is eligible only when its recorded request fingerprint
    matches the current model, dataset, plan, seed, and backend version.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", request_fingerprint):
        raise ValueError("request_fingerprint must be a lowercase SHA-256 digest")
    run_directory.mkdir(parents=True, exist_ok=True)
    previous = _attempt_directories(run_directory)
    attempt = (previous[-1][0] + 1) if previous else 1
    if attempt > _MAX_ATTEMPTS:
        raise RuntimeError(
            f"Maximum training attempts ({_MAX_ATTEMPTS}) reached; start a new run "
            "rather than overwriting prior artifacts"
        )

    resume_state: Path | None = None
    resumed_from: int | None = None
    for previous_attempt, attempt_root in reversed(previous):
        if _recorded_fingerprint(attempt_root) != request_fingerprint:
            continue
        candidate = _latest_resume_state(attempt_root)
        if candidate is not None:
            resume_state = candidate
            resumed_from = previous_attempt
            break

    attempt_root = run_directory / f"attempt-{attempt:03d}"
    attempt_root.mkdir(parents=False, exist_ok=False)
    record_path = attempt_root / _REQUEST_RECORD
    write_json_atomic(
        record_path,
        {
            "schema_version": 1,
            "attempt": attempt,
            "request_fingerprint": request_fingerprint,
            "resume_state": str(resume_state) if resume_state is not None else None,
            "resumed_from_attempt": resumed_from,
        },
    )
    return TrainingAttemptSelection(
        attempt=attempt,
        request_fingerprint=request_fingerprint,
        resume_state=resume_state,
        resumed_from_attempt=resumed_from,
        record_path=record_path,
    )


def _attempt_directories(run_directory: Path) -> list[tuple[int, Path]]:
    attempts: list[tuple[int, Path]] = []
    for path in run_directory.iterdir():
        if not path.is_dir():
            continue
        match = _ATTEMPT_DIRECTORY.fullmatch(path.name)
        if match is not None:
            attempts.append((int(match.group("number")), path))
    attempts.sort(key=lambda item: item[0])
    numbers = [number for number, _path in attempts]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError("Training attempt directory numbers are not unique")
    return attempts


def _recorded_fingerprint(attempt_root: Path) -> str | None:
    record = attempt_root / _REQUEST_RECORD
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("request_fingerprint") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _latest_resume_state(attempt_root: Path) -> Path | None:
    candidates: list[Path] = []
    fake_state = attempt_root / "states" / "training-state.json"
    if fake_state.is_file():
        candidates.append(fake_state)
    for root_name in ("checkpoints", "states"):
        root = attempt_root / root_name
        if not root.is_dir():
            continue
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.casefold().endswith("-state")
        )
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
