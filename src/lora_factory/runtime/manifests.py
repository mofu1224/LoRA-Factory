"""Validated managed-runtime and upstream backend manifests."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BackendPin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    release: str
    commit_sha: str

    @field_validator("repository")
    @classmethod
    def require_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("Backend repository must use HTTPS")
        return value.rstrip("/")

    @field_validator("commit_sha")
    @classmethod
    def require_full_commit_sha(cls, value: str) -> str:
        normalized = value.lower()
        if not _COMMIT_SHA.fullmatch(normalized):
            raise ValueError("Backend commit_sha must be a full 40-character SHA")
        return normalized


class DownloadArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    sha256: str

    @field_validator("url")
    @classmethod
    def require_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("Runtime downloads must use HTTPS")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("Download SHA256 must contain 64 hexadecimal characters")
        return normalized


class ManagedRuntimeManifest(BaseModel):
    """Source of truth for one isolated training/tagging runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    runtime_id: str
    python_version: str
    python_executable: Path
    sd_scripts_root: Path
    sd_scripts: BackendPin
    torch_version: str
    torch_cuda_version: str
    onnxruntime_version: str
    packages_lock_sha256: str
    artifacts: tuple[DownloadArtifact, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("packages_lock_sha256")
    @classmethod
    def validate_lock_hash(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("packages_lock_sha256 must contain 64 hexadecimal characters")
        return normalized


def load_runtime_manifest(path: Path) -> ManagedRuntimeManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load runtime manifest {path}: {exc}") from exc
    return ManagedRuntimeManifest.model_validate(payload)


def write_runtime_manifest(path: Path, manifest: ManagedRuntimeManifest) -> str:
    """Atomically write a manifest and return the exact file SHA256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(serialized)
    temporary.replace(path)
    return hashlib.sha256(serialized).hexdigest()
