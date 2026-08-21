"""Versioned managed backend paths and integrity state."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lora_factory.runtime.validation_record import RuntimeValidationRecord
from lora_factory.util.json import write_json_atomic


@dataclass(frozen=True, slots=True)
class ManagedRuntimeLayout:
    root: Path

    @property
    def environment(self) -> Path:
        return self.root / ".venv"

    @property
    def python(self) -> Path:
        return self.environment / "Scripts" / "python.exe"

    @property
    def sd_scripts(self) -> Path:
        return self.root / "sd-scripts"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def wd14(self) -> Path:
        return self.models / "wd14"

    @property
    def image_embedding(self) -> Path:
        return self.models / "clip-vit-large-patch14"

    @property
    def installation_record(self) -> Path:
        return self.root / "installation.json"

    @property
    def validation_record(self) -> Path:
        return self.root / "validation.json"


class RuntimeManager:
    def __init__(self, managed_root: Path, manifest_path: Path) -> None:
        self.managed_root = managed_root.resolve(strict=False)
        self.manifest_path = manifest_path.resolve(strict=True)
        self.manifest = self._load_manifest()

    @property
    def layout(self) -> ManagedRuntimeLayout:
        profile_id = str(self.manifest["profile_id"])
        return ManagedRuntimeLayout(self.managed_root / profile_id)

    def _load_manifest(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported backend manifest schema")
        return payload

    def installed(self) -> bool:
        layout = self.layout
        return (
            layout.python.is_file()
            and layout.sd_scripts.is_dir()
            and layout.installation_record.is_file()
            and (layout.wd14 / "model.onnx").is_file()
            and (layout.image_embedding / "config.json").is_file()
            and (layout.image_embedding / "model.safetensors").is_file()
        )

    def installed_commit(self) -> str | None:
        if not (self.layout.sd_scripts / ".git").is_dir():
            return None
        git_executable = shutil.which("git")
        if git_executable is None:
            return None
        safe_directory = str(self.layout.sd_scripts.resolve(strict=False))
        completed = subprocess.run(  # noqa: S603 - executable resolved; fixed read-only query.
            [git_executable, "-c", f"safe.directory={safe_directory}", "rev-parse", "HEAD"],
            cwd=self.layout.sd_scripts,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    def source_matches_manifest(self) -> bool:
        expected = str(self.manifest["sd_scripts"]["commit"])
        return self.installed_commit() == expected

    def read_validation_record(self) -> RuntimeValidationRecord | None:
        path = self.layout.validation_record
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read runtime validation record {path}: {exc}") from exc
        record = RuntimeValidationRecord.model_validate(payload)
        expected_profile = str(self.manifest["profile_id"])
        if record.profile_id != expected_profile:
            raise ValueError(
                "Runtime validation record belongs to a different compatibility profile"
            )
        return record

    def write_validation_record(self, record: RuntimeValidationRecord) -> None:
        expected_profile = str(self.manifest["profile_id"])
        if record.profile_id != expected_profile:
            raise ValueError(
                "Runtime validation record belongs to a different compatibility profile"
            )
        write_json_atomic(self.layout.validation_record, record.model_dump(mode="json"))
