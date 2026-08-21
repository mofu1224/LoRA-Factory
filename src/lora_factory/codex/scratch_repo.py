"""Dedicated sanitized Git workspace for Runtime Codex."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lora_factory.codex.schemas import CodexTaskType, StrictModel, strict_output_schema
from lora_factory.util.json import write_json_atomic

RUNTIME_AGENTS = """# Runtime Codex restrictions

- Read-only analysis only.
- Use only the sanitized files under this scratch repository.
- Do not inspect parent directories, user configuration, credentials, or unrelated files.
- Do not run arbitrary commands or modify files.
- Return exactly the requested JSON schema.
- Do not infer missing data as fact.
- Do not reveal secrets or personal paths.
"""


@dataclass(frozen=True, slots=True)
class ScratchCall:
    root: Path
    input_path: Path
    schema_path: Path
    output_path: Path
    events_path: Path
    stderr_path: Path
    image_paths: tuple[Path, ...] = ()


class ScratchRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("input", "schemas", "output", "logs"):
            (self.root / name).mkdir(exist_ok=True)
        (self.root / "AGENTS.md").write_text(RUNTIME_AGENTS, encoding="utf-8")
        if not (self.root / ".git").is_dir():
            git_executable = shutil.which("git")
            if git_executable is None:
                raise RuntimeError("Git is required for the Runtime Codex scratch repository")
            subprocess.run(  # noqa: S603 - executable was resolved with shutil.which.
                [git_executable, "init", "--quiet"],
                cwd=self.root,
                check=True,
                shell=False,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

    def prepare_call(
        self,
        *,
        call_id: str,
        task_type: CodexTaskType,
        sanitized_input: dict[str, Any],
        response_model: type[StrictModel],
        image_paths: Sequence[Path] = (),
    ) -> ScratchCall:
        self.initialize()
        resolved_image_paths = self._resolve_image_paths(task_type, image_paths)
        call_root = self.root / "output" / call_id
        if call_root.exists():
            raise FileExistsError(f"Codex call directory already exists: {call_id}")
        call_root.mkdir(parents=True)
        input_path = self.root / "input" / f"{call_id}.json"
        schema_path = self.root / "schemas" / f"{task_type.value}.schema.json"
        output_path = call_root / "result.json"
        events_path = call_root / "events.jsonl"
        stderr_path = call_root / "stderr.log"
        write_json_atomic(input_path, sanitized_input)
        write_json_atomic(schema_path, strict_output_schema(response_model))
        return ScratchCall(
            root=self.root,
            input_path=input_path,
            schema_path=schema_path,
            output_path=output_path,
            events_path=events_path,
            stderr_path=stderr_path,
            image_paths=resolved_image_paths,
        )

    def _resolve_image_paths(
        self,
        task_type: CodexTaskType,
        image_paths: Sequence[Path],
    ) -> tuple[Path, ...]:
        if task_type is CodexTaskType.DATASET_REFINEMENT and len(image_paths) > 8:
            raise ValueError("Dataset refinement supports at most 8 images per Codex call")

        resolved_root = self.root.resolve(strict=True)
        resolved_images: list[Path] = []
        for image_path in image_paths:
            try:
                resolved_image = Path(image_path).resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    "Codex image must be a regular file in the scratch repository"
                ) from exc
            if not resolved_image.is_relative_to(resolved_root):
                raise ValueError("Codex image must stay inside the dedicated scratch repository")
            if not resolved_image.is_file():
                raise ValueError("Codex image must be a regular file in the scratch repository")
            resolved_images.append(resolved_image)
        return tuple(resolved_images)
