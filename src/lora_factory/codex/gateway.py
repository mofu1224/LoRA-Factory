"""High-level structured Runtime Codex gateway with fallback and immutable audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lora_factory.codex.environment import build_codex_environment
from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.process import build_codex_arguments, run_codex_process
from lora_factory.codex.prompts import prompt_for
from lora_factory.codex.schemas import SCHEMA_MODELS, CodexTaskType, StrictModel
from lora_factory.codex.scratch_repo import ScratchRepository
from lora_factory.util.hashing import sha256_bytes, sha256_file


@dataclass(frozen=True, slots=True)
class CodexAudit:
    call_id: str
    task_type: CodexTaskType
    timestamp: datetime
    prompt_sha256: str
    schema_sha256: str
    input_sha256: str
    result_sha256: str | None
    exit_code: int | None
    duration_seconds: float
    version: str | None
    timed_out: bool
    fallback_used: bool
    attempts: int
    applied_changes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CodexGatewayResult:
    response: StrictModel
    audit: CodexAudit
    warning: str | None = None


class CodexGateway:
    def __init__(
        self,
        scratch_root: Path,
        *,
        executable: str = "codex",
        timeout_seconds: int = 180,
        max_attempts: int = 2,
    ) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("Codex attempts must be between 1 and 3")
        self.scratch = ScratchRepository(scratch_root)
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def version(self) -> str | None:
        resolved = shutil.which(self.executable)
        if resolved is None:
            return None
        completed = subprocess.run(  # noqa: S603 - executable was resolved with shutil.which.
            [resolved, "--version"],
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            env=build_codex_environment(os.environ),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    def review(
        self,
        task_type: CodexTaskType,
        sanitized_input: dict[str, Any],
        *,
        allow_fallback: bool,
    ) -> CodexGatewayResult:
        response_model = SCHEMA_MODELS[task_type]
        call_id = uuid.uuid4().hex
        call = self.scratch.prepare_call(
            call_id=call_id,
            task_type=task_type,
            sanitized_input=sanitized_input,
            response_model=response_model,
        )
        prompt = prompt_for(task_type, call.input_path.name, sanitized_input)
        version = self.version()
        started = time.monotonic()
        attempts = 0
        last_exit: int | None = None
        timed_out = False
        warning: str | None = None
        response: StrictModel | None = None

        if version is not None:
            for attempt_number in range(1, self.max_attempts + 1):
                attempts = attempt_number
                arguments = build_codex_arguments(
                    executable=self.executable,
                    call=call,
                    prompt=prompt,
                )
                process_result = run_codex_process(
                    arguments,
                    call=call,
                    timeout_seconds=self.timeout_seconds,
                    home_for_redaction=Path.home(),
                )
                last_exit = process_result.return_code
                timed_out = process_result.timed_out
                if not timed_out and last_exit == 0 and call.output_path.is_file():
                    try:
                        payload = json.loads(call.output_path.read_text(encoding="utf-8"))
                        response = response_model.model_validate(payload)
                    except (json.JSONDecodeError, ValueError) as exc:
                        warning = f"Codex returned invalid structured output: {exc}"
                    else:
                        break
                else:
                    warning = (
                        "Codex timed out" if timed_out else f"Codex exited with status {last_exit}"
                    )

        fallback_used = response is None
        if response is None:
            if not allow_fallback:
                reason = warning or "Codex CLI is unavailable"
                raise RuntimeError(reason)
            response = deterministic_fallback(task_type, sanitized_input)
            warning = warning or "Codex CLI is unavailable"

        duration = time.monotonic() - started
        result_hash = sha256_file(call.output_path) if call.output_path.is_file() else None
        audit = CodexAudit(
            call_id=call_id,
            task_type=task_type,
            timestamp=datetime.now(UTC),
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            schema_sha256=sha256_file(call.schema_path),
            input_sha256=sha256_file(call.input_path),
            result_sha256=result_hash,
            exit_code=last_exit,
            duration_seconds=duration,
            version=version,
            timed_out=timed_out,
            fallback_used=fallback_used,
            attempts=attempts,
            applied_changes={},
        )
        return CodexGatewayResult(response=response, audit=audit, warning=warning)
