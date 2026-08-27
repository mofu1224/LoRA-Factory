"""High-level structured Runtime Codex gateway with fallback and immutable audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lora_factory.codex.environment import build_codex_environment
from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.image_attachment import (
    PreparedCodexImage,
    validate_codex_image_attachments,
)
from lora_factory.codex.process import build_codex_arguments, run_codex_process
from lora_factory.codex.prompts import prompt_for
from lora_factory.codex.runtime import CodexRuntimeAdapter, CodexRuntimeProfile
from lora_factory.codex.schemas import SCHEMA_MODELS, CodexTaskType, StrictModel
from lora_factory.codex.scratch_repo import ScratchRepository
from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.packaging.metadata import sanitize_public_metadata
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
    image_input_sha256s: tuple[str, ...]
    image_count: int
    environment: str = "unknown"
    runtime_profile: str = "default"
    startup_timed_out: bool = False
    idle_timed_out: bool = False
    completion_observed: bool = False


class CodexCallError(RuntimeError):
    """Terminal Runtime Codex failure with its sanitized durable audit."""

    def __init__(self, message: str, *, audit: CodexAudit) -> None:
        super().__init__(message)
        self.audit = audit


class CodexCallCancelled(CancelledError):
    """Cancelled Runtime Codex call with its sanitized durable audit."""

    def __init__(self, *, audit: CodexAudit) -> None:
        super().__init__("Operation cancelled by the user")
        self.audit = audit


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
        startup_timeout_seconds: float | None = None,
        idle_timeout_seconds: float | None = None,
        retry_backoff_seconds: float = 0.0,
        runtime: CodexRuntimeAdapter | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Codex timeout must be positive")
        if not 1 <= max_attempts <= 3:
            raise ValueError("Codex attempts must be between 1 and 3")
        if startup_timeout_seconds is not None and startup_timeout_seconds <= 0:
            raise ValueError("Codex startup timeout must be positive")
        if idle_timeout_seconds is not None and idle_timeout_seconds <= 0:
            raise ValueError("Codex idle timeout must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("Codex retry backoff cannot be negative")
        self.scratch = ScratchRepository(scratch_root)
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.startup_timeout_seconds = startup_timeout_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.retry_backoff_seconds = retry_backoff_seconds
        self.runtime = runtime or CodexRuntimeAdapter(executable=executable)

    def version(self) -> str | None:
        resolved = self.runtime.resolve_executable()
        if resolved is None:
            resolved = shutil.which(self.executable)
        if resolved is None:
            return None
        try:
            completed = subprocess.run(  # noqa: S603 - resolved with shutil.which.
                [resolved, "--version"],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                env=build_codex_environment(os.environ),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    def review(
        self,
        task_type: CodexTaskType,
        sanitized_input: dict[str, Any],
        *,
        images: Sequence[PreparedCodexImage] = (),
        allow_fallback: bool,
        cancellation: CancellationToken | None = None,
    ) -> CodexGatewayResult:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        sanitized_payload = sanitize_public_metadata(sanitized_input)
        if not isinstance(sanitized_payload, dict):
            raise ValueError("Runtime Codex input must remain a mapping after sanitization")
        sanitized_input = sanitized_payload
        image_records = tuple(
            PreparedCodexImage.model_validate(image.model_dump(mode="python")) for image in images
        )
        if task_type is CodexTaskType.DATASET_REFINEMENT and not image_records:
            raise ValueError("Dataset refinement requires at least one image")
        if task_type is CodexTaskType.DATASET_REFINEMENT:
            raw_payload_images = sanitized_input.get("images")
            if not isinstance(raw_payload_images, (list, tuple)) or not all(
                isinstance(item, dict) for item in raw_payload_images
            ):
                raise ValueError("Dataset refinement payload must contain image mappings")
            image_records = validate_codex_image_attachments(
                image_records,
                raw_payload_images,
                scratch_root=self.scratch.root,
            )
        response_model = SCHEMA_MODELS[task_type]
        call_id = uuid.uuid4().hex
        call = self.scratch.prepare_call(
            call_id=call_id,
            task_type=task_type,
            sanitized_input=sanitized_input,
            response_model=response_model,
            image_paths=tuple(image.path for image in image_records),
        )
        image_input_sha256s = tuple(sha256_file(path) for path in call.image_paths)
        prompt = prompt_for(
            task_type,
            call.input_path.name,
            sanitized_input,
            images=image_records,
        )
        started = time.monotonic()
        version = self.version()
        attempts = 0
        last_exit: int | None = None
        timed_out = False
        startup_timed_out = False
        idle_timed_out = False
        completion_observed = False
        warning: str | None = None
        response: StrictModel | None = None
        selected_profile: CodexRuntimeProfile | None = None
        operation_deadline = started + float(self.timeout_seconds)

        def build_audit(*, fallback_used: bool) -> CodexAudit:
            result_hash = sha256_file(call.output_path) if call.output_path.is_file() else None
            return CodexAudit(
                call_id=call_id,
                task_type=task_type,
                timestamp=datetime.now(UTC),
                prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
                schema_sha256=sha256_file(call.schema_path),
                input_sha256=sha256_file(call.input_path),
                result_sha256=result_hash,
                exit_code=last_exit,
                duration_seconds=time.monotonic() - started,
                version=version,
                timed_out=timed_out,
                fallback_used=fallback_used,
                attempts=attempts,
                applied_changes={},
                image_input_sha256s=image_input_sha256s,
                image_count=len(image_input_sha256s),
                environment=(
                    selected_profile.environment.value
                    if selected_profile is not None
                    else "unknown"
                ),
                runtime_profile=(
                    selected_profile.name if selected_profile is not None else "default"
                ),
                startup_timed_out=startup_timed_out,
                idle_timed_out=idle_timed_out,
                completion_observed=completion_observed,
            )

        def wait_for_retry() -> None:
            if self.retry_backoff_seconds <= 0:
                return
            backoff_deadline = min(
                operation_deadline,
                time.monotonic() + self.retry_backoff_seconds,
            )
            while True:
                remaining = backoff_deadline - time.monotonic()
                if remaining <= 0:
                    return
                if cancellation is not None and cancellation.cancelled:
                    raise CodexCallCancelled(audit=build_audit(fallback_used=False))
                time.sleep(min(0.05, remaining))

        if version is not None or self.runtime.probe:
            profiles = self.runtime.profiles(version_hint=version)
            for attempt_number in range(1, self.max_attempts + 1):
                if cancellation is not None and cancellation.cancelled:
                    if attempts:
                        raise CodexCallCancelled(audit=build_audit(fallback_used=False))
                    cancellation.raise_if_cancelled()
                remaining = operation_deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    warning = warning or "Codex timed out"
                    break
                if not profiles:
                    warning = warning or "Codex runtime profile is unavailable"
                    break
                attempts = attempt_number
                selected_profile = profiles[(attempt_number - 1) % len(profiles)]
                if not selected_profile.usable_for_call(has_images=bool(call.image_paths)):
                    warning = (
                        "Codex CLI does not expose the required safe execution options "
                        "for this input"
                    )
                    continue
                with suppress(FileNotFoundError):
                    call.output_path.unlink()
                arguments = build_codex_arguments(
                    executable=self.executable,
                    call=call,
                    prompt=prompt,
                    profile=selected_profile,
                )
                process_kwargs: dict[str, Any] = {
                    "call": call,
                    "timeout_seconds": remaining,
                    "home_for_redaction": Path.home(),
                }
                if cancellation is not None:
                    process_kwargs["cancellation"] = cancellation
                if self.startup_timeout_seconds is not None:
                    process_kwargs["startup_timeout_seconds"] = min(
                        self.startup_timeout_seconds,
                        remaining,
                    )
                if self.idle_timeout_seconds is not None:
                    process_kwargs["idle_timeout_seconds"] = min(
                        self.idle_timeout_seconds,
                        remaining,
                    )
                try:
                    process_result = run_codex_process(arguments, **process_kwargs)
                except (OSError, subprocess.SubprocessError):
                    warning = "Codex process could not be started"
                    if attempt_number < self.max_attempts:
                        wait_for_retry()
                    continue
                last_exit = process_result.return_code
                timed_out = timed_out or process_result.timed_out
                startup_timed_out = startup_timed_out or process_result.startup_timed_out
                idle_timed_out = idle_timed_out or process_result.idle_timed_out
                completion_observed = completion_observed or process_result.completion_observed
                if process_result.cancelled or (
                    cancellation is not None and cancellation.cancelled
                ):
                    raise CodexCallCancelled(audit=build_audit(fallback_used=False))
                output_is_available = call.output_path.is_file()
                completed_successfully = (
                    not process_result.timed_out and last_exit == 0 and output_is_available
                ) or (process_result.completion_observed and output_is_available)
                if completed_successfully:
                    try:
                        payload = json.loads(call.output_path.read_text(encoding="utf-8"))
                        response = response_model.model_validate(payload)
                    except (json.JSONDecodeError, ValueError):
                        warning = "Codex returned invalid structured output"
                    else:
                        break
                else:
                    warning = (
                        "Codex timed out"
                        if process_result.timed_out
                        else f"Codex exited with status {last_exit}"
                    )
                if attempt_number < self.max_attempts:
                    wait_for_retry()

        fallback_used = response is None
        if response is None:
            if not allow_fallback:
                reason = warning or "Codex CLI is unavailable"
                raise CodexCallError(reason, audit=build_audit(fallback_used=False))
            response = deterministic_fallback(task_type, sanitized_input)
            warning = warning or "Codex CLI is unavailable"

        audit = build_audit(fallback_used=fallback_used)
        return CodexGatewayResult(response=response, audit=audit, warning=warning)
