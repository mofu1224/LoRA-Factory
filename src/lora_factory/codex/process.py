"""Timeout-safe Codex CLI process execution with audit-friendly files."""

from __future__ import annotations

import locale
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lora_factory.codex.scratch_repo import ScratchCall
from lora_factory.util.redaction import redact_text


@dataclass(frozen=True, slots=True)
class CodexProcessResult:
    arguments: tuple[str, ...]
    return_code: int
    timed_out: bool
    stdout: str
    stderr: str


def _decode_process_output(raw: bytes) -> str:
    for encoding in (
        "utf-8",
        "utf-8-sig",
        locale.getpreferredencoding(False),
        "cp932",
        "shift_jis",
    ):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def build_codex_arguments(
    *,
    executable: str,
    call: ScratchCall,
    prompt: str,
) -> list[str]:
    return [
        executable,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(call.schema_path),
        "--output-last-message",
        str(call.output_path),
        "--cd",
        str(call.root),
        prompt,
    ]


def run_codex_process(
    arguments: list[str],
    *,
    call: ScratchCall,
    timeout_seconds: int,
    home_for_redaction: Path | None = None,
) -> CodexProcessResult:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        not in {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "AZURE_OPENAI_API_KEY",
        }
    }
    # The argument array is built by ``build_codex_arguments``; no shell parses it.
    process = subprocess.Popen(  # noqa: S603
        arguments,
        cwd=call.root,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    timed_out = False
    try:
        raw_stdout, raw_stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            raw_stdout, raw_stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            raw_stdout, raw_stderr = process.communicate()
    stdout = _decode_process_output(raw_stdout or b"")
    stderr = _decode_process_output(raw_stderr or b"")

    safe_stdout = redact_text(stdout, home=home_for_redaction)
    safe_stderr = redact_text(stderr, home=home_for_redaction)
    call.events_path.write_text(safe_stdout, encoding="utf-8")
    call.stderr_path.write_text(safe_stderr, encoding="utf-8")
    return CodexProcessResult(
        arguments=tuple(arguments),
        return_code=process.returncode,
        timed_out=timed_out,
        stdout=safe_stdout,
        stderr=safe_stderr,
    )
