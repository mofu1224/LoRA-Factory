"""Bounded Codex CLI process execution with audit-friendly files."""

from __future__ import annotations

import json
import locale
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lora_factory.codex.environment import build_codex_environment
from lora_factory.codex.runtime import CodexRuntimeProfile
from lora_factory.codex.scratch_repo import ScratchCall
from lora_factory.core.cancellation import CancellationToken
from lora_factory.util.process_tree import ProcessTree, process_tree_popen_kwargs
from lora_factory.util.redaction import redact_text

_RUNTIME_CODEX_DISABLED_FEATURES: tuple[str, ...] = (
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "shell_tool",
)

_RUNTIME_CODEX_DISABLED_MCP_SERVERS: tuple[str, ...] = (
    "blender",
    "node_repl",
    "unityMCP",
)


@dataclass(frozen=True, slots=True)
class CodexProcessResult:
    arguments: tuple[str, ...]
    return_code: int
    timed_out: bool
    stdout: str
    stderr: str
    cancelled: bool = False
    completed: bool = False
    completion_observed: bool = False
    startup_timed_out: bool = False
    idle_timed_out: bool = False


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


def _profile_supports(profile: CodexRuntimeProfile | None, option: str) -> bool:
    return profile is None or profile.supports(option)


def build_codex_arguments(
    *,
    executable: str,
    call: ScratchCall,
    prompt: str,
    profile: CodexRuntimeProfile | None = None,
) -> list[str]:
    """Build an argv-only, read-only structured invocation.

    'profile' is optional for compatibility with callers that construct
    arguments directly.  The default path uses the conservative option set
    supported by the current CLI versions.
    """

    executable = profile.executable if profile is not None else executable
    image_arguments = [item for path in call.image_paths for item in ("--image", str(path))]
    configured_servers = set(_RUNTIME_CODEX_DISABLED_MCP_SERVERS)
    if profile is not None:
        configured_servers.update(profile.configured_mcp_servers)
    arguments = [executable, "exec", *image_arguments]
    if _profile_supports(profile, "--disable"):
        arguments.extend(
            option
            for feature in _RUNTIME_CODEX_DISABLED_FEATURES
            for option in ("--disable", feature)
        )
    if profile is not None and profile.ignore_user_config:
        if _profile_supports(profile, "--ignore-user-config"):
            arguments.append("--ignore-user-config")
    elif _profile_supports(profile, "--config"):
        arguments.extend(
            option
            for server in sorted(configured_servers, key=str.casefold)
            for option in ("--config", f"mcp_servers.{server}.enabled=false")
        )
    if _profile_supports(profile, "--ephemeral"):
        arguments.append("--ephemeral")
    if _profile_supports(profile, "--sandbox"):
        arguments.extend(("--sandbox", "read-only"))
    if _profile_supports(profile, "--json"):
        arguments.append("--json")
    if _profile_supports(profile, "--output-schema"):
        arguments.extend(("--output-schema", str(call.schema_path)))
    if _profile_supports(profile, "--output-last-message"):
        arguments.extend(("--output-last-message", str(call.output_path)))
    if _profile_supports(profile, "--cd"):
        arguments.extend(("--cd", str(call.root)))
    arguments.append(prompt)
    return arguments


def _has_async_process_api(process: Any) -> bool:
    return all(
        callable(getattr(process, attribute, None)) for attribute in ("poll", "wait")
    ) and all(
        callable(getattr(getattr(process, stream, None), "read", None))
        for stream in ("stdout", "stderr")
    )


def _process_return_code(process: Any) -> int:
    value = getattr(process, "returncode", None)
    return int(value) if isinstance(value, int) else -1


def _structured_output_ready(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


def _completion_event_seen(raw: bytes) -> bool:
    text = _decode_process_output(raw)
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        marker = event.get("type") or event.get("event") or event.get("status")
        if isinstance(marker, str) and marker.casefold() in {
            "turn.completed",
            "turn.failed",
            "error",
            "completed",
            "failed",
        }:
            return True
        if event.get("completed") is True:
            return True
    return False


def _wait_for_process(process: Any, timeout_seconds: float) -> bool:
    """Wait for a process without ever using an unbounded wait."""

    timeout_seconds = max(0.0, timeout_seconds)
    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            wait(timeout=timeout_seconds)
            return True
        except (subprocess.TimeoutExpired, TimeoutError):
            return False
        except (OSError, TypeError):
            pass
    poll = getattr(process, "poll", None)
    if not callable(poll):
        return False
    deadline = time.monotonic() + timeout_seconds
    while True:
        if poll() is not None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _read_stream(
    stream: Any,
    output: bytearray,
    last_activity: list[float],
    activity_seen: list[bool],
    activity_lock: threading.Lock,
) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            if not isinstance(chunk, bytes):
                return
            with activity_lock:
                output.extend(chunk)
                last_activity[0] = time.monotonic()
                activity_seen[0] = True
    except (OSError, ValueError):
        return


def _run_communicate_compat(
    process: Any,
    process_tree: ProcessTree,
    *,
    deadline: float,
    cancellation: CancellationToken | None,
) -> tuple[bytes, bytes, bool, bool, bool, bool]:
    """Keep compatibility with small process doubles used by integrations."""

    raw_stdout = b""
    raw_stderr = b""
    timed_out = False
    cancelled = False
    completed = False
    startup_timed_out = False
    while True:
        if cancellation is not None and cancellation.cancelled:
            cancelled = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            startup_timed_out = True
            break
        try:
            raw_stdout, raw_stderr = process.communicate(timeout=min(0.1, remaining))
        except (subprocess.TimeoutExpired, TimeoutError):
            continue
        if cancellation is not None and cancellation.cancelled:
            cancelled = True
        completed = True
        break

    if timed_out or (cancelled and not completed):
        process_tree.terminate()
        try:
            raw_stdout, raw_stderr = process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, TimeoutError):
            process_tree.kill()
            _wait_for_process(process, 5)
            try:
                raw_stdout, raw_stderr = process.communicate(timeout=1)
            except (OSError, subprocess.SubprocessError, TimeoutError):
                raw_stdout, raw_stderr = b"", b""
    return (
        raw_stdout or b"",
        raw_stderr or b"",
        timed_out,
        cancelled,
        completed,
        startup_timed_out,
    )


def run_codex_process(
    arguments: list[str],
    *,
    call: ScratchCall,
    timeout_seconds: float,
    home_for_redaction: Path | None = None,
    cancellation: CancellationToken | None = None,
    startup_timeout_seconds: float | None = None,
    idle_timeout_seconds: float | None = None,
    completion_grace_seconds: float = 0.75,
) -> CodexProcessResult:
    """Run Codex with total, startup, and idle budgets.

    Real subprocess pipes are drained continuously so a verbose CLI cannot
    deadlock on a full stderr/stdout pipe.  A completed structured output is
    accepted during a short grace window even if the CLI wrapper itself keeps
    a child process alive.
    """

    environment = build_codex_environment(os.environ)
    process = subprocess.Popen(  # noqa: S603 - argv is explicit and shell=False.
        arguments,
        cwd=call.root,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        **process_tree_popen_kwargs(),
    )
    process_tree = ProcessTree(process)
    process_started = time.monotonic()
    total_deadline = process_started + max(0.01, float(timeout_seconds))
    raw_stdout = b""
    raw_stderr = b""
    timed_out = False
    cancelled = False
    completed = False
    completion_observed = False
    startup_timed_out = False
    idle_timed_out = False

    try:
        if _has_async_process_api(process):
            stdout_buffer = bytearray()
            stderr_buffer = bytearray()
            last_activity = [process_started]
            activity_seen = [False]
            activity_lock = threading.Lock()
            readers = [
                threading.Thread(
                    target=_read_stream,
                    args=(
                        getattr(process, stream),
                        output,
                        last_activity,
                        activity_seen,
                        activity_lock,
                    ),
                    name=f"codex-{stream}-reader",
                    daemon=True,
                )
                for stream, output in (("stdout", stdout_buffer), ("stderr", stderr_buffer))
            ]
            for reader in readers:
                reader.start()
            output_ready_at: float | None = None
            while True:
                now = time.monotonic()
                process_return_code = process.poll()
                output_ready = _structured_output_ready(call.output_path)
                with activity_lock:
                    stdout_snapshot = bytes(stdout_buffer)
                if output_ready:
                    if output_ready_at is None:
                        output_ready_at = now
                    if _completion_event_seen(stdout_snapshot) or (
                        now - output_ready_at >= max(0.0, completion_grace_seconds)
                    ):
                        completion_observed = True
                        completed = True
                        break
                else:
                    output_ready_at = None
                if process_return_code is not None:
                    completed = True
                    completion_observed = output_ready
                    break
                if cancellation is not None and cancellation.cancelled:
                    cancelled = True
                    break
                if now >= total_deadline:
                    timed_out = True
                    with activity_lock:
                        had_activity = activity_seen[0]
                    startup_timed_out = not had_activity
                    idle_timed_out = had_activity
                    break
                with activity_lock:
                    activity_age = now - last_activity[0]
                    had_activity = activity_seen[0]
                if (
                    startup_timeout_seconds is not None
                    and not had_activity
                    and now - process_started >= max(0.01, float(startup_timeout_seconds))
                ):
                    timed_out = True
                    startup_timed_out = True
                    break
                if (
                    idle_timeout_seconds is not None
                    and had_activity
                    and activity_age >= max(0.01, float(idle_timeout_seconds))
                ):
                    timed_out = True
                    idle_timed_out = True
                    break
                time.sleep(0.05)
            with activity_lock:
                raw_stdout = bytes(stdout_buffer)
                raw_stderr = bytes(stderr_buffer)
            if timed_out or cancelled or completion_observed:
                process_tree.terminate()
                if not _wait_for_process(process, 1.0):
                    process_tree.kill()
                    _wait_for_process(process, 1.0)
            for reader in readers:
                reader.join(timeout=1.0)
            with activity_lock:
                raw_stdout = bytes(stdout_buffer)
                raw_stderr = bytes(stderr_buffer)
        else:
            (
                raw_stdout,
                raw_stderr,
                timed_out,
                cancelled,
                completed,
                startup_timed_out,
            ) = _run_communicate_compat(
                process,
                process_tree,
                deadline=total_deadline,
                cancellation=cancellation,
            )
    finally:
        process_tree.close()

    stdout = _decode_process_output(raw_stdout or b"")
    stderr = _decode_process_output(raw_stderr or b"")
    safe_stdout = redact_text(stdout, home=home_for_redaction)
    safe_stderr = redact_text(stderr, home=home_for_redaction)
    call.events_path.write_text(safe_stdout, encoding="utf-8")
    call.stderr_path.write_text(safe_stderr, encoding="utf-8")
    return CodexProcessResult(
        arguments=tuple(arguments),
        return_code=_process_return_code(process),
        timed_out=timed_out,
        stdout=safe_stdout,
        stderr=safe_stderr,
        cancelled=cancelled,
        completed=completed,
        completion_observed=completion_observed,
        startup_timed_out=startup_timed_out,
        idle_timed_out=idle_timed_out,
    )
