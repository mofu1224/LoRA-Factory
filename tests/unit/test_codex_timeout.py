from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.gateway import CodexGateway
from lora_factory.codex.process import CodexProcessResult, run_codex_process
from lora_factory.codex.runtime import CodexRuntimeAdapter
from lora_factory.codex.schemas import CodexTaskType, RecoveryReview
from lora_factory.codex.scratch_repo import ScratchRepository


def _call(tmp_path: Path):
    return ScratchRepository(tmp_path / "runtime").prepare_call(
        call_id="timeout-test",
        task_type=CodexTaskType.RECOVERY,
        sanitized_input={"classification": "CUDA_OOM"},
        response_model=RecoveryReview,
    )


class _Stream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = deque(chunks)

    def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.popleft()
        return b""


class _Process:
    pid = 12345
    returncode: int | None = None

    def __init__(self, stdout: list[bytes], stderr: list[bytes]) -> None:
        self.stdout = _Stream(stdout)
        self.stderr = _Stream(stderr)
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise TimeoutError("process is still running")
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_completed_output_is_collected_without_waiting_for_wrapper(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    call = _call(tmp_path)
    call.output_path.write_text("{}", encoding="utf-8")
    process = _Process([b'{"type":"turn.completed"}\\n'], [])
    monkeypatch.setattr(
        "lora_factory.codex.process.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )

    started = time.monotonic()
    result = run_codex_process(
        ["codex", "exec"],
        call=call,
        timeout_seconds=10,
        completion_grace_seconds=0.1,
    )

    assert time.monotonic() - started < 1
    assert result.completion_observed is True
    assert result.timed_out is False
    assert process.terminated is True
    assert process.killed is False


def test_startup_and_idle_timeouts_are_distinguished(tmp_path: Path, monkeypatch: Any) -> None:
    startup_call = _call(tmp_path / "startup")
    idle_call = _call(tmp_path / "idle")
    startup_process = _Process([], [])
    monkeypatch.setattr(
        "lora_factory.codex.process.subprocess.Popen",
        lambda *_args, **_kwargs: startup_process,
    )
    startup_result = run_codex_process(
        ["codex", "exec"],
        call=startup_call,
        timeout_seconds=2,
        startup_timeout_seconds=0.05,
    )
    assert startup_result.timed_out is True
    assert startup_result.startup_timed_out is True
    assert startup_result.idle_timed_out is False

    idle_process = _Process([b"progress"], [])
    monkeypatch.setattr(
        "lora_factory.codex.process.subprocess.Popen",
        lambda *_args, **_kwargs: idle_process,
    )
    idle_result = run_codex_process(
        ["codex", "exec"],
        call=idle_call,
        timeout_seconds=2,
        startup_timeout_seconds=1,
        idle_timeout_seconds=0.05,
    )
    assert idle_result.timed_out is True
    assert idle_result.startup_timed_out is False
    assert idle_result.idle_timed_out is True


def test_gateway_switches_from_router_to_isolated_profile(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """
base_url = "http://127.0.0.1:4202/_codex-router/session/v1"

[mcp_servers.custom]
command = "ignored"
""".strip(),
        encoding="utf-8",
    )
    environment = {"CODEX_HOME": str(codex_home), "PATH": ""}
    runtime = CodexRuntimeAdapter(source_environment=environment)
    gateway = CodexGateway(
        tmp_path / "gateway",
        runtime=runtime,
        max_attempts=2,
        retry_backoff_seconds=0,
    )
    monkeypatch.setattr(gateway, "version", lambda: "codex-cli 1.0")
    calls: list[list[str]] = []
    response = deterministic_fallback(
        CodexTaskType.DATASET_REVIEW,
        {"hard_gate_passed": True},
    ).model_dump(mode="json")

    def fake_process(arguments: list[str], **kwargs: Any) -> CodexProcessResult:
        calls.append(arguments)
        if len(calls) == 2:
            kwargs["call"].output_path.write_text(json.dumps(response), encoding="utf-8")
            return CodexProcessResult(tuple(arguments), 0, False, "", "", completed=True)
        return CodexProcessResult(tuple(arguments), -1, False, "", "")

    monkeypatch.setattr("lora_factory.codex.gateway.run_codex_process", fake_process)
    result = gateway.review(
        CodexTaskType.DATASET_REVIEW,
        {"hard_gate_passed": True},
        allow_fallback=False,
    )

    assert result.audit.fallback_used is False
    assert len(calls) == 2
    assert "--ignore-user-config" not in calls[0]
    assert "--ignore-user-config" in calls[1]
    assert result.audit.runtime_profile == "direct-isolated"


def test_gateway_retries_with_one_total_time_budget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    gateway = CodexGateway(
        tmp_path / "gateway",
        timeout_seconds=1,
        max_attempts=3,
        retry_backoff_seconds=0,
    )
    monkeypatch.setattr(gateway, "version", lambda: "codex-cli 1.0")
    clock = [0.0]
    monkeypatch.setattr("lora_factory.codex.gateway.time.monotonic", lambda: clock[0])
    seen_budgets: list[float] = []

    def fake_timeout(arguments: list[str], **_kwargs: Any) -> CodexProcessResult:
        budget = float(_kwargs["timeout_seconds"])
        seen_budgets.append(budget)
        clock[0] += min(0.6, budget)
        return CodexProcessResult(tuple(arguments), -1, True, "", "timed out")

    monkeypatch.setattr("lora_factory.codex.gateway.run_codex_process", fake_timeout)
    result = gateway.review(
        CodexTaskType.DATASET_REVIEW,
        {"hard_gate_passed": True},
        allow_fallback=True,
    )

    assert len(seen_budgets) == 2
    assert seen_budgets[0] <= 1
    assert seen_budgets[1] < seen_budgets[0]
    assert result.audit.attempts == 2
    assert result.audit.timed_out is True
