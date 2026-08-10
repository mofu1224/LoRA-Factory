from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lora_factory.codex.allowlist import validate_recovery_changes
from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.gateway import CodexGateway
from lora_factory.codex.process import CodexProcessResult, build_codex_arguments
from lora_factory.codex.schemas import CodexTaskType, DatasetReview, RecoveryReview
from lora_factory.codex.scratch_repo import ScratchRepository


def test_recovery_allowlist_rejects_paths_commands_and_locked_values() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_recovery_changes({"output_root": "elsewhere"})
    with pytest.raises(ValueError, match="forbidden"):
        validate_recovery_changes({"command": "anything"})
    with pytest.raises(ValueError, match="locked"):
        validate_recovery_changes({"batch_size": 1}, locked_fields=frozenset({"batch_size"}))


def test_recovery_allowlist_enforces_types_and_ranges() -> None:
    assert validate_recovery_changes(
        {
            "batch_size": 2,
            "gradient_accumulation": 4,
            "unet_lr": 0.0001,
            "precision": "bf16",
            "cache_latents": True,
        }
    ) == {
        "batch_size": 2,
        "gradient_accumulation": 4,
        "unet_lr": 0.0001,
        "precision": "bf16",
        "cache_latents": True,
    }
    with pytest.raises(ValueError, match="integer"):
        validate_recovery_changes({"batch_size": True})
    with pytest.raises(ValueError, match="safe range"):
        validate_recovery_changes({"network_dim": 1024})


def test_scratch_repo_is_dedicated_git_repo_with_strict_schema(tmp_path: Path) -> None:
    scratch = ScratchRepository(tmp_path / "codex-runtime")
    call = scratch.prepare_call(
        call_id="call-1",
        task_type=CodexTaskType.DATASET_REVIEW,
        sanitized_input={"hard_gate_passed": True, "accepted_count": 24},
        response_model=DatasetReview,
    )

    assert (scratch.root / ".git").is_dir()
    assert "Read-only analysis only" in (scratch.root / "AGENTS.md").read_text(encoding="utf-8")
    assert json.loads(call.input_path.read_text(encoding="utf-8"))["accepted_count"] == 24
    schema = json.loads(call.schema_path.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_codex_arguments_require_read_only_ephemeral_structured_exec(tmp_path: Path) -> None:
    call = ScratchRepository(tmp_path / "runtime").prepare_call(
        call_id="call-2",
        task_type=CodexTaskType.RECOVERY,
        sanitized_input={"classification": "CUDA_OOM"},
        response_model=RecoveryReview,
    )
    arguments = build_codex_arguments(executable="codex", call=call, prompt="Review input")

    assert arguments[:2] == ["codex", "exec"]
    assert "--model" not in arguments
    assert "--config" not in arguments
    assert "--ephemeral" in arguments
    sandbox_index = arguments.index("--sandbox")
    assert arguments[sandbox_index + 1] == "read-only"
    assert "--json" in arguments
    assert arguments[arguments.index("--output-schema") + 1] == str(call.schema_path)
    assert arguments[arguments.index("--output-last-message") + 1] == str(call.output_path)
    assert arguments[arguments.index("--cd") + 1] == str(call.root)


def test_deterministic_fallback_never_changes_training_or_recovery_settings() -> None:
    dataset = deterministic_fallback(
        CodexTaskType.DATASET_REVIEW,
        {"hard_gate_passed": True},
    )
    recovery = deterministic_fallback(
        CodexTaskType.RECOVERY,
        {"classification": "CUDA_OOM", "deterministically_recoverable": True},
    )

    assert isinstance(dataset, DatasetReview)
    assert dataset.approved is True
    assert isinstance(recovery, RecoveryReview)
    assert recovery.proposed_changes == ()


def test_codex_gateway_retries_invalid_json_then_uses_audited_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = CodexGateway(tmp_path / "runtime", max_attempts=2)
    monkeypatch.setattr(gateway, "version", lambda: "codex-cli 1.0")
    calls = 0

    def invalid_json(
        arguments: list[str],
        *,
        call: Any,
        timeout_seconds: int,
        home_for_redaction: Path,
    ) -> CodexProcessResult:
        nonlocal calls
        del timeout_seconds, home_for_redaction
        calls += 1
        call.output_path.write_text("{not-json", encoding="utf-8")
        return CodexProcessResult(tuple(arguments), 0, False, "", "")

    monkeypatch.setattr("lora_factory.codex.gateway.run_codex_process", invalid_json)
    result = gateway.review(
        CodexTaskType.DATASET_REVIEW,
        {"hard_gate_passed": True},
        allow_fallback=True,
    )

    assert calls == 2
    assert result.audit.attempts == 2
    assert result.audit.fallback_used is True
    assert result.audit.result_sha256 is not None
    assert "invalid structured output" in str(result.warning)


def test_codex_gateway_retries_timeout_and_can_require_live_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = CodexGateway(tmp_path / "runtime", max_attempts=2)
    monkeypatch.setattr(gateway, "version", lambda: "codex-cli 1.0")
    calls = 0

    def timeout(
        arguments: list[str],
        *,
        call: Any,
        timeout_seconds: int,
        home_for_redaction: Path,
    ) -> CodexProcessResult:
        nonlocal calls
        del call, timeout_seconds, home_for_redaction
        calls += 1
        return CodexProcessResult(tuple(arguments), -1, True, "", "timed out")

    monkeypatch.setattr("lora_factory.codex.gateway.run_codex_process", timeout)
    result = gateway.review(
        CodexTaskType.DATASET_REVIEW,
        {"hard_gate_passed": True},
        allow_fallback=True,
    )
    assert calls == 2
    assert result.audit.timed_out is True
    assert result.audit.fallback_used is True
    assert result.warning == "Codex timed out"

    with pytest.raises(RuntimeError, match="Codex timed out"):
        gateway.review(
            CodexTaskType.DATASET_REVIEW,
            {"hard_gate_passed": True},
            allow_fallback=False,
        )
