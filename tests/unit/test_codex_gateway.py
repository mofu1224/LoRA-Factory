from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lora_factory.codex.allowlist import validate_recovery_changes
from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.gateway import CodexGateway
from lora_factory.codex.process import (
    CodexProcessResult,
    build_codex_arguments,
    run_codex_process,
)
from lora_factory.codex.prompts import prompt_for
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


def test_codex_prompt_references_large_sanitized_input_without_copying_it() -> None:
    marker = "bounded-sanitized-value-" * 2_000

    prompt = prompt_for(
        CodexTaskType.CAPTION_REVIEW,
        "large-input.json",
        {"captions": marker},
    )

    assert "input/large-input.json" in prompt
    assert marker not in prompt
    assert len(prompt) < 2_000


def test_codex_process_passes_only_required_non_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ScratchRepository(tmp_path / "runtime").prepare_call(
        call_id="environment-boundary",
        task_type=CodexTaskType.RECOVERY,
        sanitized_input={"classification": "CUDA_OOM"},
        response_model=RecoveryReview,
    )
    source_environment = {
        "Path": r"C:\Tools",
        "PATHEXT": ".COM;.EXE",
        "SystemRoot": r"C:\Windows",
        "TEMP": r"C:\Temp",
        "USERPROFILE": r"C:\Users\tester",
        "APPDATA": r"C:\Users\tester\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\tester\AppData\Local",
        "CODEX_HOME": r"C:\Users\tester\.codex",
        "CODEX_SQLITE_HOME": r"C:\Users\tester\.codex\sqlite",
        "CODEX_CA_CERTIFICATE": r"C:\certs\ca.pem",
        "SSL_CERT_FILE": r"C:\certs\system.pem",
        "LANG": "ja_JP.UTF-8",
        "CODEX_API_KEY": "test-codex-key",
        "CODEX_ACCESS_TOKEN": "test-codex-token",
        "GITHUB_TOKEN": "test-github-token",
        "HF_TOKEN": "test-hugging-face-token",
        "DATABASE_PASSWORD": "test-database-password",
        "CUSTOM_PROVIDER_SECRET": "test-provider-secret",
        "UNRELATED_SETTING": "not required by Codex",
    }
    captured_environment: dict[str, str] = {}

    class CompletedProcess:
        returncode = 0

        @staticmethod
        def communicate(timeout: int | None = None) -> tuple[bytes, bytes]:
            del timeout
            return b"", b""

    def fake_popen(arguments: list[str], **kwargs: Any) -> CompletedProcess:
        del arguments
        captured_environment.update(kwargs["env"])
        return CompletedProcess()

    monkeypatch.setattr("lora_factory.codex.process.os.environ", source_environment)
    monkeypatch.setattr("lora_factory.codex.process.subprocess.Popen", fake_popen)

    run_codex_process(
        ["codex", "exec"],
        call=call,
        timeout_seconds=10,
        home_for_redaction=tmp_path,
    )

    assert captured_environment == {
        key: source_environment[key]
        for key in (
            "Path",
            "PATHEXT",
            "SystemRoot",
            "TEMP",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "CODEX_HOME",
            "CODEX_SQLITE_HOME",
            "CODEX_CA_CERTIFICATE",
            "SSL_CERT_FILE",
            "LANG",
        )
    }


def test_codex_version_probe_uses_the_same_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_environment = {
        "Path": r"C:\Tools",
        "SystemRoot": r"C:\Windows",
        "USERPROFILE": r"C:\Users\tester",
        "CODEX_HOME": r"C:\Users\tester\.codex",
        "LANG": "ja_JP.UTF-8",
        "CODEX_API_KEY": "test-codex-key",
        "GITHUB_TOKEN": "test-github-token",
        "DATABASE_PASSWORD": "test-database-password",
    }
    captured_environment: dict[str, str] = {}

    class CompletedVersionProbe:
        returncode = 0
        stdout = "codex-cli 1.0\n"

    def fake_run(arguments: list[str], **kwargs: Any) -> CompletedVersionProbe:
        assert arguments == ["codex", "--version"]
        captured_environment.update(kwargs["env"])
        return CompletedVersionProbe()

    monkeypatch.setattr("lora_factory.codex.gateway.os.environ", source_environment)
    monkeypatch.setattr("lora_factory.codex.gateway.shutil.which", lambda executable: executable)
    monkeypatch.setattr("lora_factory.codex.gateway.subprocess.run", fake_run)

    assert CodexGateway(tmp_path / "runtime").version() == "codex-cli 1.0"
    assert captured_environment == {
        "Path": source_environment["Path"],
        "SystemRoot": source_environment["SystemRoot"],
        "USERPROFILE": source_environment["USERPROFILE"],
        "CODEX_HOME": source_environment["CODEX_HOME"],
        "LANG": source_environment["LANG"],
    }


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
