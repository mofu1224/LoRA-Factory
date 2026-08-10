from __future__ import annotations

import os
from pathlib import Path

import pytest

from lora_factory.codex.gateway import CodexGateway
from lora_factory.codex.schemas import CodexTaskType, DatasetReview


@pytest.mark.live_codex
def test_live_codex_returns_strict_audited_dataset_review(tmp_path: Path) -> None:
    if os.environ.get("LORA_FACTORY_LIVE_CODEX") != "1":
        pytest.skip("Set LORA_FACTORY_LIVE_CODEX=1 to call the authenticated Codex CLI")

    gateway = CodexGateway(
        tmp_path / "codex-runtime",
        timeout_seconds=180,
        max_attempts=1,
    )
    result = gateway.review(
        CodexTaskType.DATASET_REVIEW,
        {
            "preset": "character",
            "accepted_count": 24,
            "hard_gate_passed": True,
            "warnings": [],
            "privacy": "sanitized synthetic live probe",
        },
        allow_fallback=False,
    )

    assert isinstance(result.response, DatasetReview)
    assert result.audit.version
    assert result.audit.exit_code == 0
    assert result.audit.fallback_used is False
    assert result.audit.result_sha256
    call_root = gateway.scratch.root / "output" / result.audit.call_id
    assert (call_root / "result.json").is_file()
    assert (call_root / "events.jsonl").is_file()
