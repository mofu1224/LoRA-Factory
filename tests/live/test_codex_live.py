from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from lora_factory.codex.gateway import CodexGateway
from lora_factory.codex.image_attachment import (
    PreparedCodexImage,
    prepare_codex_image,
    remove_codex_images,
)
from lora_factory.codex.schemas import CodexTaskType, DatasetRefinementResponse, DatasetReview

_LIVE_VISUAL_VOCABULARY = ("blue_background", "simple_background", "no_humans")


def _make_non_personal_fixture(path: Path) -> Path:
    """Create an abstract local image containing no person or private data."""

    Image.new("RGB", (96, 96), color=(30, 90, 210)).save(path, format="PNG")
    return path


def _live_refinement_payload(prepared: PreparedCodexImage) -> dict[str, object]:
    vocabulary_digest = sha256("\n".join(_LIVE_VISUAL_VOCABULARY).encode()).hexdigest()
    effective_tags = list(_LIVE_VISUAL_VOCABULARY)
    return {
        "schema_version": 1,
        "preset": "style",
        "ontology_version": 1,
        "batch_index": 1,
        "batch_count": 1,
        "generate_trigger_candidates": False,
        "vocabulary_sha256": vocabulary_digest,
        "allowed_visual_tags": effective_tags,
        "assets": [
            {
                "asset_id": prepared.asset_id,
                "original_tags": effective_tags,
                "tag_confidences": dict.fromkeys(effective_tags, 1.0),
                "fixed_tokens": [],
                "effective_tags": effective_tags,
                "draft_caption": ", ".join(effective_tags),
            }
        ],
        "images": [
            {
                "asset_id": prepared.asset_id,
                "relative_name": prepared.relative_name,
                "working_sha256": prepared.source_sha256,
                "image_sha256": prepared.output_sha256,
                "width": prepared.width,
                "height": prepared.height,
                "quality": prepared.quality,
                "byte_count": prepared.byte_count,
                "profile": prepared.profile.model_dump(mode="json"),
            }
        ],
    }


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


@pytest.mark.live_codex
def test_live_codex_dataset_refinement_reads_sanitized_image(tmp_path: Path) -> None:
    if os.environ.get("LORA_FACTORY_LIVE_CODEX") != "1":
        pytest.skip("Set LORA_FACTORY_LIVE_CODEX=1 to call the authenticated Codex CLI")

    prepared = prepare_codex_image(
        "asset-a",
        _make_non_personal_fixture(tmp_path / "fixture.png"),
        tmp_path / "runtime" / "input" / "images" / "live",
    )
    try:
        gateway = CodexGateway(
            tmp_path / "runtime",
            timeout_seconds=180,
            max_attempts=1,
        )
        result = gateway.review(
            CodexTaskType.DATASET_REFINEMENT,
            _live_refinement_payload(prepared),
            images=(prepared,),
            allow_fallback=False,
        )

        assert isinstance(result.response, DatasetRefinementResponse)
        assert result.audit.image_count == 1
        assert result.audit.image_input_sha256s == (prepared.output_sha256,)
        assert [asset.asset_id for asset in result.response.assets] == ["asset-a"]
        assert set(result.response.assets[0].effective_tags) <= set(_LIVE_VISUAL_VOCABULARY)
    finally:
        remove_codex_images(
            (prepared,),
            scratch_root=tmp_path / "runtime" / "input" / "images" / "live",
        )
