from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lora_factory.codex.schemas import SCHEMA_MODELS, CodexTaskType, strict_output_schema

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_FILENAMES = {
    CodexTaskType.DATASET_REFINEMENT: "codex_dataset_refinement.schema.json",
    CodexTaskType.DATASET_REVIEW: "codex_dataset_review.schema.json",
    CodexTaskType.CAPTION_REVIEW: "codex_caption_review.schema.json",
    CodexTaskType.TRAINING_PLAN: "codex_training_plan.schema.json",
    CodexTaskType.RECOVERY: "codex_recovery.schema.json",
    CodexTaskType.FINAL_REVIEW: "codex_final_review.schema.json",
}


def _assert_strict_objects(value: Any) -> None:
    if isinstance(value, dict):
        assert "default" not in value
        assert "title" not in value
        properties = value.get("properties")
        if value.get("type") == "object" and isinstance(properties, dict):
            assert value.get("additionalProperties") is False
            assert value.get("required") == list(properties)
        for child in value.values():
            _assert_strict_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_strict_objects(child)


def test_static_codex_schemas_exactly_match_runtime_models() -> None:
    schema_root = _ROOT / "schemas"
    assert {path.name for path in schema_root.glob("*.json")} == set(_SCHEMA_FILENAMES.values())
    for task_type, model in SCHEMA_MODELS.items():
        static_schema = json.loads(
            (schema_root / _SCHEMA_FILENAMES[task_type]).read_text(encoding="utf-8")
        )
        runtime_schema = strict_output_schema(model)
        assert static_schema == runtime_schema
        _assert_strict_objects(static_schema)
