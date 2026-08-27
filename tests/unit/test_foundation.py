from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect

from lora_factory.application.artifact_store import ApplicationArtifactStore
from lora_factory.config.models import (
    BackendMode,
    CodexRefinementMode,
    PresetKind,
    ProjectConfig,
    ProjectDraft,
    TriggerWordMode,
)
from lora_factory.config.resolver import resolve_layers
from lora_factory.config.validation import (
    ensure_descendant,
    trigger_token_collision_warning,
    validate_existing_file,
)
from lora_factory.core.cancellation import CancellationToken
from lora_factory.core.context import PipelineContext
from lora_factory.core.events import EventBus
from lora_factory.core.pipeline import PipelineEngine, StageDefinition
from lora_factory.core.stage import PipelineStage, RunStatus, StageStatus
from lora_factory.project.service import ProjectService
from lora_factory.storage.database import Database
from lora_factory.storage.orm import (
    CodexCallRow,
    ProjectRow,
    RunRow,
    StageRow,
    TrainingAttemptRow,
)
from lora_factory.storage.repositories import StageRepository

WINDOWS_RESERVED_DEVICE_NAMES = (
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
    "COM¹",
    "COM²",
    "COM³",
    "LPT¹",
    "LPT²",
    "LPT³",
)


def make_config(tmp_path: Path, **updates: object) -> ProjectConfig:
    payload: dict[str, object] = {
        "project_id": "日本語 Project",
        "lora_name": "雪のLoRA",
        "preset": PresetKind.CHARACTER,
        "trigger_token": "snow_person",
        "base_model": tmp_path / "model.safetensors",
        "input_paths": (tmp_path / "images",),
        "selected_gpu_uuids": ("GPU-12345678-abcd",),
        "output_root": tmp_path / "output",
        "backend_mode": BackendMode.FAKE,
    }
    payload.update(updates)
    return ProjectConfig.model_validate(payload)


def validate_project_name_model(
    tmp_path: Path,
    model_type: type[ProjectDraft] | type[ProjectConfig],
    field_name: str,
    value: str,
) -> None:
    if model_type is ProjectDraft:
        payload = {"project_id": "valid-project", "lora_name": "Valid LoRA"}
        payload[field_name] = value
        ProjectDraft.model_validate(payload)
        return
    make_config(tmp_path, **{field_name: value})


@pytest.mark.parametrize("model_type", [ProjectDraft, ProjectConfig], ids=["draft", "config"])
@pytest.mark.parametrize(
    ("field_name", "error_message"),
    [
        ("project_id", "project_id contains characters invalid in a Windows directory"),
        ("lora_name", "LoRA name contains characters invalid in a Windows filename"),
    ],
)
@pytest.mark.parametrize(
    ("lowercase", "suffix"),
    [(False, ""), (True, ".safetensors")],
    ids=["bare", "case-insensitive-with-extension"],
)
@pytest.mark.parametrize("reserved_name", WINDOWS_RESERVED_DEVICE_NAMES)
def test_project_names_reject_windows_reserved_device_names(
    tmp_path: Path,
    model_type: type[ProjectDraft] | type[ProjectConfig],
    field_name: str,
    error_message: str,
    lowercase: bool,
    suffix: str,
    reserved_name: str,
) -> None:
    value = (reserved_name.lower() if lowercase else reserved_name) + suffix

    with pytest.raises(ValidationError, match=error_message):
        validate_project_name_model(tmp_path, model_type, field_name, value)


@pytest.mark.parametrize("model_type", [ProjectDraft, ProjectConfig], ids=["draft", "config"])
@pytest.mark.parametrize(
    ("field_name", "error_message"),
    [
        ("project_id", "project_id contains characters invalid in a Windows directory"),
        ("lora_name", "LoRA name contains characters invalid in a Windows filename"),
    ],
)
@pytest.mark.parametrize("control_code", range(32), ids=lambda value: f"U+{value:04X}")
def test_project_names_reject_ascii_control_characters(
    tmp_path: Path,
    model_type: type[ProjectDraft] | type[ProjectConfig],
    field_name: str,
    error_message: str,
    control_code: int,
) -> None:
    with pytest.raises(ValidationError, match=error_message):
        validate_project_name_model(
            tmp_path, model_type, field_name, f"safe{chr(control_code)}name"
        )


@pytest.mark.parametrize("token", ["a,b", "a\nb", "a;b", "a  b"])
def test_trigger_token_rejects_ambiguous_caption_values(tmp_path: Path, token: str) -> None:
    with pytest.raises(ValidationError):
        make_config(tmp_path, trigger_token=token)


def test_project_config_defaults_to_user_value_or_codex_trigger_policy(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, trigger_token="")

    assert config.trigger_word_mode is TriggerWordMode.CODEX_SUGGEST
    assert config.codex_refinement_mode is CodexRefinementMode.AUTO


def test_manual_trigger_mode_requires_trigger_word(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Trigger Word is required"):
        make_config(
            tmp_path,
            trigger_token="",
            trigger_word_mode=TriggerWordMode.MANUAL,
        )


def test_codex_suggest_mode_accepts_pending_trigger_word(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        trigger_token="",
        trigger_word_mode=TriggerWordMode.CODEX_SUGGEST,
        codex_refinement_mode=CodexRefinementMode.REVIEW,
    )

    assert config.trigger_token == ""
    assert config.trigger_word_mode is TriggerWordMode.CODEX_SUGGEST
    assert config.codex_refinement_mode is CodexRefinementMode.REVIEW


def test_trigger_collision_is_a_validation_error_at_manual_project_input(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="collides with common Danbooru tag"):
        make_config(
            tmp_path,
            trigger_token="1girl",  # noqa: S106 - domain trigger, not a credential
        )

    assert "collides with common Danbooru tag" in (trigger_token_collision_warning("1girl") or "")
    assert trigger_token_collision_warning("lfx_unique_person_7f3a") is None


def test_path_validation_rejects_traversal_and_wrong_suffix(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    checkpoint = root / "model.safetensors"
    checkpoint.write_bytes(b"fixture")

    assert ensure_descendant(checkpoint, root) == checkpoint.resolve()
    assert validate_existing_file(checkpoint, suffixes={".safetensors"}) == checkpoint.resolve()
    with pytest.raises(ValueError, match="escapes"):
        ensure_descendant(tmp_path / "outside.safetensors", root)
    with pytest.raises(ValueError, match="Unsupported"):
        validate_existing_file(checkpoint, suffixes={".ckpt"})


def test_current_public_docs_reject_stale_subset_and_candidate_failure_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    current_docs = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "SECURITY.md",
            "docs/architecture.md",
            "docs/user-guide-ja.md",
        )
    }
    current_text = "\n".join(current_docs.values())

    assert "削除・canonicalization・重複除去・並べ替えだけ" not in current_text
    assert "Codex候補を作れない: Dataset画像refinementはrecoverable failure" not in current_text
    assert "pin済みWD14語彙から追加" in current_docs["docs/architecture.md"]
    assert "無効な追加tagは個別に拒否" in current_docs["docs/architecture.md"]
    assert "3件未満" in current_docs["docs/user-guide-ja.md"]
    assert "AWAITING_REVIEW" in current_docs["docs/user-guide-ja.md"]


def test_config_resolution_preserves_explicit_locks() -> None:
    resolved = resolve_layers(
        [
            ("defaults", {"resolution": 1024, "batch_size": 1}),
            ("dataset", {"resolution": 768, "batch_size": 2}),
            ("codex", {"resolution": 896, "batch_size": 4}),
        ],
        locked_values={"resolution": 1024},
    )

    assert resolved.values == {"resolution": 1024, "batch_size": 4}
    assert resolved.provenance["resolution"] == "user_lock"
    assert resolved.locked_fields == frozenset({"resolution"})


def test_project_service_creates_canonical_unicode_layout(tmp_path: Path) -> None:
    service = ProjectService(tmp_path / "projects")
    config = make_config(tmp_path)

    layout = service.create(config)

    assert layout.raw.is_dir()
    assert layout.working.is_dir()
    assert layout.captions.is_dir()
    assert layout.runs.is_dir()
    assert layout.final.is_dir()
    assert service.load_config(config.project_id) == config
    manifest = service.load_manifest(config.project_id)
    assert manifest.project_id == config.project_id
    assert manifest.raw_assets == []


def test_project_service_creates_name_only_draft_tree_idempotently(tmp_path: Path) -> None:
    service = ProjectService(tmp_path / "Project")
    draft = ProjectDraft(project_id="名前だけ Project", lora_name="名前だけ Project")

    layout, created = service.create_draft(draft)
    same_layout, created_again = service.create_draft(draft)

    assert created is True
    assert created_again is False
    assert same_layout == layout
    assert service.load_draft(draft.project_id) == draft
    assert layout.config.is_file()
    assert layout.manifest.is_file()
    assert layout.input_images.is_dir()
    assert layout.output_model.is_dir()
    assert layout.base_model.is_dir()
    assert layout.raw.is_dir()
    assert layout.import_staging.is_dir()
    assert layout.working.is_dir()
    assert layout.captions.is_dir()
    assert layout.validation.is_dir()
    assert layout.rejected.is_dir()
    assert layout.configs.is_dir()
    assert layout.runs.is_dir()
    assert layout.final.is_dir()
    assert service.load_manifest(draft.project_id).raw_assets == []

    config = make_config(
        tmp_path,
        project_id=draft.project_id,
        lora_name=draft.lora_name,
    )
    service.save_config(config)
    assert service.load_config(draft.project_id) == config


def test_database_contains_contract_tables_and_recovers_running_rows(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    expected = {
        "projects",
        "runs",
        "stages",
        "assets",
        "source_files",
        "duplicate_clusters",
        "tag_results",
        "captions",
        "dataset_splits",
        "training_plans",
        "training_attempts",
        "checkpoints",
        "samples",
        "evaluations",
        "codex_calls",
        "gpu_devices",
        "gpu_leases",
        "events",
        "destinations",
    }
    assert expected <= set(inspect(database.engine).get_table_names())

    with database.session() as session:
        session.add(ProjectRow(id="project", name="Test", root_path=str(tmp_path), config_json={}))
        session.add(RunRow(id="run", project_id="project", status=RunStatus.ACTIVE.value))
        session.add(
            StageRow(
                run_id="run",
                stage=PipelineStage.IMPORTING.value,
                status=StageStatus.RUNNING.value,
                fingerprint="0" * 64,
                stage_version="1",
                attempt=1,
            )
        )
        session.add(
            TrainingAttemptRow(
                run_id="run",
                attempt=1,
                status="running",
                command_json=[],
                recovery_json={"request_fingerprint": "f" * 64},
            )
        )

    stage_count, run_count, lease_count = database.recover_interrupted()
    assert (stage_count, run_count, lease_count) == (1, 1, 0)

    with database.session() as session:
        assert session.get(RunRow, "run").status == RunStatus.FAILED_RECOVERABLE.value  # type: ignore[union-attr]
        stage = session.query(StageRow).one()
        assert stage.status == StageStatus.FAILED.value
        assert stage.error_json == {"classification": "PROCESS_CRASH", "recoverable": True}
        attempt = session.query(TrainingAttemptRow).one()
        assert attempt.status == "failed_recoverable"
        assert attempt.recovery_json == {
            "request_fingerprint": "f" * 64,
            "classification": "PROCESS_CRASH",
            "recoverable": True,
            "retry_applied": False,
        }


def test_pipeline_reuses_matching_successful_stage(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    config = make_config(tmp_path)
    layout = ProjectService(tmp_path / "projects").create(config)
    with database.session() as session:
        session.add(
            ProjectRow(
                id=config.project_id,
                name=config.lora_name,
                root_path=str(layout.root),
                config_json=config.model_dump(mode="json"),
            )
        )
        session.add(RunRow(id="run-1", project_id=config.project_id, status=RunStatus.ACTIVE.value))

    calls: list[str] = []
    events: list[str] = []
    bus = EventBus()
    bus.subscribe("*", lambda event: events.append(event.event_type))
    context = PipelineContext(
        run_id="run-1",
        config=config,
        layout=layout,
        events=bus,
        cancellation=CancellationToken(),
    )
    stage = StageDefinition(
        stage=PipelineStage.IMPORTING,
        version="1",
        resolve_inputs=lambda _context: {"sources": ["a.png"]},
        run=lambda _context: calls.append("run") or {"asset_count": 1},
        backend_versions={"importer": "1"},
    )
    engine = PipelineEngine(StageRepository(database))

    first = engine.execute(context, [stage])
    second = engine.execute(context, [stage])

    assert first.outputs[PipelineStage.IMPORTING.value] == {"asset_count": 1}
    assert second.cache_hits == (PipelineStage.IMPORTING.value,)
    assert calls == ["run"]
    assert events == ["stage_started", "stage_completed", "stage_skipped"]


def test_codex_artifact_store_redacts_absolute_paths_from_audit(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    config = make_config(tmp_path)
    layout = ProjectService(tmp_path / "projects").create(config)
    with database.session() as session:
        session.add(
            ProjectRow(
                id=config.project_id,
                name=config.lora_name,
                root_path=str(layout.root),
                config_json=config.model_dump(mode="json"),
            )
        )
        session.add(RunRow(id="run-1", project_id=config.project_id, status=RunStatus.ACTIVE.value))
    context = PipelineContext(
        run_id="run-1",
        config=config,
        layout=layout,
        events=EventBus(),
        cancellation=CancellationToken(),
    )
    local_path = str((tmp_path / "codex" / "input" / "image.jpg").resolve())

    ApplicationArtifactStore(database).persist(
        context,
        PipelineStage.CODEX_REFINEMENT,
        {
            "audits": [
                {
                    "call_id": "call-1",
                    "task_type": "dataset_refinement",
                    "local_path": local_path,
                    "diagnostic": f"failed while reading {local_path}",
                }
            ]
        },
    )

    with database.session() as session:
        audit = session.get(CodexCallRow, "call-1").audit_json  # type: ignore[union-attr]
    assert local_path not in str(audit)
    assert audit["local_path"] == "<local-path>"
    assert "<local-path>" in audit["diagnostic"]
