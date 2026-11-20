"""Tests for selecting the safest persisted run to show and resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lora_factory.application.service import _preferred_run
from lora_factory.core.stage import PipelineStage, RunStatus, StageStatus
from lora_factory.storage.database import Database
from lora_factory.storage.orm import ProjectRow, RunRow, StageRow


def test_preferred_run_uses_most_advanced_resumable_run(tmp_path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    older = datetime.now(UTC) - timedelta(minutes=2)
    with database.session() as session:
        session.add(
            ProjectRow(
                id="project",
                name="Project",
                root_path=str(tmp_path),
                config_json={},
            )
        )
        session.add_all(
            [
                RunRow(
                    id="advanced",
                    project_id="project",
                    status=RunStatus.CANCELLED.value,
                    current_stage=PipelineStage.PLANNING.value,
                    created_at=older,
                    updated_at=older,
                    snapshot_json={},
                ),
                RunRow(
                    id="newer",
                    project_id="project",
                    status=RunStatus.FAILED_RECOVERABLE.value,
                    current_stage=PipelineStage.CODEX_REFINEMENT.value,
                    created_at=older + timedelta(minutes=1),
                    updated_at=older + timedelta(minutes=1),
                    snapshot_json={},
                ),
            ]
        )
        session.add_all(
            [
                StageRow(
                    run_id="advanced",
                    stage=PipelineStage.PLANNING.value,
                    attempt=1,
                    status=StageStatus.SUCCEEDED.value,
                    fingerprint="a",
                    stage_version="1",
                    output_json={},
                ),
                StageRow(
                    run_id="newer",
                    stage=PipelineStage.CODEX_REFINEMENT.value,
                    attempt=1,
                    status=StageStatus.FAILED.value,
                    fingerprint="b",
                    stage_version="2",
                    output_json={},
                ),
            ]
        )
        selected = _preferred_run(session, project_id="project")

    assert selected is not None
    assert selected.id == "advanced"
