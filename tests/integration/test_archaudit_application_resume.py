from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from lora_factory.application.service import LoRAFactoryController
from lora_factory.config.models import AppSettings, BackendMode, PresetKind, ProjectConfig
from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.core.stage import PipelineStage, RunStatus, StageStatus
from lora_factory.storage.database import Database
from lora_factory.storage.orm import (
    GpuDeviceRow,
    GpuLeaseRow,
    RunRow,
    StageRow,
    TrainingAttemptRow,
)
from lora_factory.testing.fixture_factory import generate_fake_fixture
from lora_factory.training.backend import (
    ProgressCallback,
    TrainingRequest,
    TrainingResult,
)
from lora_factory.training.fake_backend import FakeTrainingBackend
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json

GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"


class SimulatedApplicationCrash(BaseException):
    """Bypass normal exception cleanup like abrupt process termination would."""


class CrashAfterCheckpointBackend:
    def __init__(self) -> None:
        self.delegate = FakeTrainingBackend()

    @property
    def version(self) -> str:
        return self.delegate.version

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult:
        def crash_after_durable_state(value: Any) -> None:
            progress(value)
            if value.checkpoint is not None:
                raise SimulatedApplicationCrash("simulated hard application crash")

        return self.delegate.train(request, cancellation, crash_after_durable_state)


class RecordingFakeBackend:
    def __init__(self) -> None:
        self.delegate = FakeTrainingBackend()
        self.requests: list[TrainingRequest] = []

    @property
    def version(self) -> str:
        return self.delegate.version

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult:
        self.requests.append(request)
        return self.delegate.train(request, cancellation, progress)


def test_archaudit_application_cancel_resumes_optimizer_state_without_overwrite(
    tmp_path: Path,
) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    config = ProjectConfig(
        project_id="resume-project",
        lora_name="resume_project",
        preset=PresetKind.CHARACTER,
        trigger_token="resume_token",  # noqa: S106 - domain trigger, not a credential
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )
    cancelled = False

    def cancel_after_progress(event: dict[str, object]) -> None:
        nonlocal cancelled
        details = event.get("details")
        checkpoint = details.get("checkpoint") if isinstance(details, dict) else None
        if event["event_type"] == "training_progress" and checkpoint and not cancelled:
            cancelled = True
            controller.cancel_current()

    with pytest.raises(CancelledError):
        controller.run_pipeline(config, cancel_after_progress)

    project = settings.projects_root / config.project_id
    first_attempt = next((project / "runs").glob("*/attempt-001"))
    state = first_attempt / "states" / "training-state.json"
    assert state.is_file()
    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in (first_attempt / "checkpoints").glob("*.safetensors")
    }
    assert checkpoint_hashes

    controller.projects.save_config(
        config.model_copy(update={"trigger_token": "changed_after_snapshot"})
    )

    result = controller.resume_project(config.project_id, lambda _event: None)

    assert result["status"] == "READY"
    assert result["trigger_token"] == config.trigger_token
    assert all(
        sha256_file(first_attempt / "checkpoints" / name) == digest
        for name, digest in checkpoint_hashes.items()
    )
    database = Database(project / "state.sqlite3")
    with database.session() as session:
        rows = session.scalars(
            select(StageRow)
            .where(StageRow.stage == PipelineStage.TRAINING.value)
            .order_by(StageRow.attempt)
        ).all()
        attempts = session.scalars(
            select(TrainingAttemptRow).order_by(TrainingAttemptRow.attempt)
        ).all()
    assert [row.status for row in rows] == [
        StageStatus.CANCELLED.value,
        StageStatus.SUCCEEDED.value,
    ]
    assert rows[-1].output_json["attempt"] == 2
    assert rows[-1].output_json["resumed_from_attempt"] == 1
    assert rows[-1].output_json["resumed"] is True
    assert rows[-1].output_json["command_argv"] == []
    assert [(item.attempt, item.status) for item in attempts] == [
        (1, "cancelled"),
        (2, "completed"),
    ]


def test_application_crash_recovers_same_run_and_optimizer_state_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    crashing_backend = CrashAfterCheckpointBackend()
    controller._training_backend = (  # type: ignore[method-assign]
        lambda _context, _plan: crashing_backend
    )
    config = ProjectConfig(
        project_id="crash-resume-project",
        lora_name="crash_resume_project",
        preset=PresetKind.CHARACTER,
        trigger_token="crash_resume_token",  # noqa: S106 - domain trigger, not a credential
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )

    with pytest.raises(SimulatedApplicationCrash, match="hard application crash"):
        controller.run_pipeline(config, lambda _event: None)

    project = settings.projects_root / config.project_id
    database = Database(project / "state.sqlite3")
    first_attempt = next((project / "runs").glob("*/attempt-001"))
    run_id = first_attempt.parent.name
    state = first_attempt / "states" / "training-state.json"
    state_payload = read_json(state)
    assert isinstance(state_payload, dict)
    assert state_payload["optimizer_state_preserved"] is True
    assert int(state_payload["completed_step"]) > 0
    first_attempt_hashes = {
        str(path.relative_to(first_attempt)): sha256_file(path)
        for path in first_attempt.rglob("*")
        if path.is_file()
    }
    assert first_attempt_hashes

    # The in-process harness can still run outer exception cleanup. Restore the ACTIVE
    # run value a hard process death would have left, while preserving the genuinely
    # RUNNING stage and attempt rows created before the backend call.
    with database.session() as session:
        run = session.get(RunRow, run_id)
        assert run is not None and run.status == RunStatus.FAILED_FATAL.value
        training_stage = session.scalar(
            select(StageRow).where(
                StageRow.run_id == run_id,
                StageRow.stage == PipelineStage.TRAINING.value,
            )
        )
        attempt = session.scalar(
            select(TrainingAttemptRow).where(
                TrainingAttemptRow.run_id == run_id,
                TrainingAttemptRow.attempt == 1,
            )
        )
        assert training_stage is not None
        assert training_stage.status == StageStatus.RUNNING.value
        assert attempt is not None and attempt.status == "running"
        run.status = RunStatus.ACTIVE.value
        run.current_stage = PipelineStage.TRAINING.value
        run.error_json = None
        session.add(
            GpuDeviceRow(
                uuid=GPU_UUID,
                last_index=0,
                name="Crashed selected GPU",
                capability="sm_120",
                total_vram_mb=24 * 1024,
                compatible=True,
                details_json={},
            )
        )
        session.add(
            GpuLeaseRow(
                gpu_uuid=GPU_UUID,
                task="TRAIN",
                run_id=run_id,
                worker_id="crashed-worker",
                pid=999_999,
                estimated_vram_mb=8_192,
            )
        )

    recovered: list[tuple[int, int, int]] = []
    original_recover = Database.recover_interrupted

    def record_recovery(active_database: Database) -> tuple[int, int, int]:
        result = original_recover(active_database)
        recovered.append(result)
        return result

    monkeypatch.setattr(Database, "recover_interrupted", record_recovery)
    restarted = LoRAFactoryController(settings)
    resumed_backend = RecordingFakeBackend()
    restarted._training_backend = (  # type: ignore[method-assign]
        lambda _context, _plan: resumed_backend
    )

    result = restarted.resume_project(config.project_id, lambda _event: None)

    assert result["status"] == "READY"
    assert result["run_id"] == run_id
    assert recovered == [(1, 1, 1)]
    assert len(resumed_backend.requests) == 1
    resumed_request = resumed_backend.requests[0]
    assert resumed_request.run_id == run_id
    assert resumed_request.attempt == 2
    assert resumed_request.resume_state == state
    assert {
        str(path.relative_to(first_attempt)): sha256_file(path)
        for path in first_attempt.rglob("*")
        if path.is_file()
    } == first_attempt_hashes

    second_attempt = first_attempt.with_name("attempt-002")
    request_record = read_json(second_attempt / "training-request.json")
    assert isinstance(request_record, dict)
    assert request_record["resumed_from_attempt"] == 1
    assert Path(str(request_record["resume_state"])) == state
    with database.session() as session:
        run = session.get(RunRow, run_id)
        stages = session.scalars(
            select(StageRow)
            .where(
                StageRow.run_id == run_id,
                StageRow.stage == PipelineStage.TRAINING.value,
            )
            .order_by(StageRow.attempt)
        ).all()
        attempts = session.scalars(
            select(TrainingAttemptRow)
            .where(TrainingAttemptRow.run_id == run_id)
            .order_by(TrainingAttemptRow.attempt)
        ).all()
        leases = session.scalars(select(GpuLeaseRow)).all()
    assert run is not None and run.status == RunStatus.COMPLETED.value
    assert [(stage.attempt, stage.status) for stage in stages] == [
        (1, StageStatus.FAILED.value),
        (2, StageStatus.SUCCEEDED.value),
    ]
    assert stages[0].error_json == {
        "classification": "PROCESS_CRASH",
        "recoverable": True,
    }
    assert stages[1].output_json["resumed"] is True
    assert stages[1].output_json["resumed_from_attempt"] == 1
    assert [(attempt.attempt, attempt.status) for attempt in attempts] == [
        (1, "failed_recoverable"),
        (2, "completed"),
    ]
    assert attempts[0].recovery_json["classification"] == "PROCESS_CRASH"
    assert attempts[1].recovery_json["resumed"] is True
    assert attempts[1].recovery_json["resumed_from_attempt"] == 1
    assert leases == []
