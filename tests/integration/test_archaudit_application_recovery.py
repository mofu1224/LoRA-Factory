from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from lora_factory.application.service import LoRAFactoryController
from lora_factory.codex.image_attachment import PreparedCodexImage
from lora_factory.codex.schemas import CodexTaskType
from lora_factory.config.models import AppSettings, BackendMode, PresetKind, ProjectConfig
from lora_factory.core.cancellation import CancellationToken
from lora_factory.core.exceptions import ErrorClassification, PipelineError
from lora_factory.core.stage import PipelineStage, RunStatus
from lora_factory.storage.database import Database
from lora_factory.storage.orm import CodexCallRow, RunRow, StageRow, TrainingAttemptRow
from lora_factory.testing.fixture_factory import generate_fake_fixture
from lora_factory.training.backend import (
    ProgressCallback,
    TrainingRequest,
    TrainingResult,
)
from lora_factory.training.fake_backend import FakeTrainingBackend

GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"


class OomThenFakeBackend:
    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []

    @property
    def version(self) -> str:
        return "injected-oom-then-fake/1"

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise PipelineError(
                "CUDA out of memory in injected backend",
                classification=ErrorClassification.CUDA_OOM,
                recoverable=True,
            )
        return FakeTrainingBackend().train(request, cancellation, progress)


class AlwaysOomBackend:
    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []

    @property
    def version(self) -> str:
        return "injected-always-oom/1"

    def train(
        self,
        request: TrainingRequest,
        _cancellation: CancellationToken,
        _progress: ProgressCallback,
    ) -> TrainingResult:
        self.requests.append(request)
        raise PipelineError(
            "CUDA out of memory in persistent injected backend",
            classification=ErrorClassification.CUDA_OOM,
            recoverable=True,
        )


class NanThenFakeBackend:
    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []

    @property
    def version(self) -> str:
        return "injected-nan-then-fake/1"

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise PipelineError(
                "NaN detected in loss",
                classification=ErrorClassification.NAN_LOSS,
                recoverable=True,
            )
        return FakeTrainingBackend().train(request, cancellation, progress)


class DiskFullBackend:
    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []

    @property
    def version(self) -> str:
        return "injected-disk-full/1"

    def train(
        self,
        request: TrainingRequest,
        _cancellation: CancellationToken,
        _progress: ProgressCallback,
    ) -> TrainingResult:
        self.requests.append(request)
        raise PipelineError(
            "No space left on device",
            classification=ErrorClassification.DISK_FULL,
            recoverable=True,
        )


def test_archaudit_application_oom_recovery_uses_fresh_attempt_and_advisory_codex(
    tmp_path: Path,
) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    backend = OomThenFakeBackend()
    controller._training_backend = lambda _context, _plan: backend  # type: ignore[method-assign]
    recovery_payloads: list[dict[str, Any]] = []
    original_review = controller._codex_review

    def capture_review(
        context: Any,
        task: CodexTaskType,
        payload: dict[str, Any],
        *,
        images: Sequence[PreparedCodexImage] = (),
        allow_fallback: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        if task is CodexTaskType.RECOVERY:
            recovery_payloads.append(payload)
        return original_review(
            context,
            task,
            payload,
            images=images,
            allow_fallback=allow_fallback,
        )

    controller._codex_review = capture_review  # type: ignore[method-assign]
    config = ProjectConfig(
        project_id="recovery-project",
        lora_name="recovery_project",
        preset=PresetKind.CHARACTER,
        trigger_token="recovery_token",  # noqa: S106 - domain trigger, not a credential
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )

    result = controller.run_pipeline(config, lambda _event: None)

    assert result["status"] == "READY"
    assert [request.attempt for request in backend.requests] == [1, 2]
    assert backend.requests[1].resume_state is None
    assert backend.requests[1].dataset_config != backend.requests[0].dataset_config
    assert backend.requests[1].dataset_config.is_file()

    project = settings.projects_root / config.project_id
    database = Database(project / "state.sqlite3")
    with database.session() as session:
        training_stage = session.scalar(
            select(StageRow).where(StageRow.stage == PipelineStage.TRAINING.value)
        )
        attempts = session.scalars(
            select(TrainingAttemptRow).order_by(TrainingAttemptRow.attempt)
        ).all()
        recovery_codex = session.scalars(
            select(CodexCallRow).where(CodexCallRow.task_type == "recovery")
        ).all()
    assert training_stage is not None
    output = training_stage.output_json
    assert output["attempt"] == 2
    assert len(output["recovery_records"]) == 1
    recovery = output["recovery_records"][0]
    assert recovery["classification"] == ErrorClassification.CUDA_OOM.value
    assert recovery["retry_applied"] is True
    assert recovery["codex"]["advisory_only"] is True
    assert recovery["codex"]["applied_changes"] == {}
    assert recovery["plan_before"] != recovery["plan_after"]
    assert len(recovery_payloads) == 1
    assert recovery_payloads[0]["command_argv"] == []
    assert recovery_payloads[0]["dependency_versions"]["training_backend"] == backend.version
    assert recovery_payloads[0]["resolved_config"]["preset"] == PresetKind.CHARACTER.value
    assert [(item.attempt, item.status) for item in attempts] == [
        (1, "failed_recoverable"),
        (2, "completed"),
    ]
    assert len(recovery_codex) == 1
    assert recovery_codex[0].applied_changes_json == {}


def test_archaudit_exhausted_recovery_is_fatal_and_persists_all_attempts(
    tmp_path: Path,
) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    backend = AlwaysOomBackend()
    controller._training_backend = lambda _context, _plan: backend  # type: ignore[method-assign]
    config = ProjectConfig(
        project_id="exhausted-recovery-project",
        lora_name="exhausted_recovery_project",
        preset=PresetKind.CHARACTER,
        trigger_token="exhausted_recovery",  # noqa: S106 - domain trigger, not a credential
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )

    with pytest.raises(PipelineError) as failure:
        controller.run_pipeline(config, lambda _event: None)

    assert failure.value.classification is ErrorClassification.CUDA_OOM
    assert failure.value.recoverable is False
    assert [request.attempt for request in backend.requests] == [1, 2, 3]
    database = Database(settings.projects_root / config.project_id / "state.sqlite3")
    with database.session() as session:
        run = session.scalar(select(RunRow))
        attempts = session.scalars(
            select(TrainingAttemptRow).order_by(TrainingAttemptRow.attempt)
        ).all()
        recovery_codex = session.scalars(
            select(CodexCallRow)
            .where(CodexCallRow.task_type == "recovery")
            .order_by(CodexCallRow.id)
        ).all()
    assert run is not None
    assert run.status == RunStatus.FAILED_FATAL.value
    assert [(item.attempt, item.status) for item in attempts] == [
        (1, "failed_recoverable"),
        (2, "failed_recoverable"),
        (3, "failed_fatal"),
    ]
    assert len(recovery_codex) == 3


def test_application_nan_recovery_changes_plan_and_completes(tmp_path: Path) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    backend = NanThenFakeBackend()
    controller._training_backend = lambda _context, _plan: backend  # type: ignore[method-assign]
    config = ProjectConfig(
        project_id="nan-recovery-project",
        lora_name="nan_recovery_project",
        preset=PresetKind.CHARACTER,
        trigger_token="nan_recovery_token",  # noqa: S106 - domain trigger
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )

    result = controller.run_pipeline(config, lambda _event: None)

    assert result["status"] == "READY"
    assert [request.attempt for request in backend.requests] == [1, 2]
    assert backend.requests[0].plan.no_half_vae is False
    assert backend.requests[1].plan.no_half_vae is True
    database = Database(settings.projects_root / config.project_id / "state.sqlite3")
    with database.session() as session:
        attempts = session.scalars(
            select(TrainingAttemptRow).order_by(TrainingAttemptRow.attempt)
        ).all()
    assert [(item.attempt, item.status) for item in attempts] == [
        (1, "failed_recoverable"),
        (2, "completed"),
    ]


def test_application_disk_full_can_resume_after_space_is_freed(tmp_path: Path) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    backend = DiskFullBackend()
    controller._training_backend = lambda _context, _plan: backend  # type: ignore[method-assign]
    config = ProjectConfig(
        project_id="disk-full-project",
        lora_name="disk_full_project",
        preset=PresetKind.CHARACTER,
        trigger_token="disk_full_token",  # noqa: S106 - domain trigger
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )

    with pytest.raises(PipelineError) as failure:
        controller.run_pipeline(config, lambda _event: None)

    assert failure.value.classification is ErrorClassification.DISK_FULL
    assert failure.value.recoverable is True
    assert len(backend.requests) == 1
    database = Database(settings.projects_root / config.project_id / "state.sqlite3")
    with database.session() as session:
        run = session.scalar(select(RunRow))
        attempts = session.scalars(select(TrainingAttemptRow)).all()
    assert run is not None and run.status == RunStatus.FAILED_RECOVERABLE.value
    assert [(item.attempt, item.status) for item in attempts] == [(1, "failed_recoverable")]

    controller._training_backend = (  # type: ignore[method-assign]
        lambda _context, _plan: FakeTrainingBackend()
    )
    result = controller.resume_project(config.project_id, lambda _event: None)

    assert result["status"] == "READY"
    with database.session() as session:
        attempts = session.scalars(
            select(TrainingAttemptRow).order_by(TrainingAttemptRow.attempt)
        ).all()
    assert [(item.attempt, item.status) for item in attempts] == [
        (1, "failed_recoverable"),
        (2, "completed"),
    ]
