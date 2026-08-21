from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pytest
import tomlkit
from PIL import Image
from safetensors.numpy import save_file
from sqlalchemy import func, select

import lora_factory.application.service as application_service
from lora_factory.application.service import LoRAFactoryController
from lora_factory.caption.tag_vocabulary import WD14TagVocabulary
from lora_factory.caption.tagger import FAKE_WD14_VOCABULARY
from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.process import CodexProcessResult
from lora_factory.codex.schemas import CodexTaskType
from lora_factory.config.models import (
    AppSettings,
    BackendMode,
    CodexRefinementMode,
    DestinationConfig,
    DestinationKind,
    PresetKind,
    ProjectConfig,
    TriggerWordMode,
)
from lora_factory.core.cancellation import CancelledError
from lora_factory.core.exceptions import PipelineError
from lora_factory.core.stage import PipelineStage, RunStatus, StageStatus
from lora_factory.sampling.prompts import load_benchmark_prompts
from lora_factory.storage.database import Database
from lora_factory.storage.orm import (
    AssetRow,
    CaptionRow,
    CheckpointRow,
    CodexCallRow,
    DatasetSplitRow,
    EvaluationRow,
    RunRow,
    SampleRow,
    StageRow,
    TagResultRow,
    TrainingAttemptRow,
    TrainingPlanRow,
)
from lora_factory.util.hashing import sha256_file


def _fake_sdxl(path: Path) -> None:
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": np.zeros((1, 1), dtype=np.float32),
            "conditioner.embedders.1.model.text_projection": np.ones((1, 1), dtype=np.float32),
        },
        path,
        metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )


def _images(root: Path, *, count: int = 8) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        generator = np.random.default_rng(1000 + index)
        pixels = generator.integers(0, 256, size=(544, 640, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = np.clip(
            pixels[:, :, index % 3].astype(np.int16) + index * 7, 0, 255
        ).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(root / f"画像 {index + 1}.png")


def _stage_output(settings: AppSettings, project_id: str, stage: PipelineStage) -> dict[str, Any]:
    database = Database(settings.projects_root / project_id / "state.sqlite3")
    with database.session() as session:
        row = session.scalar(
            select(StageRow)
            .where(StageRow.stage == stage.value, StageRow.status == StageStatus.SUCCEEDED.value)
            .order_by(StageRow.attempt.desc())
        )
    assert row is not None
    return dict(row.output_json)


def _latest_run(settings: AppSettings, project_id: str) -> RunRow:
    database = Database(settings.projects_root / project_id / "state.sqlite3")
    with database.session() as session:
        row = session.scalar(select(RunRow).order_by(RunRow.created_at.desc()).limit(1))
        assert row is not None
        session.expunge(row)
    return row


def _training_attempt_count(settings: AppSettings, project_id: str) -> int:
    database = Database(settings.projects_root / project_id / "state.sqlite3")
    with database.session() as session:
        return int(session.scalar(select(func.count()).select_from(TrainingAttemptRow)) or 0)


def _refinement_fixture(
    tmp_path: Path,
    *,
    project_id: str,
    refinement_mode: CodexRefinementMode,
    trigger_mode: TriggerWordMode,
    image_count: int = 8,
) -> tuple[LoRAFactoryController, ProjectConfig]:
    source = tmp_path / "refinement-images"
    _images(source, count=image_count)
    base_model = tmp_path / "refinement-base.safetensors"
    _fake_sdxl(base_model)
    controller = LoRAFactoryController(
        AppSettings(
            projects_root=tmp_path / "projects",
            managed_runtime_root=tmp_path / "runtime",
            codex_runtime_root=tmp_path / "codex",
        )
    )
    return controller, ProjectConfig(
        project_id=project_id,
        lora_name=project_id,
        preset=PresetKind.CHARACTER,
        trigger_token="manual_lfx" if trigger_mode is TriggerWordMode.MANUAL else "",
        base_model=base_model,
        input_paths=(source,),
        selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
        codex_refinement_mode=refinement_mode,
        trigger_word_mode=trigger_mode,
    )


def test_fake_refinement_prepares_every_accepted_image_in_batches_and_cleans_up(
    tmp_path: Path,
) -> None:
    source = tmp_path / "images"
    _images(source, count=17)
    base_model = tmp_path / "base.safetensors"
    _fake_sdxl(base_model)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    config = ProjectConfig(
        project_id="all-image-refinement",
        lora_name="all_image_refinement",
        preset=PresetKind.CHARACTER,
        trigger_token="all_image_token",  # noqa: S106 - domain trigger
        base_model=base_model,
        input_paths=(source,),
        selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )

    events: list[dict[str, Any]] = []
    result = controller.run_pipeline(config, events.append)

    assert result["status"] == "READY"
    output = _stage_output(settings, config.project_id, PipelineStage.CODEX_REFINEMENT)
    assert [audit["image_count"] for audit in output["audits"]] == [8, 8, 1]
    assert sum(audit["image_count"] for audit in output["audits"]) == 17
    assert {digest for audit in output["audits"] for digest in audit["image_input_sha256s"]} == set(
        output["image_sha256s"].values()
    )
    assert set(output["working_sha256s"]) == set(output["image_sha256s"])
    assert all(
        sha256_file(
            settings.projects_root / config.project_id / "dataset" / "working" / f"{asset_id}.png"
        )
        == digest
        for asset_id, digest in output["working_sha256s"].items()
    )
    assert {event["event_type"] for event in events} >= {
        "codex_image_progress",
        "codex_batch_progress",
    }
    assert str(tmp_path) not in json.dumps(output, ensure_ascii=False)
    record_root = (
        settings.projects_root
        / config.project_id
        / "runs"
        / str(result["run_id"])
        / "codex-refinement"
    )
    assert all(
        str(tmp_path) not in path.read_text(encoding="utf-8")
        for path in record_root.glob("batch-*.json")
    )
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["payload_byte_count"] <= 128 * 1024
        for path in record_root.glob("batch-*.json")
    )
    assert not list(settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_partial_image_preparation_retries_twice_cleans_up_and_blocks_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="partial-preparation-failure",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    config = config.model_copy(update={"allow_without_codex": True})
    source_hashes = {
        path.name: sha256_file(path) for path in Path(config.input_paths[0]).glob("*.png")
    }
    original_prepare = application_service.prepare_codex_image
    calls = 0

    def fail_on_second_image(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls in {2, 4}:
            raise ValueError("cannot sanitize image")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(application_service, "prepare_codex_image", fail_on_second_image)

    with pytest.raises(PipelineError, match="Runtime Codex image refinement failed") as raised:
        controller.run_pipeline(config, lambda _event: None)

    assert raised.value.recoverable is True
    assert calls == 4
    assert _latest_run(controller.settings, config.project_id).status == (
        RunStatus.FAILED_RECOVERABLE.value
    )
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert source_hashes == {
        path.name: sha256_file(path) for path in Path(config.input_paths[0]).glob("*.png")
    }
    raw_root = controller.settings.projects_root / config.project_id / "dataset" / "raw"
    assert {sha256_file(path) for path in raw_root.iterdir()} == set(source_hashes.values())
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


@pytest.mark.parametrize(
    "gateway_reason",
    ["Codex timed out", "Codex CLI is unavailable"],
)
def test_codex_failure_never_uses_dataset_fallback_or_starts_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway_reason: str,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="codex-hard-stop",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    config = config.model_copy(update={"allow_without_codex": True})
    original_review = controller._codex_review
    leaked_path = str(Path(config.input_paths[0]).resolve())

    def unavailable(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        if task is CodexTaskType.DATASET_REFINEMENT:
            raise RuntimeError(f"{gateway_reason} at {leaked_path}; token=secret")
        return original_review(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(controller, "_codex_review", unavailable)

    with pytest.raises(PipelineError) as raised:
        controller.run_pipeline(config, lambda _event: None)

    assert raised.value.recoverable is True
    assert leaked_path not in str(raised.value)
    assert "secret" not in str(raised.value)
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_invalid_refinement_schema_retries_then_blocks_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="invalid-refinement-schema",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    config = config.model_copy(update={"allow_without_codex": True})
    calls = 0

    def invalid_schema(
        _context: object,
        task: CodexTaskType,
        _payload: dict[str, Any],
        **_kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal calls
        assert task is CodexTaskType.DATASET_REFINEMENT
        calls += 1
        return {"schema_version": 1, "assets": []}, {}, None

    monkeypatch.setattr(controller, "_codex_review", invalid_schema)

    with pytest.raises(PipelineError) as raised:
        controller.run_pipeline(config, lambda _event: None)

    assert raised.value.recoverable is True
    assert calls == 2
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_prepared_asset_image_mismatch_is_recoverable_and_blocks_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="asset-image-mismatch",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    original_prepare = application_service.prepare_codex_image

    def mismatch(*args: object, **kwargs: object) -> object:
        prepared = original_prepare(*args, **kwargs)
        return prepared.model_copy(update={"asset_id": "different-asset"})

    monkeypatch.setattr(application_service, "prepare_codex_image", mismatch)

    with pytest.raises(PipelineError, match="asset") as raised:
        controller.run_pipeline(config, lambda _event: None)

    assert raised.value.recoverable is True
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_refinement_cache_survives_restart_and_reruns_only_incomplete_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="restart-cache",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
        image_count=17,
    )
    original_review = controller._codex_review
    first_calls = 0

    def fail_third_batch(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal first_calls
        if task is CodexTaskType.DATASET_REFINEMENT:
            first_calls += 1
            if first_calls == 3:
                raise RuntimeError("third batch unavailable")
        return original_review(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(controller, "_codex_review", fail_third_batch)
    with pytest.raises(PipelineError):
        controller.run_pipeline(config, lambda _event: None)
    run_id = _latest_run(controller.settings, config.project_id).id

    restarted = LoRAFactoryController(controller.settings)
    restarted_original = restarted._codex_review
    resumed_calls = 0

    def record_resumed_calls(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal resumed_calls
        if task is CodexTaskType.DATASET_REFINEMENT:
            resumed_calls += 1
        return restarted_original(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(restarted, "_codex_review", record_resumed_calls)
    result = restarted.resume_project(config.project_id, lambda _event: None)

    assert result["run_id"] == run_id
    assert resumed_calls == 1
    output = _stage_output(controller.settings, config.project_id, PipelineStage.CODEX_REFINEMENT)
    assert [bool(audit.get("reused")) for audit in output["audits"]] == [True, True, False]
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_changed_working_hash_invalidates_only_its_cached_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="working-hash-cache",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
        image_count=17,
    )
    original_review = controller._codex_review
    calls = 0

    def fail_last(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal calls
        if task is CodexTaskType.DATASET_REFINEMENT:
            calls += 1
            if calls == 3:
                raise RuntimeError("last batch unavailable")
        return original_review(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(controller, "_codex_review", fail_last)
    with pytest.raises(PipelineError):
        controller.run_pipeline(config, lambda _event: None)

    working = sorted(
        (controller.settings.projects_root / config.project_id / "dataset" / "working").glob(
            "*.png"
        )
    )[0]
    with Image.open(working) as opened:
        pixels = np.asarray(opened).copy()
    pixels[0, 0] = (pixels[0, 0].astype(np.uint16) + 1).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(working)

    restarted = LoRAFactoryController(controller.settings)
    restarted_original = restarted._codex_review
    resumed_calls = 0

    def record_calls(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal resumed_calls
        if task is CodexTaskType.DATASET_REFINEMENT:
            resumed_calls += 1
        return restarted_original(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(restarted, "_codex_review", record_calls)
    restarted.resume_project(config.project_id, lambda _event: None)

    assert resumed_calls == 2
    output = _stage_output(controller.settings, config.project_id, PipelineStage.CODEX_REFINEMENT)
    assert sum(bool(audit.get("reused")) for audit in output["audits"]) == 1
    assert output["working_sha256s"][working.stem] == sha256_file(working)


def test_incomplete_refinement_cache_record_is_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="incomplete-cache",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
        image_count=17,
    )
    original_review = controller._codex_review
    calls = 0

    def fail_last(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal calls
        if task is CodexTaskType.DATASET_REFINEMENT:
            calls += 1
            if calls == 3:
                raise RuntimeError("last batch unavailable")
        return original_review(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(controller, "_codex_review", fail_last)
    with pytest.raises(PipelineError):
        controller.run_pipeline(config, lambda _event: None)
    run_id = _latest_run(controller.settings, config.project_id).id
    record = (
        controller.settings.projects_root
        / config.project_id
        / "runs"
        / run_id
        / "codex-refinement"
        / "batch-0001.json"
    )
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["complete"] = False
    record.write_text(json.dumps(payload), encoding="utf-8")

    restarted = LoRAFactoryController(controller.settings)
    restarted_original = restarted._codex_review
    resumed_calls = 0

    def record_calls(
        context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        nonlocal resumed_calls
        if task is CodexTaskType.DATASET_REFINEMENT:
            resumed_calls += 1
        return restarted_original(context, task, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(restarted, "_codex_review", record_calls)
    restarted.resume_project(config.project_id, lambda _event: None)

    assert resumed_calls == 2


def test_refinement_cancellation_cleans_prepared_images_before_training(tmp_path: Path) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="cancel-refinement",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    cancelled = False

    def cancel_after_first_image(event: dict[str, Any]) -> None:
        nonlocal cancelled
        details = event.get("details")
        if (
            event["event_type"] == "codex_image_progress"
            and isinstance(details, dict)
            and details.get("action") == "prepared"
            and not cancelled
        ):
            cancelled = True
            controller.cancel_current()

    with pytest.raises(CancelledError):
        controller.run_pipeline(config, cancel_after_first_image)

    assert _latest_run(controller.settings, config.project_id).status == RunStatus.CANCELLED.value
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_active_codex_process_cancellation_terminates_child_and_cleans_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="cancel-active-codex-process",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    process_started = Event()
    application_service.CodexGateway(controller.settings.codex_runtime_root).scratch.initialize()

    class BlockingProcess:
        returncode: int | None = None
        terminated = False
        killed = False

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            process_started.set()
            if self.terminated or self.killed:
                return b"", b""
            raise subprocess.TimeoutExpired("codex", timeout)

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = BlockingProcess()
    monkeypatch.setattr(application_service.CodexGateway, "version", lambda _self: "codex 1.0")
    monkeypatch.setattr(
        "lora_factory.codex.process.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    original_review = controller._codex_review

    def use_live_gateway_for_refinement(
        context: Any,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        if task is not CodexTaskType.DATASET_REFINEMENT:
            return original_review(context, task, payload, **kwargs)
        original_config = context.config
        context.config = original_config.model_copy(update={"backend_mode": BackendMode.REAL})
        try:
            return original_review(context, task, payload, **kwargs)
        finally:
            context.config = original_config

    monkeypatch.setattr(controller, "_codex_review", use_live_gateway_for_refinement)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(controller.run_pipeline, config, lambda _event: None)
        assert process_started.wait(timeout=10)
        controller.cancel_current()
        with pytest.raises(CancelledError):
            future.result(timeout=10)

    assert process.terminated is True
    assert process.killed is False
    assert _latest_run(controller.settings, config.project_id).status == RunStatus.CANCELLED.value
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_terminal_codex_failure_audit_is_persisted_after_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="terminal-codex-audit",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    calls = 0
    monkeypatch.setattr(application_service.CodexGateway, "version", lambda _self: "codex 1.0")

    def timeout(
        arguments: list[str],
        **_kwargs: Any,
    ) -> CodexProcessResult:
        nonlocal calls
        calls += 1
        return CodexProcessResult(tuple(arguments), -1, True, "", "")

    monkeypatch.setattr("lora_factory.codex.gateway.run_codex_process", timeout)
    original_review = controller._codex_review

    def use_live_gateway_for_refinement(
        context: Any,
        task: CodexTaskType,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        if task is not CodexTaskType.DATASET_REFINEMENT:
            return original_review(context, task, payload, **kwargs)
        original_config = context.config
        context.config = original_config.model_copy(update={"backend_mode": BackendMode.REAL})
        try:
            return original_review(context, task, payload, **kwargs)
        finally:
            context.config = original_config

    monkeypatch.setattr(controller, "_codex_review", use_live_gateway_for_refinement)

    with pytest.raises(PipelineError, match="Codex timed out"):
        controller.run_pipeline(config, lambda _event: None)

    database = Database(controller.settings.projects_root / config.project_id / "state.sqlite3")
    with database.session() as session:
        rows = tuple(session.scalars(select(CodexCallRow)))
    assert calls == 2
    assert len(rows) == 1
    audit = rows[0].audit_json
    assert rows[0].task_type == CodexTaskType.DATASET_REFINEMENT.value
    assert audit["attempts"] == 2
    assert audit["timed_out"] is True
    assert audit["fallback_used"] is False
    assert audit["version"] == "codex 1.0"
    assert audit["image_count"] == 8
    assert len(audit["image_input_sha256s"]) == 8
    assert str(tmp_path) not in json.dumps(audit)
    assert "stderr" not in json.dumps(audit).casefold()
    assert _training_attempt_count(controller.settings, config.project_id) == 0
    assert not list(controller.settings.codex_runtime_root.glob("input/images/**/*.jpg"))


def test_missing_trigger_candidates_waits_for_trigger_review_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="missing-trigger-candidates",
        refinement_mode=CodexRefinementMode.AUTO,
        trigger_mode=TriggerWordMode.CODEX_SUGGEST,
    )
    monkeypatch.setattr(
        application_service,
        "normalize_trigger_candidates",
        lambda _values: (_ for _ in ()).throw(ValueError("no valid candidates")),
    )

    result = controller.run_pipeline(config, lambda _event: None)

    assert result["status"] == RunStatus.AWAITING_REVIEW.value
    review = controller.refinement_review(config.project_id)
    assert review["requires_trigger_selection"] is True
    assert review["trigger_candidates"] == []
    assert _training_attempt_count(controller.settings, config.project_id) == 0


@pytest.mark.parametrize(
    ("refinement_mode", "trigger_mode"),
    [
        (CodexRefinementMode.AUTO, TriggerWordMode.CODEX_SUGGEST),
        (CodexRefinementMode.REVIEW, TriggerWordMode.MANUAL),
        (CodexRefinementMode.REVIEW, TriggerWordMode.CODEX_SUGGEST),
    ],
)
def test_refinement_waits_before_training_and_resumes_same_run(
    tmp_path: Path,
    refinement_mode: CodexRefinementMode,
    trigger_mode: TriggerWordMode,
) -> None:
    project_id = f"wait-{refinement_mode.value}-{trigger_mode.value}"
    controller, config = _refinement_fixture(
        tmp_path,
        project_id=project_id,
        refinement_mode=refinement_mode,
        trigger_mode=trigger_mode,
    )
    source_hashes = {
        path.name: sha256_file(path) for path in Path(config.input_paths[0]).glob("*.png")
    }

    waiting = controller.run_pipeline(config, lambda _event: None)

    assert waiting["status"] == "AWAITING_REVIEW"
    controller = LoRAFactoryController(controller.settings)
    review = controller.refinement_review(project_id)
    assert review["run_id"] == waiting["run_id"]
    assert review["requires_refinement_review"] is (refinement_mode is CodexRefinementMode.REVIEW)
    assert review["requires_trigger_selection"] is (trigger_mode is TriggerWordMode.CODEX_SUGGEST)
    batch_root = (
        tmp_path / "projects" / project_id / "runs" / str(waiting["run_id"]) / "codex-refinement"
    )
    batch_records = tuple(batch_root.glob("batch-*.json"))
    assert batch_records
    assert all(
        len(json.loads(path.read_text(encoding="utf-8"))["input_hash"]) == 64
        for path in batch_records
    )
    database = Database(tmp_path / "projects" / project_id / "state.sqlite3")
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(TrainingAttemptRow)) == 0
    trigger_word = (
        review["trigger_candidates"][0]["value"]
        if trigger_mode is TriggerWordMode.CODEX_SUGGEST
        else "manual_lfx"
    )
    decisions = (
        [{"asset_id": item["asset_id"], "decision": "accept"} for item in review["items"]]
        if refinement_mode is CodexRefinementMode.REVIEW
        else []
    )
    controller.submit_refinement_review(
        project_id,
        {
            "upstream_fingerprint": review["upstream_fingerprint"],
            "trigger_word": trigger_word,
            "items": decisions,
        },
    )

    result = controller.resume_project(project_id, lambda _event: None)

    assert result["status"] == "READY"
    assert result["run_id"] == waiting["run_id"]
    assert result["trigger_token"] == trigger_word
    captions = tuple((tmp_path / "projects" / project_id / "dataset" / "captions").glob("*.txt"))
    assert captions
    assert all(path.read_text(encoding="utf-8").startswith(f"{trigger_word},") for path in captions)
    assert source_hashes == {
        path.name: sha256_file(path) for path in Path(config.input_paths[0]).glob("*.png")
    }
    output = Path(str(result["final_model"])).parent
    training_info = json.loads((output / "training_info.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert training_info["trigger_token"] == trigger_word
    assert manifest["trigger_word"] == trigger_word
    assert manifest["codex_refinement"]["mode"] == refinement_mode.value
    assert manifest["codex_refinement"]["trigger_word_mode"] == trigger_mode.value
    with database.session() as session:
        sample_metadata = tuple(session.scalars(select(SampleRow.metadata_json)))
    assert sample_metadata
    assert all(trigger_word in str(item["request"]["prompt"]) for item in sample_metadata)


def test_refinement_rejects_stale_approval_fingerprint(tmp_path: Path) -> None:
    controller, config = _refinement_fixture(
        tmp_path,
        project_id="stale-refinement",
        refinement_mode=CodexRefinementMode.REVIEW,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    controller.run_pipeline(config, lambda _event: None)

    with pytest.raises(ValueError, match="stale"):
        controller.submit_refinement_review(
            config.project_id,
            {
                "upstream_fingerprint": "0" * 64,
                "trigger_word": "manual_lfx",
                "items": [],
            },
        )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("unknown_tag", "effective_tags"),
        ("forbidden_tag", "forbidden category"),
        ("missing_trigger", "Trigger token must be the first"),
        ("missing_class", "Expected fixed class token"),
        ("trigger_collision", "collides with common Danbooru tag"),
    ],
)
def test_invalid_refinement_approval_stays_waiting_without_saving_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_error: str,
) -> None:
    if case == "forbidden_tag":
        vocabulary = WD14TagVocabulary.from_names((*FAKE_WD14_VOCABULARY, "best_quality"))
        monkeypatch.setattr(
            WD14TagVocabulary,
            "fake",
            classmethod(lambda _cls: vocabulary),
        )
    controller, config = _refinement_fixture(
        tmp_path,
        project_id=f"invalid-approval-{case}",
        refinement_mode=CodexRefinementMode.REVIEW,
        trigger_mode=TriggerWordMode.MANUAL,
    )
    waiting = controller.run_pipeline(config, lambda _event: None)
    review = controller.refinement_review(config.project_id)
    review_path = (
        controller.settings.projects_root
        / config.project_id
        / "runs"
        / str(waiting["run_id"])
        / "refinement-review.json"
    )
    approval_path = review_path.with_name("refinement-approval.json")
    review_before = review_path.read_bytes()
    first = review["items"][0]
    items = [{"asset_id": item["asset_id"], "decision": "accept"} for item in review["items"]]
    edited: dict[str, Any] = {
        "asset_id": first["asset_id"],
        "decision": "edit",
        "effective_tags": first["proposed_tags"],
    }
    if case == "unknown_tag":
        edited["effective_tags"] = [*first["proposed_tags"], "invented_tag"]
    elif case == "forbidden_tag":
        edited["effective_tags"] = [*first["proposed_tags"], "best_quality"]
    elif case == "missing_trigger":
        edited["caption"] = "1girl, smile"
    else:
        edited["caption"] = "manual_lfx, smile"
    items[0] = edited

    approval_trigger = "1girl" if case == "trigger_collision" else "manual_lfx"
    with pytest.raises(ValueError, match=expected_error):
        controller.submit_refinement_review(
            config.project_id,
            {
                "upstream_fingerprint": review["upstream_fingerprint"],
                "trigger_word": approval_trigger,
                "items": items,
            },
        )

    assert approval_path.exists() is False
    assert review_path.read_bytes() == review_before
    assert _latest_run(controller.settings, config.project_id).status == (
        RunStatus.AWAITING_REVIEW.value
    )
    assert _training_attempt_count(controller.settings, config.project_id) == 0


def test_invalid_codex_suggestion_is_audited_without_stopping_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "images"
    _images(source)
    base_model = tmp_path / "base.safetensors"
    _fake_sdxl(base_model)
    controller = LoRAFactoryController(
        AppSettings(
            projects_root=tmp_path / "projects",
            managed_runtime_root=tmp_path / "runtime",
            codex_runtime_root=tmp_path / "codex",
        )
    )

    def review_with_invalid_training_change(
        _context: object,
        task: CodexTaskType,
        payload: dict[str, Any],
        **_kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        response = deterministic_fallback(task, payload).model_dump(mode="json")
        if task is CodexTaskType.TRAINING_PLAN:
            response["proposed_changes"] = [
                {
                    "field": "batch_size",
                    "value": 999,
                    "reason": "schema-valid but outside the Factory allowlist",
                }
            ]
        return response, {"fallback_used": False, "task_type": task.value, "attempts": 1}, None

    monkeypatch.setattr(controller, "_codex_review", review_with_invalid_training_change)
    events: list[dict[str, Any]] = []

    result = controller.run_pipeline(
        ProjectConfig(
            project_id="invalid-codex-advisory",
            lora_name="Invalid Codex Advisory",
            preset=PresetKind.CHARACTER,
            trigger_token="advisory_person",  # noqa: S106 - domain trigger, not a credential
            base_model=base_model,
            input_paths=(source,),
            selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
            output_root=tmp_path / "output",
            backend_mode=BackendMode.FAKE,
        ),
        events.append,
    )

    assert result["status"] == "READY"
    invalid_change_warnings = [
        event
        for event in events
        if event["event_type"] == "warning" and "invalid training changes" in str(event["message"])
    ]
    assert len(invalid_change_warnings) == 1
    assert invalid_change_warnings[0]["details"] == {"validated_suggestions": {}}


def test_full_fake_pipeline_uses_real_formats_and_stage_graph(tmp_path: Path) -> None:
    source = tmp_path / "入力 画像"
    _images(source)
    base_model = tmp_path / "tiny SDXL.safetensors"
    _fake_sdxl(base_model)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
        destinations=[
            DestinationConfig(
                kind=DestinationKind.A1111,
                root=tmp_path / "stable-diffusion-webui",
            )
        ],
    )
    controller = LoRAFactoryController(settings)
    events: list[dict[str, object]] = []
    result = controller.run_pipeline(
        ProjectConfig(
            project_id="Fake 統合",
            lora_name="Fake 統合",
            preset=PresetKind.CHARACTER,
            trigger_token="lfx_person",  # noqa: S106 - domain trigger, not a credential
            base_model=base_model,
            input_paths=(source,),
            selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
            output_root=tmp_path / "output",
            backend_mode=BackendMode.FAKE,
        ),
        events.append,
    )

    assert result["status"] == "READY"
    assert result["raw_integrity_verified"] is True
    final_model = Path(str(result["final_model"]))
    assert final_model.is_file() and final_model.suffix == ".safetensors"
    assert Path(str(result["preview"])).is_file()
    assert Path(str(result["comparison"])).is_file()
    with Image.open(result["preview"]) as preview:
        assert preview.format == "PNG"
        assert preview.size == (512, 512)
    manifest = json.loads(
        (final_model.parent / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["backend_manifest"]["sd_scripts"]["commit"]
    assert manifest["codex_pretrain_audit"]["fallback_used"] is True
    assert len(manifest["source_images"]) == 8
    assert all(len(item["sha256"]) == 64 for item in manifest["source_images"])
    assert len(manifest["dataset_decisions"]) == 8
    assert (
        manifest["dataset_statistics"]["diversity_score"]
        == manifest["dataset_diversity"]["content_diversity"]
    )
    assert manifest["validation_split"]["enabled"] is False
    assert "training_command_arguments" not in manifest
    assert manifest["gpu_capabilities"][0]["uuid"] == "gpu-1"
    assert set(manifest["gpu_assignments"]) == {
        "tagging",
        "reference_embedding",
        "training",
        "screening",
        "weight_sweep",
        "multi_seed_validation",
        "generated_tagging",
        "image_embedding",
    }
    assert {
        gpu_uuid for assignments in manifest["gpu_assignments"].values() for gpu_uuid in assignments
    } == {"gpu-1"}
    evaluation = json.loads((final_model.parent / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["image_embedding"]["model_id"] == ("lora-factory/fake-image-embedding")
    assert evaluation["image_embedding"]["dimension"] == 192
    assert all(item["evidence"]["semantic_embedding_used"] for item in evaluation["metrics"])
    reference_embedding = manifest["dataset_reference_embedding"]
    assert reference_embedding["status"] == "complete"
    assert reference_embedding["model_id"] == "lora-factory/fake-image-embedding"
    assert reference_embedding["dimension"] == 192
    assert len(reference_embedding["centroid_similarity_by_asset"]) == 8
    assert reference_embedding["path"] == "embeddings.json"
    assert set(reference_embedding["outlier_asset_ids"]) <= {
        item["asset_id"] for item in manifest["dataset_decisions"] if item["accepted"]
    }
    assert set(manifest["sampling"]["prompt_ids"]) == {
        prompt.id for prompt in load_benchmark_prompts(PresetKind.CHARACTER)
    }
    assert manifest["destination_copies"] == []

    copied = controller.copy_output("Fake 統合", DestinationKind.A1111.value)
    assert Path(str(copied["destination"])).is_file()
    manifest = json.loads(
        (final_model.parent / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["destination_copies"][-1]["sha256"] == copied["sha256"]
    alternatives = result["alternatives"]
    assert isinstance(alternatives, dict) and alternatives
    promoted_id = next(iter(alternatives))
    promoted = controller.promote_alternative("Fake 統合", promoted_id)
    assert promoted["sha256"]
    manifest = json.loads(
        (final_model.parent / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["manual_selected_checkpoint_id"] == promoted_id
    assert manifest["final_lora"]["sha256"] == promoted["sha256"]
    project_root = settings.projects_root / "Fake 統合"
    assert (project_root / "run_snapshot.yaml").is_file()
    assert (project_root / "runs" / str(result["run_id"]) / "run_snapshot.yaml").is_file()
    dataset_config = tomlkit.parse(
        (project_root / "configs" / "dataset.toml").read_text(encoding="utf-8")
    )
    assert "validation_split" not in dataset_config["datasets"][0]
    raw_files = list((project_root / "dataset" / "raw").glob("*"))
    assert len(raw_files) == 8
    assert all(path.name.startswith(path.stem[:16]) for path in raw_files)
    completed_stages = {
        str(event["stage"]) for event in events if event["event_type"] == "stage_completed"
    }
    assert {"IMPORTING", "TAGGING", "TRAINING", "PACKAGING", "READY"} <= completed_stages

    database = Database(project_root / "state.sqlite3")
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(AssetRow)) == 16
        assert session.scalar(select(func.count()).select_from(TagResultRow)) == 8
        assert session.scalar(select(func.count()).select_from(CaptionRow)) == 8
        assert session.scalar(select(func.count()).select_from(DatasetSplitRow)) == 8
        assert session.scalar(select(func.count()).select_from(TrainingPlanRow)) == 1
        assert session.scalar(select(func.count()).select_from(TrainingAttemptRow)) == 1
        assert session.scalar(select(func.count()).select_from(CheckpointRow)) >= 1
        assert session.scalar(select(func.count()).select_from(SampleRow)) >= 1
        observed_prompt_ids = set(session.scalars(select(SampleRow.prompt_id)))
        expected_prompt_ids = {prompt.id for prompt in load_benchmark_prompts(PresetKind.CHARACTER)}
        assert observed_prompt_ids == expected_prompt_ids
        assert session.scalar(select(func.count()).select_from(EvaluationRow)) >= 1
        assert session.scalar(select(func.count()).select_from(CodexCallRow)) == 5


def test_pre_snapshot_include_override_can_restore_a_quality_reject(tmp_path: Path) -> None:
    source = tmp_path / "input"
    _images(source)
    rejected_source = source / "manual include.png"
    Image.new("RGB", (512, 512), (0, 0, 0)).save(rejected_source)
    rejected_asset_id = sha256_file(rejected_source)
    base_model = tmp_path / "base.safetensors"
    _fake_sdxl(base_model)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    config = ProjectConfig(
        project_id="include-override",
        lora_name="include_override",
        preset=PresetKind.CHARACTER,
        trigger_token="include_override_token",  # noqa: S106 - domain trigger
        base_model=base_model,
        input_paths=(source,),
        selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )
    controller.projects.create(config)
    controller.set_dataset_override(
        config.project_id,
        {"asset_id": rejected_asset_id, "included": True},
    )

    events: list[dict[str, object]] = []
    result = controller.run_pipeline(config, events.append)

    assert result["status"] == "READY"
    manifest = json.loads(
        (Path(str(result["final_model"])).parent / "reproducibility_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    restored = next(
        item for item in manifest["dataset_decisions"] if item["asset_id"] == rejected_asset_id
    )
    assert restored["accepted"] is True
    assert "Warning" in restored["categories"]
    assert manifest["dataset_statistics"]["accepted_count"] == 9
    review_event = next(event for event in events if event["event_type"] == "dataset_review")
    review_details = review_event["details"]
    assert isinstance(review_details, dict)
    restored_review = next(
        item for item in review_details["items"] if item["asset_id"] == rejected_asset_id
    )
    assert restored_review["bucket"] == "512x512"
    assert review_details["gate"]["accepted_count"] == 9
    database = Database(settings.projects_root / config.project_id / "state.sqlite3")
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(TagResultRow)) == 9
        assert session.scalar(select(func.count()).select_from(CaptionRow)) == 9
