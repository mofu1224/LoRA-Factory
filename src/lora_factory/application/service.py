"""Application orchestration for the complete LoRA Factory workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from sqlalchemy import select

from lora_factory import __version__
from lora_factory.application.artifact_store import ApplicationArtifactStore
from lora_factory.application.progress_reporting import progress_details
from lora_factory.application.refinement_review import (
    RefinementApproval,
    RefinementReviewItem,
    RefinementReviewState,
    refinement_fingerprint,
)
from lora_factory.application.setup_status import build_setup_checks
from lora_factory.caption.audit import audit_captions
from lora_factory.caption.managed_wd14 import ManagedWD14Tagger
from lora_factory.caption.refinement import (
    AssetRefinementProposal,
    CaptionDraftAsset,
    CaptionDraftBatch,
    RefinementDecisionKind,
    TriggerCandidate,
    batch_refinement_assets,
    build_caption_drafts,
    finalize_captions,
    normalize_trigger_candidates,
    validate_refinement_proposals,
)
from lora_factory.caption.tag_vocabulary import WD14TagVocabulary
from lora_factory.caption.tagger import (
    FakeTagger,
    ImageTagResult,
    RawTagStore,
    TaggerProtocol,
    TagScore,
)
from lora_factory.caption.writer import CaptionWriter
from lora_factory.codex.allowlist import validate_recovery_changes
from lora_factory.codex.fallback import deterministic_fallback
from lora_factory.codex.gateway import CodexCallCancelled, CodexCallError, CodexGateway
from lora_factory.codex.image_attachment import (
    DEFAULT_CODEX_IMAGE_PROFILE,
    PreparedCodexImage,
    prepare_codex_image,
    remove_codex_images,
)
from lora_factory.codex.runtime import CodexRuntimeAdapter
from lora_factory.codex.schemas import CodexTaskType, DatasetRefinementResponse
from lora_factory.config.loader import dump_yaml_mapping, load_yaml_mapping
from lora_factory.config.models import (
    AppSettings,
    BackendMode,
    CodexRefinementMode,
    DatasetStatistics,
    DestinationConfig,
    DestinationKind,
    GpuCapability,
    PresetKind,
    ProjectConfig,
    ProjectDraft,
    TrainingPlan,
    TriggerWordMode,
)
from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.core.context import PipelineContext
from lora_factory.core.events import EventBus, PipelineEvent
from lora_factory.core.exceptions import ErrorClassification, PipelineError
from lora_factory.core.pipeline import PipelineEngine, StageDefinition
from lora_factory.core.stage import PipelineStage, RunStatus, StageStatus
from lora_factory.dataset.diversity import analyze_diversity
from lora_factory.dataset.duplicates import DuplicateCandidate, detect_duplicates
from lora_factory.dataset.embedding_review import (
    analyze_dataset_embeddings,
    combine_duplicate_cluster_ids,
)
from lora_factory.dataset.image_normalizer import normalize_image
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY
from lora_factory.dataset.quality import (
    DatasetGateThresholds,
    QualityAssessment,
    QualityDisposition,
    assess_image,
    evaluate_quality_gate,
)
from lora_factory.dataset.report import build_dataset_report
from lora_factory.dataset.review_categories import (
    primary_review_category,
    review_categories,
)
from lora_factory.evaluation.embedding import (
    FakeImageEmbeddingBackend,
    ImageEmbedding,
    ImageEmbeddingBackend,
    ImageEmbeddingBatch,
    ManagedClipImageEmbeddingBackend,
)
from lora_factory.evaluation.metrics import SampleObservation, aggregate_candidate_signals
from lora_factory.evaluation.models import CandidateMetrics
from lora_factory.evaluation.ranking import rank_candidates
from lora_factory.gpu.discovery import bind_gpu_for_child, discover_nvidia_gpus
from lora_factory.gpu.execution import lease_selected_gpu
from lora_factory.gpu.models import GpuBinding, GpuDevice, GpuTaskKind
from lora_factory.gpu.scheduler import SelectedGpuScheduler
from lora_factory.model.inspector import inspect_sdxl_safetensors
from lora_factory.packaging.destinations import copy_without_overwrite
from lora_factory.packaging.finalizer import (
    CheckpointCandidate,
    FinalizationRequest,
    Finalizer,
)
from lora_factory.packaging.metadata import artifact_record, sanitize_public_metadata
from lora_factory.project.import_service import ImmutableImportService
from lora_factory.project.integrity import compare_raw_snapshots, verify_raw_store
from lora_factory.project.layout import ProjectLayout
from lora_factory.project.manifest import DatasetManifest
from lora_factory.project.service import ProjectService
from lora_factory.runtime.doctor import inspect_runtime
from lora_factory.runtime.installer import ManagedRuntimeInstaller
from lora_factory.runtime.manager import RuntimeManager
from lora_factory.runtime.validation_record import RuntimeValidationRecord, ValidationStatus
from lora_factory.sampling.backend import SamplerBackend, SampleRequest
from lora_factory.sampling.fake_backend import FakeSampler
from lora_factory.sampling.grid import render_grid
from lora_factory.sampling.prompts import BenchmarkPrompt, load_benchmark_prompts
from lora_factory.storage.database import Database
from lora_factory.storage.orm import DestinationRow, EventRow, ProjectRow, RunRow, StageRow
from lora_factory.storage.repositories import StageRepository
from lora_factory.training.backend import TrainingBackend, TrainingProgress, TrainingRequest
from lora_factory.training.batch_probe import (
    BatchProbeRequest,
    BatchProbeResult,
    FakeBatchProbe,
    TorchCudaBatchProbe,
    descending_batch_candidates,
    estimate_training_vram_mb,
)
from lora_factory.training.buckets import format_assigned_bucket
from lora_factory.training.dataset_toml import (
    verify_materialized_validation_partition,
    write_attempt_dataset_config,
    write_dataset_configs,
)
from lora_factory.training.fake_backend import FakeTrainingBackend
from lora_factory.training.planner import plan_training, plan_validation_split
from lora_factory.training.profiles import load_preset_profile
from lora_factory.training.recovery import RecoveryDecision, next_recovery
from lora_factory.training.resume import prepare_training_attempt, training_request_fingerprint
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json, write_json_atomic
from lora_factory.util.redaction import redact_text

EventCallback = Callable[[dict[str, Any]], None]

_REFINEMENT_MAX_ITEMS = 8
_REFINEMENT_DRAFT_BYTES = 120 * 1024
_REFINEMENT_PAYLOAD_BYTES = 128 * 1024


def remove_empty_image_root(image_root: Path, *, boundary: Path) -> None:
    """Remove an empty per-batch image tree without crossing its runtime boundary."""

    resolved_boundary = boundary.resolve(strict=False)
    current = image_root.resolve(strict=False)
    if current == resolved_boundary or not current.is_relative_to(resolved_boundary):
        raise ValueError("Codex image root is outside its runtime boundary")
    while current != resolved_boundary:
        try:
            current.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            break
        current = current.parent


@dataclass(frozen=True, slots=True)
class _TagBatch:
    results: tuple[ImageTagResult, ...]
    gpu_uuids: tuple[str, ...]
    model_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class _EmbeddingShardedBatch:
    result: ImageEmbeddingBatch
    gpu_uuids: tuple[str, ...]


def _dataset_statistics(
    accepted_ids: Sequence[str],
    analyzed_by_id: Mapping[str, Mapping[str, Any]],
    *,
    validation_count: int,
    diversity_score: float,
) -> DatasetStatistics:
    if not accepted_ids:
        raise ValueError("Dataset statistics require at least one accepted image")
    accepted_items = [analyzed_by_id[asset_id] for asset_id in accepted_ids]
    short_sides = sorted(min(int(item["width"]), int(item["height"])) for item in accepted_items)
    areas = sorted(int(item["width"]) * int(item["height"]) for item in accepted_items)
    ratios = [
        max(
            int(item["width"]) / int(item["height"]),
            int(item["height"]) / int(item["width"]),
        )
        for item in accepted_items
    ]
    p10_index = max(0, min(len(short_sides) - 1, int(len(short_sides) * 0.1)))
    return DatasetStatistics(
        accepted_count=len(accepted_ids),
        validation_count=validation_count,
        short_side_p10=short_sides[p10_index],
        short_side_median=int(median(short_sides)),
        area_median=int(median(areas)),
        aspect_ratio_min=min(ratios),
        aspect_ratio_max=max(ratios),
        diversity_score=diversity_score,
    )


def _local_app_data() -> Path:
    configured = os.environ.get("LOCALAPPDATA")
    if configured:
        return Path(configured)
    return Path.home() / "AppData" / "Local"


def repository_root() -> Path:
    if bool(getattr(sys, "frozen", False)):
        bundle_root = getattr(sys, "_MEIPASS", None)
        if isinstance(bundle_root, str):
            return Path(bundle_root).resolve(strict=True)
    return Path(__file__).resolve().parents[3]


def application_root() -> Path:
    """Return the portable app root where user-visible ``Project`` data lives."""

    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return repository_root()


def _settings_path() -> Path:
    return _local_app_data() / "LoRAFactory" / "settings.json"


def default_app_settings() -> AppSettings:
    data_root = _local_app_data() / "LoRAFactory"
    defaults = AppSettings(
        projects_root=application_root() / "Project",
        managed_runtime_root=data_root / "runtimes",
        codex_runtime_root=data_root / "codex-scratch",
    )
    settings_path = _settings_path()
    if not settings_path.is_file():
        return defaults
    try:
        loaded = AppSettings.model_validate(read_json(settings_path))
        return loaded.model_copy(update={"projects_root": defaults.projects_root})
    except (OSError, ValueError):
        return defaults


def _jsonable(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stage_output(context: PipelineContext, stage: PipelineStage) -> dict[str, Any]:
    value = context.artifacts.get(stage.value)
    if not isinstance(value, dict):
        raise RuntimeError(f"Pipeline stage output is unavailable: {stage.value}")
    return value


def _safe_output_directory(root: Path, name: str) -> Path:
    candidate = root.resolve(strict=False) / name
    if not candidate.exists() or not any(candidate.iterdir()):
        return candidate
    counter = 2
    while True:
        versioned = candidate.with_name(f"{candidate.name}_v{counter}")
        if not versioned.exists() or not any(versioned.iterdir()):
            return versioned
        counter += 1


def _training_vram_estimate_mb(plan: TrainingPlan) -> int:
    """Return a conservative scheduler reservation for one SDXL training process."""

    return estimate_training_vram_mb(
        resolution=plan.resolution,
        batch_size=plan.batch_size,
        network_dim=plan.network_dim,
        precision=plan.precision,
    )


class LoRAFactoryController:
    """Synchronous headless controller; the Qt layer invokes it in a worker thread."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or default_app_settings()
        self.settings.projects_root.mkdir(parents=True, exist_ok=True)
        self.settings.managed_runtime_root.mkdir(parents=True, exist_ok=True)
        self.settings.codex_runtime_root.mkdir(parents=True, exist_ok=True)
        self.projects = ProjectService(self.settings.projects_root)
        self.runtime = RuntimeManager(
            self.settings.managed_runtime_root,
            repository_root() / "backend-manifest.json",
        )
        self._codex_runtime = CodexRuntimeAdapter(probe=True)
        self._cancellation: CancellationToken | None = None
        self._destinations = tuple(self.settings.destinations)

    def discover_gpus(self) -> Sequence[GpuDevice]:
        return discover_nvidia_gpus()

    def setup_checks(self) -> Sequence[Mapping[str, Any]]:
        return build_setup_checks(
            settings=self.settings,
            runtime=self.runtime,
            destinations=self._destinations,
        )

    def recent_projects(self) -> Sequence[Mapping[str, Any]]:
        values: list[dict[str, Any]] = []
        for path in sorted(
            self.settings.projects_root.glob("*/project.yaml"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                config = self.projects.load_config(path.parent.name)
            except (OSError, ValueError):
                try:
                    draft = self.projects.load_draft(path.parent.name)
                except (OSError, ValueError):
                    continue
                values.append(
                    {
                        **draft.model_dump(mode="json"),
                        "name": draft.lora_name,
                        "status": "draft",
                        "run_id": None,
                        "completion": None,
                        "project_root": str(path.parent.resolve(strict=False)),
                        "modified_at": datetime.fromtimestamp(
                            path.stat().st_mtime, tz=UTC
                        ).isoformat(),
                    }
                )
                continue
            status = "created"
            run_id: str | None = None
            database_path = path.parent / "state.sqlite3"
            if database_path.is_file():
                database = Database(database_path)
                database.initialize()
                with database.session() as session:
                    latest = session.scalar(
                        select(RunRow).order_by(RunRow.created_at.desc()).limit(1)
                    )
                if latest is not None:
                    status = latest.status.casefold()
                    run_id = latest.id
            completion: dict[str, Any] | None = None
            completion_path = path.parent / "completion.json"
            if completion_path.is_file():
                loaded = read_json(completion_path)
                if isinstance(loaded, dict):
                    completion = loaded
            values.append(
                {
                    **config.model_dump(mode="json"),
                    "name": config.lora_name,
                    "status": status,
                    "run_id": run_id,
                    "completion": completion,
                    "project_root": str(path.parent.resolve(strict=False)),
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
                }
            )
        return tuple(values[:20])

    def create_project(self, name: str) -> Mapping[str, Any]:
        """Create an idempotent name-only project and initialize its durable base tree."""

        draft = ProjectDraft(project_id=name, lora_name=name)
        layout, created = self.projects.create_draft(draft)
        database = Database(layout.database)
        database.initialize()
        return {
            **draft.model_dump(mode="json"),
            "created": created,
            "project_root": str(layout.root),
        }

    def run_pipeline(self, config: ProjectConfig, emit: EventCallback) -> Mapping[str, Any]:
        config.base_model.resolve(strict=True)
        for source in config.input_paths:
            source.resolve(strict=True)
        layout = self.projects.layout_for(config.project_id)
        if layout.config.exists():
            self.projects.save_config(config)
            layout.create()
        else:
            layout = self.projects.create(config)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        return self._execute(config, layout, run_id, emit)

    def resume_project(self, project_id: str, emit: EventCallback) -> Mapping[str, Any]:
        config = self.projects.load_config(project_id)
        layout = self.projects.layout_for(project_id)
        database = Database(layout.database)
        database.initialize()
        database.recover_interrupted()
        with database.session() as session:
            latest = session.scalar(select(RunRow).order_by(RunRow.created_at.desc()).limit(1))
        if latest is None:
            return self._execute(
                config,
                layout,
                datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8],
                emit,
            )
        if latest.status == RunStatus.COMPLETED.value:
            package = layout.run(latest.id).root / "completion.json"
            if package.is_file():
                result = read_json(package)
                if isinstance(result, dict):
                    return result
        if latest.status == RunStatus.FAILED_FATAL.value:
            raise RuntimeError("The latest run failed fatally and cannot be resumed unchanged")
        snapshot_config = latest.snapshot_json.get("config")
        if isinstance(snapshot_config, dict):
            config = ProjectConfig.model_validate(snapshot_config)
        return self._execute(config, layout, latest.id, emit, resume=True)

    def cancel_current(self) -> None:
        if self._cancellation is not None:
            self._cancellation.cancel()

    def save_destinations(self, destinations: Sequence[Mapping[str, Any]]) -> None:
        self._destinations = tuple(DestinationConfig.model_validate(item) for item in destinations)
        self.settings = self.settings.model_copy(update={"destinations": list(self._destinations)})
        write_json_atomic(
            _settings_path(),
            self.settings.model_dump(mode="json"),
        )

    def repair_setup(self, check_name: str) -> None:
        if check_name != "Managed Training Runtime":
            raise ValueError(f"No automatic repair is defined for {check_name}")
        progress_path = self.settings.managed_runtime_root / "install-progress.json"

        def progress(stage: str, fraction: float, message: str) -> None:
            write_json_atomic(
                progress_path,
                {"stage": stage, "progress": fraction, "message": message},
            )

        if not self.runtime.installed() or not self.runtime.source_matches_manifest():
            ManagedRuntimeInstaller(self.runtime).install(CancellationToken(), progress)
        if not self.runtime.installed() or not self.runtime.source_matches_manifest():
            raise RuntimeError(
                "Managed runtime installation did not produce the pinned complete runtime"
            )

        devices = tuple(device for device in discover_nvidia_gpus() if device.compatible)
        if not devices:
            raise RuntimeError("No compatible NVIDIA CUDA GPU is available for deep validation")
        device = max(
            devices,
            key=lambda item: (item.free_vram_mb, item.total_vram_mb, -item.index),
        )
        binding = bind_gpu_for_child(
            device.uuid,
            selected_uuids=(device.uuid,),
            devices=devices,
        )
        report = inspect_runtime(
            python_executable=self.runtime.layout.python,
            binding=binding,
            sd_scripts_root=self.runtime.layout.sd_scripts,
        )
        record = RuntimeValidationRecord.from_doctor_report(
            str(self.runtime.manifest["profile_id"]), report
        )
        self.runtime.write_validation_record(record)
        if not record.ready:
            failures = "; ".join(
                f"{name}: {check.detail}"
                for name, check in record.checks.items()
                if check.status is ValidationStatus.ERROR
            )
            raise RuntimeError(
                f"Managed runtime deep validation failed on {device.uuid}: "
                f"{failures or 'doctor reported NOT READY'}"
            )

    def open_output_folder(self, project_id: str) -> None:
        directory = Path(str(self._completion(project_id)["output_directory"])).resolve(strict=True)
        if os.name != "nt":
            raise RuntimeError("The native output-folder action is available on Windows only")
        explorer = Path(os.environ.get("WINDIR", r"C:\Windows")) / "explorer.exe"
        if not explorer.is_file():
            raise FileNotFoundError(f"Windows Explorer was not found: {explorer}")
        subprocess.Popen(  # noqa: S603 - fixed executable and validated directory path.
            [str(explorer), str(directory)],
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def duplicate_project(self, project_id: str) -> Mapping[str, Any]:
        original = self.projects.load_config(project_id)
        duplicate_name = f"{original.lora_name} Copy"
        return original.model_copy(
            update={"project_id": duplicate_name, "lora_name": duplicate_name}
        ).model_dump(mode="json")

    def set_dataset_override(self, project_id: str, override: Mapping[str, Any]) -> None:
        """Persist one pre-run dataset decision for the next immutable run snapshot."""

        asset_id = str(override.get("asset_id", ""))
        if not asset_id or any(character not in "0123456789abcdef" for character in asset_id):
            raise ValueError("Dataset override requires a hexadecimal content asset_id")
        allowed = {"included", "final_caption"}
        unknown = set(override) - {"asset_id", *allowed}
        if unknown:
            raise ValueError(f"Unsupported dataset override field: {sorted(unknown)[0]}")
        path = self.projects.layout_for(project_id).dataset / "review-overrides.json"
        payload: dict[str, Any] = {}
        if path.is_file():
            loaded = read_json(path)
            if isinstance(loaded, dict):
                payload = loaded
        current = payload.get(asset_id, {})
        item = dict(current) if isinstance(current, dict) else {}
        if "included" in override:
            item["included"] = bool(override["included"])
        if "final_caption" in override:
            caption = str(override["final_caption"]).strip()
            if not caption or "\n" in caption or "\r" in caption:
                raise ValueError("Caption override must contain one non-empty line")
            item["final_caption"] = caption
        payload[asset_id] = item
        write_json_atomic(path, payload)

    def refinement_review(self, project_id: str) -> Mapping[str, Any]:
        """Load the latest durable Runtime Codex review without exposing project paths."""

        layout = self.projects.layout_for(project_id)
        database = Database(layout.database)
        database.initialize()
        with database.session() as session:
            latest = session.scalar(select(RunRow).order_by(RunRow.created_at.desc()).limit(1))
        if latest is None or latest.status != RunStatus.AWAITING_REVIEW.value:
            raise ValueError("The project has no run awaiting refinement review")
        path = layout.run(latest.id).root / "refinement-review.json"
        return RefinementReviewState.model_validate(read_json(path)).model_dump(mode="json")

    def submit_refinement_review(
        self,
        project_id: str,
        decision: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Atomically save a validated approval for the latest waiting run."""

        layout = self.projects.layout_for(project_id)
        database = Database(layout.database)
        database.initialize()
        with database.session() as session:
            latest = session.scalar(select(RunRow).order_by(RunRow.created_at.desc()).limit(1))
        if latest is None or latest.status != RunStatus.AWAITING_REVIEW.value:
            raise ValueError("The project has no run awaiting refinement review")
        run_root = layout.run(latest.id).root
        review = RefinementReviewState.model_validate(
            read_json(run_root / "refinement-review.json")
        )
        approval = RefinementApproval.model_validate(decision)
        if approval.upstream_fingerprint != review.upstream_fingerprint:
            raise ValueError("The refinement approval is stale; reload the current review")
        expected = {item.asset_id for item in review.items}
        received = [item.asset_id for item in approval.items]
        if len(received) != len(set(received)) or set(received) - expected:
            raise ValueError("Refinement approval contains duplicate or unknown assets")
        if review.requires_refinement_review and set(received) != expected:
            raise ValueError("Refinement approval must decide every accepted asset")
        snapshot_config = latest.snapshot_json.get("config")
        if not isinstance(snapshot_config, dict):
            raise ValueError("The waiting run has no valid project snapshot")
        config = ProjectConfig.model_validate(snapshot_config)
        with database.session() as session:
            draft_output = self._successful_stage_output(
                session,
                latest.id,
                PipelineStage.CAPTION_DRAFTING,
            )
            refinement_output = self._successful_stage_output(
                session,
                latest.id,
                PipelineStage.CODEX_REFINEMENT,
            )
        self._validate_refinement_approval_values(
            config=config,
            review=review,
            approval=approval,
            drafts=CaptionDraftBatch.model_validate(draft_output),
            refinement=refinement_output,
            vocabulary=self._refinement_vocabulary_for_config(config),
        )
        write_json_atomic(run_root / "refinement-approval.json", approval.model_dump(mode="json"))
        return {
            "project_id": project_id,
            "run_id": latest.id,
            "status": RunStatus.AWAITING_REVIEW.value,
            "approval_saved": True,
        }

    @staticmethod
    def _successful_stage_output(
        session: Any,
        run_id: str,
        stage: PipelineStage,
    ) -> dict[str, Any]:
        row = session.scalar(
            select(StageRow)
            .where(
                StageRow.run_id == run_id,
                StageRow.stage == stage.value,
                StageRow.status == StageStatus.SUCCEEDED.value,
            )
            .order_by(StageRow.attempt.desc())
            .limit(1)
        )
        if row is None or not isinstance(row.output_json, dict):
            raise ValueError(f"Waiting run is missing successful {stage.value} output")
        return dict(row.output_json)

    def copy_output(self, project_id: str, destination_kind: str) -> Mapping[str, Any]:
        completion = self._completion(project_id)
        source = Path(str(completion["final_model"]))
        kind = DestinationKind(destination_kind)
        destination = next(
            (item for item in self._destinations if item.kind is kind and item.enabled), None
        )
        if destination is None:
            raise ValueError(f"No enabled {kind.value} destination is configured")
        copied = copy_without_overwrite(source, configured_root=destination.root, kind=kind)
        result = {
            "source": str(copied.source),
            "destination": str(copied.destination),
            "sha256": copied.sha256,
            "versioned": copied.versioned,
            "kind": kind.value,
            "configured_root": str(destination.root),
            "copied_at": datetime.now(UTC).isoformat(),
        }
        layout = self.projects.layout_for(project_id)
        database = Database(layout.database)
        database.initialize()
        with database.session() as session:
            row = session.scalar(
                select(DestinationRow).where(
                    DestinationRow.project_id == project_id,
                    DestinationRow.kind == kind.value,
                    DestinationRow.root_path == str(destination.root),
                )
            )
            if row is None:
                session.add(
                    DestinationRow(
                        project_id=project_id,
                        kind=kind.value,
                        root_path=str(destination.root),
                        enabled=True,
                        last_copy_json=result,
                    )
                )
            else:
                row.enabled = True
                row.last_copy_json = result
        manifest_path = source.parent / "reproducibility_manifest.json"
        manifest_payload = read_json(manifest_path)
        if not isinstance(manifest_payload, dict):
            raise ValueError(f"Invalid reproducibility manifest: {manifest_path}")
        copies = manifest_payload.get("destination_copies", [])
        if not isinstance(copies, list):
            copies = []
        manifest_payload["destination_copies"] = [*copies, sanitize_public_metadata(result)]
        write_json_atomic(manifest_path, manifest_payload)
        return result

    def promote_alternative(self, project_id: str, checkpoint_id: str) -> Mapping[str, Any]:
        completion = self._completion(project_id)
        alternatives = completion.get("alternatives", {})
        if not isinstance(alternatives, dict) or checkpoint_id not in alternatives:
            raise KeyError(f"Unknown packaged alternative: {checkpoint_id}")
        final_model = Path(str(completion["final_model"]))
        new_hash = Finalizer().promote_alternative(
            final_model=final_model,
            alternative=Path(str(alternatives[checkpoint_id])),
            history_directory=final_model.parent / "history",
        )
        completion["sha256"] = new_hash
        promoted_at = datetime.now(UTC).isoformat()
        manifest_path = final_model.parent / "reproducibility_manifest.json"
        manifest_payload = read_json(manifest_path)
        if not isinstance(manifest_payload, dict):
            raise ValueError(f"Invalid reproducibility manifest: {manifest_path}")
        promotions = manifest_payload.get("manual_promotions", [])
        if not isinstance(promotions, list):
            promotions = []
        manifest_payload.update(
            {
                "final_lora": artifact_record(final_model),
                "manual_selected_checkpoint_id": checkpoint_id,
                "manual_promotions": [
                    *promotions,
                    {
                        "checkpoint_id": checkpoint_id,
                        "source_filename": Path(str(alternatives[checkpoint_id])).name,
                        "promoted_at": promoted_at,
                        "final_sha256": new_hash,
                    },
                ],
            }
        )
        write_json_atomic(manifest_path, manifest_payload)
        training_info_path = final_model.parent / "training_info.json"
        training_info = read_json(training_info_path)
        if not isinstance(training_info, dict):
            raise ValueError(f"Invalid training info: {training_info_path}")
        training_info.update(
            {
                "best_checkpoint": checkpoint_id,
                "final_model": artifact_record(final_model),
                "manually_promoted_at": promoted_at,
            }
        )
        write_json_atomic(training_info_path, training_info)
        write_json_atomic(self.projects.layout_for(project_id).root / "completion.json", completion)
        return completion

    def _completion(self, project_id: str) -> dict[str, Any]:
        path = self.projects.layout_for(project_id).root / "completion.json"
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid completion record: {path}")
        return payload

    def _execute(
        self,
        config: ProjectConfig,
        layout: ProjectLayout,
        run_id: str,
        emit: EventCallback,
        *,
        resume: bool = False,
    ) -> Mapping[str, Any]:
        run_layout = layout.run(run_id)
        run_layout.create()
        database = Database(layout.database)
        database.initialize()
        config_json = config.model_dump(mode="json")
        snapshot_path = run_layout.root / "run_snapshot.yaml"
        if snapshot_path.is_file():
            run_snapshot = load_yaml_mapping(snapshot_path)
        else:
            run_snapshot = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "config": config_json,
                "dataset_overrides": self._dataset_overrides(layout),
                "backend_manifest": self.runtime.manifest,
                "factory_version": __version__,
            }
            dump_yaml_mapping(snapshot_path, run_snapshot)
            dump_yaml_mapping(layout.run_snapshot, run_snapshot)
        token = CancellationToken()
        self._cancellation = token
        bus = EventBus()

        def publish(event: PipelineEvent) -> None:
            payload = {
                "event_type": event.event_type,
                "stage": event.stage or "",
                "message": event.message,
                "overall_progress": event.progress,
                "details": _jsonable(event.details),
                "created_at": event.created_at.isoformat(),
            }
            with database.session() as session:
                session.add(
                    EventRow(
                        run_id=run_id,
                        event_type=event.event_type,
                        stage=event.stage,
                        message=event.message,
                        details_json=_jsonable(event.details),
                    )
                )
                row = session.get(RunRow, run_id)
                if row is not None and event.stage:
                    row.current_stage = event.stage
            emit(payload)

        bus.subscribe("*", publish)
        with database.session() as session:
            project = session.get(ProjectRow, config.project_id)
            if project is None:
                session.add(
                    ProjectRow(
                        id=config.project_id,
                        name=config.lora_name,
                        root_path=str(layout.root),
                        config_json=config_json,
                    )
                )
            else:
                project.name = config.lora_name
                project.config_json = config_json
            run = session.get(RunRow, run_id)
            if run is None:
                session.add(
                    RunRow(
                        id=run_id,
                        project_id=config.project_id,
                        status=RunStatus.ACTIVE.value,
                        snapshot_json=run_snapshot,
                    )
                )
            else:
                run.status = RunStatus.ACTIVE.value
                run.error_json = None

        context = PipelineContext(
            run_id=run_id,
            config=config,
            layout=layout,
            events=bus,
            cancellation=token,
            artifacts={"_RUN_SNAPSHOT": run_snapshot},
        )
        engine = PipelineEngine(
            StageRepository(database),
            on_stage_success=ApplicationArtifactStore(database).persist,
        )
        try:
            stages = self._stages(context)
            barrier = next(
                index
                for index, definition in enumerate(stages)
                if definition.stage is PipelineStage.CODEX_REFINEMENT
            )
            engine.execute(context, stages[: barrier + 1])
            review = self._prepare_refinement_review(context)
            approval_path = run_layout.root / "refinement-approval.json"
            approval: RefinementApproval | None = None
            if approval_path.is_file():
                approval = RefinementApproval.model_validate(read_json(approval_path))
                if approval.upstream_fingerprint != review.upstream_fingerprint:
                    approval = None
            if (review.requires_refinement_review or review.requires_trigger_selection) and (
                approval is None
            ):
                write_json_atomic(
                    run_layout.root / "refinement-review.json",
                    review.model_dump(mode="json"),
                )
                with database.session() as session:
                    row = session.get(RunRow, run_id)
                    if row is not None:
                        row.status = RunStatus.AWAITING_REVIEW.value
                        row.current_stage = PipelineStage.CODEX_REFINEMENT.value
                        row.error_json = None
                bus.publish(
                    PipelineEvent(
                        event_type="awaiting_review",
                        stage=PipelineStage.CODEX_REFINEMENT.value,
                        message="Trigger Word or caption/tag approval is required before training",
                        progress=(barrier + 1) / len(stages),
                        details={
                            "run_id": run_id,
                            "requires_trigger_selection": review.requires_trigger_selection,
                            "requires_refinement_review": review.requires_refinement_review,
                        },
                    )
                )
                return {
                    "project_id": config.project_id,
                    "run_id": run_id,
                    "status": RunStatus.AWAITING_REVIEW.value,
                    "requires_trigger_selection": review.requires_trigger_selection,
                    "requires_refinement_review": review.requires_refinement_review,
                }
            if approval is None:
                approval = RefinementApproval(
                    upstream_fingerprint=review.upstream_fingerprint,
                    trigger_word=review.trigger_word,
                )
            self._apply_refinement_approval(context, review, approval)
            if run_snapshot.get("confirmed_trigger_word") != approval.trigger_word:
                run_snapshot = {
                    **run_snapshot,
                    "confirmed_trigger_word": approval.trigger_word,
                    "refinement_approval_fingerprint": approval.upstream_fingerprint,
                }
                dump_yaml_mapping(snapshot_path, run_snapshot)
                dump_yaml_mapping(layout.run_snapshot, run_snapshot)
                with database.session() as session:
                    row = session.get(RunRow, run_id)
                    if row is not None:
                        row.snapshot_json = run_snapshot
            result = engine.execute(context, stages[barrier + 1 :])
            ready = result.outputs[PipelineStage.READY.value]
            with database.session() as session:
                row = session.get(RunRow, run_id)
                if row is not None:
                    row.status = RunStatus.COMPLETED.value
                    row.current_stage = PipelineStage.READY.value
            write_json_atomic(layout.root / "completion.json", ready)
            write_json_atomic(run_layout.root / "completion.json", ready)
            return ready
        except BaseException as exc:
            with database.session() as session:
                row = session.get(RunRow, run_id)
                if row is not None:
                    recoverable = isinstance(exc, PipelineError) and exc.recoverable
                    row.status = (
                        RunStatus.CANCELLED.value
                        if token.cancelled
                        else RunStatus.FAILED_RECOVERABLE.value
                        if recoverable
                        else RunStatus.FAILED_FATAL.value
                    )
                    row.error_json = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "recoverable": recoverable,
                    }
            raise
        finally:
            self._cancellation = None

    def _stages(self, context: PipelineContext) -> tuple[StageDefinition, ...]:
        config = context.config
        fake = config.backend_mode is BackendMode.FAKE
        backend_versions = {
            "tagger": "fake-wd14/1" if fake else str(self.runtime.manifest["wd14"]["revision"]),
            "trainer": "fake-trainer/1"
            if fake
            else str(self.runtime.manifest["sd_scripts"]["commit"]),
            "sampler": "fake-sampler/1"
            if fake
            else str(self.runtime.manifest["sd_scripts"]["commit"]),
            "image_embedding": (
                "fake-image-embedding/1"
                if fake
                else ":".join(
                    (
                        str(self.runtime.manifest["image_embedding"]["revision"]),
                        str(
                            self.runtime.manifest["image_embedding"]["artifacts"][
                                "model.safetensors"
                            ]["sha256"]
                        ),
                    )
                )
            ),
        }

        def definition(
            stage: PipelineStage,
            runner: Callable[[PipelineContext], dict[str, Any]],
            dependencies: tuple[PipelineStage, ...] = (),
        ) -> StageDefinition:
            def resolve_inputs(ctx: PipelineContext) -> Mapping[str, Any]:
                return {item.value: ctx.artifacts.get(item.value, {}) for item in dependencies}

            return StageDefinition(
                stage=stage,
                version="1",
                resolve_inputs=resolve_inputs,
                run=runner,
                backend_versions=backend_versions,
            )

        def refinement_inputs(ctx: PipelineContext) -> Mapping[str, Any]:
            try:
                accepted = {
                    str(value)
                    for value in ctx.artifacts.get(PipelineStage.DEDUPLICATING.value, {}).get(
                        "accepted_asset_ids", ()
                    )
                }
                normalized = ctx.artifacts.get(PipelineStage.NORMALIZING.value, {}).get("items", ())
                working_sha256s = {
                    str(item["asset_id"]): sha256_file(Path(str(item["path"])))
                    for item in normalized
                    if str(item["asset_id"]) in accepted
                }
                return {
                    PipelineStage.CAPTION_DRAFTING.value: ctx.artifacts.get(
                        PipelineStage.CAPTION_DRAFTING.value, {}
                    ),
                    "accepted_asset_ids": sorted(accepted),
                    "working_sha256s": working_sha256s,
                }
            except Exception as exc:
                raise self._refinement_pipeline_error(exc) from exc

        return (
            definition(PipelineStage.IMPORTING, self._import_stage),
            definition(
                PipelineStage.NORMALIZING,
                self._normalize_stage,
                (PipelineStage.IMPORTING,),
            ),
            definition(
                PipelineStage.ANALYZING,
                self._analyze_stage,
                (PipelineStage.NORMALIZING,),
            ),
            definition(
                PipelineStage.DEDUPLICATING,
                self._deduplicate_stage,
                (PipelineStage.ANALYZING,),
            ),
            definition(
                PipelineStage.TAGGING,
                self._tag_stage,
                (PipelineStage.DEDUPLICATING,),
            ),
            definition(
                PipelineStage.CAPTION_DRAFTING,
                self._caption_draft_stage,
                (PipelineStage.TAGGING, PipelineStage.DEDUPLICATING),
            ),
            StageDefinition(
                stage=PipelineStage.CODEX_REFINEMENT,
                version="2",
                resolve_inputs=refinement_inputs,
                run=self._codex_refinement_stage,
                backend_versions=backend_versions,
            ),
            definition(
                PipelineStage.CAPTIONING,
                self._caption_stage,
                (PipelineStage.CAPTION_DRAFTING, PipelineStage.CODEX_REFINEMENT),
            ),
            definition(
                PipelineStage.DATASET_REVIEW,
                self._dataset_review_stage,
                (PipelineStage.CAPTIONING,),
            ),
            definition(
                PipelineStage.PLANNING,
                self._planning_stage,
                (PipelineStage.DATASET_REVIEW,),
            ),
            definition(
                PipelineStage.CODEX_PRETRAIN_REVIEW,
                self._codex_pretrain_stage,
                (PipelineStage.PLANNING,),
            ),
            definition(
                PipelineStage.PREFLIGHT,
                self._preflight_stage,
                (PipelineStage.CODEX_PRETRAIN_REVIEW,),
            ),
            definition(
                PipelineStage.TRAINING,
                self._training_stage,
                (PipelineStage.PREFLIGHT,),
            ),
            definition(
                PipelineStage.SCREENING,
                lambda ctx: self._sampling_stage(ctx, pass_number=1),
                (PipelineStage.TRAINING,),
            ),
            definition(
                PipelineStage.WEIGHT_SWEEP,
                lambda ctx: self._sampling_stage(ctx, pass_number=2),
                (PipelineStage.SCREENING,),
            ),
            definition(
                PipelineStage.MULTI_SEED_VALIDATION,
                lambda ctx: self._sampling_stage(ctx, pass_number=3),
                (PipelineStage.WEIGHT_SWEEP,),
            ),
            definition(
                PipelineStage.FINAL_EVALUATION,
                self._evaluation_stage,
                (PipelineStage.MULTI_SEED_VALIDATION,),
            ),
            definition(
                PipelineStage.CODEX_FINAL_REVIEW,
                self._codex_final_stage,
                (PipelineStage.FINAL_EVALUATION,),
            ),
            definition(
                PipelineStage.SELECTING,
                self._selection_stage,
                (PipelineStage.CODEX_FINAL_REVIEW,),
            ),
            definition(
                PipelineStage.PACKAGING,
                self._packaging_stage,
                (PipelineStage.SELECTING,),
            ),
            definition(PipelineStage.READY, self._ready_stage, (PipelineStage.PACKAGING,)),
        )

    def _import_stage(self, context: PipelineContext) -> dict[str, Any]:
        result = ImmutableImportService(
            context.layout, project_id=context.config.project_id
        ).import_paths(
            context.config.input_paths,
            recursive=bool(context.config.advanced.recursive_import),
            check_cancelled=context.cancellation.raise_if_cancelled,
        )
        if result.failures or result.scan_issues:
            context.events.publish(
                PipelineEvent(
                    event_type="warning",
                    stage=PipelineStage.IMPORTING.value,
                    message=(
                        f"Import completed with {len(result.failures)} failures and "
                        f"{len(result.scan_issues)} scan issues"
                    ),
                )
            )
        if not result.manifest.raw_assets:
            raise ValueError("No supported images were imported")
        raw_snapshot = verify_raw_store(context.layout, result.manifest)
        return {
            "asset_count": len(result.manifest.raw_assets),
            "imported_asset_ids": list(result.imported_asset_ids),
            "reused_asset_ids": list(result.reused_asset_ids),
            "failure_count": len(result.failures),
            "manifest": str(context.layout.manifest),
            "raw_snapshot": raw_snapshot,
        }

    def _manifest(self, context: PipelineContext) -> DatasetManifest:
        return DatasetManifest.model_validate(read_json(context.layout.manifest))

    def _normalize_stage(self, context: PipelineContext) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        raw_assets = self._manifest(context).raw_assets
        total = len(raw_assets)
        for index, asset in enumerate(raw_assets, start=1):
            context.cancellation.raise_if_cancelled()
            source = context.layout.raw / asset.stored_filename
            destination = context.layout.working / f"{asset.asset_id}.png"
            normalized = normalize_image(source, destination)
            items.append(
                {
                    "asset_id": asset.asset_id,
                    "path": str(normalized.output_path),
                    "original_filename": asset.sources[0].original_filename,
                    "width": normalized.width,
                    "height": normalized.height,
                    "sha256": normalized.output_sha256,
                    "warnings": list(normalized.warnings),
                }
            )
            context.events.publish(
                PipelineEvent(
                    event_type="normalize_progress",
                    stage=PipelineStage.NORMALIZING.value,
                    message=f"Normalized image {index}/{total}",
                    progress=index / max(total, 1),
                    details=progress_details(
                        context,
                        interval_seconds=self.settings.telemetry_interval_seconds,
                        current_task="NORMALIZING (CPU)",
                        images_current=index,
                        images_total=total,
                    ),
                )
            )
        return {"items": items, "count": len(items)}

    def _analyze_stage(self, context: PipelineContext) -> dict[str, Any]:
        normalized = _stage_output(context, PipelineStage.NORMALIZING)
        assessments = [assess_image(Path(item["path"])) for item in normalized["items"]]
        report = build_dataset_report(assessments)
        items = []
        for source, assessment in zip(normalized["items"], assessments, strict=True):
            items.append(
                {
                    **source,
                    "assessment": assessment.model_dump(mode="json"),
                    "accepted": assessment.accepted,
                    "reasons": [reason.message for reason in assessment.reasons],
                }
            )
        return {"items": items, "report": report.model_dump(mode="json")}

    def _deduplicate_stage(self, context: PipelineContext) -> dict[str, Any]:
        analyzed = _stage_output(context, PipelineStage.ANALYZING)
        overrides = self._run_dataset_overrides(context)
        forced_included = {
            asset_id for asset_id, override in overrides.items() if override.get("included") is True
        }
        forced_excluded = {
            asset_id
            for asset_id, override in overrides.items()
            if override.get("included") is False
        }
        usable = [
            item
            for item in analyzed["items"]
            if (item["accepted"] or item["asset_id"] in forced_included)
            and item["asset_id"] not in forced_excluded
        ]
        candidates = [
            DuplicateCandidate.from_assessment(
                item["asset_id"],
                QualityAssessment.model_validate(item["assessment"]),
                sha256=item["sha256"],
            )
            for item in usable
        ]
        report = detect_duplicates(candidates)
        excluded = {
            asset_id for cluster in report.clusters for asset_id in cluster.rejected_exact_ids
        }
        accepted_ids = [
            item["asset_id"]
            for item in usable
            if item["asset_id"] not in excluded or item["asset_id"] in forced_included
        ]
        cluster_ids = {
            member: cluster.cluster_id for cluster in report.clusters for member in cluster.members
        }
        return {
            "accepted_asset_ids": accepted_ids,
            "excluded_exact_asset_ids": sorted(excluded),
            "forced_included_asset_ids": sorted(forced_included & set(accepted_ids)),
            "forced_excluded_asset_ids": sorted(forced_excluded),
            "cluster_ids": cluster_ids,
            "largest_cluster_ratio": report.largest_cluster_ratio,
            "clusters": [item.model_dump(mode="json") for item in report.clusters],
        }

    def _tagger(
        self,
        context: PipelineContext,
        *,
        binding: GpuBinding | None = None,
        work_directory: Path | None = None,
    ) -> TaggerProtocol:
        if context.config.backend_mode is BackendMode.FAKE:
            return FakeTagger()
        if not self.runtime.installed():
            raise RuntimeError("Managed runtime is not installed; complete Setup first")
        if binding is None:
            raise RuntimeError("A selected-GPU lease binding is required for real WD14 tagging")
        wd14 = self.runtime.manifest["wd14"]
        return ManagedWD14Tagger(
            python_executable=self.runtime.layout.python,
            helper_script=repository_root()
            / "src"
            / "lora_factory"
            / "runtime_scripts"
            / "wd14_infer.py",
            model_path=self.runtime.layout.wd14 / "model.onnx",
            tags_path=self.runtime.layout.wd14 / "selected_tags.csv",
            revision=str(wd14["revision"]),
            binding=binding,
            work_directory=work_directory or context.layout.run(context.run_id).root / "tagging",
            cancellation=context.cancellation,
            general_threshold=float(wd14["general_threshold"]),
            character_threshold=float(wd14["character_threshold"]),
        )

    @contextmanager
    def _tagger_session(
        self,
        context: PipelineContext,
        *,
        task: GpuTaskKind,
        work_directory: Path | None = None,
        preferred_gpu_uuid: str | None = None,
    ) -> Iterator[tuple[TaggerProtocol, str]]:
        selected = context.config.selected_gpu_uuids
        task_offset = 1 if task is GpuTaskKind.GENERATED_TAG else 0
        preferred = preferred_gpu_uuid or selected[task_offset % len(selected)]
        if preferred not in selected:
            raise ValueError("Preferred tagging GPU is outside the selected pool")
        if context.config.backend_mode is BackendMode.FAKE:
            yield (
                self._tagger(context, work_directory=work_directory),
                preferred,
            )
            return
        with lease_selected_gpu(
            database=Database(context.layout.database),
            selected_uuids=context.config.selected_gpu_uuids,
            run_id=context.run_id,
            task=task,
            estimated_vram_mb=4096,
            preferred_gpu_uuid=preferred,
        ) as (binding, _lease):
            yield (
                self._tagger(context, binding=binding, work_directory=work_directory),
                binding.uuid,
            )

    def _tag_many_sharded(
        self,
        context: PipelineContext,
        *,
        task: GpuTaskKind,
        paths: Sequence[Path],
        asset_ids: Sequence[str],
        work_directory: Path,
    ) -> _TagBatch:
        if len(paths) != len(asset_ids):
            raise ValueError("asset_ids must align with paths")
        if not paths:
            raise ValueError("At least one image is required for tagging")
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("Tagging asset IDs must be unique")

        selected = context.config.selected_gpu_uuids
        if context.config.backend_mode is BackendMode.FAKE:
            shard_ids: dict[str, tuple[str, ...]] = {
                gpu_uuid: tuple(
                    asset_id
                    for index, asset_id in enumerate(asset_ids)
                    if selected[index % len(selected)] == gpu_uuid
                )
                for gpu_uuid in selected
            }
            shard_ids = {uuid: ids for uuid, ids in shard_ids.items() if ids}
        else:
            shard_ids = SelectedGpuScheduler(selected).shard_items(
                asset_ids,
                discover_nvidia_gpus(),
            )

        index_by_id = {asset_id: index for index, asset_id in enumerate(asset_ids)}
        ordered: list[ImageTagResult | None] = [None] * len(paths)
        observed_uuids: list[str] = []
        backend_identity: set[tuple[str, str]] = set()

        def run_shard(
            preferred_uuid: str, ids: tuple[str, ...]
        ) -> tuple[tuple[tuple[int, ImageTagResult], ...], str, str, str]:
            indices = tuple(index_by_id[asset_id] for asset_id in ids)
            shard_directory = work_directory / preferred_uuid.replace("-", "_")
            with self._tagger_session(
                context,
                task=task,
                work_directory=shard_directory,
                preferred_gpu_uuid=preferred_uuid,
            ) as (tagger, actual_uuid):
                results = tagger.tag_many(
                    [paths[index] for index in indices],
                    asset_ids=[asset_ids[index] for index in indices],
                )
                if len(results) != len(indices):
                    raise RuntimeError("Tagger shard returned an invalid result count")
                return (
                    tuple(zip(indices, results, strict=True)),
                    actual_uuid,
                    tagger.model_id,
                    tagger.revision,
                )

        with ThreadPoolExecutor(
            max_workers=len(shard_ids),
            thread_name_prefix=f"lora-factory-{task.value.lower()}",
        ) as executor:
            futures = {
                executor.submit(run_shard, gpu_uuid, ids): gpu_uuid
                for gpu_uuid, ids in shard_ids.items()
            }
            for future in as_completed(futures):
                indexed, actual_uuid, model_id, revision = future.result()
                if actual_uuid not in selected:
                    raise RuntimeError("Tagger used a GPU outside the selected pool")
                observed_uuids.append(actual_uuid)
                backend_identity.add((model_id, revision))
                for index, result in indexed:
                    ordered[index] = result

        if any(result is None for result in ordered):
            raise RuntimeError("Tagger shard results did not cover every input image")
        if len(backend_identity) != 1:
            raise RuntimeError("Tagger shards used inconsistent backend versions")
        model_id, revision = backend_identity.pop()
        return _TagBatch(
            results=tuple(result for result in ordered if result is not None),
            gpu_uuids=tuple(uuid for uuid in selected if uuid in observed_uuids),
            model_id=model_id,
            revision=revision,
        )

    def _embedding_backend(
        self,
        context: PipelineContext,
        *,
        binding: GpuBinding | None = None,
        work_directory: Path,
    ) -> ImageEmbeddingBackend:
        if context.config.backend_mode is BackendMode.FAKE:
            return FakeImageEmbeddingBackend()
        if not self.runtime.installed():
            raise RuntimeError(
                "Managed runtime or pinned CLIP image embedding model is incomplete; "
                "repair Managed Training Runtime in Setup"
            )
        if binding is None:
            raise RuntimeError("A selected-GPU lease binding is required for real CLIP embedding")
        embedding = self.runtime.manifest["image_embedding"]
        artifacts = embedding["artifacts"]
        return ManagedClipImageEmbeddingBackend(
            python_executable=self.runtime.layout.python,
            helper_script=repository_root()
            / "src"
            / "lora_factory"
            / "runtime_scripts"
            / "clip_embed.py",
            model_directory=self.runtime.layout.image_embedding,
            model_id=str(embedding["model_id"]),
            revision=str(embedding["revision"]),
            config_sha256=str(artifacts["config.json"]["sha256"]),
            model_sha256=str(artifacts["model.safetensors"]["sha256"]),
            binding=binding,
            work_directory=work_directory,
            cancellation=context.cancellation,
        )

    @contextmanager
    def _embedding_session(
        self,
        context: PipelineContext,
        *,
        task: GpuTaskKind,
        work_directory: Path,
        preferred_gpu_uuid: str,
    ) -> Iterator[tuple[ImageEmbeddingBackend, str]]:
        if task not in {GpuTaskKind.REFERENCE_EMBED, GpuTaskKind.GENERATED_EMBED}:
            raise ValueError(f"Unsupported image embedding task: {task.value}")
        selected = context.config.selected_gpu_uuids
        if preferred_gpu_uuid not in selected:
            raise ValueError("Preferred embedding GPU is outside the selected pool")
        if context.config.backend_mode is BackendMode.FAKE:
            yield (
                self._embedding_backend(context, work_directory=work_directory),
                preferred_gpu_uuid,
            )
            return
        with lease_selected_gpu(
            database=Database(context.layout.database),
            selected_uuids=selected,
            run_id=context.run_id,
            task=task,
            estimated_vram_mb=5120,
            preferred_gpu_uuid=preferred_gpu_uuid,
        ) as (binding, _lease):
            yield (
                self._embedding_backend(
                    context,
                    binding=binding,
                    work_directory=work_directory,
                ),
                binding.uuid,
            )

    def _embed_many_sharded(
        self,
        context: PipelineContext,
        *,
        task: GpuTaskKind,
        paths: Sequence[Path],
        work_directory: Path,
    ) -> _EmbeddingShardedBatch:
        if not paths:
            raise ValueError("At least one image is required for embedding")
        resolved = tuple(path.resolve(strict=True) for path in paths)
        unique_paths = tuple(dict.fromkeys(resolved))
        item_ids = tuple(f"image-{index:06d}" for index in range(len(unique_paths)))
        selected = context.config.selected_gpu_uuids
        if context.config.backend_mode is BackendMode.FAKE:
            shard_ids: dict[str, tuple[str, ...]] = {
                gpu_uuid: tuple(
                    item_id
                    for index, item_id in enumerate(item_ids)
                    if selected[index % len(selected)] == gpu_uuid
                )
                for gpu_uuid in selected
            }
            shard_ids = {uuid: ids for uuid, ids in shard_ids.items() if ids}
        else:
            shard_ids = SelectedGpuScheduler(selected).shard_items(
                item_ids,
                discover_nvidia_gpus(),
            )
        index_by_id = {item_id: index for index, item_id in enumerate(item_ids)}
        ordered: list[ImageEmbedding | None] = [None] * len(unique_paths)
        observed_uuids: list[str] = []
        backend_identity: set[tuple[str, str, int, str]] = set()

        def run_shard(
            preferred_uuid: str, ids: tuple[str, ...]
        ) -> tuple[tuple[tuple[int, ImageEmbedding], ...], str, str, str, int, str]:
            indices = tuple(index_by_id[item_id] for item_id in ids)
            shard_directory = work_directory / preferred_uuid.replace("-", "_")
            with self._embedding_session(
                context,
                task=task,
                work_directory=shard_directory,
                preferred_gpu_uuid=preferred_uuid,
            ) as (backend, actual_uuid):
                batch = backend.embed_many([unique_paths[index] for index in indices])
                if len(batch.items) != len(indices):
                    raise RuntimeError("Embedding shard returned an invalid result count")
                return (
                    tuple(zip(indices, batch.items, strict=True)),
                    actual_uuid,
                    batch.model_id,
                    batch.revision,
                    batch.dimension,
                    batch.device,
                )

        with ThreadPoolExecutor(
            max_workers=len(shard_ids),
            thread_name_prefix=f"lora-factory-{task.value.lower()}",
        ) as executor:
            futures = {
                executor.submit(run_shard, gpu_uuid, ids): gpu_uuid
                for gpu_uuid, ids in shard_ids.items()
            }
            for future in as_completed(futures):
                indexed, actual_uuid, model_id, revision, dimension, device = future.result()
                if actual_uuid not in selected:
                    raise RuntimeError("Embedding used a GPU outside the selected pool")
                observed_uuids.append(actual_uuid)
                backend_identity.add((model_id, revision, dimension, device))
                for index, result in indexed:
                    ordered[index] = result

        if any(result is None for result in ordered):
            raise RuntimeError("Embedding shard results did not cover every unique input image")
        if len(backend_identity) != 1:
            raise RuntimeError("Embedding shards used inconsistent model identities or dimensions")
        model_id, revision, dimension, device = backend_identity.pop()
        return _EmbeddingShardedBatch(
            result=ImageEmbeddingBatch(
                model_id=model_id,
                revision=revision,
                dimension=dimension,
                device=device,
                items=tuple(result for result in ordered if result is not None),
            ),
            gpu_uuids=tuple(uuid for uuid in selected if uuid in observed_uuids),
        )

    def _tag_stage(self, context: PipelineContext) -> dict[str, Any]:
        dedup = _stage_output(context, PipelineStage.DEDUPLICATING)
        accepted = set(dedup["accepted_asset_ids"])
        normalized = _stage_output(context, PipelineStage.NORMALIZING)
        items = [item for item in normalized["items"] if item["asset_id"] in accepted]
        total = len(items)
        first_gpu_uuid = context.config.selected_gpu_uuids[0]
        context.events.publish(
            PipelineEvent(
                event_type="tag_progress",
                stage=PipelineStage.TAGGING.value,
                message=f"Tagging {total} images",
                progress=0.0,
                details=progress_details(
                    context,
                    interval_seconds=self.settings.telemetry_interval_seconds,
                    current_task="TAGGING",
                    images_current=0,
                    images_total=total,
                    active_gpu_uuid=first_gpu_uuid,
                ),
            )
        )
        batch = self._tag_many_sharded(
            context,
            task=GpuTaskKind.WD14_TAG,
            paths=[Path(item["path"]) for item in items],
            asset_ids=[str(item["asset_id"]) for item in items],
            work_directory=context.layout.run(context.run_id).root / "tagging",
        )
        results = batch.results
        context.events.publish(
            PipelineEvent(
                event_type="tag_progress",
                stage=PipelineStage.TAGGING.value,
                message=f"Tagged {len(results)}/{total} images on {len(batch.gpu_uuids)} GPU(s)",
                progress=1.0,
                details=progress_details(
                    context,
                    interval_seconds=self.settings.telemetry_interval_seconds,
                    current_task="TAGGING",
                    images_current=len(results),
                    images_total=total,
                    active_gpu_uuid=batch.gpu_uuids[0],
                ),
            )
        )
        store = RawTagStore(context.layout.dataset / "tags" / "raw")
        for result in results:
            context.cancellation.raise_if_cancelled()
            store.write(result)
        return {
            "model_id": batch.model_id,
            "revision": batch.revision,
            "gpu_uuid": batch.gpu_uuids[0],
            "gpu_uuids": list(batch.gpu_uuids),
            "results": [result.model_dump(mode="json") for result in results],
        }

    def _caption_draft_stage(self, context: PipelineContext) -> dict[str, Any]:
        tag_output = _stage_output(context, PipelineStage.TAGGING)
        image_tags = {
            item["asset_id"]: tuple(
                TagScore.model_validate(tag) for tag in item["tags"] if tag.get("selected", True)
            )
            for item in tag_output["results"]
        }
        cluster_ids = _stage_output(context, PipelineStage.DEDUPLICATING)["cluster_ids"]
        return build_caption_drafts(
            context.config.preset,
            image_tags,
            duplicate_cluster_ids=cluster_ids,
        ).model_dump(mode="json")

    def _codex_refinement_stage(self, context: PipelineContext) -> dict[str, Any]:
        try:
            return self._run_codex_refinement_stage(context)
        except CancelledError:
            raise
        except Exception as exc:
            raise self._refinement_pipeline_error(exc) from exc

    @staticmethod
    def _refinement_pipeline_error(exc: Exception) -> PipelineError:
        reason = redact_text(str(exc), home=Path.home())
        sanitized = sanitize_public_metadata(reason)
        if not isinstance(sanitized, str):
            sanitized = type(exc).__name__
        sanitized = " ".join(sanitized.split())[:1000] or type(exc).__name__
        return PipelineError(
            f"Runtime Codex image refinement failed: {sanitized}",
            recoverable=True,
        )

    def _run_codex_refinement_stage(self, context: PipelineContext) -> dict[str, Any]:
        drafts = CaptionDraftBatch.model_validate(
            _stage_output(context, PipelineStage.CAPTION_DRAFTING)
        )
        normalized_items = _stage_output(context, PipelineStage.NORMALIZING)["items"]
        normalized_by_asset: dict[str, dict[str, Any]] = {}
        for raw_item in normalized_items:
            normalized_item = dict(raw_item)
            asset_id = str(normalized_item["asset_id"])
            if asset_id in normalized_by_asset:
                raise ValueError(f"Normalized image mapping contains duplicate asset {asset_id}")
            normalized_by_asset[asset_id] = normalized_item
        accepted_ids = {
            str(value)
            for value in _stage_output(context, PipelineStage.DEDUPLICATING)["accepted_asset_ids"]
        }
        draft_ids = set(drafts.assets)
        if draft_ids != accepted_ids:
            raise ValueError("Caption drafts do not exactly cover accepted normalized images")
        normalized_by_asset = {
            asset_id: normalized_by_asset[asset_id]
            for asset_id in sorted(draft_ids)
            if asset_id in normalized_by_asset
        }
        if set(normalized_by_asset) != draft_ids:
            raise ValueError("Normalized image mapping does not exactly cover caption drafts")

        vocabulary = self._refinement_vocabulary(context)
        batches = batch_refinement_assets(
            tuple(drafts.assets.values()),
            max_items=_REFINEMENT_MAX_ITEMS,
            max_bytes=_REFINEMENT_DRAFT_BYTES,
        )
        decisions: dict[str, Any] = {}
        effective_tags: dict[str, list[str]] = {}
        added_tags: dict[str, list[str]] = {}
        removed_tags: dict[str, list[str]] = {}
        image_sha256s: dict[str, str] = {}
        working_sha256s: dict[str, str] = {}
        candidate_values: list[TriggerCandidate] = []
        audits: list[dict[str, Any]] = []
        warnings: list[str] = []
        fallback_used = False
        record_root = context.layout.run(context.run_id).root / "codex-refinement"
        total_image_count = len(drafts.assets)
        prepared_before_batch = 0

        for batch_index, batch in enumerate(batches):
            batch_key = f"{context.run_id}-batch-{batch_index:04d}"
            image_root = self.settings.codex_runtime_root / "input" / "images" / batch_key
            record_path = record_root / f"batch-{batch_index:04d}.json"
            prepared: list[PreparedCodexImage] = []
            validated = None
            response: DatasetRefinementResponse | None = None
            audit: dict[str, Any] = {}
            warning: str | None = None
            reused = False
            input_hash = ""
            try:
                prepared.extend(
                    self._prepare_codex_images_with_two_attempts(
                        context,
                        batch,
                        normalized_by_asset,
                        image_root,
                        batch_index=batch_index,
                        batch_count=len(batches),
                        prepared_before_batch=prepared_before_batch,
                        total_image_count=total_image_count,
                    )
                )
                payload = self._refinement_payload(
                    context,
                    batch,
                    prepared,
                    vocabulary,
                    batch_index=batch_index,
                    batch_count=len(batches),
                )
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(encoded) > _REFINEMENT_PAYLOAD_BYTES:
                    raise ValueError(
                        f"Sanitized refinement batch exceeds {_REFINEMENT_PAYLOAD_BYTES} bytes"
                    )
                input_hash = refinement_fingerprint(payload)
                cached = self._load_matching_refinement_batch(
                    record_path,
                    input_hash,
                    prepared,
                )
                if cached is not None:
                    response_json, audit, warning = cached
                    reused = True
                    self._publish_codex_batch_progress(
                        context,
                        batch_index=batch_index,
                        batch_count=len(batches),
                        action="cache_hit",
                        attempt=0,
                    )
                    sanitized_response = sanitize_public_metadata(response_json)
                    if not isinstance(sanitized_response, dict):
                        raise ValueError("Sanitized Runtime Codex response is not a mapping")
                    response = DatasetRefinementResponse.model_validate(sanitized_response)
                    proposals = tuple(
                        AssetRefinementProposal.model_validate(item.model_dump(mode="json"))
                        for item in response.assets
                    )
                    subset = drafts.model_copy(
                        update={"assets": {item.asset_id: item for item in batch}}
                    )
                    validated = validate_refinement_proposals(
                        subset,
                        proposals,
                        vocabulary=vocabulary,
                    )
                else:
                    last_contract_error: ValueError | None = None
                    for contract_attempt in range(1, 3):
                        self._publish_codex_batch_progress(
                            context,
                            batch_index=batch_index,
                            batch_count=len(batches),
                            action="call" if contract_attempt == 1 else "retry",
                            attempt=contract_attempt,
                        )
                        response_json, audit, warning = self._codex_review(
                            context,
                            CodexTaskType.DATASET_REFINEMENT,
                            payload,
                            images=prepared,
                            allow_fallback=False,
                        )
                        try:
                            sanitized_response = sanitize_public_metadata(response_json)
                            if not isinstance(sanitized_response, dict):
                                raise ValueError(
                                    "Sanitized Runtime Codex response is not a mapping"
                                )
                            response = DatasetRefinementResponse.model_validate(sanitized_response)
                            proposals = tuple(
                                AssetRefinementProposal.model_validate(item.model_dump(mode="json"))
                                for item in response.assets
                            )
                            subset = drafts.model_copy(
                                update={"assets": {item.asset_id: item for item in batch}}
                            )
                            validated = validate_refinement_proposals(
                                subset,
                                proposals,
                                vocabulary=vocabulary,
                            )
                        except ValueError as exc:
                            last_contract_error = exc
                            continue
                        break
                    if validated is None or response is None:
                        raise last_contract_error or ValueError(
                            f"Batch {batch_index} failed the Runtime Codex asset contract"
                        )

                path_free_audit = sanitize_public_metadata(audit)
                if not isinstance(path_free_audit, dict):
                    raise ValueError("Sanitized Runtime Codex audit is not a mapping")
                audit = {
                    **path_free_audit,
                    "batch_index": batch_index,
                    "input_hash": input_hash,
                    "payload_byte_count": len(encoded),
                }
                if reused:
                    audit["reused"] = True
                audits.append(audit)
                fallback_used = fallback_used or bool(audit.get("fallback_used", False))
                path_free_warning: str | None = None
                if warning:
                    sanitized_warning = sanitize_public_metadata(
                        redact_text(warning, home=Path.home())
                    )
                    path_free_warning = (
                        sanitized_warning
                        if isinstance(sanitized_warning, str)
                        else "Runtime Codex returned a warning"
                    )
                    warnings.append(f"batch {batch_index}: {path_free_warning}")
                if not context.config.trigger_token:
                    candidate_values.extend(
                        TriggerCandidate.model_validate(item.model_dump(mode="json"))
                        for item in response.trigger_word_candidates
                    )
                if not reused:
                    write_json_atomic(
                        record_path,
                        {
                            "schema_version": 1,
                            "complete": True,
                            "input_hash": input_hash,
                            "payload_byte_count": len(encoded),
                            "image_profile": prepared[0].profile.model_dump(mode="json"),
                            "image_sha256s": [item.output_sha256 for item in prepared],
                            "working_sha256s": [item.source_sha256 for item in prepared],
                            "response": response.model_dump(mode="json"),
                            "audit": audit,
                            "warning": path_free_warning,
                        },
                    )
                for item in prepared:
                    image_sha256s[item.asset_id] = item.output_sha256
                    working_sha256s[item.asset_id] = item.source_sha256
            finally:
                remove_codex_images(prepared, scratch_root=image_root)
                remove_empty_image_root(
                    image_root,
                    boundary=self.settings.codex_runtime_root,
                )

            if validated is None:
                raise ValueError(f"Batch {batch_index} did not produce validated decisions")
            for asset_id, decision in validated.decisions.items():
                decisions[asset_id] = decision.model_dump(mode="json")
                effective = tuple(validated.effective_tags[asset_id])
                effective_tags[asset_id] = list(effective)
                original = tuple(drafts.assets[asset_id].original_tags)
                original_set = set(original)
                effective_set = set(effective)
                added_tags[asset_id] = [tag for tag in effective if tag not in original_set]
                removed_tags[asset_id] = [tag for tag in original if tag not in effective_set]
            prepared_before_batch += len(batch)

        if (
            context.config.backend_mode is BackendMode.FAKE
            and context.config.trigger_word_mode is TriggerWordMode.CODEX_SUGGEST
            and not context.config.trigger_token
        ):
            seed = hashlib.sha256(context.config.project_id.encode("utf-8")).hexdigest()[:8]
            candidate_values.extend(
                TriggerCandidate(
                    value=f"lfx_{seed}_{suffix}",
                    reason="Deterministic Fake backend Trigger Word candidate.",
                )
                for suffix in ("nova", "ember", "quartz", "cobalt")
            )
        candidates: tuple[TriggerCandidate, ...] = ()
        if candidate_values:
            try:
                candidates = normalize_trigger_candidates(candidate_values)
            except ValueError as exc:
                warnings.append(str(exc))
        return {
            "effective_tags": effective_tags,
            "decisions": decisions,
            "added_tags": added_tags,
            "removed_tags": removed_tags,
            "trigger_candidates": [item.model_dump(mode="json") for item in candidates],
            "audits": audits,
            "warnings": warnings,
            "fallback_used": fallback_used,
            "batch_count": len(batches),
            "chunk_count": len(batches),
            "image_profile": DEFAULT_CODEX_IMAGE_PROFILE.model_dump(mode="json"),
            "image_sha256s": image_sha256s,
            "working_sha256s": working_sha256s,
            "ordered_image_hashes": [image_sha256s[asset_id] for asset_id in sorted(image_sha256s)],
            "cleanup_status": "removed_after_each_call",
        }

    def _refinement_vocabulary(self, context: PipelineContext) -> WD14TagVocabulary:
        return self._refinement_vocabulary_for_config(context.config)

    def _refinement_vocabulary_for_config(
        self,
        config: ProjectConfig,
    ) -> WD14TagVocabulary:
        if config.backend_mode is BackendMode.FAKE:
            return WD14TagVocabulary.fake()
        return WD14TagVocabulary.from_csv(self.runtime.layout.wd14 / "selected_tags.csv")

    def _prepare_codex_images_with_two_attempts(
        self,
        context: PipelineContext,
        batch: Sequence[CaptionDraftAsset],
        normalized_by_asset: Mapping[str, Mapping[str, Any]],
        image_root: Path,
        *,
        batch_index: int,
        batch_count: int,
        prepared_before_batch: int,
        total_image_count: int,
    ) -> tuple[PreparedCodexImage, ...]:
        for attempt in range(1, 3):
            partial: list[PreparedCodexImage] = []
            try:
                for image_index, draft in enumerate(batch, start=1):
                    context.cancellation.raise_if_cancelled()
                    item = normalized_by_asset.get(draft.asset_id)
                    if item is None:
                        raise ValueError(
                            f"Normalized image is missing for draft asset {draft.asset_id}"
                        )
                    source_path = Path(str(item["path"])).resolve(strict=True)
                    expected_source = (context.layout.working / f"{draft.asset_id}.png").resolve(
                        strict=True
                    )
                    if source_path != expected_source:
                        raise ValueError("Normalized asset/image mapping does not match")
                    self._publish_codex_image_progress(
                        context,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        image_index=image_index,
                        image_count=len(batch),
                        images_current=prepared_before_batch + image_index,
                        images_total=total_image_count,
                        asset_id=draft.asset_id,
                        action="preparing",
                        attempt=attempt,
                    )
                    prepared = prepare_codex_image(
                        draft.asset_id,
                        source_path,
                        image_root,
                    )
                    partial.append(prepared)
                    prepared = PreparedCodexImage.model_validate(prepared.model_dump(mode="python"))
                    if prepared.asset_id != draft.asset_id:
                        raise ValueError("Prepared image asset mapping does not match its draft")
                    partial[-1] = prepared
                    self._publish_codex_image_progress(
                        context,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        image_index=image_index,
                        image_count=len(batch),
                        images_current=prepared_before_batch + image_index,
                        images_total=total_image_count,
                        asset_id=draft.asset_id,
                        action="prepared",
                        attempt=attempt,
                    )
                return tuple(partial)
            except CancelledError:
                remove_codex_images(partial, scratch_root=image_root)
                raise
            except Exception:
                remove_codex_images(partial, scratch_root=image_root)
                if attempt == 2:
                    raise
                self._publish_codex_batch_progress(
                    context,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    action="image_retry",
                    attempt=attempt + 1,
                )
        raise RuntimeError("Image preparation attempts were exhausted")

    def _refinement_payload(
        self,
        context: PipelineContext,
        batch: Sequence[CaptionDraftAsset],
        prepared: Sequence[PreparedCodexImage],
        vocabulary: WD14TagVocabulary,
        *,
        batch_index: int,
        batch_count: int,
    ) -> dict[str, Any]:
        if len(batch) != len(prepared) or not prepared:
            raise ValueError("Refinement assets and prepared images must align exactly")
        if [item.asset_id for item in batch] != [item.asset_id for item in prepared]:
            raise ValueError("Refinement asset/image mapping is out of order")
        return {
            "schema_version": 1,
            "preset": context.config.preset.value,
            "ontology_version": 1,
            "batch_index": batch_index,
            "batch_count": batch_count,
            "generate_trigger_candidates": (
                context.config.trigger_word_mode is TriggerWordMode.CODEX_SUGGEST
                and not context.config.trigger_token
            ),
            "vocabulary_sha256": refinement_fingerprint(sorted(vocabulary.tags)),
            "assets": [item.model_dump(mode="json") for item in batch],
            "images": [
                {
                    "asset_id": item.asset_id,
                    "relative_name": item.relative_name,
                    "working_sha256": item.source_sha256,
                    "image_sha256": item.output_sha256,
                    "width": item.width,
                    "height": item.height,
                    "quality": item.quality,
                    "byte_count": item.byte_count,
                    "profile": item.profile.model_dump(mode="json"),
                }
                for item in prepared
            ],
        }

    def _load_matching_refinement_batch(
        self,
        record_path: Path,
        input_hash: str,
        prepared: Sequence[PreparedCodexImage],
    ) -> tuple[dict[str, Any], dict[str, Any], str | None] | None:
        if not record_path.is_file():
            return None
        loaded = read_json(record_path)
        if not isinstance(loaded, dict):
            return None
        expected_profile = prepared[0].profile.model_dump(mode="json")
        if (
            loaded.get("complete") is not True
            or loaded.get("input_hash") != input_hash
            or not isinstance(loaded.get("payload_byte_count"), int)
            or int(loaded["payload_byte_count"]) > _REFINEMENT_PAYLOAD_BYTES
            or loaded.get("image_profile") != expected_profile
            or loaded.get("image_sha256s") != [item.output_sha256 for item in prepared]
            or loaded.get("working_sha256s") != [item.source_sha256 for item in prepared]
            or not isinstance(loaded.get("response"), dict)
            or not isinstance(loaded.get("audit"), dict)
        ):
            return None
        warning = loaded.get("warning")
        return (
            dict(loaded["response"]),
            dict(loaded["audit"]),
            str(warning) if warning is not None else None,
        )

    def _publish_codex_image_progress(
        self,
        context: PipelineContext,
        *,
        batch_index: int,
        batch_count: int,
        image_index: int,
        image_count: int,
        images_current: int,
        images_total: int,
        asset_id: str,
        action: str,
        attempt: int,
    ) -> None:
        context.events.publish(
            PipelineEvent(
                event_type="codex_image_progress",
                stage=PipelineStage.CODEX_REFINEMENT.value,
                message=(
                    f"Preparing Codex images {images_current}/{images_total}"
                    if action == "preparing"
                    else f"Prepared Codex images {images_current}/{images_total}"
                ),
                progress=image_index / max(image_count, 1),
                details={
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "image_index": image_index,
                    "image_count": image_count,
                    "images_current": images_current,
                    "images_total": images_total,
                    "asset_id": asset_id,
                    "action": action,
                    "attempt": attempt,
                },
            )
        )

    def _publish_codex_batch_progress(
        self,
        context: PipelineContext,
        *,
        batch_index: int,
        batch_count: int,
        action: str,
        attempt: int,
    ) -> None:
        context.events.publish(
            PipelineEvent(
                event_type="codex_batch_progress",
                stage=PipelineStage.CODEX_REFINEMENT.value,
                message=f"Codex image batch {batch_index + 1}/{batch_count}",
                progress=(batch_index + 1) / max(batch_count, 1),
                details={
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "action": action,
                    "attempt": attempt,
                },
            )
        )

    def _prepare_refinement_review(self, context: PipelineContext) -> RefinementReviewState:
        drafts = CaptionDraftBatch.model_validate(
            _stage_output(context, PipelineStage.CAPTION_DRAFTING)
        )
        refinement = _stage_output(context, PipelineStage.CODEX_REFINEMENT)
        candidates = tuple(
            TriggerCandidate.model_validate(item)
            for item in refinement.get("trigger_candidates", ())
        )
        vocabulary = self._refinement_vocabulary(context)
        resolved_trigger = context.config.trigger_token or (
            candidates[0].value if candidates else ""
        )
        requires_trigger = (
            context.config.trigger_word_mode is TriggerWordMode.CODEX_SUGGEST
            and not resolved_trigger
        )
        requires_review = context.config.codex_refinement_mode is CodexRefinementMode.REVIEW
        preview_trigger = resolved_trigger or "<Trigger Word required>"
        items: list[RefinementReviewItem] = []
        for asset_id, draft in drafts.assets.items():
            raw_decision = refinement["decisions"][asset_id]
            proposal = AssetRefinementProposal.model_validate(raw_decision["proposal"])
            proposed_tags = tuple(refinement["effective_tags"][asset_id])
            proposed_caption = ", ".join(
                (
                    preview_trigger,
                    *draft.fixed_tokens,
                    *(DEFAULT_ONTOLOGY.display(tag) for tag in proposed_tags),
                )
            )
            items.append(
                RefinementReviewItem(
                    asset_id=asset_id,
                    original_tags=draft.original_tags,
                    baseline_tags=draft.effective_tags,
                    proposed_tags=proposed_tags,
                    effective_tags=proposed_tags,
                    draft_caption=draft.draft_caption,
                    proposed_caption=proposed_caption,
                    reason=proposal.reason,
                    confidence=proposal.confidence,
                    factory_accepted=bool(raw_decision["accepted"]),
                    rejection_reason=raw_decision.get("rejection_reason"),
                )
            )
        fingerprint = refinement_fingerprint(
            {
                "run_id": context.run_id,
                "preset": context.config.preset.value,
                "codex_refinement_mode": context.config.codex_refinement_mode.value,
                "trigger_word_mode": context.config.trigger_word_mode.value,
                "trigger_token": context.config.trigger_token,
                "drafts": drafts.model_dump(mode="json"),
                "effective_tags": refinement["effective_tags"],
                "decisions": refinement["decisions"],
                "trigger_candidates": refinement["trigger_candidates"],
            }
        )
        warnings = list(refinement.get("warnings", ()))
        if requires_trigger and not candidates:
            warnings.append("Runtime Codex did not provide valid candidates; enter a Trigger Word")
        return RefinementReviewState(
            run_id=context.run_id,
            upstream_fingerprint=fingerprint,
            requires_refinement_review=requires_review,
            requires_trigger_selection=requires_trigger,
            preset=context.config.preset,
            class_token=drafts.class_token,
            invariants=drafts.invariants,
            pinned_tag_vocabulary=tuple(sorted(vocabulary.tags)),
            pinned_tag_categories=dict(sorted(vocabulary.model_categories.items())),
            trigger_word=resolved_trigger,
            trigger_candidates=candidates,
            items=tuple(items),
            warnings=tuple(warnings),
        )

    def _apply_refinement_approval(
        self,
        context: PipelineContext,
        review: RefinementReviewState,
        approval: RefinementApproval,
    ) -> None:
        drafts = CaptionDraftBatch.model_validate(
            _stage_output(context, PipelineStage.CAPTION_DRAFTING)
        )
        refinement = _stage_output(context, PipelineStage.CODEX_REFINEMENT)
        effective, caption_edits = self._validate_refinement_approval_values(
            config=context.config,
            review=review,
            approval=approval,
            drafts=drafts,
            refinement=refinement,
            vocabulary=self._refinement_vocabulary(context),
        )
        context.config = context.config.model_copy(update={"trigger_token": approval.trigger_word})
        context.artifacts["_REFINEMENT_APPROVAL"] = {
            "approval": approval.model_dump(mode="json"),
            "upstream_fingerprint": review.upstream_fingerprint,
            "effective_tags": {key: list(value) for key, value in effective.items()},
            "caption_edits": caption_edits,
        }

    @staticmethod
    def _validate_refinement_approval_values(
        *,
        config: ProjectConfig,
        review: RefinementReviewState,
        approval: RefinementApproval,
        drafts: CaptionDraftBatch,
        refinement: Mapping[str, Any],
        vocabulary: WD14TagVocabulary,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        if approval.upstream_fingerprint != review.upstream_fingerprint:
            raise ValueError("Refinement approval fingerprint does not match the current review")
        raw_effective = refinement.get("effective_tags")
        if not isinstance(raw_effective, Mapping) or set(raw_effective) != set(drafts.assets):
            raise ValueError("Refinement approval source does not cover every accepted asset")
        effective = {asset_id: tuple(raw_effective[asset_id]) for asset_id in drafts.assets}
        caption_edits: dict[str, str] = {}
        if review.requires_refinement_review:
            by_id = {item.asset_id: item for item in approval.items}
            if len(by_id) != len(approval.items) or set(by_id) != set(drafts.assets):
                raise ValueError("Refinement approval must decide every accepted asset")
            proposals: list[AssetRefinementProposal] = []
            for asset_id, draft in drafts.assets.items():
                item = by_id[asset_id]
                if item.decision == "accept":
                    tags = effective[asset_id]
                elif item.decision == "reject":
                    tags = draft.effective_tags
                else:
                    if item.effective_tags is None:
                        raise ValueError(f"{asset_id}.effective_tags: edited tags are required")
                    tags = item.effective_tags
                proposals.append(
                    AssetRefinementProposal(
                        asset_id=asset_id,
                        decision=RefinementDecisionKind.REPLACE,
                        effective_tags=tags,
                        reason="User-approved refinement decision.",
                        confidence=1.0,
                    )
                )
                if item.caption is not None:
                    caption_edits[asset_id] = item.caption
            validated = validate_refinement_proposals(
                drafts,
                proposals,
                vocabulary=vocabulary,
            )
            for asset_id, decision in validated.decisions.items():
                if not decision.accepted:
                    raise ValueError(
                        f"{asset_id}.effective_tags: {decision.rejection_reason or 'invalid tags'}"
                    )
                edited = by_id[asset_id].decision == "edit"
                if edited and decision.rejected_tags:
                    tag, reason = next(iter(decision.rejected_tags.items()))
                    raise ValueError(f"{asset_id}.effective_tags: {tag!r} {reason}")
            effective = validated.effective_tags

        captions = finalize_captions(drafts, effective, approval.trigger_word)
        captions.update(caption_edits)
        for asset_id, caption in captions.items():
            audit = audit_captions(
                {asset_id: caption},
                approval.trigger_word,
                config.preset,
                class_token=drafts.class_token,
                invariant_tags=drafts.invariants,
            )
            errors = [issue.message for issue in audit.issues if issue.severity.value == "error"]
            if errors:
                raise ValueError(f"{asset_id}.caption: {errors[0]}")
            if audit.semantic_content_coverage < 1.0:
                raise ValueError(
                    f"{asset_id}.caption: caption must contain at least one semantic tag"
                )
        return effective, caption_edits

    def _caption_stage(self, context: PipelineContext) -> dict[str, Any]:
        drafts = CaptionDraftBatch.model_validate(
            _stage_output(context, PipelineStage.CAPTION_DRAFTING)
        )
        approved = context.artifacts.get("_REFINEMENT_APPROVAL")
        if not isinstance(approved, dict):
            raise RuntimeError("Caption finalization requires a validated refinement approval")
        effective_tags = approved["effective_tags"]
        captions = finalize_captions(
            drafts,
            effective_tags,
            context.config.trigger_token,
        )
        for asset_id, caption in approved.get("caption_edits", {}).items():
            if asset_id in captions:
                captions[asset_id] = str(caption)
        overrides = self._run_dataset_overrides(context)
        for asset_id, override in overrides.items():
            caption = override.get("final_caption")
            if asset_id in captions and isinstance(caption, str):
                captions[asset_id] = caption
        audit = audit_captions(
            captions,
            context.config.trigger_token,
            context.config.preset,
            class_token=drafts.class_token,
            invariant_tags=drafts.invariants,
        )
        if not audit.passed:
            failures = "; ".join(issue.message for issue in audit.issues[:5])
            raise ValueError(f"Caption QA failed: {failures}")
        CaptionWriter(context.layout.captions, raw_root=context.layout.raw).write(captions)
        return {
            "captions": captions,
            "keep_tokens": drafts.keep_tokens,
            "class_token": drafts.class_token,
            "invariants": list(drafts.invariants),
            "warnings": list(drafts.warnings),
            "effective_tags": effective_tags,
            "refinement_fingerprint": approved["upstream_fingerprint"],
            "audit": audit.model_dump(mode="json"),
        }

    def _dataset_review_stage(self, context: PipelineContext) -> dict[str, Any]:
        analyzed = _stage_output(context, PipelineStage.ANALYZING)
        auto_accepted_ids = set(
            _stage_output(context, PipelineStage.DEDUPLICATING)["accepted_asset_ids"]
        )
        overrides = self._run_dataset_overrides(context)
        accepted_ids = {
            asset_id
            for asset_id in auto_accepted_ids
            if overrides.get(asset_id, {}).get("included", True)
        }
        assessments: list[QualityAssessment] = []
        review_items: list[dict[str, Any]] = []
        captions = _stage_output(context, PipelineStage.CAPTIONING)["captions"]
        tag_results = _stage_output(context, PipelineStage.TAGGING)["results"]
        tags_by_id = {
            item["asset_id"]: [tag["name"] for tag in item["tags"] if tag.get("selected", True)]
            for item in tag_results
        }
        tag_scores_by_id = {
            item["asset_id"]: tuple(
                TagScore.model_validate(tag) for tag in item["tags"] if tag.get("selected", True)
            )
            for item in tag_results
            if item["asset_id"] in accepted_ids
        }
        accepted_asset_ids = tuple(sorted(accepted_ids))
        profile = load_preset_profile(context.config.preset)
        validation_split = plan_validation_split(accepted_asset_ids, profile)
        validation_ids = set(validation_split.validation_asset_ids)
        analyzed_by_id = {str(item["asset_id"]): item for item in analyzed["items"]}
        reference_embedding_root = (
            context.layout.run(context.run_id).root / "dataset-review" / "reference-embedding"
        )
        existing_cluster_ids = _stage_output(context, PipelineStage.DEDUPLICATING)["cluster_ids"]
        centroid_similarity_by_asset: dict[str, float] = {}
        embedding_outlier_ids: set[str] = set()
        if accepted_asset_ids:
            reference_embedding_result = self._embed_many_sharded(
                context,
                task=GpuTaskKind.REFERENCE_EMBED,
                paths=[
                    Path(str(analyzed_by_id[asset_id]["path"])) for asset_id in accepted_asset_ids
                ],
                work_directory=reference_embedding_root,
            )
            reference_embedding_batch = reference_embedding_result.result
            reference_embedding_path = reference_embedding_root / "embeddings.json"
            write_json_atomic(
                reference_embedding_path,
                reference_embedding_batch.model_dump(mode="json"),
            )
            embedding_signals = analyze_dataset_embeddings(
                accepted_asset_ids,
                np.stack(
                    [
                        np.asarray(item.values, dtype=np.float32)
                        for item in reference_embedding_batch.items
                    ]
                ),
            )
            centroid_similarity_by_asset = dict(embedding_signals.centroid_similarity_by_asset)
            embedding_outlier_ids = set(embedding_signals.outlier_asset_ids)
            combined_cluster_ids = combine_duplicate_cluster_ids(
                accepted_asset_ids,
                existing_cluster_ids,
                embedding_signals.near_duplicate_clusters,
            )
            reference_embedding_review = {
                "status": "complete",
                "model_id": reference_embedding_batch.model_id,
                "revision": reference_embedding_batch.revision,
                "dimension": reference_embedding_batch.dimension,
                "device": reference_embedding_batch.device,
                "path": str(reference_embedding_path),
                "gpu_uuids": list(reference_embedding_result.gpu_uuids),
                **embedding_signals.model_dump(mode="json"),
            }
        else:
            combined_cluster_ids = {}
            reference_embedding_review = {
                "status": "not_run_no_accepted_images",
                "model_id": "",
                "revision": "",
                "dimension": 0,
                "device": "not_run",
                "path": None,
                "gpu_uuids": [],
                "centroid_similarity_by_asset": {},
                "near_duplicate_threshold": 0.995,
                "near_duplicate_clusters": [],
                "outlier_threshold": 0.0,
                "outlier_asset_ids": [],
                "warnings": ["Reference embedding skipped because no images were accepted"],
            }
        reference_embedding_review_path = reference_embedding_root / "review.json"
        write_json_atomic(reference_embedding_review_path, reference_embedding_review)
        reference_embedding_review["report_path"] = str(reference_embedding_review_path)
        for item in analyzed["items"]:
            assessment = QualityAssessment.model_validate(item["assessment"])
            source_disposition = assessment.disposition
            if item["asset_id"] in accepted_ids and not assessment.accepted:
                assessment = assessment.model_copy(update={"disposition": QualityDisposition.WARN})
            elif item["asset_id"] not in accepted_ids and assessment.accepted:
                assessment = assessment.model_copy(
                    update={"disposition": QualityDisposition.REJECT}
                )
            assessments.append(assessment)
            duplicate_cluster_id = combined_cluster_ids.get(
                item["asset_id"]
            ) or existing_cluster_ids.get(item["asset_id"])
            categories = review_categories(
                included=item["asset_id"] in accepted_ids,
                disposition=(
                    QualityDisposition.WARN
                    if item["asset_id"] in accepted_ids
                    and source_disposition is QualityDisposition.REJECT
                    else source_disposition
                ),
                duplicate=duplicate_cluster_id is not None,
                validation=item["asset_id"] in validation_ids,
                embedding_outlier=item["asset_id"] in embedding_outlier_ids,
            )
            reasons = [
                *item["reasons"],
                *(
                    ["Learned embedding outlier; manual review recommended"]
                    if item["asset_id"] in embedding_outlier_ids
                    else []
                ),
                *(
                    [f"Duplicate cluster {duplicate_cluster_id}; representative rules applied"]
                    if duplicate_cluster_id is not None
                    else []
                ),
            ]
            review_items.append(
                {
                    "asset_id": item["asset_id"],
                    "category": primary_review_category(categories).value,
                    "categories": [category.value for category in categories],
                    "original_filename": item["original_filename"],
                    "width": item["width"],
                    "height": item["height"],
                    "bucket": "Pending planner",
                    "reasons": reasons,
                    "raw_tags": tags_by_id.get(item["asset_id"], []),
                    "final_caption": captions.get(item["asset_id"], ""),
                    "included": item["asset_id"] in accepted_ids,
                    "thumbnail_path": item["path"],
                    "reference_embedding_similarity": centroid_similarity_by_asset.get(
                        item["asset_id"]
                    ),
                    "embedding_outlier": item["asset_id"] in embedding_outlier_ids,
                    "duplicate_cluster_id": duplicate_cluster_id,
                }
            )
        diversity = analyze_diversity(
            tag_scores_by_id,
            duplicate_cluster_ids=combined_cluster_ids,
            style_similarity_scores=(
                tuple(centroid_similarity_by_asset[asset_id] for asset_id in accepted_asset_ids)
                if context.config.preset is PresetKind.STYLE
                else None
            ),
        )
        statistics = _dataset_statistics(
            accepted_asset_ids,
            analyzed_by_id,
            validation_count=len(validation_ids),
            diversity_score=diversity.content_diversity,
        )
        caption_info = _stage_output(context, PipelineStage.CAPTIONING)
        preliminary_plan = plan_training(
            context.config,
            statistics,
            self._gpu_capabilities(
                context.config.selected_gpu_uuids,
                fake=context.config.backend_mode is BackendMode.FAKE,
            ),
            profile,
            has_class_token=caption_info["class_token"] is not None,
            available_optimizers=("AdamW8bit", "AdamW"),
        )
        for review_item in review_items:
            review_item["bucket"] = format_assigned_bucket(
                int(review_item["width"]),
                int(review_item["height"]),
                max_resolution=preliminary_plan.resolution,
            )
        gate = evaluate_quality_gate(
            assessments,
            context.config.preset,
            thresholds=DatasetGateThresholds(
                hard_minimum=profile.quality_gate.hard_minimum,
                warning_below=profile.quality_gate.warning_below,
                max_near_duplicate_dominance=profile.quality_gate.max_near_duplicate_dominance,
                max_dominant_character_ratio=profile.quality_gate.max_dominant_character_ratio,
                min_content_diversity=profile.quality_gate.min_content_diversity,
            ),
            near_duplicate_dominance=diversity.near_duplicate_cluster_dominance,
            content_diversity=diversity.content_diversity,
            dominant_character_ratio=diversity.dominant_character_ratio,
            style_consistency=diversity.style_consistency,
        )
        if not gate.passed:
            raise ValueError(
                "Dataset quality gate failed: " + "; ".join(i.message for i in gate.issues)
            )
        context.events.publish(
            PipelineEvent(
                event_type="dataset_review",
                stage=PipelineStage.DATASET_REVIEW.value,
                message=f"{gate.accepted_count} images accepted",
                details={
                    "items": review_items,
                    "gate": gate.model_dump(mode="json"),
                    "diversity": diversity.model_dump(mode="json"),
                    "reference_embedding": reference_embedding_review,
                    "validation_split": validation_split.model_dump(mode="json"),
                },
            )
        )
        return {
            "gate": gate.model_dump(mode="json"),
            "items": review_items,
            "accepted_asset_ids": sorted(accepted_ids),
            "override_count": len(overrides),
            "diversity": diversity.model_dump(mode="json"),
            "reference_embedding": reference_embedding_review,
            "validation_split": validation_split.model_dump(mode="json"),
            "bucket_resolution": preliminary_plan.resolution,
        }

    @staticmethod
    def _dataset_overrides(layout: ProjectLayout) -> dict[str, dict[str, Any]]:
        path = layout.dataset / "review-overrides.json"
        if not path.is_file():
            return {}
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Dataset override file must contain an object: {path}")
        return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}

    @staticmethod
    def _run_dataset_overrides(context: PipelineContext) -> dict[str, dict[str, Any]]:
        snapshot = context.artifacts.get("_RUN_SNAPSHOT", {})
        if not isinstance(snapshot, dict):
            return {}
        payload = snapshot.get("dataset_overrides", {})
        if not isinstance(payload, dict):
            raise ValueError("Run snapshot dataset_overrides must contain an object")
        return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}

    def _gpu_capabilities(
        self, selected: Sequence[str], *, fake: bool
    ) -> tuple[GpuCapability, ...]:
        if fake:
            return tuple(
                GpuCapability(
                    uuid=gpu_uuid,
                    index=index,
                    name="Deterministic Fake CUDA Device",
                    total_vram_mb=24 * 1024,
                    free_vram_mb=24 * 1024,
                    capability_major=12,
                    capability_minor=0,
                    bf16_supported=True,
                    compatible=True,
                    compatibility_reason="Fake E2E capability",
                )
                for index, gpu_uuid in enumerate(selected)
            )
        devices = discover_nvidia_gpus()
        selected_set = set(selected)
        return tuple(
            GpuCapability(
                uuid=item.uuid,
                index=item.index,
                name=item.name,
                total_vram_mb=item.total_vram_mb,
                free_vram_mb=item.free_vram_mb,
                capability_major=item.capability_major,
                capability_minor=item.capability_minor,
                bf16_supported=item.bf16_supported,
                compatible=item.compatible and item.uuid in selected_set,
                compatibility_reason=item.compatibility_reason,
            )
            for item in devices
        )

    def _run_batch_probe(
        self,
        context: PipelineContext,
        preliminary_plan: TrainingPlan,
        gpu_capabilities: Sequence[GpuCapability],
    ) -> tuple[BatchProbeResult, dict[str, Any]]:
        """Probe the planned GPU without creating model, optimizer, or dataset state."""

        capability = next(
            (item for item in gpu_capabilities if item.uuid == preliminary_plan.training_gpu_uuid),
            None,
        )
        if capability is None:
            raise RuntimeError("Training GPU capability disappeared before batch probe")
        candidates = (
            (preliminary_plan.batch_size,)
            if context.config.advanced.batch_size is not None
            else descending_batch_candidates(preliminary_plan.batch_size)
        )
        work_directory = context.layout.run(context.run_id).root / "batch-probe"
        request = BatchProbeRequest(
            gpu_uuid=preliminary_plan.training_gpu_uuid,
            resolution=preliminary_plan.resolution,
            network_dim=preliminary_plan.network_dim,
            precision=preliminary_plan.precision,
            candidate_batch_sizes=candidates,
            work_directory=work_directory,
        )
        lease_record: dict[str, Any] | None = None
        if context.config.backend_mode is BackendMode.FAKE:
            result = FakeBatchProbe(
                free_vram_mb=capability.free_vram_mb,
                total_vram_mb=capability.total_vram_mb,
            ).probe(request, context.cancellation)
        else:
            if not self.runtime.installed() or not self.runtime.source_matches_manifest():
                raise RuntimeError("Managed runtime is absent or its source pin differs")
            database = Database(context.layout.database)
            database.initialize()
            devices = discover_nvidia_gpus()
            with lease_selected_gpu(
                database=database,
                selected_uuids=(preliminary_plan.training_gpu_uuid,),
                run_id=context.run_id,
                task=GpuTaskKind.BATCH_PROBE,
                estimated_vram_mb=256,
                preferred_gpu_uuid=preliminary_plan.training_gpu_uuid,
                devices=devices,
            ) as (binding, lease):
                result = TorchCudaBatchProbe(
                    python_executable=self.runtime.layout.python,
                    binding=binding,
                    selected_gpu_uuids=context.config.selected_gpu_uuids,
                ).probe(request, context.cancellation)
                lease_record = {
                    "lease_id": lease.lease_id,
                    "gpu_uuid": lease.gpu_uuid,
                    "physical_index": binding.physical_index,
                    "logical_index": binding.logical_index,
                    "estimated_vram_mb": lease.estimated_vram_mb,
                }
        report_path = work_directory / "batch-probe.json"
        evidence = {
            "result": result.model_dump(mode="json"),
            "lease": lease_record,
            "report_path": str(report_path),
        }
        write_json_atomic(report_path, evidence)
        return result, evidence

    def _planning_stage(self, context: PipelineContext) -> dict[str, Any]:
        review_output = _stage_output(context, PipelineStage.DATASET_REVIEW)
        accepted_ids = review_output["accepted_asset_ids"]
        profile = load_preset_profile(context.config.preset)
        split = plan_validation_split(accepted_ids, profile)
        if review_output.get("validation_split") != split.model_dump(mode="json"):
            raise RuntimeError("Dataset Review validation categories differ from Planning split")
        analyzed_items = {
            item["asset_id"]: item
            for item in _stage_output(context, PipelineStage.ANALYZING)["items"]
        }
        review_diversity = _stage_output(context, PipelineStage.DATASET_REVIEW)["diversity"]
        statistics = _dataset_statistics(
            accepted_ids,
            analyzed_items,
            validation_count=len(split.validation_asset_ids),
            diversity_score=float(review_diversity["content_diversity"]),
        )
        caption_info = _stage_output(context, PipelineStage.CAPTIONING)
        gpu_capabilities = self._gpu_capabilities(
            context.config.selected_gpu_uuids,
            fake=context.config.backend_mode is BackendMode.FAKE,
        )
        preliminary_plan = plan_training(
            context.config,
            statistics,
            gpu_capabilities,
            profile,
            has_class_token=caption_info["class_token"] is not None,
            available_optimizers=("AdamW8bit", "AdamW"),
        )
        if int(review_output.get("bucket_resolution", 0)) != preliminary_plan.resolution:
            raise RuntimeError("Dataset Review bucket resolution differs from Planning")
        batch_probe_result, batch_probe = self._run_batch_probe(
            context,
            preliminary_plan,
            gpu_capabilities,
        )
        plan = plan_training(
            context.config,
            statistics,
            gpu_capabilities,
            profile,
            has_class_token=caption_info["class_token"] is not None,
            available_optimizers=("AdamW8bit", "AdamW"),
            batch_probe_result=batch_probe_result,
        )
        train_dir = context.layout.dataset / "training"
        validation_dir = context.layout.validation
        all_dir = context.layout.dataset / "all-accepted"
        for directory in (train_dir, validation_dir, all_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for pattern in ("*.png", "*.txt"):
                for stale in directory.glob(pattern):
                    stale.unlink()
        captions = caption_info["captions"]
        validation_set = set(split.validation_asset_ids)
        for asset_id in accepted_ids:
            context.cancellation.raise_if_cancelled()
            target_dir = validation_dir if asset_id in validation_set else train_dir
            source = context.layout.working / f"{asset_id}.png"
            destination = target_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            caption_path = destination.with_suffix(".txt")
            caption_path.write_text(str(captions[asset_id]).strip() + "\n", encoding="utf-8")
            combined_image = all_dir / source.name
            shutil.copy2(source, combined_image)
            combined_image.with_suffix(".txt").write_text(
                str(captions[asset_id]).strip() + "\n", encoding="utf-8"
            )
        verify_materialized_validation_partition(
            image_dir=all_dir,
            expected_training_ids=split.train_asset_ids,
            expected_validation_ids=split.validation_asset_ids,
            seed=split.seed,
        )
        configs = write_dataset_configs(
            output_dir=context.layout.configs,
            training_image_dir=all_dir,
            validation_image_dir=validation_dir if split.enabled else None,
            plan=plan,
            validation_total_count=len(accepted_ids) if split.enabled else None,
        )
        write_json_atomic(context.layout.training_config, plan.model_dump(mode="json"))
        return {
            "plan": plan.model_dump(mode="json"),
            "statistics": statistics.model_dump(mode="json"),
            "split": split.model_dump(mode="json"),
            "dataset_config": str(configs.training),
            "validation_config": str(configs.validation) if configs.validation else None,
            "train_dir": str(train_dir),
            "validation_dir": str(validation_dir) if split.enabled else None,
            "gpu_capabilities": [item.model_dump(mode="json") for item in gpu_capabilities],
            "batch_probe": batch_probe,
        }

    def _codex_review(
        self,
        context: PipelineContext,
        task: CodexTaskType,
        payload: dict[str, Any],
        *,
        images: Sequence[PreparedCodexImage] = (),
        allow_fallback: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        sanitized_payload = sanitize_public_metadata(payload)
        if not isinstance(sanitized_payload, dict):
            raise ValueError("Runtime Codex payload must remain a mapping after sanitization")
        payload = sanitized_payload
        if context.config.backend_mode is BackendMode.FAKE:
            response = deterministic_fallback(task, payload)
            return (
                response.model_dump(mode="json"),
                {
                    "fallback_used": True,
                    "task_type": task.value,
                    "attempts": 0,
                    "image_input_sha256s": [item.output_sha256 for item in images],
                    "image_count": len(images),
                },
                "Runtime Codex intentionally replaced by deterministic Fake backend",
            )
        fallback = context.config.allow_without_codex if allow_fallback is None else allow_fallback
        if task is CodexTaskType.DATASET_REFINEMENT:
            fallback = False
        try:
            result = CodexGateway(
                self.settings.codex_runtime_root,
                timeout_seconds=self.settings.codex_timeout_seconds,
                startup_timeout_seconds=self.settings.codex_startup_timeout_seconds,
                idle_timeout_seconds=self.settings.codex_idle_timeout_seconds,
                retry_backoff_seconds=self.settings.codex_retry_backoff_seconds,
                runtime=self._codex_runtime,
            ).review(
                task,
                payload,
                images=images,
                allow_fallback=fallback,
                cancellation=context.cancellation,
            )
        except (CodexCallError, CodexCallCancelled) as exc:
            ApplicationArtifactStore(Database(context.layout.database)).persist_codex_audit(
                context,
                _jsonable(exc.audit),
            )
            raise
        ApplicationArtifactStore(Database(context.layout.database)).persist_codex_audit(
            context,
            _jsonable(result.audit),
        )
        return (
            result.response.model_dump(mode="json"),
            _jsonable(result.audit),
            result.warning,
        )

    def _codex_pretrain_stage(self, context: PipelineContext) -> dict[str, Any]:
        review = _stage_output(context, PipelineStage.DATASET_REVIEW)
        caption = _stage_output(context, PipelineStage.CAPTIONING)
        planning = _stage_output(context, PipelineStage.PLANNING)
        dataset_response, dataset_audit, dataset_warning = self._codex_review(
            context,
            CodexTaskType.DATASET_REVIEW,
            {
                "preset": context.config.preset.value,
                "hard_gate_passed": bool(review["gate"]["passed"]),
                "accepted_count": review["gate"]["accepted_count"],
                "rejected_count": review["gate"]["rejected_count"],
                "warning_codes": [item["code"] for item in review["gate"]["issues"]],
                "diversity": review["diversity"],
                "duplicate_cluster_count": len(
                    _stage_output(context, PipelineStage.DEDUPLICATING)["clusters"]
                ),
            },
        )
        caption_response, caption_audit, caption_warning = self._codex_review(
            context,
            CodexTaskType.CAPTION_REVIEW,
            {
                "preset": context.config.preset.value,
                "caption_qa_passed": bool(caption["audit"]["passed"]),
                "caption_count": len(caption["captions"]),
                "keep_tokens": caption["keep_tokens"],
                "class_token": caption["class_token"],
                "invariants": caption["invariants"],
                "warnings": caption["warnings"],
                "audit_issue_codes": [item["code"] for item in caption["audit"].get("issues", [])],
            },
        )
        training_payload = {
            "preset": context.config.preset.value,
            "hard_gate_passed": bool(review["gate"]["passed"]),
            "caption_qa_passed": bool(caption["audit"]["passed"]),
            "preflight_passed": True,
            "accepted_count": review["gate"]["accepted_count"],
            "warning_codes": [item["code"] for item in review["gate"]["issues"]],
            "training_plan": planning["plan"],
        }
        response, audit, training_warning = self._codex_review(
            context, CodexTaskType.TRAINING_PLAN, training_payload
        )
        advisory_messages: list[str] = []
        try:
            proposed_items = response.get("proposed_changes", [])
            proposed: dict[str, Any] = {}
            for item in proposed_items:
                field = str(item["field"])
                if field in proposed:
                    raise ValueError(f"Runtime Codex proposed {field!r} more than once")
                proposed[field] = item["value"]
            validated_suggestions = validate_recovery_changes(
                proposed,
                locked_fields=context.config.locked_fields,
            )
        except ValueError as exc:
            validated_suggestions = {}
            advisory_messages.append(
                f"Runtime Codex proposed invalid training changes; they were ignored: {exc}"
            )

        if not dataset_response.get("approved", False):
            advisory_messages.append(
                "Runtime Codex raised advisory dataset concerns; the deterministic quality "
                "gate remains authoritative"
            )
        if not caption_response.get("approved", False):
            advisory_messages.append(
                "Runtime Codex raised advisory caption concerns; validated Factory captions "
                "remain unchanged"
            )
        if not response.get("approved", False):
            advisory_messages.append(
                "Runtime Codex raised advisory concerns about the deterministic training plan; "
                "the Factory-owned plan remains unchanged"
            )
        for advisory in advisory_messages:
            context.events.publish(
                PipelineEvent(
                    event_type="warning",
                    stage=PipelineStage.CODEX_PRETRAIN_REVIEW.value,
                    message=advisory,
                    details={"validated_suggestions": validated_suggestions},
                )
            )
        combined_warning = (
            "; ".join(
                [
                    item
                    for item in (dataset_warning, caption_warning, training_warning)
                    if item is not None
                ]
                + advisory_messages
            )
            or None
        )
        return {
            "response": response,
            "audit": audit,
            "reviews": {
                CodexTaskType.DATASET_REVIEW.value: dataset_response,
                CodexTaskType.CAPTION_REVIEW.value: caption_response,
                CodexTaskType.TRAINING_PLAN.value: response,
            },
            "audits": [dataset_audit, caption_audit, audit],
            "warning": combined_warning,
            "validated_suggestions": validated_suggestions,
            "applied_changes": {},
        }

    def _preflight_stage(self, context: PipelineContext) -> dict[str, Any]:
        inspection = inspect_sdxl_safetensors(context.config.base_model)
        if not inspection.is_sdxl:
            raise ValueError("The selected base model is not structurally SDXL compatible")
        validation: dict[str, Any] | None = None
        if context.config.backend_mode is BackendMode.REAL:
            if not self.runtime.installed() or not self.runtime.source_matches_manifest():
                raise RuntimeError("Managed training runtime is absent or its source pin differs")
            context.cancellation.raise_if_cancelled()
            plan = TrainingPlan.model_validate(
                _stage_output(context, PipelineStage.PLANNING)["plan"]
            )
            devices = discover_nvidia_gpus()
            binding = bind_gpu_for_child(
                plan.training_gpu_uuid,
                selected_uuids=context.config.selected_gpu_uuids,
                devices=devices,
            )
            report = inspect_runtime(
                python_executable=self.runtime.layout.python,
                binding=binding,
                sd_scripts_root=self.runtime.layout.sd_scripts,
            )
            record = RuntimeValidationRecord.from_doctor_report(
                str(self.runtime.manifest["profile_id"]), report
            )
            self.runtime.write_validation_record(record)
            if not record.ready:
                failures = "; ".join(
                    f"{name}: {check.detail}"
                    for name, check in record.checks.items()
                    if check.status is ValidationStatus.ERROR
                )
                raise RuntimeError(
                    f"Managed runtime deep validation failed on {plan.training_gpu_uuid}: "
                    f"{failures or 'doctor reported NOT READY'}"
                )
            validation = record.model_dump(mode="json")
            context.cancellation.raise_if_cancelled()
        return {
            "model": inspection.model_dump(mode="json"),
            "runtime_mode": context.config.backend_mode.value,
            "runtime_root": str(self.runtime.layout.root),
            "runtime_validation": validation,
            "ready": True,
        }

    def _training_backend(
        self,
        context: PipelineContext,
        plan: TrainingPlan,
        *,
        binding: GpuBinding | None = None,
    ) -> TrainingBackend:
        if context.config.backend_mode is BackendMode.FAKE:
            return FakeTrainingBackend()
        from lora_factory.training.sd_scripts_backend import SdScriptsTrainingBackend

        if binding is None:
            raise RuntimeError("Real training requires an active selected-GPU lease")
        return SdScriptsTrainingBackend(
            python_executable=self.runtime.layout.python,
            sd_scripts_root=self.runtime.layout.sd_scripts,
            binding=binding,
            selected_gpu_uuids=context.config.selected_gpu_uuids,
        )

    @contextmanager
    def _training_attempt_backend(
        self,
        context: PipelineContext,
        plan: TrainingPlan,
    ) -> Iterator[tuple[TrainingBackend, TrainingPlan, dict[str, Any] | None]]:
        """Bind one attempt to a durable lease after fresh UUID-to-index discovery."""

        if context.config.backend_mode is BackendMode.FAKE:
            yield self._training_backend(context, plan), plan, None
            return

        database = Database(context.layout.database)
        database.initialize()
        estimated_vram_mb = _training_vram_estimate_mb(plan)
        with lease_selected_gpu(
            database=database,
            selected_uuids=context.config.selected_gpu_uuids,
            run_id=context.run_id,
            task=GpuTaskKind.TRAIN,
            estimated_vram_mb=estimated_vram_mb,
            preferred_gpu_uuid=plan.training_gpu_uuid,
        ) as (binding, lease):
            effective_plan = plan
            if binding.uuid != plan.training_gpu_uuid:
                effective_plan = plan.model_copy(
                    update={
                        "training_gpu_uuid": binding.uuid,
                        "provenance": {
                            **plan.provenance,
                            "training_gpu_uuid": "attempt_scheduler_lease",
                        },
                    }
                )
            backend = self._training_backend(
                context,
                effective_plan,
                binding=binding,
            )
            yield (
                backend,
                effective_plan,
                {
                    "lease_id": lease.lease_id,
                    "gpu_uuid": lease.gpu_uuid,
                    "physical_index": binding.physical_index,
                    "logical_index": binding.logical_index,
                    "estimated_vram_mb": lease.estimated_vram_mb,
                },
            )

    def _codex_recovery_review(
        self,
        context: PipelineContext,
        *,
        error: PipelineError,
        backend_version: str,
        decision: RecoveryDecision,
        current_plan: TrainingPlan,
        selected_gpus: tuple[GpuCapability, ...],
        previous_records: Sequence[Mapping[str, Any]],
        retry_allowed: bool,
    ) -> dict[str, Any]:
        command_argv = [
            str(
                sanitize_public_metadata(
                    redact_text(str(item), home=Path.home())
                    .replace(str(context.layout.root), "%PROJECT_ROOT%")
                    .replace(str(context.config.base_model), "%BASE_MODEL%")
                    .replace(str(context.config.output_root), "%OUTPUT_ROOT%")
                )
            )
            for item in getattr(error, "command_argv", ())
        ]
        current_plan_payload = current_plan.model_dump(mode="json")
        current_plan_payload.pop("training_gpu_uuid", None)
        selected_gpu_summary = {
            "count": len(selected_gpus),
            "compatible_count": sum(1 for item in selected_gpus if item.compatible),
            "total_vram_mb": sum(item.total_vram_mb for item in selected_gpus),
            "free_vram_mb": sum(item.free_vram_mb for item in selected_gpus),
        }
        deterministic_decision = sanitize_public_metadata(decision.model_dump(mode="json"))
        if not isinstance(deterministic_decision, dict):
            raise TypeError("Runtime Codex recovery decision must remain a mapping")
        decision_plan = deterministic_decision.get("plan")
        if isinstance(decision_plan, dict):
            decision_plan.pop("training_gpu_uuid", None)
            provenance = decision_plan.get("provenance")
            if isinstance(provenance, dict):
                provenance.pop("training_gpu_uuid", None)
        payload = {
            "classification": error.classification.value,
            "diagnostics": str(
                sanitize_public_metadata(redact_text(str(error), home=Path.home())[-2000:])
            ),
            "command_argv": command_argv,
            "deterministically_recoverable": retry_allowed,
            "deterministic_decision": deterministic_decision,
            "current_plan": current_plan_payload,
            "resolved_config": {
                "preset": context.config.preset.value,
                "advanced": context.config.advanced.model_dump(mode="json"),
                "locked_fields": sorted(context.config.locked_fields),
            },
            "dependency_versions": {
                "training_backend": backend_version,
                "runtime_profile": str(self.runtime.manifest["profile_id"]),
                "sd_scripts_commit": str(self.runtime.manifest["sd_scripts"]["commit"]),
            },
            "selected_gpu_summary": selected_gpu_summary,
            "locked_fields": sorted(context.config.locked_fields),
            "previous_attempts": [
                {
                    "attempt": item.get("attempt"),
                    "classification": item.get("classification"),
                    "change": item.get("change"),
                    "quality_impact": item.get("quality_impact"),
                }
                for item in previous_records
            ],
        }
        response, audit, warning = self._codex_review(
            context,
            CodexTaskType.RECOVERY,
            payload,
        )
        proposed: dict[str, Any] = {}
        validation_error: str | None = None
        try:
            for item in response.get("proposed_changes", []):
                field = str(item["field"])
                if field in proposed:
                    raise ValueError(f"Runtime Codex proposed {field!r} more than once")
                proposed[field] = item["value"]
            validated = validate_recovery_changes(
                proposed,
                locked_fields=context.config.locked_fields,
            )
        except (KeyError, TypeError, ValueError) as exc:
            validated = {}
            validation_error = str(exc)
        warnings = [item for item in (warning, validation_error) if item]
        return {
            "response": response,
            "audit": audit,
            "warning": "; ".join(warnings) if warnings else None,
            "validated_suggestions": validated,
            "applied_changes": {},
            "advisory_only": True,
        }

    def _training_stage(self, context: PipelineContext) -> dict[str, Any]:
        planning = _stage_output(context, PipelineStage.PLANNING)
        plan = TrainingPlan.model_validate(planning["plan"])
        dataset_config = Path(planning["dataset_config"])
        run_directory = context.layout.run(context.run_id).root
        base_model_sha256 = str(_stage_output(context, PipelineStage.PREFLIGHT)["model"]["sha256"])
        training_image_count = int(planning["statistics"]["accepted_count"])
        recovery_records: list[dict[str, Any]] = []
        previous_changes: list[str] = []

        def progress(value: TrainingProgress) -> None:
            context.events.publish(
                PipelineEvent(
                    event_type="training_progress",
                    stage=PipelineStage.TRAINING.value,
                    message=value.message,
                    progress=min(1.0, value.step / max(value.total_steps, 1)),
                    details=progress_details(
                        context,
                        interval_seconds=self.settings.telemetry_interval_seconds,
                        current_task="TRAINING",
                        images_current=training_image_count,
                        images_total=training_image_count,
                        active_gpu_uuid=plan.training_gpu_uuid,
                        extra=value.model_dump(mode="json"),
                    ),
                )
            )

        while True:
            context.cancellation.raise_if_cancelled()
            lease_audit: dict[str, Any] | None = None
            try:
                with self._training_attempt_backend(context, plan) as (
                    backend,
                    leased_plan,
                    lease_audit,
                ):
                    plan = leased_plan
                    request_fingerprint = training_request_fingerprint(
                        run_id=context.run_id,
                        output_name=context.config.lora_name,
                        base_model_sha256=base_model_sha256,
                        dataset_config=dataset_config,
                        plan=plan,
                        seed=42,
                        backend_version=backend.version,
                    )
                    attempt = prepare_training_attempt(
                        run_directory=run_directory,
                        request_fingerprint=request_fingerprint,
                    )
                    request = TrainingRequest(
                        run_id=context.run_id,
                        attempt=attempt.attempt,
                        output_name=context.config.lora_name,
                        base_model=context.config.base_model,
                        dataset_config=dataset_config,
                        run_directory=run_directory,
                        plan=plan,
                        seed=42,
                        resume_state=attempt.resume_state,
                    )
                    ApplicationArtifactStore(
                        Database(context.layout.database)
                    ).persist_training_started(
                        context,
                        attempt=attempt.attempt,
                        request_fingerprint=attempt.request_fingerprint,
                        resume_state=attempt.resume_state,
                        resumed_from_attempt=attempt.resumed_from_attempt,
                        gpu_lease=lease_audit,
                    )
                    result = backend.train(request, context.cancellation, progress)
            except CancelledError as exc:
                cancellation_record = {
                    "attempt": attempt.attempt,
                    "classification": "CANCELLED",
                    "status": "cancelled",
                    "recoverable": True,
                    "retry_applied": False,
                    "change": None,
                    "reason": (
                        "Training was cancelled by the user; resume requires a fresh attempt."
                    ),
                    "quality_impact": False,
                    "plan_before": plan.model_dump(mode="json"),
                    "plan_after": plan.model_dump(mode="json"),
                    "dataset_config": str(dataset_config),
                    "request_fingerprint": attempt.request_fingerprint,
                    "command_argv": list(getattr(exc, "command_argv", ())),
                    "gpu_lease": lease_audit,
                    "codex": None,
                }
                ApplicationArtifactStore(
                    Database(context.layout.database)
                ).persist_training_failure(
                    context,
                    cancellation_record,
                )
                raise
            except PipelineError as exc:
                selected = set(context.config.selected_gpu_uuids)
                capabilities = tuple(
                    item
                    for item in self._gpu_capabilities(
                        context.config.selected_gpu_uuids,
                        fake=context.config.backend_mode is BackendMode.FAKE,
                    )
                    if item.uuid in selected
                )
                decision = next_recovery(
                    classification=exc.classification,
                    plan=plan,
                    selected_gpus=capabilities,
                    locked_fields=context.config.locked_fields,
                    previous_changes=tuple(previous_changes),
                    attempt=attempt.attempt,
                )
                retry_allowed = bool(
                    exc.recoverable and decision.recoverable and decision.change is not None
                )
                externally_resumable = bool(
                    exc.recoverable
                    and exc.classification is ErrorClassification.DISK_FULL
                    and attempt.attempt < 3
                )
                codex_review = self._codex_recovery_review(
                    context,
                    error=exc,
                    backend_version=backend.version,
                    decision=decision,
                    current_plan=plan,
                    selected_gpus=capabilities,
                    previous_records=recovery_records,
                    retry_allowed=retry_allowed,
                )
                recovery_record = {
                    "attempt": attempt.attempt,
                    "classification": exc.classification.value,
                    "recoverable": exc.recoverable,
                    "status": (
                        "failed_recoverable"
                        if retry_allowed or externally_resumable
                        else "failed_fatal"
                    ),
                    "retry_applied": retry_allowed,
                    "resume_required": externally_resumable,
                    "change": decision.change,
                    "reason": decision.reason,
                    "quality_impact": decision.quality_impact,
                    "plan_before": plan.model_dump(mode="json"),
                    "plan_after": decision.plan.model_dump(mode="json"),
                    "dataset_config": str(dataset_config),
                    "request_fingerprint": attempt.request_fingerprint,
                    "command_argv": list(getattr(exc, "command_argv", ())),
                    "gpu_lease": lease_audit,
                    "codex": codex_review,
                }
                recovery_records.append(recovery_record)
                ApplicationArtifactStore(
                    Database(context.layout.database)
                ).persist_training_failure(
                    context,
                    recovery_record,
                )
                if not retry_allowed:
                    if exc.recoverable:
                        if externally_resumable:
                            raise PipelineError(
                                f"{exc} Free disk space, then resume this project.",
                                classification=exc.classification,
                                recoverable=True,
                            ) from exc
                        raise PipelineError(
                            f"{exc} Automatic recovery stopped: {decision.reason}",
                            classification=exc.classification,
                            recoverable=False,
                        ) from exc
                    raise
                if decision.change is not None:
                    previous_changes.append(decision.change)
                plan = decision.plan
                dataset_config = write_attempt_dataset_config(
                    run_directory=run_directory,
                    image_dir=context.layout.dataset / "all-accepted",
                    plan=plan,
                    validation_total_count=(
                        training_image_count if plan.validation_enabled else None
                    ),
                )
                continue
            break

        if not result.checkpoints:
            raise RuntimeError("Training produced no valid checkpoints")
        return {
            "backend_version": backend.version,
            "checkpoints": [item.model_dump(mode="json") for item in result.checkpoints],
            "state_path": str(result.state_path),
            "log_path": str(result.log_path),
            "completed_steps": result.completed_steps,
            "resumed": result.resumed,
            "attempt": attempt.attempt,
            "resumed_from_attempt": attempt.resumed_from_attempt,
            "request_fingerprint": attempt.request_fingerprint,
            "command_argv": list(result.command_argv),
            "plan": plan.model_dump(mode="json"),
            "dataset_config": str(dataset_config),
            "gpu_lease": lease_audit,
            "recovery_records": recovery_records,
            "quality_impact": any(item["quality_impact"] for item in recovery_records),
        }

    def _sampler(
        self,
        context: PipelineContext,
        *,
        binding: GpuBinding | None = None,
    ) -> SamplerBackend:
        if context.config.backend_mode is BackendMode.FAKE:
            return FakeSampler()
        from lora_factory.sampling.sd_scripts_backend import SdScriptsSampler

        plan = TrainingPlan.model_validate(_stage_output(context, PipelineStage.TRAINING)["plan"])
        if binding is None:
            raise RuntimeError("A selected-GPU lease binding is required for real sampling")
        return SdScriptsSampler(
            python_executable=self.runtime.layout.python,
            sd_scripts_root=self.runtime.layout.sd_scripts,
            commit=str(self.runtime.manifest["sd_scripts"]["commit"]),
            binding=binding,
            precision=plan.precision,
            cancellation=context.cancellation,
        )

    @contextmanager
    def _sampler_session(
        self, context: PipelineContext, *, pass_number: int
    ) -> Iterator[tuple[SamplerBackend, str]]:
        if context.config.backend_mode is BackendMode.FAKE:
            selected = context.config.selected_gpu_uuids
            yield self._sampler(context), selected[(pass_number - 1) % len(selected)]
            return
        plan = TrainingPlan.model_validate(_stage_output(context, PipelineStage.TRAINING)["plan"])
        selected = context.config.selected_gpu_uuids
        training_index = selected.index(plan.training_gpu_uuid)
        preferred_gpu_uuid = selected[(training_index + pass_number - 1) % len(selected)]
        with lease_selected_gpu(
            database=Database(context.layout.database),
            selected_uuids=context.config.selected_gpu_uuids,
            run_id=context.run_id,
            task=GpuTaskKind.SAMPLE,
            estimated_vram_mb=6144,
            preferred_gpu_uuid=preferred_gpu_uuid,
        ) as (binding, _lease):
            yield self._sampler(context, binding=binding), binding.uuid

    def _run_sampling_matrix(
        self,
        context: PipelineContext,
        *,
        sampler: SamplerBackend,
        sampling_gpu_uuid: str,
        pass_number: int,
        checkpoint_subset: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        seeds: Sequence[int],
        prompt_subset: Sequence[BenchmarkPrompt],
        output: Path,
        total: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for checkpoint in checkpoint_subset:
            for weight in weights:
                for seed in seeds:
                    for prompt in prompt_subset:
                        if len(results) >= self.settings.max_sample_images_per_pass:
                            return results
                        context.cancellation.raise_if_cancelled()
                        request = SampleRequest(
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                            checkpoint_path=Path(str(checkpoint["path"])),
                            base_model_path=context.config.base_model,
                            prompt_id=prompt.id,
                            prompt=prompt.prompt.replace("{trigger}", context.config.trigger_token),
                            negative_prompt=prompt.negative_prompt,
                            seed=seed,
                            weight=weight,
                            width=512 if context.config.backend_mode is BackendMode.FAKE else 768,
                            height=512 if context.config.backend_mode is BackendMode.FAKE else 768,
                            steps=8 if context.config.backend_mode is BackendMode.FAKE else 24,
                        )
                        result = sampler.sample(request, output)
                        results.append(result.model_dump(mode="json"))
                        completed = len(results)
                        context.events.publish(
                            PipelineEvent(
                                event_type="sampling_progress",
                                stage=(
                                    PipelineStage.SCREENING.value
                                    if pass_number == 1
                                    else PipelineStage.WEIGHT_SWEEP.value
                                    if pass_number == 2
                                    else PipelineStage.MULTI_SEED_VALIDATION.value
                                ),
                                message=f"Generated sample {completed}/{total}",
                                progress=completed / total,
                                details=progress_details(
                                    context,
                                    interval_seconds=self.settings.telemetry_interval_seconds,
                                    current_task="SAMPLING",
                                    images_current=completed,
                                    images_total=total,
                                    active_gpu_uuid=sampling_gpu_uuid,
                                ),
                            )
                        )
        return results

    def _sampling_stage(self, context: PipelineContext, *, pass_number: int) -> dict[str, Any]:
        checkpoints = _stage_output(context, PipelineStage.TRAINING)["checkpoints"]
        prompts = load_benchmark_prompts(context.config.preset)
        weights: tuple[float, ...]
        seeds: tuple[int, ...]
        if pass_number == 1:
            checkpoint_subset = checkpoints
            weights, seeds, prompt_subset = (0.8,), (101,), prompts[:2]
        elif pass_number == 2:
            checkpoint_subset = checkpoints[-3:]
            weights, seeds, prompt_subset = (0.6, 0.8, 1.0), (202,), prompts[2:5]
        else:
            checkpoint_subset = checkpoints[-2:]
            weights, seeds, prompt_subset = (0.7, 0.9), (303, 404), prompts[5:]
        output = context.layout.run(context.run_id).samples / f"pass-{pass_number}"
        uncapped_total = len(checkpoint_subset) * len(weights) * len(seeds) * len(prompt_subset)
        total = max(1, min(uncapped_total, self.settings.max_sample_images_per_pass))
        with self._sampler_session(context, pass_number=pass_number) as (
            sampler,
            sampling_gpu_uuid,
        ):
            results = self._run_sampling_matrix(
                context,
                sampler=sampler,
                sampling_gpu_uuid=sampling_gpu_uuid,
                pass_number=pass_number,
                checkpoint_subset=checkpoint_subset,
                weights=weights,
                seeds=seeds,
                prompt_subset=prompt_subset,
                output=output,
                total=total,
            )
        return {
            "backend_version": sampler.version,
            "pass_number": pass_number,
            "samples": results,
            "success_count": sum(item["success"] for item in results),
            "configured_image_cap": self.settings.max_sample_images_per_pass,
            "uncapped_image_count": uncapped_total,
            "gpu_uuid": sampling_gpu_uuid,
        }

    def _evaluation_stage(self, context: PipelineContext) -> dict[str, Any]:
        checkpoints = _stage_output(context, PipelineStage.TRAINING)["checkpoints"]
        samples = [
            *_stage_output(context, PipelineStage.SCREENING)["samples"],
            *_stage_output(context, PipelineStage.WEIGHT_SWEEP)["samples"],
            *_stage_output(context, PipelineStage.MULTI_SEED_VALIDATION)["samples"],
        ]
        successful = [
            item for item in samples if item["success"] and Path(str(item["image_path"])).is_file()
        ]
        generated_batch = self._tag_many_sharded(
            context,
            task=GpuTaskKind.GENERATED_TAG,
            paths=[Path(str(item["image_path"])) for item in successful],
            asset_ids=[f"generated-{index:04d}" for index in range(len(successful))],
            work_directory=context.layout.run(context.run_id).evaluation / "generated-tagging",
        )
        generated_results = generated_batch.results
        tags_by_path = {
            str(result.source_path): tuple(tag.name for tag in result.tags if tag.selected)
            for result in generated_results
        }
        generated_tags_path = context.layout.run(context.run_id).evaluation / "generated-tags.json"
        write_json_atomic(
            generated_tags_path,
            [result.model_dump(mode="json") for result in generated_results],
        )
        accepted_ids = _stage_output(context, PipelineStage.DATASET_REVIEW)["accepted_asset_ids"]
        reference_paths = [context.layout.working / f"{asset_id}.png" for asset_id in accepted_ids]
        embedding_paths = [
            *reference_paths,
            *(Path(str(item["image_path"])) for item in successful),
        ]
        embedding_work = context.layout.run(context.run_id).evaluation / "image-embedding"
        embedding_result = self._embed_many_sharded(
            context,
            task=GpuTaskKind.GENERATED_EMBED,
            paths=embedding_paths,
            work_directory=embedding_work,
        )
        embedding_batch = embedding_result.result
        embedding_gpu_uuid = embedding_result.gpu_uuids[0]
        embedding_path = embedding_work / "embeddings.json"
        write_json_atomic(embedding_path, embedding_batch.model_dump(mode="json"))
        learned_embeddings = embedding_batch.vectors_by_path()
        invariant_tags = tuple(
            str(item) for item in _stage_output(context, PipelineStage.CAPTIONING)["invariants"]
        )
        metrics: list[CandidateMetrics] = []
        for checkpoint in checkpoints:
            candidate_samples = [
                item
                for item in samples
                if item["request"]["checkpoint_id"] == checkpoint["checkpoint_id"]
            ]
            signals = aggregate_candidate_signals(
                reference_paths=reference_paths,
                observations=[
                    SampleObservation(
                        path=Path(str(item["image_path"])),
                        prompt_id=str(item["request"]["prompt_id"]),
                        prompt=str(item["request"]["prompt"]),
                        seed=int(item["request"]["seed"]),
                        weight=float(item["request"]["weight"]),
                        success=bool(item["success"]),
                        tags=tags_by_path.get(str(Path(str(item["image_path"])).resolve()), ()),
                    )
                    for item in candidate_samples
                ],
                trigger_token=context.config.trigger_token,
                invariant_tags=invariant_tags,
                learned_embeddings=learned_embeddings,
                embedding_model_id=embedding_batch.model_id,
                embedding_revision=embedding_batch.revision,
            )
            validation_loss = checkpoint.get("validation_loss")
            validation = 0.5 if validation_loss is None else max(0.0, 1.0 - float(validation_loss))
            metrics.append(
                CandidateMetrics(
                    checkpoint_id=checkpoint["checkpoint_id"],
                    identity=(
                        signals.reference_similarity
                        if context.config.preset is PresetKind.CHARACTER
                        else 0.0
                    ),
                    style_similarity=(
                        signals.reference_similarity
                        if context.config.preset is PresetKind.STYLE
                        else 0.0
                    ),
                    prompt_compliance=signals.prompt_compliance,
                    flexibility=signals.flexibility,
                    consistency=signals.consistency,
                    validation=validation,
                    technical=signals.technical,
                    visual_heuristic=signals.visual_heuristic,
                    overfit_risk=signals.memorization_risk,
                    generation_failure_rate=signals.generation_failure_rate,
                    recommended_weight=signals.recommended_weight,
                    evidence={
                        **signals.evidence,
                        "epoch": checkpoint["epoch"],
                        "step": checkpoint["step"],
                    },
                )
            )
        evaluation_profile = load_preset_profile(context.config.preset).evaluation
        ranked = rank_candidates(
            metrics,
            context.config.preset,
            weights=evaluation_profile.weights,
            overfit_penalty_weight=evaluation_profile.overfit_penalty_weight,
        )
        evaluation_path = context.layout.run(context.run_id).evaluation / "ranking.json"
        write_json_atomic(
            evaluation_path,
            {
                "metrics": [item.model_dump(mode="json") for item in metrics],
                "ranking": [item.model_dump(mode="json") for item in ranked],
                "generated_tags": str(generated_tags_path),
                "image_embedding": {
                    "model_id": embedding_batch.model_id,
                    "revision": embedding_batch.revision,
                    "dimension": embedding_batch.dimension,
                    "device": embedding_batch.device,
                    "path": str(embedding_path),
                    "gpu_uuid": embedding_gpu_uuid,
                },
            },
        )
        return {
            "metrics": [item.model_dump(mode="json") for item in metrics],
            "ranking": [item.model_dump(mode="json") for item in ranked],
            "path": str(evaluation_path),
            "generated_tags": str(generated_tags_path),
            "generated_tag_gpu_uuid": generated_batch.gpu_uuids[0],
            "generated_tag_gpu_uuids": list(generated_batch.gpu_uuids),
            "image_embedding": {
                "model_id": embedding_batch.model_id,
                "revision": embedding_batch.revision,
                "dimension": embedding_batch.dimension,
                "device": embedding_batch.device,
                "path": str(embedding_path),
            },
            "image_embedding_gpu_uuid": embedding_gpu_uuid,
            "image_embedding_gpu_uuids": list(embedding_result.gpu_uuids),
        }

    def _codex_final_stage(self, context: PipelineContext) -> dict[str, Any]:
        ranking = _stage_output(context, PipelineStage.FINAL_EVALUATION)["ranking"]
        response, audit, warning = self._codex_review(
            context,
            CodexTaskType.FINAL_REVIEW,
            {
                "preset": context.config.preset.value,
                "numeric_ranking": [item["checkpoint_id"] for item in ranking],
                "candidate_scores": {item["checkpoint_id"]: item["score"] for item in ranking},
            },
        )
        return {"response": response, "audit": audit, "warning": warning}

    def _selection_stage(self, context: PipelineContext) -> dict[str, Any]:
        ranking = _stage_output(context, PipelineStage.FINAL_EVALUATION)["ranking"]
        checkpoints = {
            item["checkpoint_id"]: item
            for item in _stage_output(context, PipelineStage.TRAINING)["checkpoints"]
        }
        valid = [item for item in ranking if item["hard_gate_passed"]]
        if not valid:
            raise RuntimeError("No checkpoint passed the technical selection gates")
        selected = valid[0]
        alternatives = valid[1:4]
        return {
            "selected": {**selected, "path": checkpoints[selected["checkpoint_id"]]["path"]},
            "alternatives": [
                {**item, "path": checkpoints[item["checkpoint_id"]]["path"]}
                for item in alternatives
            ],
        }

    def _packaging_stage(self, context: PipelineContext) -> dict[str, Any]:
        selection = _stage_output(context, PipelineStage.SELECTING)
        all_samples = [
            *_stage_output(context, PipelineStage.SCREENING)["samples"],
            *_stage_output(context, PipelineStage.WEIGHT_SWEEP)["samples"],
            *_stage_output(context, PipelineStage.MULTI_SEED_VALIDATION)["samples"],
        ]
        selected_id = selection["selected"]["checkpoint_id"]
        selected_samples = [
            item
            for item in all_samples
            if item["request"]["checkpoint_id"] == selected_id and item["success"]
        ]
        if not selected_samples:
            raise RuntimeError("Selected checkpoint has no successful preview sample")
        preview = Path(selected_samples[0]["image_path"])
        comparison = context.layout.run(context.run_id).evaluation / "comparison.png"
        render_grid(
            [
                (
                    Path(item["image_path"]),
                    f"{item['request']['checkpoint_id']} w={item['request']['weight']}",
                )
                for item in all_samples[:12]
                if item["success"]
            ],
            comparison,
        )
        plan = TrainingPlan.model_validate(_stage_output(context, PipelineStage.TRAINING)["plan"])
        model = _stage_output(context, PipelineStage.PREFLIGHT)["model"]
        training_output = _stage_output(context, PipelineStage.TRAINING)
        planning_output = _stage_output(context, PipelineStage.PLANNING)
        review_output = _stage_output(context, PipelineStage.DATASET_REVIEW)
        raw_manifest = self._manifest(context)
        source_images = [
            {
                "asset_id": asset.asset_id,
                "sha256": asset.sha256,
                "size_bytes": asset.size_bytes,
                "original_filenames": [source.original_filename for source in asset.sources],
            }
            for asset in raw_manifest.raw_assets
        ]
        dataset_decisions = [
            {
                "asset_id": item["asset_id"],
                "accepted": bool(item["included"]),
                "category": item["category"],
                "categories": item.get("categories", [item["category"]]),
                "reasons": item["reasons"],
            }
            for item in review_output["items"]
        ]
        sampling_seeds = sorted(
            {
                int(item["request"]["seed"])
                for item in all_samples
                if isinstance(item.get("request"), dict)
            }
        )
        sampling_weights = sorted(
            {
                float(item["request"]["weight"])
                for item in all_samples
                if isinstance(item.get("request"), dict)
            }
        )
        sampling_prompt_ids = sorted(
            {
                str(item["request"]["prompt_id"])
                for item in all_samples
                if isinstance(item.get("request"), dict)
            }
        )
        gpu_assignments = {
            "tagging": _stage_output(context, PipelineStage.TAGGING).get("gpu_uuids", []),
            "reference_embedding": review_output["reference_embedding"].get("gpu_uuids", []),
            "training": [plan.training_gpu_uuid],
            "screening": [_stage_output(context, PipelineStage.SCREENING).get("gpu_uuid")],
            "weight_sweep": [_stage_output(context, PipelineStage.WEIGHT_SWEEP).get("gpu_uuid")],
            "multi_seed_validation": [
                _stage_output(context, PipelineStage.MULTI_SEED_VALIDATION).get("gpu_uuid")
            ],
            "generated_tagging": _stage_output(context, PipelineStage.FINAL_EVALUATION).get(
                "generated_tag_gpu_uuids", []
            ),
            "image_embedding": [
                *_stage_output(context, PipelineStage.FINAL_EVALUATION).get(
                    "image_embedding_gpu_uuids", []
                )
            ],
        }
        output_directory = _safe_output_directory(
            context.config.output_root, context.config.lora_name
        )
        selected = selection["selected"]
        alternatives = selection["alternatives"]
        finalized = Finalizer().finalize(
            FinalizationRequest(
                output_directory=output_directory,
                lora_name=context.config.lora_name,
                trigger_token=context.config.trigger_token,
                preset=context.config.preset,
                base_model=context.config.base_model,
                base_model_sha256=model["sha256"],
                selected=CheckpointCandidate(
                    checkpoint_id=selected["checkpoint_id"],
                    path=Path(selected["path"]),
                    score=selected["score"],
                    recommended_weight=selected["recommended_weight"],
                ),
                alternatives=tuple(
                    CheckpointCandidate(
                        checkpoint_id=item["checkpoint_id"],
                        path=Path(item["path"]),
                        score=item["score"],
                        recommended_weight=item["recommended_weight"],
                    )
                    for item in alternatives
                ),
                preview=preview,
                comparison=comparison,
                resolved_config=context.config.model_dump(mode="json"),
                evaluation=_stage_output(context, PipelineStage.FINAL_EVALUATION),
                training_info={
                    "resolution": plan.resolution,
                    "network_dim": plan.network_dim,
                    "network_alpha": plan.network_alpha,
                    "training_image_count": planning_output["statistics"]["accepted_count"],
                    "recommended_range": "0.60-1.00",
                    "backend": context.config.backend_mode.value,
                    "epochs": plan.epochs,
                    "estimated_steps": plan.estimated_steps,
                    "optimizer": plan.optimizer,
                    "precision": plan.precision,
                    "validation_enabled": plan.validation_enabled,
                },
                reproducibility={
                    "schema_version": 1,
                    "run_id": context.run_id,
                    "trigger_word": context.config.trigger_token,
                    "codex_refinement": {
                        "mode": context.config.codex_refinement_mode.value,
                        "trigger_word_mode": context.config.trigger_word_mode.value,
                        "codex_image_profile": _stage_output(
                            context, PipelineStage.CODEX_REFINEMENT
                        ).get("image_profile", {}),
                        "codex_image_count": len(
                            _stage_output(context, PipelineStage.CODEX_REFINEMENT).get(
                                "ordered_image_hashes", ()
                            )
                        ),
                        "ordered_image_hashes": _stage_output(
                            context, PipelineStage.CODEX_REFINEMENT
                        ).get("ordered_image_hashes", []),
                        "batch_input_hashes": [
                            audit["input_hash"]
                            for audit in _stage_output(context, PipelineStage.CODEX_REFINEMENT).get(
                                "audits", []
                            )
                            if isinstance(audit, Mapping)
                            and isinstance(audit.get("input_hash"), str)
                        ],
                        "cleanup_status": _stage_output(
                            context, PipelineStage.CODEX_REFINEMENT
                        ).get("cleanup_status", "not_applicable"),
                        "fallback_used": bool(
                            _stage_output(context, PipelineStage.CODEX_REFINEMENT).get(
                                "fallback_used", False
                            )
                        ),
                        "chunk_count": int(
                            _stage_output(context, PipelineStage.CODEX_REFINEMENT).get(
                                "chunk_count", 0
                            )
                        ),
                        "audits": _stage_output(context, PipelineStage.CODEX_REFINEMENT).get(
                            "audits", []
                        ),
                        "approval_fingerprint": _stage_output(context, PipelineStage.CAPTIONING)[
                            "refinement_fingerprint"
                        ],
                    },
                    "factory_version": __version__,
                    "factory_python": (
                        f"{sys.version_info.major}.{sys.version_info.minor}."
                        f"{sys.version_info.micro}"
                    ),
                    "backend_manifest": self.runtime.manifest,
                    "preset": context.config.preset.value,
                    "preset_schema_version": load_preset_profile(
                        context.config.preset
                    ).schema_version,
                    "source_images": source_images,
                    "dataset_decisions": dataset_decisions,
                    "dataset_diversity": review_output["diversity"],
                    "dataset_statistics": planning_output["statistics"],
                    "dataset_reference_embedding": review_output["reference_embedding"],
                    "training_plan": plan.model_dump(mode="json"),
                    "batch_probe": planning_output["batch_probe"],
                    "training_attempt": training_output["attempt"],
                    "training_resumed": training_output["resumed"],
                    "training_command_arguments": training_output["command_argv"],
                    "training_dataset_config": training_output["dataset_config"],
                    "checkpoints": training_output["checkpoints"],
                    "validation_split": planning_output["split"],
                    "seeds": {
                        "training": 42,
                        "validation": plan.validation_seed,
                        "sampling": sampling_seeds,
                    },
                    "sampling": {
                        "prompt_ids": sampling_prompt_ids,
                        "weights": sampling_weights,
                        "image_count": len(all_samples),
                    },
                    "gpu_capabilities": planning_output["gpu_capabilities"],
                    "gpu_assignments": gpu_assignments,
                    "training_gpu_lease": training_output.get("gpu_lease"),
                    "recovery_changes": training_output.get("recovery_records", []),
                    "codex_pretrain_audit": _stage_output(
                        context, PipelineStage.CODEX_PRETRAIN_REVIEW
                    )["audit"],
                    "codex_pretrain_reviews": _stage_output(
                        context, PipelineStage.CODEX_PRETRAIN_REVIEW
                    )["reviews"],
                    "codex_pretrain_audits": _stage_output(
                        context, PipelineStage.CODEX_PRETRAIN_REVIEW
                    )["audits"],
                    "codex_final_audit": _stage_output(context, PipelineStage.CODEX_FINAL_REVIEW)[
                        "audit"
                    ],
                    "destination_copies": [],
                },
                warnings=tuple(
                    warning
                    for warning in (
                        *tuple(
                            str(item)
                            for item in _stage_output(context, PipelineStage.CODEX_REFINEMENT).get(
                                "warnings", ()
                            )
                        ),
                        _stage_output(context, PipelineStage.CODEX_PRETRAIN_REVIEW)["warning"],
                        _stage_output(context, PipelineStage.CODEX_FINAL_REVIEW)["warning"],
                    )
                    if warning
                ),
            )
        )
        alternative_map = {
            item["checkpoint_id"]: str(path)
            for item, path in zip(alternatives, finalized.alternative_models, strict=True)
        }
        return {
            "output_directory": str(finalized.directory),
            "final_model": str(finalized.final_model),
            "preview": str(finalized.preview),
            "comparison": str(finalized.comparison),
            "alternatives": alternative_map,
            "sha256": finalized.sha256,
        }

    def _ready_stage(self, context: PipelineContext) -> dict[str, Any]:
        before = _stage_output(context, PipelineStage.IMPORTING)["raw_snapshot"]
        after = verify_raw_store(context.layout)
        compare_raw_snapshots(dict(before), after)
        package = _stage_output(context, PipelineStage.PACKAGING)
        selected = _stage_output(context, PipelineStage.SELECTING)["selected"]
        alternatives = _stage_output(context, PipelineStage.SELECTING)["alternatives"]
        return {
            **package,
            "final_model_path": package["final_model"],
            "preview_path": package["preview"],
            "comparison_path": package["comparison"],
            "comparison_grid_path": package["comparison"],
            "project_id": context.config.project_id,
            "run_id": context.run_id,
            "status": "READY",
            "backend_mode": context.config.backend_mode.value,
            "trigger_token": context.config.trigger_token,
            "preset": context.config.preset.value,
            "base_model": str(context.config.base_model),
            "recommended_weight": selected["recommended_weight"],
            "recommended_range": [0.6, 1.0],
            "score_summary": {
                "selected_checkpoint": selected["checkpoint_id"],
                "score": selected["score"],
            },
            "top_alternatives": alternatives,
            "raw_integrity_verified": True,
        }
