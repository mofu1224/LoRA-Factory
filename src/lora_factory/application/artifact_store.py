"""Materialize successful pipeline artifacts into the project audit database."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from lora_factory.core.context import PipelineContext
from lora_factory.core.stage import PipelineStage
from lora_factory.project.manifest import DatasetManifest
from lora_factory.storage.database import Database
from lora_factory.storage.orm import (
    AssetRow,
    CaptionRow,
    CheckpointRow,
    CodexCallRow,
    DatasetSplitRow,
    DuplicateClusterRow,
    EvaluationRow,
    SampleRow,
    SourceFileRow,
    TagResultRow,
    TrainingAttemptRow,
    TrainingPlanRow,
)
from lora_factory.util.json import read_json


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(root.resolve(strict=False)))
    except ValueError:
        return str(resolved)


class ApplicationArtifactStore:
    """Idempotent per-stage projections for queryable audit/history tables."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def persist(
        self,
        context: PipelineContext,
        stage: PipelineStage,
        output: dict[str, Any],
    ) -> None:
        handler = getattr(self, f"_persist_{stage.value.casefold()}", None)
        if callable(handler):
            handler(context, output)

    def _persist_importing(self, context: PipelineContext, _output: dict[str, Any]) -> None:
        manifest = DatasetManifest.model_validate(read_json(context.layout.manifest))
        with self.database.session() as session:
            for asset in manifest.raw_assets:
                session.merge(
                    AssetRow(
                        id=asset.asset_id,
                        project_id=context.config.project_id,
                        kind="raw",
                        relative_path=str(Path("dataset") / "raw" / asset.stored_filename),
                        sha256=asset.sha256,
                        size_bytes=asset.size_bytes,
                        metadata_json={"extension": asset.extension, "immutable": True},
                    )
                )
                for source in asset.sources:
                    existing = session.scalar(
                        select(SourceFileRow.id).where(
                            SourceFileRow.asset_id == asset.asset_id,
                            SourceFileRow.original_path == str(source.original_path),
                        )
                    )
                    if existing is None:
                        session.add(
                            SourceFileRow(
                                asset_id=asset.asset_id,
                                original_path=str(source.original_path),
                                original_filename=source.original_filename,
                            )
                        )

    def _persist_normalizing(self, context: PipelineContext, output: dict[str, Any]) -> None:
        with self.database.session() as session:
            for item in output["items"]:
                path = Path(str(item["path"]))
                session.merge(
                    AssetRow(
                        id=f"working-{item['asset_id']}",
                        project_id=context.config.project_id,
                        kind="normalized",
                        relative_path=_relative_or_absolute(path, context.layout.root),
                        sha256=str(item["sha256"]),
                        size_bytes=path.stat().st_size,
                        metadata_json={
                            "source_asset_id": str(item["asset_id"]),
                            "width": int(item["width"]),
                            "height": int(item["height"]),
                            "warnings": list(item["warnings"]),
                        },
                    )
                )

    def _persist_deduplicating(self, context: PipelineContext, output: dict[str, Any]) -> None:
        with self.database.session() as session:
            session.execute(
                delete(DuplicateClusterRow).where(
                    DuplicateClusterRow.project_id == context.config.project_id
                )
            )
            for item in output["clusters"]:
                scores = item.get("scores", [])
                confidence = float(scores[0]["total"]) if scores else 0.0
                session.add(
                    DuplicateClusterRow(
                        id=str(item["cluster_id"]),
                        project_id=context.config.project_id,
                        representative_asset_id=str(item["representative_id"]),
                        member_ids_json=[str(value) for value in item["members"]],
                        method=str(item["kind"]),
                        confidence=confidence,
                    )
                )

    def _persist_tagging(self, _context: PipelineContext, output: dict[str, Any]) -> None:
        asset_ids = [str(item["asset_id"]) for item in output["results"]]
        with self.database.session() as session:
            if asset_ids:
                session.execute(delete(TagResultRow).where(TagResultRow.asset_id.in_(asset_ids)))
            for item in output["results"]:
                session.add(
                    TagResultRow(
                        asset_id=str(item["asset_id"]),
                        model_id=str(item["model_id"]),
                        model_revision=str(item["revision"]),
                        tags_json={str(tag["name"]): float(tag["score"]) for tag in item["tags"]},
                    )
                )

    def _persist_captioning(self, context: PipelineContext, output: dict[str, Any]) -> None:
        captions = {str(key): str(value) for key, value in output["captions"].items()}
        with self.database.session() as session:
            if captions:
                session.execute(delete(CaptionRow).where(CaptionRow.asset_id.in_(captions)))
            for asset_id, caption in captions.items():
                session.add(
                    CaptionRow(
                        asset_id=asset_id,
                        preset=context.config.preset.value,
                        trigger_token=context.config.trigger_token,
                        caption=caption,
                        audit_json=dict(output["audit"]),
                    )
                )

    def _persist_planning(self, context: PipelineContext, output: dict[str, Any]) -> None:
        split = output["split"]
        with self.database.session() as session:
            session.execute(delete(DatasetSplitRow).where(DatasetSplitRow.run_id == context.run_id))
            for split_name, key in (
                ("training", "train_asset_ids"),
                ("validation", "validation_asset_ids"),
            ):
                for asset_id in split[key]:
                    session.add(
                        DatasetSplitRow(
                            run_id=context.run_id,
                            asset_id=str(asset_id),
                            split=split_name,
                            seed=int(split["seed"]),
                        )
                    )
            session.execute(delete(TrainingPlanRow).where(TrainingPlanRow.run_id == context.run_id))
            session.add(
                TrainingPlanRow(
                    run_id=context.run_id,
                    plan_json=dict(output["plan"]),
                    selected=True,
                )
            )

    def _persist_codex_pretrain_review(
        self, context: PipelineContext, output: dict[str, Any]
    ) -> None:
        self._persist_codex(context, PipelineStage.CODEX_PRETRAIN_REVIEW, output)

    def _persist_codex_final_review(self, context: PipelineContext, output: dict[str, Any]) -> None:
        self._persist_codex(context, PipelineStage.CODEX_FINAL_REVIEW, output)

    def _persist_codex(
        self,
        context: PipelineContext,
        stage: PipelineStage,
        output: dict[str, Any],
    ) -> None:
        raw_audits = output.get("audits")
        audits = (
            [dict(item) for item in raw_audits if isinstance(item, dict)]
            if isinstance(raw_audits, list)
            else [dict(output["audit"])]
        )
        with self.database.session() as session:
            for audit in audits:
                task_type = str(audit.get("task_type") or stage.value.casefold())
                call_id = str(
                    audit.get("call_id") or f"{context.run_id}-{stage.value.casefold()}-{task_type}"
                )
                session.merge(
                    CodexCallRow(
                        id=call_id,
                        run_id=context.run_id,
                        task_type=task_type,
                        audit_json=audit,
                        applied_changes_json=dict(output.get("applied_changes", {})),
                    )
                )

    def _persist_training(self, context: PipelineContext, output: dict[str, Any]) -> None:
        recovery_records = [
            dict(item) for item in output.get("recovery_records", []) if isinstance(item, dict)
        ]
        attempt_number = int(output.get("attempt", 1))
        command = [str(item) for item in output.get("command_argv", [])]
        with self.database.session() as session:
            for record in recovery_records:
                self._merge_training_failure(session, context, record)
            attempt = session.scalar(
                select(TrainingAttemptRow).where(
                    TrainingAttemptRow.run_id == context.run_id,
                    TrainingAttemptRow.attempt == attempt_number,
                )
            )
            if attempt is None:
                attempt = TrainingAttemptRow(
                    run_id=context.run_id,
                    attempt=attempt_number,
                    status="completed",
                    command_json=command,
                    recovery_json={
                        "resumed": bool(output.get("resumed", False)),
                        "resumed_from_attempt": output.get("resumed_from_attempt"),
                        "request_fingerprint": output.get("request_fingerprint"),
                        "gpu_lease": output.get("gpu_lease"),
                        "prior_recovery_attempts": [
                            int(item["attempt"]) for item in recovery_records
                        ],
                    },
                    return_code=0,
                )
                session.add(attempt)
            else:
                attempt.status = "completed"
                attempt.command_json = command
                attempt.recovery_json = {
                    "resumed": bool(output.get("resumed", False)),
                    "resumed_from_attempt": output.get("resumed_from_attempt"),
                    "request_fingerprint": output.get("request_fingerprint"),
                    "gpu_lease": output.get("gpu_lease"),
                    "prior_recovery_attempts": [int(item["attempt"]) for item in recovery_records],
                }
                attempt.return_code = 0
            for item in output["checkpoints"]:
                session.merge(
                    CheckpointRow(
                        id=str(item["checkpoint_id"]),
                        run_id=context.run_id,
                        path=str(item["path"]),
                        epoch=int(item["epoch"]),
                        step=int(item["step"]),
                        size_bytes=int(item["size_bytes"]),
                        sha256=str(item["sha256"]),
                        metadata_json=dict(item.get("metadata", {})),
                        validation_loss=item.get("validation_loss"),
                        train_loss=item.get("train_loss"),
                        status="valid",
                    )
                )

    def persist_training_started(
        self,
        context: PipelineContext,
        *,
        attempt: int,
        request_fingerprint: str,
        resume_state: Path | None,
        resumed_from_attempt: int | None,
        gpu_lease: dict[str, Any] | None,
    ) -> None:
        """Record an attempt before the backend process can terminate unexpectedly."""

        with self.database.session() as session:
            existing = session.scalar(
                select(TrainingAttemptRow).where(
                    TrainingAttemptRow.run_id == context.run_id,
                    TrainingAttemptRow.attempt == attempt,
                )
            )
            if existing is not None:
                raise RuntimeError(
                    f"Training attempt history already contains attempt {attempt}; "
                    "refusing to overwrite it"
                )
            session.add(
                TrainingAttemptRow(
                    run_id=context.run_id,
                    attempt=attempt,
                    status="running",
                    command_json=[],
                    recovery_json={
                        "request_fingerprint": request_fingerprint,
                        "resume_state": str(resume_state) if resume_state is not None else None,
                        "resumed_from_attempt": resumed_from_attempt,
                        "gpu_lease": gpu_lease,
                    },
                    return_code=None,
                )
            )

    def persist_training_failure(
        self,
        context: PipelineContext,
        record: dict[str, Any],
    ) -> None:
        """Durably record an attempt even when the TRAINING stage never succeeds."""

        with self.database.session() as session:
            self._merge_training_failure(session, context, record)

    @staticmethod
    def _merge_training_failure(
        session: Any,
        context: PipelineContext,
        record: dict[str, Any],
    ) -> None:
        failed_number = int(record["attempt"])
        codex = record.get("codex")
        if isinstance(codex, dict) and isinstance(codex.get("audit"), dict):
            audit = dict(codex["audit"])
            call_id = str(audit.get("call_id") or f"{context.run_id}-recovery-{failed_number}")
            session.merge(
                CodexCallRow(
                    id=call_id,
                    run_id=context.run_id,
                    task_type="recovery",
                    audit_json=audit,
                    applied_changes_json=dict(codex.get("applied_changes", {})),
                )
            )
        failed = session.scalar(
            select(TrainingAttemptRow).where(
                TrainingAttemptRow.run_id == context.run_id,
                TrainingAttemptRow.attempt == failed_number,
            )
        )
        requested_status = record.get("status")
        status = (
            str(requested_status)
            if requested_status in {"cancelled", "failed_recoverable", "failed_fatal"}
            else "failed_recoverable"
            if record.get("retry_applied")
            else "failed_fatal"
        )
        if failed is None:
            session.add(
                TrainingAttemptRow(
                    run_id=context.run_id,
                    attempt=failed_number,
                    status=status,
                    command_json=[str(item) for item in record.get("command_argv", [])],
                    recovery_json=record,
                    return_code=1,
                )
            )
            return
        failed.status = status
        failed.command_json = [str(item) for item in record.get("command_argv", [])]
        failed.recovery_json = record
        failed.return_code = 1

    def _persist_screening(self, context: PipelineContext, output: dict[str, Any]) -> None:
        self._persist_samples(context, output)

    def _persist_weight_sweep(self, context: PipelineContext, output: dict[str, Any]) -> None:
        self._persist_samples(context, output)

    def _persist_multi_seed_validation(
        self, context: PipelineContext, output: dict[str, Any]
    ) -> None:
        self._persist_samples(context, output)

    def _persist_samples(self, _context: PipelineContext, output: dict[str, Any]) -> None:
        with self.database.session() as session:
            for item in output["samples"]:
                request = dict(item["request"])
                identity = json.dumps(request, ensure_ascii=True, sort_keys=True).encode("utf-8")
                sample_id = f"sample-{hashlib.sha256(identity).hexdigest()[:32]}"
                session.merge(
                    SampleRow(
                        id=sample_id,
                        checkpoint_id=str(request["checkpoint_id"]),
                        path=str(item["image_path"]),
                        prompt_id=str(request["prompt_id"]),
                        seed=int(request["seed"]),
                        weight=float(request["weight"]),
                        metadata_json={
                            "request": request,
                            "metadata_path": str(item["metadata_path"]),
                            "success": bool(item["success"]),
                            "error": item.get("error"),
                        },
                    )
                )

    def _persist_final_evaluation(self, _context: PipelineContext, output: dict[str, Any]) -> None:
        ranking = {str(item["checkpoint_id"]): item for item in output["ranking"]}
        checkpoint_ids = [str(item["checkpoint_id"]) for item in output["metrics"]]
        with self.database.session() as session:
            if checkpoint_ids:
                session.execute(
                    delete(EvaluationRow).where(EvaluationRow.checkpoint_id.in_(checkpoint_ids))
                )
            for metrics in output["metrics"]:
                checkpoint_id = str(metrics["checkpoint_id"])
                ranked = ranking[checkpoint_id]
                session.add(
                    EvaluationRow(
                        checkpoint_id=checkpoint_id,
                        score=float(ranked["score"]),
                        recommended_weight=float(ranked["recommended_weight"]),
                        metrics_json=dict(metrics),
                        warnings_json=[str(item) for item in ranked.get("warnings", [])],
                    )
                )
