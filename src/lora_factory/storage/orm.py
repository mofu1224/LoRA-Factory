"""SQLAlchemy schema for durable projects, stages, artifacts, and audits."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the per-project database."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)


class ProjectRow(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    root_path: Mapped[str] = mapped_column(Text, unique=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class RunRow(TimestampMixin, Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(32), index=True)
    current_stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class StageRow(TimestampMixin, Base):
    __tablename__ = "stages"
    __table_args__ = (UniqueConstraint("run_id", "stage", "attempt", name="uq_stage_attempt"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(40), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    stage_version: Mapped[str] = mapped_column(String(32))
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)


class AssetRow(TimestampMixin, Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), index=True)
    relative_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SourceFileRow(TimestampMixin, Base):
    __tablename__ = "source_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    original_path: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str] = mapped_column(Text)


class DuplicateClusterRow(TimestampMixin, Base):
    __tablename__ = "duplicate_clusters"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    representative_asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    member_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    method: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)


class TagResultRow(TimestampMixin, Base):
    __tablename__ = "tag_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[str] = mapped_column(Text)
    model_revision: Mapped[str] = mapped_column(String(64))
    tags_json: Mapped[dict[str, float]] = mapped_column(JSON)


class CaptionRow(TimestampMixin, Base):
    __tablename__ = "captions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    preset: Mapped[str] = mapped_column(String(16))
    trigger_token: Mapped[str] = mapped_column(String(64))
    caption: Mapped[str] = mapped_column(Text)
    audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class DatasetSplitRow(TimestampMixin, Base):
    __tablename__ = "dataset_splits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), index=True)
    split: Mapped[str] = mapped_column(String(16))
    seed: Mapped[int] = mapped_column(Integer)


class TrainingPlanRow(TimestampMixin, Base):
    __tablename__ = "training_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    selected: Mapped[bool] = mapped_column(Boolean, default=True)


class TrainingAttemptRow(TimestampMixin, Base):
    __tablename__ = "training_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24))
    command_json: Mapped[list[str]] = mapped_column(JSON)
    recovery_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CheckpointRow(TimestampMixin, Base):
    __tablename__ = "checkpoints"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    path: Mapped[str] = mapped_column(Text)
    epoch: Mapped[int] = mapped_column(Integer)
    step: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    validation_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24))


class SampleRow(TimestampMixin, Base):
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(
        ForeignKey("checkpoints.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(Text)
    prompt_id: Mapped[str] = mapped_column(String(80))
    seed: Mapped[int] = mapped_column(Integer)
    weight: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvaluationRow(TimestampMixin, Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id: Mapped[str] = mapped_column(
        ForeignKey("checkpoints.id", ondelete="CASCADE"), index=True
    )
    score: Mapped[float] = mapped_column(Float)
    recommended_weight: Mapped[float] = mapped_column(Float)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    warnings_json: Mapped[list[str]] = mapped_column(JSON, default=list)


class CodexCallRow(TimestampMixin, Base):
    __tablename__ = "codex_calls"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    task_type: Mapped[str] = mapped_column(String(40))
    audit_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    applied_changes_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class GpuDeviceRow(TimestampMixin, Base):
    __tablename__ = "gpu_devices"

    uuid: Mapped[str] = mapped_column(String(80), primary_key=True)
    last_index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(Text)
    capability: Mapped[str] = mapped_column(String(16))
    total_vram_mb: Mapped[int] = mapped_column(Integer)
    compatible: Mapped[bool] = mapped_column(Boolean)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class GpuLeaseRow(TimestampMixin, Base):
    __tablename__ = "gpu_leases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gpu_uuid: Mapped[str] = mapped_column(
        ForeignKey("gpu_devices.uuid", ondelete="CASCADE"), index=True
    )
    task: Mapped[str] = mapped_column(String(40))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    worker_id: Mapped[str] = mapped_column(String(80))
    pid: Mapped[int] = mapped_column(Integer)
    estimated_vram_mb: Mapped[int] = mapped_column(Integer)
    acquired_at: Mapped[datetime] = mapped_column(default=utc_now)
    heartbeat_at: Mapped[datetime] = mapped_column(default=utc_now, index=True)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, index=True)


class DestinationRow(TimestampMixin, Base):
    __tablename__ = "destinations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20))
    root_path: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_copy_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
