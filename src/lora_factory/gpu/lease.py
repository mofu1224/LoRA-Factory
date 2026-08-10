"""SQLite-backed GPU lease acquisition, heartbeat, and stale recovery."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import psutil
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select

from lora_factory.gpu.models import GpuTaskKind
from lora_factory.storage.database import Database
from lora_factory.storage.orm import GpuLeaseRow


class LeaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lease_id: int = Field(ge=1)
    gpu_uuid: str
    task: GpuTaskKind
    run_id: str
    worker_id: str
    pid: int = Field(ge=0)
    estimated_vram_mb: int = Field(ge=0)
    acquired_at: datetime
    heartbeat_at: datetime


class LeaseConflict(RuntimeError):
    """Raised when a device already has an active lease."""


def _record(row: GpuLeaseRow) -> LeaseRecord:
    return LeaseRecord(
        lease_id=row.id,
        gpu_uuid=row.gpu_uuid,
        task=GpuTaskKind(row.task),
        run_id=row.run_id,
        worker_id=row.worker_id,
        pid=row.pid,
        estimated_vram_mb=row.estimated_vram_mb,
        acquired_at=row.acquired_at,
        heartbeat_at=row.heartbeat_at,
    )


class GpuLeaseManager:
    """Persist GPU ownership for the single scheduler process."""

    def __init__(self, database: Database, selected_uuids: Sequence[str]) -> None:
        if not selected_uuids:
            raise ValueError("At least one GPU UUID must be selected")
        if len(set(selected_uuids)) != len(selected_uuids):
            raise ValueError("Selected GPU UUIDs must be unique")
        self.database = database
        self.selected_uuids = frozenset(selected_uuids)

    def acquire(
        self,
        *,
        gpu_uuid: str,
        task: GpuTaskKind,
        run_id: str,
        worker_id: str,
        pid: int,
        estimated_vram_mb: int,
        now: datetime | None = None,
    ) -> LeaseRecord:
        if gpu_uuid not in self.selected_uuids:
            raise ValueError(f"GPU {gpu_uuid} is outside the selected pool")
        if pid < 0 or estimated_vram_mb < 0:
            raise ValueError("pid and estimated VRAM must be non-negative")
        timestamp = now or datetime.now(UTC)
        with self.database.session() as session:
            existing = session.scalar(
                select(GpuLeaseRow).where(GpuLeaseRow.gpu_uuid == gpu_uuid).limit(1)
            )
            if existing is not None:
                raise LeaseConflict(f"GPU {gpu_uuid} is leased by worker {existing.worker_id}")
            row = GpuLeaseRow(
                gpu_uuid=gpu_uuid,
                task=task.value,
                run_id=run_id,
                worker_id=worker_id,
                pid=pid,
                estimated_vram_mb=estimated_vram_mb,
                acquired_at=timestamp,
                heartbeat_at=timestamp,
            )
            session.add(row)
            session.flush()
            return _record(row)

    def heartbeat(
        self, lease_id: int, *, worker_id: str, now: datetime | None = None
    ) -> LeaseRecord:
        with self.database.session() as session:
            row = session.get(GpuLeaseRow, lease_id)
            if row is None or row.worker_id != worker_id:
                raise KeyError(f"Unknown GPU lease for worker {worker_id!r}: {lease_id}")
            row.heartbeat_at = now or datetime.now(UTC)
            session.flush()
            return _record(row)

    def release(self, lease_id: int, *, worker_id: str) -> None:
        with self.database.session() as session:
            row = session.get(GpuLeaseRow, lease_id)
            if row is None or row.worker_id != worker_id:
                raise KeyError(f"Unknown GPU lease for worker {worker_id!r}: {lease_id}")
            session.delete(row)

    def active(self) -> tuple[LeaseRecord, ...]:
        with self.database.session() as session:
            rows = session.scalars(select(GpuLeaseRow).order_by(GpuLeaseRow.id)).all()
            return tuple(_record(row) for row in rows)

    def recover_stale(
        self,
        *,
        heartbeat_timeout: timedelta,
        now: datetime | None = None,
        process_alive: Callable[[int], bool] = psutil.pid_exists,
    ) -> tuple[int, ...]:
        """Release timed-out leases only after confirming the owning PID is gone."""

        if heartbeat_timeout <= timedelta(0):
            raise ValueError("heartbeat_timeout must be positive")
        cutoff = (now or datetime.now(UTC)) - heartbeat_timeout
        with self.database.session() as session:
            stale_rows = session.scalars(
                select(GpuLeaseRow).where(GpuLeaseRow.heartbeat_at < cutoff)
            ).all()
            stale_ids = tuple(row.id for row in stale_rows if not process_alive(row.pid))
            if stale_ids:
                session.execute(delete(GpuLeaseRow).where(GpuLeaseRow.id.in_(stale_ids)))
            return stale_ids
