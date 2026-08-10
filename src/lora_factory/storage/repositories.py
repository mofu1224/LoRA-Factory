"""Focused repositories used by pipeline orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from lora_factory.core.stage import StageStatus
from lora_factory.storage.database import Database
from lora_factory.storage.orm import StageRow


class StageRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def successful_output(self, run_id: str, stage: str, fingerprint: str) -> dict[str, Any] | None:
        with self.database.session() as session:
            row = session.scalar(
                select(StageRow)
                .where(
                    StageRow.run_id == run_id,
                    StageRow.stage == stage,
                    StageRow.fingerprint == fingerprint,
                    StageRow.status == StageStatus.SUCCEEDED.value,
                )
                .order_by(StageRow.attempt.desc())
                .limit(1)
            )
            return None if row is None else dict(row.output_json)

    def start(
        self,
        *,
        run_id: str,
        stage: str,
        fingerprint: str,
        stage_version: str,
        inputs: dict[str, Any],
    ) -> int:
        with self.database.session() as session:
            latest = session.scalar(
                select(StageRow.attempt)
                .where(StageRow.run_id == run_id, StageRow.stage == stage)
                .order_by(StageRow.attempt.desc())
                .limit(1)
            )
            row = StageRow(
                run_id=run_id,
                stage=stage,
                attempt=(latest or 0) + 1,
                status=StageStatus.RUNNING.value,
                fingerprint=fingerprint,
                stage_version=stage_version,
                input_json=inputs,
                started_at=datetime.now(UTC),
            )
            session.add(row)
            session.flush()
            return row.id

    def finish(self, row_id: int, *, output: dict[str, Any]) -> None:
        with self.database.session() as session:
            row = session.get(StageRow, row_id)
            if row is None:
                raise KeyError(f"Unknown stage row: {row_id}")
            row.status = StageStatus.SUCCEEDED.value
            row.output_json = output
            row.finished_at = datetime.now(UTC)

    def fail(self, row_id: int, *, error: dict[str, Any], cancelled: bool = False) -> None:
        with self.database.session() as session:
            row = session.get(StageRow, row_id)
            if row is None:
                raise KeyError(f"Unknown stage row: {row_id}")
            row.status = StageStatus.CANCELLED.value if cancelled else StageStatus.FAILED.value
            row.error_json = error
            row.finished_at = datetime.now(UTC)
