"""SQLite engine setup, migrations, and crash recovery."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Engine, create_engine, event, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from lora_factory.core.stage import RunStatus, StageStatus
from lora_factory.storage.orm import Base, GpuLeaseRow, RunRow, StageRow, TrainingAttemptRow


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite+pysqlite:///{path}", future=True)
        event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def recover_interrupted(self) -> tuple[int, int, int]:
        """Mark abandoned work recoverable and release process-bound GPU leases."""

        now = datetime.now(UTC)
        with self.session() as session:
            active_run_ids = select(RunRow.id).where(RunRow.status == RunStatus.ACTIVE.value)
            attempts = session.scalars(
                select(TrainingAttemptRow).where(
                    TrainingAttemptRow.run_id.in_(active_run_ids),
                    TrainingAttemptRow.status == "running",
                )
            ).all()
            for attempt in attempts:
                attempt.status = "failed_recoverable"
                attempt.recovery_json = {
                    **attempt.recovery_json,
                    "classification": "PROCESS_CRASH",
                    "recoverable": True,
                    "retry_applied": False,
                }
            stages = cast(
                CursorResult[Any],
                session.execute(
                    update(StageRow)
                    .where(StageRow.status == StageStatus.RUNNING.value)
                    .values(
                        status=StageStatus.FAILED.value,
                        finished_at=now,
                        error_json={"classification": "PROCESS_CRASH", "recoverable": True},
                    )
                ),
            ).rowcount
            runs = cast(
                CursorResult[Any],
                session.execute(
                    update(RunRow)
                    .where(RunRow.status == RunStatus.ACTIVE.value)
                    .values(
                        status=RunStatus.FAILED_RECOVERABLE.value,
                        error_json={"classification": "PROCESS_CRASH", "recoverable": True},
                    )
                ),
            ).rowcount
            leases = session.query(GpuLeaseRow).delete()
        return int(stages or 0), int(runs or 0), int(leases or 0)
