from __future__ import annotations

from pathlib import Path
from threading import Event

from sqlalchemy import func, select

from lora_factory.core.stage import RunStatus
from lora_factory.gpu.execution import lease_selected_gpu
from lora_factory.gpu.lease import GpuLeaseManager
from lora_factory.gpu.models import GpuDevice, GpuTaskKind
from lora_factory.storage.database import Database
from lora_factory.storage.orm import GpuDeviceRow, GpuLeaseRow, ProjectRow, RunRow


def _device(uuid: str, index: int, free: int) -> GpuDevice:
    return GpuDevice(
        uuid=uuid,
        index=index,
        name=f"GPU {index}",
        total_vram_mb=16_384,
        free_vram_mb=free,
        capability_major=12,
        capability_minor=0,
    )


def test_selected_gpu_execution_persists_heartbeats_and_releases_lease(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    first_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    second_uuid = "GPU-00000000-0000-0000-0000-000000000002"
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.session() as session:
        session.add(ProjectRow(id="p", name="p", root_path=str(tmp_path), config_json={}))
        session.add(RunRow(id="r", project_id="p", status=RunStatus.ACTIVE.value))

    heartbeat_seen = Event()
    original_heartbeat = GpuLeaseManager.heartbeat

    def observed_heartbeat(
        manager: GpuLeaseManager,
        lease_id: int,
        *,
        worker_id: str,
        now: object = None,
    ) -> object:
        result = original_heartbeat(
            manager,
            lease_id,
            worker_id=worker_id,
            now=now,
        )
        heartbeat_seen.set()
        return result

    monkeypatch.setattr(GpuLeaseManager, "heartbeat", observed_heartbeat)  # type: ignore[attr-defined]
    with lease_selected_gpu(
        database=database,
        selected_uuids=(first_uuid, second_uuid),
        run_id="r",
        task=GpuTaskKind.TRAIN,
        estimated_vram_mb=8_000,
        preferred_gpu_uuid=second_uuid,
        devices=(_device(first_uuid, 0, 15_000), _device(second_uuid, 1, 12_000)),
        heartbeat_interval_seconds=0.01,
    ) as (binding, lease):
        assert binding.uuid == second_uuid
        assert binding.physical_index == 1
        assert binding.logical_index == 0
        assert lease.gpu_uuid == second_uuid
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(GpuDeviceRow)) == 2
            assert session.scalar(select(func.count()).select_from(GpuLeaseRow)) == 1
        assert heartbeat_seen.wait(timeout=1)

    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(GpuLeaseRow)) == 0
