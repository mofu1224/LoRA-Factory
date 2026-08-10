"""Application-facing selected-GPU scheduling, binding, and durable leasing."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import timedelta
from threading import Event, Thread

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from lora_factory.gpu.discovery import bind_gpu_for_child, discover_nvidia_gpus
from lora_factory.gpu.lease import GpuLeaseManager, LeaseRecord
from lora_factory.gpu.models import GpuBinding, GpuDevice, GpuTaskKind, GpuTaskRequest
from lora_factory.gpu.scheduler import SelectedGpuScheduler
from lora_factory.storage.database import Database
from lora_factory.storage.orm import GpuDeviceRow, utc_now


@contextmanager
def lease_selected_gpu(
    *,
    database: Database,
    selected_uuids: Sequence[str],
    run_id: str,
    task: GpuTaskKind,
    estimated_vram_mb: int,
    preferred_gpu_uuid: str | None = None,
    devices: Sequence[GpuDevice] | None = None,
    heartbeat_interval_seconds: float = 30.0,
) -> Iterator[tuple[GpuBinding, LeaseRecord]]:
    """Lease one selected physical GPU and resolve logical cuda:0 immediately."""

    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    observed = tuple(devices) if devices is not None else discover_nvidia_gpus()
    if not observed:
        raise RuntimeError("No NVIDIA CUDA GPU is currently available")
    with database.session() as session:
        for device in observed:
            now = utc_now()
            values = {
                "uuid": device.uuid,
                "last_index": device.index,
                "name": device.name,
                "capability": f"sm_{device.capability_major}{device.capability_minor}",
                "total_vram_mb": device.total_vram_mb,
                "compatible": device.compatible,
                "details_json": {
                    "free_vram_mb": device.free_vram_mb,
                    "utilization_percent": device.utilization_percent,
                    "compatibility_reason": device.compatibility_reason,
                },
                "created_at": now,
                "updated_at": now,
            }
            statement = sqlite_insert(GpuDeviceRow).values(**values)
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[GpuDeviceRow.uuid],
                    set_={
                        "last_index": statement.excluded.last_index,
                        "name": statement.excluded.name,
                        "capability": statement.excluded.capability,
                        "total_vram_mb": statement.excluded.total_vram_mb,
                        "compatible": statement.excluded.compatible,
                        "details_json": statement.excluded.details_json,
                        "updated_at": now,
                    },
                )
            )

    leases = GpuLeaseManager(database, selected_uuids)
    leases.recover_stale(heartbeat_timeout=timedelta(minutes=2))
    assignment = SelectedGpuScheduler(selected_uuids).assign(
        GpuTaskRequest(
            task_id=f"{run_id}:{task.value}",
            kind=task,
            estimated_vram_mb=estimated_vram_mb,
            preferred_gpu_uuid=preferred_gpu_uuid,
        ),
        observed,
        leased_gpu_uuids=(record.gpu_uuid for record in leases.active()),
    )
    binding = bind_gpu_for_child(
        assignment.gpu_uuid,
        selected_uuids=selected_uuids,
        devices=observed,
    )
    worker_id = f"{run_id}:{task.value}:{uuid.uuid4().hex[:12]}"
    lease = leases.acquire(
        gpu_uuid=assignment.gpu_uuid,
        task=task,
        run_id=run_id,
        worker_id=worker_id,
        pid=os.getpid(),
        estimated_vram_mb=estimated_vram_mb,
    )
    stop_heartbeat = Event()
    heartbeat_errors: list[BaseException] = []

    def heartbeat_worker() -> None:
        while not stop_heartbeat.wait(heartbeat_interval_seconds):
            try:
                leases.heartbeat(lease.lease_id, worker_id=worker_id)
            except KeyError:
                return
            except BaseException as exc:
                heartbeat_errors.append(exc)
                return

    heartbeat_thread = Thread(
        target=heartbeat_worker,
        name=f"gpu-lease-heartbeat-{lease.lease_id}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield binding, lease
    finally:
        unwinding_exception = sys.exc_info()[0] is not None
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=5)
        # Crash recovery may have released the record while unwinding.
        with suppress(KeyError):
            leases.release(lease.lease_id, worker_id=worker_id)
        if heartbeat_errors and not unwinding_exception:
            raise RuntimeError("GPU lease heartbeat failed") from heartbeat_errors[0]
