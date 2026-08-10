"""Deterministic scheduler constrained to the user's selected GPU pool."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from lora_factory.gpu.models import GpuAssignment, GpuDevice, GpuTaskKind, GpuTaskRequest


class NoGpuAvailable(RuntimeError):
    """Raised when no selected compatible device can satisfy a task."""


class SelectedGpuScheduler:
    """Assign one task to one selected physical GPU.

    A standard or candidate training request is never spread across multiple GPUs.
    Independent requests may be assigned independently by the caller.
    """

    def __init__(self, selected_uuids: Sequence[str]) -> None:
        if not selected_uuids:
            raise ValueError("At least one GPU UUID must be selected")
        if len(set(selected_uuids)) != len(selected_uuids):
            raise ValueError("Selected GPU UUIDs must be unique")
        self.selected_uuids = tuple(selected_uuids)

    def assign(
        self,
        request: GpuTaskRequest,
        devices: Sequence[GpuDevice],
        *,
        leased_gpu_uuids: Iterable[str] = (),
    ) -> GpuAssignment:
        leased = set(leased_gpu_uuids)
        selected_order = {uuid: position for position, uuid in enumerate(self.selected_uuids)}
        candidates = [
            device
            for device in devices
            if device.uuid in selected_order
            and device.uuid not in leased
            and device.compatible
            and device.free_vram_mb >= request.estimated_vram_mb
        ]
        if request.preferred_gpu_uuid is not None:
            if request.preferred_gpu_uuid not in selected_order:
                raise ValueError("Preferred GPU is outside the selected pool")
            preferred = [
                device for device in candidates if device.uuid == request.preferred_gpu_uuid
            ]
            if preferred:
                candidates = preferred

        if not candidates:
            raise NoGpuAvailable(
                f"No selected GPU has {request.estimated_vram_mb} MiB free for {request.kind.value}"
            )

        if request.kind in {GpuTaskKind.TRAIN, GpuTaskKind.OPTIONAL_CANDIDATE_TRAIN}:
            candidates.sort(
                key=lambda device: (
                    -device.free_vram_mb,
                    device.utilization_percent,
                    selected_order[device.uuid],
                )
            )
        else:
            candidates.sort(
                key=lambda device: (
                    device.utilization_percent,
                    -device.free_vram_mb,
                    selected_order[device.uuid],
                )
            )
        chosen = candidates[0]
        return GpuAssignment(
            task_id=request.task_id,
            kind=request.kind,
            gpu_uuid=chosen.uuid,
            physical_index=chosen.index,
            estimated_vram_mb=request.estimated_vram_mb,
        )

    def shard_items(
        self, item_ids: Sequence[str], devices: Sequence[GpuDevice]
    ) -> dict[str, tuple[str, ...]]:
        """Round-robin independent items over selected compatible devices."""

        by_uuid = {device.uuid: device for device in devices}
        missing = [uuid for uuid in self.selected_uuids if uuid not in by_uuid]
        if missing:
            raise NoGpuAvailable(f"Selected GPU is no longer present: {missing[0]}")
        candidates = [by_uuid[uuid] for uuid in self.selected_uuids if by_uuid[uuid].compatible]
        candidates.sort(key=lambda device: (-device.free_vram_mb, device.index))
        if not candidates:
            raise NoGpuAvailable("No compatible selected GPU is available for sharding")
        shards: dict[str, list[str]] = {device.uuid: [] for device in candidates}
        for position, item_id in enumerate(item_ids):
            shards[candidates[position % len(candidates)].uuid].append(item_id)
        return {uuid: tuple(items) for uuid, items in shards.items() if items}
