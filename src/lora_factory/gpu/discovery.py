"""NVIDIA GPU discovery and UUID-first child-process isolation."""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from io import StringIO

from lora_factory.gpu.models import GpuBinding, GpuDevice

type DiscoveryRunner = Callable[[tuple[str, ...], float], subprocess.CompletedProcess[str]]


class GpuDiscoveryError(RuntimeError):
    """Raised when the NVIDIA inventory cannot be obtained or trusted."""


def _run_discovery(
    argv: tuple[str, ...], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )


def parse_nvidia_smi_inventory(output: str) -> tuple[GpuDevice, ...]:
    """Parse the stable CSV query emitted by :func:`discover_nvidia_gpus`."""

    devices: list[GpuDevice] = []
    for line_number, row in enumerate(csv.reader(StringIO(output)), start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 7:
            raise GpuDiscoveryError(
                f"nvidia-smi row {line_number} has {len(row)} fields; expected 7"
            )
        index, uuid, name, total, free, utilization, compute_capability = (
            field.strip() for field in row
        )
        try:
            capability_major, capability_minor = (
                int(part) for part in compute_capability.split(".", maxsplit=1)
            )
            device = GpuDevice(
                uuid=uuid,
                index=int(index),
                name=name,
                total_vram_mb=int(float(total)),
                free_vram_mb=int(float(free)),
                utilization_percent=float(utilization),
                capability_major=capability_major,
                capability_minor=capability_minor,
            )
        except (TypeError, ValueError) as exc:
            raise GpuDiscoveryError(f"Invalid nvidia-smi row {line_number}: {row!r}") from exc
        devices.append(device)

    if not devices:
        raise GpuDiscoveryError("nvidia-smi returned no CUDA devices")
    uuids = [device.uuid for device in devices]
    indices = [device.index for device in devices]
    if len(set(uuids)) != len(uuids):
        raise GpuDiscoveryError("nvidia-smi returned duplicate GPU UUIDs")
    if len(set(indices)) != len(indices):
        raise GpuDiscoveryError("nvidia-smi returned duplicate CUDA indices")
    return tuple(sorted(devices, key=lambda item: item.index))


def discover_nvidia_gpus(
    *,
    executable: str = "nvidia-smi",
    timeout_seconds: float = 10.0,
    runner: DiscoveryRunner = _run_discovery,
) -> tuple[GpuDevice, ...]:
    """Query UUID, current index, memory, utilization, and compute capability."""

    argv = (
        executable,
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_cap",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = runner(argv, timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuDiscoveryError(f"Unable to execute {executable!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise GpuDiscoveryError(
            f"{executable!r} exited with {completed.returncode}: {detail or 'no diagnostics'}"
        )
    return parse_nvidia_smi_inventory(completed.stdout)


def resolve_gpu_indices(
    selected_uuids: Sequence[str], devices: Sequence[GpuDevice]
) -> dict[str, int]:
    """Resolve a selected UUID pool against one fresh discovery snapshot."""

    if not selected_uuids:
        raise ValueError("At least one GPU UUID must be selected")
    if len(set(selected_uuids)) != len(selected_uuids):
        raise ValueError("Selected GPU UUIDs must be unique")
    by_uuid = {device.uuid: device.index for device in devices}
    missing = [uuid for uuid in selected_uuids if uuid not in by_uuid]
    if missing:
        raise GpuDiscoveryError(f"Selected GPU is no longer present: {missing[0]}")
    return {uuid: by_uuid[uuid] for uuid in selected_uuids}


def bind_gpu_for_child(
    gpu_uuid: str,
    *,
    selected_uuids: Sequence[str],
    devices: Sequence[GpuDevice],
    base_environment: Mapping[str, str] | None = None,
) -> GpuBinding:
    """Make one selected physical GPU visible as logical ``cuda:0`` to a child.

    The current CUDA index is resolved immediately before launch.  No caller-provided
    ``CUDA_VISIBLE_DEVICES`` value survives this boundary.
    """

    if gpu_uuid not in selected_uuids:
        raise ValueError(f"GPU {gpu_uuid} is outside the selected pool")
    physical_index = resolve_gpu_indices(selected_uuids, devices)[gpu_uuid]
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_index),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    return GpuBinding(
        uuid=gpu_uuid,
        physical_index=physical_index,
        logical_index=0,
        environment=environment,
    )
