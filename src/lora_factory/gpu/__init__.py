"""UUID-first GPU discovery, scheduling, leasing, and telemetry."""

from lora_factory.gpu.discovery import (
    GpuDiscoveryError,
    bind_gpu_for_child,
    discover_nvidia_gpus,
    parse_nvidia_smi_inventory,
    resolve_gpu_indices,
)
from lora_factory.gpu.lease import GpuLeaseManager, LeaseConflict, LeaseRecord
from lora_factory.gpu.models import (
    GpuAssignment,
    GpuBinding,
    GpuDevice,
    GpuTaskKind,
    GpuTaskRequest,
    GpuTelemetrySample,
)
from lora_factory.gpu.scheduler import NoGpuAvailable, SelectedGpuScheduler
from lora_factory.gpu.telemetry import TelemetryError, sample_telemetry

__all__ = [
    "GpuAssignment",
    "GpuBinding",
    "GpuDevice",
    "GpuDiscoveryError",
    "GpuLeaseManager",
    "GpuTaskKind",
    "GpuTaskRequest",
    "GpuTelemetrySample",
    "LeaseConflict",
    "LeaseRecord",
    "NoGpuAvailable",
    "SelectedGpuScheduler",
    "TelemetryError",
    "bind_gpu_for_child",
    "discover_nvidia_gpus",
    "parse_nvidia_smi_inventory",
    "resolve_gpu_indices",
    "sample_telemetry",
]
