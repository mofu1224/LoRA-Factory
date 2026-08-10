"""Lightweight nvidia-smi telemetry sampling without importing CUDA libraries."""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Callable, Sequence
from io import StringIO

from lora_factory.gpu.models import GpuTelemetrySample

type TelemetryRunner = Callable[[tuple[str, ...], float], subprocess.CompletedProcess[str]]


class TelemetryError(RuntimeError):
    """Raised when a telemetry snapshot is malformed or unavailable."""


def _run_telemetry(
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
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def sample_telemetry(
    selected_uuids: Sequence[str],
    *,
    executable: str = "nvidia-smi",
    timeout_seconds: float = 10.0,
    runner: TelemetryRunner = _run_telemetry,
) -> tuple[GpuTelemetrySample, ...]:
    if not selected_uuids:
        return ()
    argv = (
        executable,
        "--query-gpu=uuid,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = runner(argv, timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TelemetryError(f"Unable to execute {executable!r}: {exc}") from exc
    if completed.returncode != 0:
        raise TelemetryError(f"nvidia-smi telemetry failed with exit code {completed.returncode}")
    selected = set(selected_uuids)
    samples: list[GpuTelemetrySample] = []
    for line_number, row in enumerate(csv.reader(StringIO(completed.stdout)), start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 6:
            raise TelemetryError(f"Telemetry row {line_number} has {len(row)} fields")
        uuid, utilization, memory_used, memory_free, temperature, power = (
            field.strip() for field in row
        )
        if uuid not in selected:
            continue
        try:
            samples.append(
                GpuTelemetrySample(
                    uuid=uuid,
                    utilization_percent=float(utilization),
                    memory_used_mb=int(float(memory_used)),
                    memory_free_mb=int(float(memory_free)),
                    temperature_c=float(temperature),
                    power_watts=float(power),
                )
            )
        except ValueError as exc:
            raise TelemetryError(f"Invalid telemetry row {line_number}: {row!r}") from exc
    missing = selected - {sample.uuid for sample in samples}
    if missing:
        raise TelemetryError(f"No telemetry returned for selected GPU: {sorted(missing)[0]}")
    return tuple(samples)
