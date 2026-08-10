"""Run-scoped image progress and throttled selected-GPU telemetry payloads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import monotonic
from typing import Any

from lora_factory.config.models import BackendMode
from lora_factory.core.context import PipelineContext
from lora_factory.gpu.models import GpuTelemetrySample
from lora_factory.gpu.telemetry import TelemetryError, sample_telemetry

type TelemetrySampler = Callable[[Sequence[str]], tuple[GpuTelemetrySample, ...]]
type Clock = Callable[[], float]

_REPORTER_KEY = "_application_progress_reporter"


class PipelineProgressReporter:
    """Cache telemetry for the configured interval while updating current task details."""

    def __init__(
        self,
        selected_gpu_uuids: Sequence[str],
        *,
        interval_seconds: float,
        fake_backend: bool,
        sampler: TelemetrySampler = sample_telemetry,
        clock: Clock = monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Telemetry interval must be positive")
        self.selected_gpu_uuids = tuple(selected_gpu_uuids)
        self.interval_seconds = interval_seconds
        self.fake_backend = fake_backend
        self.sampler = sampler
        self.clock = clock
        self._last_sample_at: float | None = None
        self._cached: list[dict[str, Any]] = []

    def details(
        self,
        *,
        current_task: str,
        images_current: int,
        images_total: int,
        active_gpu_uuid: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if images_current < 0 or images_total < 0 or images_current > images_total:
            raise ValueError("Image progress must satisfy 0 <= current <= total")
        warning = self._refresh_if_due()
        gpus: list[dict[str, Any]] = []
        for cached in self._cached:
            row = dict(cached)
            row["task"] = (
                current_task
                if active_gpu_uuid is None or row["uuid"] == active_gpu_uuid
                else "Idle"
            )
            gpus.append(row)
        payload: dict[str, Any] = {
            "images_current": images_current,
            "images_total": images_total,
            "gpus": gpus,
        }
        if warning:
            payload["telemetry_warning"] = warning
        if extra:
            payload.update(extra)
        return payload

    def _refresh_if_due(self) -> str:
        now = self.clock()
        if (
            self._cached
            and self._last_sample_at is not None
            and now - self._last_sample_at < self.interval_seconds
        ):
            return ""
        self._last_sample_at = now
        if self.fake_backend:
            self._cached = [
                {
                    "uuid": uuid,
                    "utilization_percent": 0.0,
                    "memory_used_mb": 0,
                    "memory_free_mb": 24 * 1024,
                    "temperature_c": 0.0,
                    "power_watts": 0.0,
                    "simulated": True,
                }
                for uuid in self.selected_gpu_uuids
            ]
            return ""
        try:
            samples = self.sampler(self.selected_gpu_uuids)
        except TelemetryError as exc:
            if not self._cached:
                self._cached = [
                    {
                        "uuid": uuid,
                        "utilization_percent": 0.0,
                        "memory_used_mb": 0,
                        "memory_free_mb": 0,
                        "telemetry_unavailable": True,
                    }
                    for uuid in self.selected_gpu_uuids
                ]
            return str(exc)
        self._cached = [sample.model_dump(mode="json") for sample in samples]
        return ""


def progress_details(
    context: PipelineContext,
    *,
    interval_seconds: float,
    current_task: str,
    images_current: int,
    images_total: int,
    active_gpu_uuid: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reporter = context.artifacts.get(_REPORTER_KEY)
    if not isinstance(reporter, PipelineProgressReporter):
        reporter = PipelineProgressReporter(
            context.config.selected_gpu_uuids,
            interval_seconds=interval_seconds,
            fake_backend=context.config.backend_mode is BackendMode.FAKE,
        )
        context.artifacts[_REPORTER_KEY] = reporter
    return reporter.details(
        current_task=current_task,
        images_current=images_current,
        images_total=images_total,
        active_gpu_uuid=active_gpu_uuid,
        extra=extra,
    )
