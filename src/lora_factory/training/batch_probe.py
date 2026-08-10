"""Bounded, selected-GPU batch-size probes used before SDXL training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.process_manager import ProcessManager


class BatchProbeError(RuntimeError):
    """Raised when no requested batch size fits the live selected GPU."""


class BatchProbeRequest(BaseModel):
    """Validated inputs for one side-effect-free VRAM probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_uuid: str
    resolution: Literal[768, 896, 1024]
    network_dim: Annotated[int, Field(ge=1, le=512)]
    precision: Literal["bf16", "fp16", "fp32"]
    candidate_batch_sizes: tuple[Annotated[int, Field(ge=1, le=32)], ...]
    work_directory: Path
    safety_margin_mb: Annotated[int, Field(ge=256, le=8192)] = 1024
    bounded_allocation_mb: Annotated[int, Field(ge=8, le=128)] = 32

    @model_validator(mode="after")
    def validate_candidates(self) -> BatchProbeRequest:
        if not self.candidate_batch_sizes:
            raise ValueError("Batch probe requires at least one candidate")
        if tuple(sorted(set(self.candidate_batch_sizes), reverse=True)) != (
            self.candidate_batch_sizes
        ):
            raise ValueError("Batch probe candidates must be unique and strictly descending")
        return self


class BatchProbeAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: Annotated[int, Field(ge=1, le=32)]
    estimated_vram_mb: Annotated[int, Field(gt=0)]
    observed_free_vram_mb: Annotated[int, Field(ge=0)]
    succeeded: bool
    detail: str


class BatchProbeResult(BaseModel):
    """Auditable probe result consumed by the final Training Planner pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_version: str
    method: Literal["deterministic_fake_vram", "torch_cuda_live_vram"]
    gpu_uuid: str
    selected_batch_size: Annotated[int, Field(ge=1, le=32)]
    observed_total_vram_mb: Annotated[int, Field(gt=0)]
    observed_initial_free_vram_mb: Annotated[int, Field(ge=0)]
    safety_margin_mb: Annotated[int, Field(ge=256, le=8192)]
    bounded_allocation_mb: Annotated[int, Field(ge=0, le=128)]
    attempts: tuple[BatchProbeAttempt, ...]
    command_argv: tuple[str, ...] = ()
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    training_artifacts_created: bool = False

    @model_validator(mode="after")
    def validate_selection(self) -> BatchProbeResult:
        successful = [attempt for attempt in self.attempts if attempt.succeeded]
        if len(successful) != 1 or successful[0].batch_size != self.selected_batch_size:
            raise ValueError("Batch probe result must contain exactly one selected attempt")
        if self.training_artifacts_created:
            raise ValueError("A pre-training batch probe cannot create training artifacts")
        return self


def estimate_training_vram_mb(
    *,
    resolution: Literal[768, 896, 1024],
    batch_size: int,
    network_dim: int,
    precision: Literal["bf16", "fp16", "fp32"],
) -> int:
    """Conservative reservation model shared by probe and GPU lease scheduling."""

    if batch_size < 1 or network_dim < 1:
        raise ValueError("Batch size and network dimension must be positive")
    base_by_resolution = {768: 6144, 896: 8192, 1024: 10240}
    batch_overhead = (batch_size - 1) * 1536
    rank_overhead = max(0, network_dim - 32) * 8
    precision_overhead = 2048 if precision == "fp32" else 0
    return base_by_resolution[resolution] + batch_overhead + rank_overhead + precision_overhead


def descending_batch_candidates(maximum: int) -> tuple[int, ...]:
    """Return a bounded descending halving ladder that always reaches batch one."""

    if maximum < 1 or maximum > 32:
        raise ValueError("Maximum batch size must be between 1 and 32")
    candidates: list[int] = []
    value = maximum
    while value > 1:
        candidates.append(value)
        value = max(1, value // 2)
    candidates.append(1)
    return tuple(dict.fromkeys(candidates))


def _attempts(
    request: BatchProbeRequest,
    *,
    observed_free_vram_mb: int,
) -> tuple[tuple[BatchProbeAttempt, ...], int | None]:
    attempts: list[BatchProbeAttempt] = []
    selected: int | None = None
    for batch_size in request.candidate_batch_sizes:
        required = estimate_training_vram_mb(
            resolution=request.resolution,
            batch_size=batch_size,
            network_dim=request.network_dim,
            precision=request.precision,
        )
        succeeded = observed_free_vram_mb >= required + request.safety_margin_mb
        attempts.append(
            BatchProbeAttempt(
                batch_size=batch_size,
                estimated_vram_mb=required,
                observed_free_vram_mb=observed_free_vram_mb,
                succeeded=succeeded,
                detail=(
                    "Live free VRAM satisfies the conservative reservation and safety margin"
                    if succeeded
                    else "Conservative reservation plus safety margin exceeds live free VRAM"
                ),
            )
        )
        if succeeded:
            selected = batch_size
            break
    return tuple(attempts), selected


class FakeBatchProbe:
    """Deterministic test provider with the same selection semantics as the real probe."""

    version = "fake-batch-probe/1"

    def __init__(self, *, free_vram_mb: int, total_vram_mb: int) -> None:
        if free_vram_mb < 0 or total_vram_mb <= 0 or free_vram_mb > total_vram_mb:
            raise ValueError("Invalid Fake batch-probe VRAM snapshot")
        self.free_vram_mb = free_vram_mb
        self.total_vram_mb = total_vram_mb

    def probe(
        self,
        request: BatchProbeRequest,
        cancellation: CancellationToken,
    ) -> BatchProbeResult:
        cancellation.raise_if_cancelled()
        attempts, selected = _attempts(request, observed_free_vram_mb=self.free_vram_mb)
        if selected is None:
            raise BatchProbeError(
                f"No candidate batch fits Fake GPU {request.gpu_uuid}; "
                f"free VRAM is {self.free_vram_mb} MiB"
            )
        return BatchProbeResult(
            provider_version=self.version,
            method="deterministic_fake_vram",
            gpu_uuid=request.gpu_uuid,
            selected_batch_size=selected,
            observed_total_vram_mb=self.total_vram_mb,
            observed_initial_free_vram_mb=self.free_vram_mb,
            safety_margin_mb=request.safety_margin_mb,
            bounded_allocation_mb=0,
            attempts=attempts,
        )


_TORCH_PROBE = r"""
import json
import sys

import torch

payload = json.loads(sys.argv[1])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the selected-GPU batch probe")
torch.cuda.set_device(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}[payload["precision"]]
element_size = torch.empty((), dtype=dtype).element_size()
elements = max(1, payload["bounded_allocation_mb"] * 1024 * 1024 // element_size)
allocation = torch.empty(elements, device="cuda:0", dtype=dtype)
allocation.add_(1)
torch.cuda.synchronize()
del allocation
torch.cuda.empty_cache()
print(json.dumps({
    "free_vram_mb": free_bytes // (1024 * 1024),
    "total_vram_mb": total_bytes // (1024 * 1024),
}, sort_keys=True))
""".strip()


class TorchCudaBatchProbe:
    """Production live-VRAM provider isolated to one selected physical GPU."""

    version = "torch-cuda-batch-probe/1"

    def __init__(
        self,
        *,
        python_executable: Path,
        binding: GpuBinding,
        selected_gpu_uuids: tuple[str, ...],
        process_manager: ProcessManager | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        if binding.uuid not in selected_gpu_uuids:
            raise ValueError("Batch-probe GPU binding is outside the selected pool")
        if timeout_seconds <= 0:
            raise ValueError("Batch-probe timeout must be positive")
        self.python_executable = python_executable.resolve(strict=False)
        self.binding = binding
        self.process_manager = process_manager or ProcessManager()
        self.timeout_seconds = timeout_seconds

    def probe(
        self,
        request: BatchProbeRequest,
        cancellation: CancellationToken,
    ) -> BatchProbeResult:
        if request.gpu_uuid != self.binding.uuid:
            raise ValueError("Batch-probe request and selected GPU binding differ")
        cancellation.raise_if_cancelled()
        request.work_directory.mkdir(parents=True, exist_ok=True)
        stdout_path = request.work_directory / "batch-probe.stdout.log"
        stderr_path = request.work_directory / "batch-probe.stderr.log"
        child_payload = json.dumps(
            {
                "precision": request.precision,
                "bounded_allocation_mb": request.bounded_allocation_mb,
            },
            sort_keys=True,
        )
        actual_argv = [str(self.python_executable), "-c", _TORCH_PROBE, child_payload]
        process = self.process_manager.run(
            actual_argv,
            cwd=request.work_directory,
            environment=self.binding.environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cancellation=cancellation,
            timeout_seconds=self.timeout_seconds,
            on_line=lambda _channel, _line: None,
        )
        if process.cancelled:
            raise CancelledError("Selected-GPU batch probe was cancelled")
        if process.timed_out:
            raise TimeoutError(f"Selected-GPU batch probe exceeded {self.timeout_seconds} seconds")
        if process.return_code != 0:
            detail = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            raise BatchProbeError(
                f"Selected-GPU batch probe exited with {process.return_code}: "
                f"{detail[-2000:] or 'no diagnostics'}"
            )
        try:
            output_lines = [
                line
                for line in stdout_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            payload = json.loads(output_lines[-1])
            observed_free = int(payload["free_vram_mb"])
            observed_total = int(payload["total_vram_mb"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BatchProbeError("Selected-GPU batch probe returned invalid JSON") from exc
        attempts, selected = _attempts(request, observed_free_vram_mb=observed_free)
        if selected is None:
            raise BatchProbeError(
                f"No candidate batch fits selected GPU {request.gpu_uuid}; "
                f"live free VRAM is {observed_free} MiB"
            )
        redacted_argv = (
            str(self.python_executable),
            "-c",
            "<lora-factory bounded CUDA batch probe>",
            child_payload,
        )
        return BatchProbeResult(
            provider_version=self.version,
            method="torch_cuda_live_vram",
            gpu_uuid=request.gpu_uuid,
            selected_batch_size=selected,
            observed_total_vram_mb=observed_total,
            observed_initial_free_vram_mb=observed_free,
            safety_margin_mb=request.safety_margin_mb,
            bounded_allocation_mb=request.bounded_allocation_mb,
            attempts=attempts,
            command_argv=redacted_argv,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
