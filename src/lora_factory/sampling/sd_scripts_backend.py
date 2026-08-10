"""Pinned sd-scripts SDXL sampler adapter with one selected visible GPU."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.models import GpuBinding
from lora_factory.sampling.backend import SampleRequest, SampleResult
from lora_factory.training.process_manager import ManagedProcessResult, ProcessManager
from lora_factory.util.json import write_json_atomic


class ProcessRunner(Protocol):
    def run(
        self,
        arguments: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
        cancellation: CancellationToken,
        timeout_seconds: int | None,
        on_line: Callable[[str, str], None],
    ) -> ManagedProcessResult: ...


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip(".-")
    return cleaned[:80] or "sample"


class SdScriptsSampler:
    def __init__(
        self,
        *,
        python_executable: Path,
        sd_scripts_root: Path,
        commit: str,
        gpu_uuid: str | None = None,
        binding: GpuBinding | None = None,
        precision: str = "bf16",
        process_manager: ProcessManager | None = None,
        cancellation: CancellationToken | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        if precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("Unsupported sampler precision")
        resolved_uuid = binding.uuid if binding is not None else gpu_uuid
        if resolved_uuid is None or not resolved_uuid.startswith("GPU-"):
            raise ValueError("A physical NVIDIA GPU UUID is required")
        self.python_executable = python_executable.resolve(strict=False)
        self.sd_scripts_root = sd_scripts_root.resolve(strict=False)
        self.gpu_uuid = resolved_uuid
        self.binding = binding
        self.commit = commit
        self.precision = precision
        self.process_manager = process_manager or ProcessManager()
        self.cancellation = cancellation or CancellationToken()
        self.timeout_seconds = timeout_seconds

    @property
    def version(self) -> str:
        return f"sd-scripts/{self.commit}"

    def build_arguments(self, request: SampleRequest, output_directory: Path) -> list[str]:
        prompt_expression = request.prompt
        if request.negative_prompt:
            prompt_expression = f"{prompt_expression} --n {request.negative_prompt}"
        arguments = [
            str(self.python_executable),
            str(self.sd_scripts_root / "sdxl_gen_img.py"),
            "--ckpt",
            str(request.base_model_path),
            "--outdir",
            str(output_directory),
            "--prompt",
            prompt_expression,
            "--W",
            str(request.width),
            "--H",
            str(request.height),
            "--steps",
            str(request.steps),
            "--sampler",
            request.sampler,
            "--scale",
            str(request.cfg_scale),
            "--seed",
            str(request.seed),
            "--images_per_prompt",
            "1",
            "--batch_size",
            "1",
            "--sdpa",
            "--network_module",
            "networks.lora",
            "--network_weights",
            str(request.checkpoint_path),
            "--network_mul",
            str(request.weight),
        ]
        if self.precision == "bf16":
            arguments.append("--bf16")
        elif self.precision == "fp16":
            arguments.append("--fp16")
        return arguments

    def sample(self, request: SampleRequest, output_directory: Path) -> SampleResult:
        request_root = output_directory / _safe_component(
            f"{request.checkpoint_id}-{request.prompt_id}-{request.seed}-{request.weight:.2f}"
        )
        request_root.mkdir(parents=True, exist_ok=True)
        arguments = self.build_arguments(request, request_root)
        process_result = self.process_manager.run(
            arguments,
            cwd=self.sd_scripts_root,
            environment=(
                self.binding.environment
                if self.binding is not None
                else {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            ),
            stdout_path=request_root / "stdout.log",
            stderr_path=request_root / "stderr.log",
            cancellation=self.cancellation,
            timeout_seconds=self.timeout_seconds,
            on_line=lambda _channel, _line: None,
        )
        images = sorted(request_root.rglob("*.png"), key=lambda path: path.stat().st_mtime_ns)
        if process_result.return_code != 0 or process_result.cancelled or process_result.timed_out:
            error = (
                "sd-scripts sampling was cancelled"
                if process_result.cancelled
                else "sd-scripts sampling timed out"
                if process_result.timed_out
                else f"sd-scripts sampler exited with {process_result.return_code}"
            )
            metadata_path = request_root / "failure.json"
            write_json_atomic(metadata_path, {"error": error, "arguments": arguments})
            return SampleResult(
                request=request,
                image_path=request_root / "missing.png",
                metadata_path=metadata_path,
                success=False,
                error=error,
            )
        if not images:
            raise RuntimeError("sd-scripts sampling succeeded but produced no PNG")
        image_path = images[-1]
        metadata_path = image_path.with_suffix(".json")
        write_json_atomic(
            metadata_path,
            {
                **request.model_dump(mode="json"),
                "backend": self.version,
                "gpu_uuid": self.gpu_uuid,
                "physical_index": (
                    self.binding.physical_index if self.binding is not None else None
                ),
                "logical_index": 0,
                "arguments": arguments,
                "image_filename": image_path.name,
            },
        )
        return SampleResult(
            request=request,
            image_path=image_path,
            metadata_path=metadata_path,
            success=True,
        )
