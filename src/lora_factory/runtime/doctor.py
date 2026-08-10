"""Managed runtime capability probes executed in selected-GPU child processes."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.gpu.models import GpuBinding
from lora_factory.runtime.compatibility_matrix import (
    CompatibilityDecision,
    evaluate_runtime_compatibility,
)

type ProbeRunner = Callable[
    [tuple[str, ...], Path | None, Mapping[str, str], float],
    subprocess.CompletedProcess[str],
]

_TORCH_PROBE = r"""
import json
import torch

result = {
    "torch_version": str(torch.__version__),
    "torch_cuda_version": str(torch.version.cuda or ""),
    "cuda_available": bool(torch.cuda.is_available()),
    "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    "tensor_smoke_ok": False,
    "bf16_supported": False,
    "bf16_smoke_ok": False,
    "capability_major": 0,
    "capability_minor": 0,
    "device_name": "",
}
if result["cuda_available"]:
    major, minor = torch.cuda.get_device_capability(0)
    result["capability_major"] = int(major)
    result["capability_minor"] = int(minor)
    result["device_name"] = torch.cuda.get_device_name(0)
    tensor = torch.tensor([2.0, 3.0], device="cuda")
    expected = torch.tensor([4.0, 9.0], device="cuda")
    result["tensor_smoke_ok"] = bool(torch.allclose(tensor * tensor, expected))
    result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    if result["bf16_supported"]:
        bf16 = torch.ones(32, dtype=torch.bfloat16, device="cuda")
        result["bf16_smoke_ok"] = bool(float((bf16 + bf16).float().mean().cpu()) == 2.0)
print(json.dumps(result, ensure_ascii=True))
""".strip()

_ONNX_PROBE = r"""
import json
import numpy as np
import onnxruntime as ort

providers = list(ort.get_available_providers())
cuda_provider = "CUDAExecutionProvider" in providers
transfer_ok = False
if cuda_provider:
    value = ort.OrtValue.ortvalue_from_numpy(np.asarray([1.0, 2.0], dtype=np.float32), "cuda", 0)
    transfer_ok = bool(np.array_equal(value.numpy(), np.asarray([1.0, 2.0], dtype=np.float32)))
print(json.dumps({
    "onnxruntime_version": str(ort.__version__),
    "providers": providers,
    "cuda_provider": cuda_provider,
    "cuda_transfer_smoke_ok": transfer_ok,
}, ensure_ascii=True))
""".strip()


class ProbeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    ok: bool
    argv: tuple[str, ...]
    return_code: int | None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class RuntimeDoctorReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_uuid: str
    torch: ProbeResult
    onnx: ProbeResult
    sd_scripts: ProbeResult
    compatibility: CompatibilityDecision | None
    ready: bool


def _run_probe(
    argv: tuple[str, ...],
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        list(argv),
        cwd=None if cwd is None else str(cwd),
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )


def _json_probe(
    *,
    name: str,
    argv: tuple[str, ...],
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float,
    runner: ProbeRunner,
) -> ProbeResult:
    try:
        completed = runner(argv, cwd, environment, timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(
            name=name,
            ok=False,
            argv=argv,
            return_code=None,
            error=str(exc),
        )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return ProbeResult(
            name=name,
            ok=False,
            argv=argv,
            return_code=completed.returncode,
            error=detail or "probe exited without diagnostics",
        )
    if not stdout_lines:
        return ProbeResult(
            name=name,
            ok=False,
            argv=argv,
            return_code=completed.returncode,
            error="probe returned no JSON result",
        )
    try:
        details = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as exc:
        return ProbeResult(
            name=name,
            ok=False,
            argv=argv,
            return_code=completed.returncode,
            error=f"invalid probe JSON: {exc}",
        )
    if not isinstance(details, dict):
        return ProbeResult(
            name=name,
            ok=False,
            argv=argv,
            return_code=completed.returncode,
            error="probe JSON must be an object",
        )
    return ProbeResult(
        name=name,
        ok=True,
        argv=argv,
        return_code=completed.returncode,
        details=details,
    )


def _sd_scripts_probe(module_name: str) -> str:
    if not module_name.replace("_", "").isalnum():
        raise ValueError(
            "sd-scripts module name must contain only letters, digits, and underscores"
        )
    return (
        "import importlib,json; "
        f"module=importlib.import_module({module_name!r}); "
        "print(json.dumps({'module': module.__name__, 'file': str(module.__file__)}, "
        "ensure_ascii=True))"
    )


def inspect_runtime(
    *,
    python_executable: Path,
    binding: GpuBinding,
    sd_scripts_root: Path,
    sd_scripts_module: str = "sdxl_train_network",
    timeout_seconds: float = 60.0,
    runner: ProbeRunner = _run_probe,
) -> RuntimeDoctorReport:
    """Run torch, bf16, ONNX CUDA, and sd-scripts import probes."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if binding.environment.get("CUDA_VISIBLE_DEVICES") != str(binding.physical_index):
        raise ValueError("GPU binding isolation environment was modified after validation")
    if binding.environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("GPU binding device order was modified after validation")
    python = str(python_executable)
    torch_result = _json_probe(
        name="torch_cuda",
        argv=(python, "-c", _TORCH_PROBE),
        cwd=None,
        environment=binding.environment,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    onnx_result = _json_probe(
        name="onnx_cuda",
        argv=(python, "-c", _ONNX_PROBE),
        cwd=None,
        environment=binding.environment,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    sd_scripts_result = _json_probe(
        name="sd_scripts_import",
        argv=(python, "-c", _sd_scripts_probe(sd_scripts_module)),
        cwd=sd_scripts_root,
        environment=binding.environment,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )

    compatibility: CompatibilityDecision | None = None
    if torch_result.ok:
        details = torch_result.details
        try:
            arch_list = details.get("arch_list", [])
            if not isinstance(arch_list, list):
                raise ValueError("arch_list must be an array")
            compatibility = evaluate_runtime_compatibility(
                capability_major=int(details.get("capability_major", 0)),
                capability_minor=int(details.get("capability_minor", 0)),
                torch_version=str(details.get("torch_version", "")),
                torch_cuda_version=str(details.get("torch_cuda_version", "")),
                torch_arch_list=tuple(str(item) for item in arch_list),
                tensor_smoke_ok=bool(details.get("tensor_smoke_ok", False)),
                bf16_smoke_ok=bool(details.get("bf16_smoke_ok", False)),
            )
        except (TypeError, ValueError) as exc:
            torch_result = torch_result.model_copy(
                update={"ok": False, "error": f"invalid torch probe result: {exc}"}
            )

    onnx_ready = bool(onnx_result.details.get("cuda_provider")) and bool(
        onnx_result.details.get("cuda_transfer_smoke_ok")
    )
    ready = bool(
        torch_result.ok
        and compatibility is not None
        and compatibility.compatible
        and onnx_result.ok
        and onnx_ready
        and sd_scripts_result.ok
    )
    return RuntimeDoctorReport(
        gpu_uuid=binding.uuid,
        torch=torch_result,
        onnx=onnx_result,
        sd_scripts=sd_scripts_result,
        compatibility=compatibility,
        ready=ready,
    )
