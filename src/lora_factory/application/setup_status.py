"""Fast setup-status aggregation backed by cached deep-validation evidence."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lora_factory.config.models import AppSettings, DestinationConfig, DestinationKind
from lora_factory.gpu.discovery import discover_nvidia_gpus
from lora_factory.gpu.models import GpuDevice
from lora_factory.runtime.manager import RuntimeManager
from lora_factory.runtime.validation_record import (
    RuntimeValidationRecord,
    ValidationStatus,
)

type CommandRunner = Callable[[tuple[str, ...], float], subprocess.CompletedProcess[str]]
type Which = Callable[[str], str | None]
type GpuDiscovery = Callable[[], Sequence[GpuDevice]]


def _run_command(argv: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executables are resolved before fixed read-only calls.
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _detail(completed: subprocess.CompletedProcess[str]) -> str:
    text = (completed.stdout or completed.stderr).strip()
    if not text:
        return f"Exited with code {completed.returncode} without diagnostics"
    return " | ".join(text.splitlines()[-3:])[:800]


def _check(
    name: str,
    status: str,
    detail: str,
    *,
    repairable: bool = False,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "repairable": repairable,
        "required": required,
    }


def _cached_check(
    record: RuntimeValidationRecord | None,
    key: str,
    name: str,
    *,
    unavailable_detail: str,
) -> dict[str, Any]:
    if record is None or key not in record.checks:
        return _check(name, "needs setup", unavailable_detail)
    value = record.checks[key]
    status = {
        ValidationStatus.OK: "ok",
        ValidationStatus.ERROR: "error",
        ValidationStatus.NOT_RUN: "needs setup",
    }[value.status]
    return _check(name, status, value.detail)


def _destination_checks(
    destinations: Sequence[DestinationConfig],
) -> list[dict[str, Any]]:
    configured = {item.kind: item for item in destinations}
    checks: list[dict[str, Any]] = []
    labels = {
        DestinationKind.A1111: "AUTOMATIC1111 destination",
        DestinationKind.FORGE: "Forge destination",
        DestinationKind.COMFYUI: "ComfyUI destination",
    }
    for kind, label in labels.items():
        destination = configured.get(kind)
        if destination is None:
            checks.append(_check(label, "optional", "Not configured (optional)", required=False))
            continue
        root = destination.root.resolve(strict=False)
        if not destination.enabled:
            checks.append(_check(label, "optional", f"Disabled: {root}", required=False))
        elif root.exists():
            checks.append(_check(label, "ok", str(root), required=False))
        else:
            checks.append(
                _check(
                    label,
                    "warning",
                    f"Configured path does not exist yet: {root}",
                    required=False,
                )
            )
    return checks


def build_setup_checks(
    *,
    settings: AppSettings,
    runtime: RuntimeManager,
    destinations: Sequence[DestinationConfig],
    which: Which = shutil.which,
    runner: CommandRunner = _run_command,
    gpu_discovery: GpuDiscovery = discover_nvidia_gpus,
) -> tuple[Mapping[str, Any], ...]:
    """Return quick checks without rerunning Torch, ONNX, or sd-scripts deep probes."""

    checks: list[dict[str, Any]] = [
        _check(
            "Factory Python",
            "ok" if sys.version_info[:2] == (3, 12) else "error",
            sys.version.split()[0],
        )
    ]
    for name, command in (("uv", "uv"), ("Git", "git")):
        found = which(command)
        checks.append(
            _check(name, "ok" if found else "error", found or f"{command} was not found on PATH")
        )

    codex = which("codex")
    codex_required = settings.codex_required
    checks.append(
        _check(
            "Runtime Codex",
            "ok" if codex else "error",
            codex or "codex was not found on PATH",
            required=codex_required,
        )
    )
    if codex:
        try:
            login = runner((codex, "login", "status"), 10.0)
            checks.append(
                _check(
                    "Codex Authentication",
                    "ok" if login.returncode == 0 else "error",
                    _detail(login),
                    required=codex_required,
                )
            )
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(
                _check(
                    "Codex Authentication",
                    "error",
                    f"Unable to query codex login status: {exc}",
                    required=codex_required,
                )
            )
    else:
        checks.append(
            _check(
                "Codex Authentication",
                "error",
                "Cannot check authentication until Codex CLI is installed",
                required=codex_required,
            )
        )

    managed_python = runtime.layout.python
    checks.append(
        _check(
            "Managed Python",
            "ok" if managed_python.is_file() else "needs setup",
            str(managed_python) if managed_python.is_file() else "Managed Python is not installed",
            repairable=not managed_python.is_file(),
        )
    )

    nvidia_smi = which("nvidia-smi")
    if nvidia_smi is None:
        checks.extend(
            (
                _check("NVIDIA Driver", "error", "nvidia-smi was not found on PATH"),
                _check("NVML", "error", "NVML cannot be queried without nvidia-smi"),
            )
        )
    else:
        try:
            driver = runner(
                (
                    nvidia_smi,
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ),
                10.0,
            )
            driver_ok = driver.returncode == 0
            checks.append(
                _check(
                    "NVIDIA Driver",
                    "ok" if driver_ok else "error",
                    _detail(driver),
                )
            )
            checks.append(
                _check(
                    "NVML",
                    "ok" if driver_ok else "error",
                    "nvidia-smi NVML query succeeded" if driver_ok else _detail(driver),
                )
            )
        except (OSError, subprocess.SubprocessError) as exc:
            checks.extend(
                (
                    _check("NVIDIA Driver", "error", f"nvidia-smi query failed: {exc}"),
                    _check("NVML", "error", f"NVML query failed: {exc}"),
                )
            )

    devices: Sequence[GpuDevice] = ()
    try:
        devices = gpu_discovery()
        detail = ", ".join(
            f"{item.name} ({item.uuid}, sm_{item.capability_major}{item.capability_minor})"
            for item in devices
        )
        checks.append(_check("NVIDIA CUDA GPUs", "ok", detail))
    except RuntimeError as exc:
        checks.append(_check("NVIDIA CUDA GPUs", "error", str(exc)))

    installed = runtime.installed()
    source_matches = installed and runtime.source_matches_manifest()
    validation_error = ""
    try:
        record = runtime.read_validation_record()
    except ValueError as exc:
        record = None
        validation_error = str(exc)
    if record is not None and record.gpu_uuid is not None:
        current_uuids = {device.uuid for device in devices}
        if record.gpu_uuid not in current_uuids:
            validation_error = (
                f"Cached validation GPU is not present: {record.gpu_uuid}; run doctor --deep"
            )
            record = None
    runtime_ready = installed and source_matches and record is not None and record.ready
    runtime_detail = (
        str(runtime.layout.root)
        if runtime_ready
        else validation_error
        or "Install the pinned runtime and run `lora-factory doctor --deep` once"
    )
    checks.append(
        _check(
            "Managed Training Runtime",
            "ok" if runtime_ready else "needs setup",
            runtime_detail,
            repairable=not runtime_ready,
        )
    )
    checks.append(
        _cached_check(
            record,
            "pytorch_gpu",
            "PyTorch GPU Smoke Test",
            unavailable_detail="No cached selected-GPU PyTorch validation; run doctor --deep",
        )
    )
    checks.append(
        _cached_check(
            record,
            "onnx_cuda",
            "ONNX Runtime CUDA Provider",
            unavailable_detail="No cached ONNX CUDA provider validation; run doctor --deep",
        )
    )
    sd_scripts = _cached_check(
        record,
        "sd_scripts",
        "sd-scripts backend",
        unavailable_detail="No cached sd-scripts import validation; run doctor --deep",
    )
    if installed and not source_matches:
        sd_scripts = _check(
            "sd-scripts backend",
            "error",
            "Installed sd-scripts commit differs from backend-manifest.json",
        )
    checks.append(sd_scripts)

    free_gib = shutil.disk_usage(runtime.managed_root).free / 1024**3
    checks.append(
        _check(
            "Disk Space",
            "ok" if free_gib >= 15 else "error",
            f"{free_gib:.1f} GiB free",
        )
    )
    checks.extend(_destination_checks(destinations))
    return tuple(checks)
