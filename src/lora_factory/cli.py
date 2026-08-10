"""Developer diagnostics and reproducible acceptance commands for LoRA Factory."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from lora_factory.application.service import (
    LoRAFactoryController,
    default_app_settings,
)
from lora_factory.config.models import (
    GPU_UUID_PATTERN,
    AdvancedOverrides,
    AppSettings,
    BackendMode,
    PresetKind,
    ProjectConfig,
)
from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.discovery import bind_gpu_for_child
from lora_factory.gpu.models import GpuDevice
from lora_factory.model.inspector import ModelInspection, inspect_sdxl_safetensors
from lora_factory.packaging.metadata import inspect_safetensors
from lora_factory.runtime.doctor import RuntimeDoctorReport, inspect_runtime
from lora_factory.runtime.installer import ManagedRuntimeInstaller
from lora_factory.runtime.validation_record import RuntimeValidationRecord
from lora_factory.testing.fixture_factory import generate_fake_fixture
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json, write_json_atomic

app = typer.Typer(
    name="lora-factory",
    help="LoRA Factory developer diagnostics and reproducible smoke commands.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console(stderr=True)


class FakeE2EReport(BaseModel):
    """Auditable result of the same-path deterministic Fake pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    project_id: str
    run_id: str
    workspace: Path
    fixture_root: Path
    output_directory: Path
    final_model: Path
    final_sha256: str
    preview: Path
    comparison: Path
    event_count: int = Field(ge=1)
    observed_stages: tuple[str, ...]
    source_hashes_unchanged: bool
    gpu_uuid: str
    gpu_uuids: tuple[str, ...]
    reproducibility_manifest: Path
    report_path: Path


def _json_dump(value: Any) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _settings_with_overrides(
    *,
    projects_root: Path | None = None,
    managed_runtime_root: Path | None = None,
    codex_runtime_root: Path | None = None,
) -> AppSettings:
    defaults = default_app_settings()
    return defaults.model_copy(
        update={
            "projects_root": projects_root or defaults.projects_root,
            "managed_runtime_root": managed_runtime_root or defaults.managed_runtime_root,
            "codex_runtime_root": codex_runtime_root or defaults.codex_runtime_root,
        }
    )


def _new_run_workspace(parent: Path, prefix: str) -> Path:
    parent = parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for _attempt in range(100):
        candidate = parent / f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Unable to allocate an unused run directory under {parent}")


def _select_gpu_uuid(
    controller: LoRAFactoryController,
    requested: str | None,
    *,
    allow_fake_without_gpu: bool,
) -> tuple[str, tuple[GpuDevice, ...]]:
    selected, devices = _select_gpu_uuids(
        controller,
        () if requested is None else (requested,),
        allow_fake_without_gpu=allow_fake_without_gpu,
    )
    return selected[0], devices


def _select_gpu_uuids(
    controller: LoRAFactoryController,
    requested: Sequence[str] | None,
    *,
    allow_fake_without_gpu: bool,
) -> tuple[tuple[str, ...], tuple[GpuDevice, ...]]:
    requested_uuids = tuple(requested or ())
    if len(set(requested_uuids)) != len(requested_uuids):
        raise typer.BadParameter("--gpu-uuid values must be unique")
    try:
        devices = tuple(controller.discover_gpus())
    except RuntimeError:
        if not allow_fake_without_gpu:
            raise
        devices = ()
    if requested_uuids:
        if not devices and not allow_fake_without_gpu:
            raise typer.BadParameter("No NVIDIA CUDA GPU was discovered")
        available = {device.uuid for device in devices}
        missing = tuple(uuid for uuid in requested_uuids if uuid not in available)
        if missing and devices:
            raise typer.BadParameter(f"Selected GPU UUID is not present: {missing[0]}")
        return requested_uuids, devices
    if devices:
        selected = max(devices, key=lambda item: (item.free_vram_mb, item.total_vram_mb, item.uuid))
        return (selected.uuid,), devices
    if allow_fake_without_gpu:
        return ("GPU-00000000-0000-0000-0000-000000000001",), ()
    raise typer.BadParameter("No NVIDIA CUDA GPU was discovered")


def _select_fake_gpu_uuids(requested: Sequence[str] | None) -> tuple[str, ...]:
    requested_uuids = tuple(requested or ())
    if len(set(requested_uuids)) != len(requested_uuids):
        raise typer.BadParameter("--gpu-uuid values must be unique")
    invalid = next((uuid for uuid in requested_uuids if not GPU_UUID_PATTERN.match(uuid)), None)
    if invalid is not None:
        raise typer.BadParameter(f"Invalid NVIDIA GPU UUID: {invalid}")
    return requested_uuids or ("GPU-00000000-0000-0000-0000-000000000001",)


def _event_printer(payload: dict[str, Any]) -> None:
    stage = str(payload.get("stage", "pipeline"))
    message = str(payload.get("message", payload.get("event_type", "event")))
    progress = payload.get("overall_progress")
    suffix = f" ({float(progress) * 100:.0f}%)" if isinstance(progress, int | float) else ""
    console.print(f"[cyan]{stage}[/cyan]: {message}{suffix}")


def _verify_pipeline_result(result: Mapping[str, Any]) -> tuple[Path, Path, Path, str]:
    if result.get("status") != "READY":
        raise RuntimeError(f"Pipeline did not reach READY: {result.get('status')!r}")
    final_model = Path(str(result["final_model"])).resolve(strict=True)
    preview = Path(str(result["preview"])).resolve(strict=True)
    comparison = Path(str(result["comparison"])).resolve(strict=True)
    metadata = inspect_safetensors(final_model)
    if not metadata:
        raise RuntimeError("Final safetensors contains no metadata")
    digest = sha256_file(final_model)
    if digest != str(result["sha256"]):
        raise RuntimeError("Final model hash differs from the packaged completion record")
    return final_model, preview, comparison, digest


def run_fake_e2e(
    workspace: Path,
    *,
    preset: PresetKind = PresetKind.CHARACTER,
    image_count: int = 18,
    requested_gpu_uuid: str | None = None,
    requested_gpu_uuids: Sequence[str] | None = None,
    controller: LoRAFactoryController | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> FakeE2EReport:
    """Generate immutable fixtures and execute the production ApplicationController path."""

    minimum = 8 if preset is PresetKind.CHARACTER else 16
    if image_count < minimum:
        raise ValueError(f"{preset.value} Fake E2E requires at least {minimum} images")
    workspace = workspace.resolve(strict=True)
    fixture = generate_fake_fixture(workspace / "fixture", image_count=image_count)
    settings = _settings_with_overrides(
        projects_root=workspace / "app-data" / "projects",
        managed_runtime_root=workspace / "app-data" / "runtimes",
        codex_runtime_root=workspace / "app-data" / "codex-scratch",
    )
    active_controller = controller or LoRAFactoryController(settings)
    if requested_gpu_uuid is not None and requested_gpu_uuids:
        raise ValueError("Use requested_gpu_uuid or requested_gpu_uuids, not both")
    requested_pool = (
        tuple(requested_gpu_uuids or ()) if requested_gpu_uuid is None else (requested_gpu_uuid,)
    )
    gpu_uuids = _select_fake_gpu_uuids(requested_pool)
    project_id = f"fake-e2e-{uuid.uuid4().hex[:12]}"
    output_root = workspace / "output"
    output_root.mkdir()
    fixture_trigger = "lfx_demo"
    config = ProjectConfig(
        project_id=project_id,
        lora_name="FakeCharacterLoRA" if preset is PresetKind.CHARACTER else "FakeStyleLoRA",
        preset=preset,
        trigger_token=fixture_trigger,
        base_model=fixture.base_model,
        input_paths=(fixture.image_directory,),
        selected_gpu_uuids=gpu_uuids,
        output_root=output_root,
        backend_mode=BackendMode.FAKE,
        allow_without_codex=True,
        advanced=AdvancedOverrides(
            resolution=768,
            network_dim=8,
            network_alpha=4,
            batch_size=1,
            gradient_accumulation=1,
            repeats=1,
            epochs=4,
        ),
    )
    events: list[dict[str, Any]] = []

    def emit(payload: dict[str, Any]) -> None:
        events.append(payload)
        if event_callback is not None:
            event_callback(payload)

    result = active_controller.run_pipeline(config, emit)
    final_model, preview, comparison, digest = _verify_pipeline_result(result)
    after_hashes = {path.name: sha256_file(path) for path in fixture.image_paths}
    unchanged = fixture.source_sha256 == after_hashes
    if not unchanged:
        raise RuntimeError("Fake E2E source image hashes changed during the pipeline")
    reproducibility_manifest = (final_model.parent / "reproducibility_manifest.json").resolve(
        strict=True
    )
    manifest = read_json(reproducibility_manifest)
    if not isinstance(manifest, dict):
        raise RuntimeError("Fake E2E reproducibility manifest is not an object")
    capabilities = manifest.get("gpu_capabilities")
    if not isinstance(capabilities, list):
        raise RuntimeError("Fake E2E reproducibility manifest has no GPU capabilities")
    manifest_gpu_uuids = tuple(
        str(item.get("uuid")) for item in capabilities if isinstance(item, dict)
    )
    expected_public_gpu_ids = tuple(f"gpu-{index}" for index in range(1, len(gpu_uuids) + 1))
    if manifest_gpu_uuids != expected_public_gpu_ids:
        raise RuntimeError("Fake E2E selected GPU pool differs from its manifest")
    observed_stages = tuple(
        dict.fromkeys(str(event.get("stage", "")) for event in events if event.get("stage"))
    )
    report_path = workspace / "fake-e2e-report.json"
    report = FakeE2EReport(
        status=str(result["status"]),
        project_id=project_id,
        run_id=str(result["run_id"]),
        workspace=workspace,
        fixture_root=fixture.root,
        output_directory=Path(str(result["output_directory"])),
        final_model=final_model,
        final_sha256=digest,
        preview=preview,
        comparison=comparison,
        event_count=len(events),
        observed_stages=observed_stages,
        source_hashes_unchanged=unchanged,
        gpu_uuid=gpu_uuids[0],
        gpu_uuids=gpu_uuids,
        reproducibility_manifest=reproducibility_manifest,
        report_path=report_path,
    )
    write_json_atomic(report_path, report.model_dump(mode="json"))
    return report


def _render_checks(checks: Sequence[Mapping[str, Any]]) -> None:
    table = Table(title="LoRA Factory Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        status = str(check.get("status", "unknown"))
        style = "green" if status == "ok" else "yellow" if status == "needs setup" else "red"
        table.add_row(
            str(check.get("name", "unknown")),
            f"[{style}]{status}[/{style}]",
            str(check.get("detail", "")),
        )
    console.print(table)


def _run_runtime_doctor(
    controller: LoRAFactoryController, requested_gpu_uuid: str | None
) -> RuntimeDoctorReport:
    requested = () if requested_gpu_uuid is None else (requested_gpu_uuid,)
    return _run_runtime_doctors(controller, requested)[0]


def _run_runtime_doctors(
    controller: LoRAFactoryController, requested_gpu_uuids: Sequence[str] | None
) -> tuple[RuntimeDoctorReport, ...]:
    if not controller.runtime.installed() or not controller.runtime.source_matches_manifest():
        raise RuntimeError("Managed runtime is absent or does not match backend-manifest.json")
    gpu_uuids, devices = _select_gpu_uuids(
        controller,
        requested_gpu_uuids,
        allow_fake_without_gpu=False,
    )
    return tuple(
        inspect_runtime(
            python_executable=controller.runtime.layout.python,
            binding=bind_gpu_for_child(
                gpu_uuid,
                selected_uuids=gpu_uuids,
                devices=devices,
            ),
            sd_scripts_root=controller.runtime.layout.sd_scripts,
        )
        for gpu_uuid in gpu_uuids
    )


@app.command("doctor")
def doctor_command(
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Run managed CUDA, bf16, ONNX and sd-scripts probes."),
    ] = False,
    gpu_uuid: Annotated[
        str | None,
        typer.Option("--gpu-uuid", help="Physical NVIDIA UUID used by --deep."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON to stdout."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero for errors or incomplete setup."),
    ] = False,
    managed_runtime_root: Annotated[
        Path | None,
        typer.Option("--runtime-root", help="Override the managed runtime parent directory."),
    ] = None,
) -> None:
    """Report Factory prerequisites, with optional selected-GPU runtime probes."""

    settings = _settings_with_overrides(managed_runtime_root=managed_runtime_root)
    controller = LoRAFactoryController(settings)
    checks = tuple(controller.setup_checks())
    runtime_report: RuntimeDoctorReport | None = None
    deep_error: str | None = None
    if deep:
        try:
            runtime_report = _run_runtime_doctor(controller, gpu_uuid)
            controller.runtime.write_validation_record(
                RuntimeValidationRecord.from_doctor_report(
                    str(controller.runtime.manifest["profile_id"]), runtime_report
                )
            )
            checks = tuple(controller.setup_checks())
        except (OSError, RuntimeError, ValueError) as exc:
            deep_error = f"{type(exc).__name__}: {exc}"
    payload = {
        "checks": [dict(item) for item in checks],
        "runtime": runtime_report.model_dump(mode="json") if runtime_report else None,
        "runtime_error": deep_error,
    }
    if json_output:
        typer.echo(_json_dump(payload))
    else:
        _render_checks(checks)
        if runtime_report is not None:
            console.print(
                f"Managed runtime probe: "
                f"[{'green' if runtime_report.ready else 'red'}]"
                f"{'READY' if runtime_report.ready else 'NOT READY'}[/]"
            )
        if deep_error:
            console.print(f"[red]Managed runtime probe failed:[/red] {deep_error}")
    failed = any(
        bool(item.get("required", True)) and str(item.get("status")) == "error" for item in checks
    )
    incomplete = any(
        bool(item.get("required", True)) and str(item.get("status")) == "needs setup"
        for item in checks
    )
    runtime_not_ready = deep and (runtime_report is None or not runtime_report.ready)
    if strict and (failed or incomplete or deep_error or runtime_not_ready):
        raise typer.Exit(code=2)


@app.command("install-runtime")
def install_runtime_command(
    managed_runtime_root: Annotated[
        Path | None,
        typer.Option("--runtime-root", help="Override the managed runtime parent directory."),
    ] = None,
) -> None:
    """Install and hash-verify the pinned, isolated backend runtime."""

    controller = LoRAFactoryController(
        _settings_with_overrides(managed_runtime_root=managed_runtime_root)
    )
    cancellation = CancellationToken()

    def progress(stage: str, fraction: float, message: str) -> None:
        console.print(f"[cyan]{stage}[/cyan] {fraction * 100:5.1f}% {message}")

    try:
        ManagedRuntimeInstaller(controller.runtime).install(cancellation, progress)
    except KeyboardInterrupt as exc:
        cancellation.cancel()
        console.print("[yellow]Runtime installation cancelled between safe stages.[/yellow]")
        raise typer.Exit(code=130) from exc
    console.print(f"[green]Installed:[/green] {controller.runtime.layout.root}")


@app.command("inspect-model")
def inspect_model_command(
    path: Annotated[Path, typer.Argument(help="SDXL/Illustrious .safetensors checkpoint.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete inspection as JSON."),
    ] = False,
) -> None:
    """Inspect only a bounded safetensors header; model tensors are not loaded."""

    inspection: ModelInspection = inspect_sdxl_safetensors(path)
    if json_output:
        typer.echo(_json_dump(inspection))
        return
    table = Table(title="Base model inspection")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Path", str(inspection.path))
    table.add_row("SHA-256", inspection.sha256)
    table.add_row("Size", f"{inspection.file_size / 1024**3:.2f} GiB")
    table.add_row("Architecture", inspection.architecture_family)
    table.add_row("SDXL", str(inspection.is_sdxl))
    table.add_row("Possible Illustrious", str(inspection.possible_illustrious))
    table.add_row("Compatibility", inspection.compatibility.value)
    table.add_row("Warnings", "; ".join(inspection.warnings) or "none")
    console.print(table)
    if not inspection.is_sdxl:
        raise typer.Exit(code=2)


@app.command("fake-e2e")
def fake_e2e_command(
    workspace_root: Annotated[
        Path,
        typer.Option(
            "--workspace-root",
            help="Parent for a fresh non-overwriting run directory.",
        ),
    ] = Path(".artifacts/fake-e2e"),
    preset: Annotated[PresetKind, typer.Option("--preset")] = PresetKind.CHARACTER,
    image_count: Annotated[
        int,
        typer.Option("--image-count", min=8, help="Generated diverse PNG source count."),
    ] = 18,
    gpu_uuids: Annotated[
        list[str] | None,
        typer.Option(
            "--gpu-uuid",
            help="Synthetic UUID; repeat for a Fake GPU pool. CUDA is never executed.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress per-stage progress messages."),
    ] = False,
) -> None:
    """Run import through packaging with real formats and deterministic Fake adapters."""

    minimum = 8 if preset is PresetKind.CHARACTER else 16
    if image_count < minimum:
        raise typer.BadParameter(
            f"--image-count must be at least {minimum} for the {preset.value} preset"
        )
    workspace = _new_run_workspace(workspace_root, "run")
    report = run_fake_e2e(
        workspace,
        preset=preset,
        image_count=image_count,
        requested_gpu_uuids=gpu_uuids,
        event_callback=None if quiet or json_output else _event_printer,
    )
    if json_output:
        typer.echo(_json_dump(report))
    else:
        console.print(f"[green]Fake E2E READY[/green]: {report.final_model}")
        console.print(f"Report: {report.report_path}")
        console.print(f"Source hashes unchanged: {report.source_hashes_unchanged}")


@app.command("live-smoke")
def live_smoke_command(
    base_model: Annotated[
        Path,
        typer.Option("--base-model", exists=True, file_okay=True, dir_okay=False),
    ],
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, help="Tiny dataset image file or folder."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Fresh or reusable parent for versioned outputs."),
    ],
    gpu_uuids: Annotated[
        list[str] | None,
        typer.Option(
            "--gpu-uuid",
            help="Physical NVIDIA UUID; repeat for a pool, or omit to use the most-free GPU.",
        ),
    ] = None,
    activation_token: Annotated[
        str | None,
        typer.Option("--trigger", help="Activation token (default: lfx_smoke)."),
    ] = None,
    allow_codex_fallback: Annotated[
        bool,
        typer.Option("--allow-codex-fallback/--require-codex"),
    ] = True,
    managed_runtime_root: Annotated[
        Path | None,
        typer.Option("--runtime-root", help="Override the managed runtime parent directory."),
    ] = None,
) -> None:
    """Capability-check and run a bounded one-epoch Real Backend pipeline smoke."""

    inspection = inspect_sdxl_safetensors(base_model)
    if not inspection.is_sdxl:
        raise typer.BadParameter("--base-model is not structurally SDXL compatible")
    controller = LoRAFactoryController(
        _settings_with_overrides(managed_runtime_root=managed_runtime_root)
    )
    reports = _run_runtime_doctors(controller, gpu_uuids)
    not_ready = tuple(report for report in reports if not report.ready)
    if not_ready:
        console.print(_json_dump([report.model_dump(mode="json") for report in reports]))
        failed = ", ".join(report.gpu_uuid for report in not_ready)
        raise typer.BadParameter(f"Managed runtime doctor did not reach READY for: {failed}")
    selected_uuids = tuple(report.gpu_uuid for report in reports)
    run_suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_id=f"live-smoke-{run_suffix}",
        lora_name=f"LoRAFactorySmoke-{run_suffix}",
        preset=PresetKind.CHARACTER,
        trigger_token=activation_token or "lfx_smoke",
        base_model=base_model.resolve(strict=True),
        input_paths=(input_path.resolve(strict=True),),
        selected_gpu_uuids=selected_uuids,
        output_root=output_root,
        backend_mode=BackendMode.REAL,
        allow_without_codex=allow_codex_fallback,
        advanced=AdvancedOverrides(
            resolution=768,
            network_dim=4,
            network_alpha=1,
            batch_size=1,
            gradient_accumulation=1,
            repeats=1,
            epochs=1,
            optimizer="AdamW",
        ),
    )
    result = controller.run_pipeline(config, _event_printer)
    final_model, preview, comparison, digest = _verify_pipeline_result(result)
    typer.echo(
        _json_dump(
            {
                "status": result["status"],
                "project_id": result["project_id"],
                "run_id": result["run_id"],
                "gpu_uuid": selected_uuids[0],
                "gpu_uuids": selected_uuids,
                "final_model": str(final_model),
                "final_sha256": digest,
                "preview": str(preview),
                "comparison": str(comparison),
            }
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic entry point used by ``python -m lora_factory.cli``."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    app(args=arguments, prog_name="lora-factory", standalone_mode=False)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
