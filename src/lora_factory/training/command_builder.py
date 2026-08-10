"""Safe argument-array construction for a pinned sd-scripts training runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from lora_factory.config.models import TrainingPlan
from lora_factory.gpu.models import GpuBinding

_SAFE_IDENTIFIER = re.compile(r"^[\w.+-]{1,128}$", re.UNICODE)
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class TrainingCommand(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    gpu_uuid: str


def _validated_identifier(value: str, *, label: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(value) or value.endswith((" ", ".")):
        raise ValueError(f"Unsafe {label}: {value!r}")
    if value.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"Unsafe {label}: Windows reserves {value!r}")
    return value


def _validate_argv(argv: Sequence[str]) -> None:
    for argument in argv:
        if not argument or any(character in argument for character in ("\x00", "\r", "\n")):
            raise ValueError(f"Invalid command argument: {argument!r}")


def build_sd_scripts_command(
    *,
    python_executable: Path,
    sd_scripts_root: Path,
    pretrained_model: Path,
    dataset_config: Path,
    output_dir: Path,
    output_name: str,
    logging_dir: Path,
    plan: TrainingPlan,
    binding: GpuBinding,
    selected_gpu_uuids: Sequence[str],
    seed: int,
    resume_state: Path | None = None,
    extra_environment: Mapping[str, str] | None = None,
) -> TrainingCommand:
    """Build one-process Accelerate invocation without shell interpretation."""

    if not selected_gpu_uuids:
        raise ValueError("At least one GPU UUID must be selected")
    if len(set(selected_gpu_uuids)) != len(selected_gpu_uuids):
        raise ValueError("Selected GPU UUIDs must be unique")
    if binding.uuid not in selected_gpu_uuids:
        raise ValueError("Training GPU binding is outside the selected pool")
    if plan.training_gpu_uuid != binding.uuid:
        raise ValueError("Training plan and GPU binding refer to different UUIDs")
    if binding.environment.get("CUDA_VISIBLE_DEVICES") != str(binding.physical_index):
        raise ValueError("GPU binding isolation environment was modified after validation")
    if binding.environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("GPU binding device order was modified after validation")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    output_name = _validated_identifier(output_name, label="output name")
    optimizer = _validated_identifier(plan.optimizer, label="optimizer")
    scheduler = _validated_identifier(plan.scheduler, label="scheduler")
    training_script = sd_scripts_root / "sdxl_train_network.py"
    required_files = {
        "managed Python": python_executable,
        "sd-scripts training entrypoint": training_script,
        "base model": pretrained_model,
        "dataset config": dataset_config,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if resume_state is not None and not resume_state.exists():
        raise ValueError(f"Resume state does not exist: {resume_state}")

    argv: list[str] = [
        str(python_executable),
        "-m",
        "accelerate.commands.launch",
        "--num_processes=1",
        "--num_machines=1",
        f"--mixed_precision={plan.precision}",
        str(training_script),
        f"--pretrained_model_name_or_path={pretrained_model}",
        f"--dataset_config={dataset_config}",
        f"--output_dir={output_dir}",
        f"--output_name={output_name}",
        "--save_model_as=safetensors",
        "--network_module=networks.lora",
        f"--network_dim={plan.network_dim}",
        f"--network_alpha={plan.network_alpha}",
        f"--unet_lr={plan.unet_lr:.12g}",
        f"--text_encoder_lr={plan.text_encoder_lr:.12g}",
        f"--optimizer_type={optimizer}",
        f"--lr_scheduler={scheduler}",
        f"--mixed_precision={plan.precision}",
        f"--save_precision={plan.precision}",
        f"--gradient_accumulation_steps={plan.gradient_accumulation}",
        f"--max_train_epochs={plan.epochs}",
        f"--save_every_n_epochs={plan.save_every_n_epochs}",
        f"--logging_dir={logging_dir}",
        f"--seed={seed}",
        "--save_state",
    ]
    if plan.gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if plan.cache_latents:
        argv.append("--cache_latents")
    if plan.cache_text_encoder_outputs:
        argv.append("--cache_text_encoder_outputs")
    if plan.sdpa:
        argv.append("--sdpa")
    if plan.no_half_vae:
        argv.append("--no_half_vae")
    if plan.validation_enabled:
        argv.extend(
            (
                f"--validation_seed={plan.validation_seed}",
                "--validate_every_n_epochs=1",
                f"--max_validation_steps={max(1, plan.validation_image_count)}",
            )
        )
    if resume_state is not None:
        argv.append(f"--resume={resume_state}")
    _validate_argv(argv)

    environment = dict(binding.environment)
    for key, value in (extra_environment or {}).items():
        if key in {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}:
            raise ValueError(f"GPU isolation environment cannot be overridden: {key}")
        if not key or any(character in key + value for character in ("\x00", "\r", "\n")):
            raise ValueError(f"Invalid environment entry: {key!r}")
        environment[key] = value
    return TrainingCommand(
        argv=tuple(argv),
        cwd=sd_scripts_root,
        environment=environment,
        gpu_uuid=binding.uuid,
    )
