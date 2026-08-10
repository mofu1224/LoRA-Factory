from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from lora_factory.application.service import default_app_settings, repository_root
from lora_factory.caption.managed_wd14 import ManagedWD14Tagger
from lora_factory.config.models import TrainingPlan
from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.discovery import bind_gpu_for_child, discover_nvidia_gpus
from lora_factory.gpu.models import GpuDevice
from lora_factory.runtime.doctor import inspect_runtime
from lora_factory.runtime.manager import RuntimeManager
from lora_factory.sampling.backend import SampleRequest
from lora_factory.sampling.sd_scripts_backend import SdScriptsSampler
from lora_factory.training.backend import TrainingRequest
from lora_factory.training.command_builder import build_sd_scripts_command
from lora_factory.training.dataset_toml import (
    render_dataset_toml,
    sd_scripts_validation_partition,
    verify_materialized_validation_partition,
)
from lora_factory.training.sd_scripts_backend import SdScriptsTrainingBackend

_VALIDATION_PROBE_MARKER = "LORA_FACTORY_VALIDATION_PROBE="
_VALIDATION_LOSS = re.compile(r"val_epoch_avg_loss=([0-9.eE+-]+)")


def _base_model() -> Path:
    raw = os.environ.get("LORA_FACTORY_LIVE_BASE_MODEL")
    if not raw:
        pytest.skip("LORA_FACTORY_LIVE_BASE_MODEL is not configured")
    path = Path(raw).resolve(strict=True)
    if path.suffix.casefold() != ".safetensors":
        pytest.skip("Live base model must be a safetensors checkpoint")
    return path


def _runtime() -> RuntimeManager:
    settings = default_app_settings()
    manager = RuntimeManager(
        settings.managed_runtime_root,
        repository_root() / "backend-manifest.json",
    )
    if not manager.installed() or not manager.source_matches_manifest():
        pytest.skip("Pinned managed runtime is not installed")
    return manager


def _selected_device() -> tuple[GpuDevice, tuple[GpuDevice, ...]]:
    devices = discover_nvidia_gpus()
    preferred = os.environ.get("LORA_FACTORY_LIVE_GPU_UUID")
    if preferred:
        return next(item for item in devices if item.uuid == preferred), devices
    return max(devices, key=lambda item: item.free_vram_mb), devices


def _make_dataset(root: Path, count: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        y, x = np.mgrid[0:768, 0:768]
        red = ((x + index * 31) % 256).astype(np.uint8)
        green = ((y * (index + 1) // 3) % 256).astype(np.uint8)
        blue = (((x // 2 + y // 3) + index * 47) % 256).astype(np.uint8)
        pixels = np.stack((red, green, blue), axis=2)
        image_path = root / f"live-{index:02d}.png"
        Image.fromarray(pixels, mode="RGB").save(image_path)
        image_path.with_suffix(".txt").write_text(
            f"lf_live, abstract geometric pattern, color variation {index}\n",
            encoding="utf-8",
        )
    return root


def _probe_pinned_partition(
    *,
    manager: RuntimeManager,
    binding_environment: dict[str, str],
    command_argv: tuple[str, ...],
    work_directory: Path,
) -> dict[str, Any]:
    command_path = work_directory / "trainer-command.json"
    command_path.write_text(
        json.dumps(list(command_argv), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    probe = repository_root() / "tests" / "live" / "helpers" / "probe_sd_scripts_validation.py"
    completed = subprocess.run(  # noqa: S603 - fixed managed Python and test helper.
        [str(manager.layout.python), str(probe), str(command_path)],
        cwd=manager.layout.sd_scripts,
        env={**os.environ, **binding_environment},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Pinned sd-scripts partition probe failed:\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    payload_line = next(
        (
            line
            for line in completed.stdout.splitlines()
            if line.startswith(_VALIDATION_PROBE_MARKER)
        ),
        None,
    )
    if payload_line is None:
        pytest.fail("Pinned sd-scripts partition probe produced no structured payload")
    payload = json.loads(payload_line.removeprefix(_VALIDATION_PROBE_MARKER))
    if not isinstance(payload, dict):
        pytest.fail("Pinned sd-scripts partition probe payload is not an object")
    return payload


@pytest.mark.live_gpu
def test_managed_runtime_doctor_on_selected_gpu() -> None:
    manager = _runtime()
    device, devices = _selected_device()
    selected = tuple(item.uuid for item in devices)
    binding = bind_gpu_for_child(device.uuid, selected_uuids=selected, devices=devices)
    report = inspect_runtime(
        python_executable=manager.layout.python,
        binding=binding,
        sd_scripts_root=manager.layout.sd_scripts,
        timeout_seconds=180,
    )
    assert report.ready
    assert report.torch.details["bf16_smoke_ok"] is True
    assert "sm_120" in report.torch.details["arch_list"]
    assert report.onnx.details["cuda_transfer_smoke_ok"] is True


@pytest.mark.live_gpu
def test_managed_wd14_uses_cuda_provider(tmp_path: Path) -> None:
    manager = _runtime()
    device, devices = _selected_device()
    selected = tuple(item.uuid for item in devices)
    binding = bind_gpu_for_child(device.uuid, selected_uuids=selected, devices=devices)
    image_dir = _make_dataset(tmp_path / "images", count=1)
    tagger = ManagedWD14Tagger(
        python_executable=manager.layout.python,
        helper_script=repository_root()
        / "src"
        / "lora_factory"
        / "runtime_scripts"
        / "wd14_infer.py",
        model_path=manager.layout.wd14 / "model.onnx",
        tags_path=manager.layout.wd14 / "selected_tags.csv",
        revision=str(manager.manifest["wd14"]["revision"]),
        binding=binding,
        work_directory=tmp_path / "tagging",
        cancellation=CancellationToken(),
    )
    result = tagger.tag(image_dir / "live-00.png", asset_id="a" * 64)
    assert result.tags
    stdout = (tmp_path / "tagging" / "stdout.log").read_text(encoding="utf-8")
    assert '"provider": "CUDAExecutionProvider"' in stdout


@pytest.mark.live_gpu
@pytest.mark.live_sd_scripts
def test_tiny_real_training_and_sampling(tmp_path: Path) -> None:
    manager = _runtime()
    base_model = _base_model()
    device, devices = _selected_device()
    selected = tuple(item.uuid for item in devices)
    binding = bind_gpu_for_child(device.uuid, selected_uuids=selected, devices=devices)
    dataset = _make_dataset(tmp_path / "dataset")
    plan = TrainingPlan(
        resolution=768,
        batch_size=1,
        gradient_accumulation=1,
        repeats=1,
        epochs=1,
        estimated_steps=8,
        network_dim=4,
        network_alpha=4,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=False,
        training_gpu_uuid=device.uuid,
        estimated_disk_mb=512,
    )
    dataset_config = tmp_path / "dataset.toml"
    dataset_config.write_text(render_dataset_toml(image_dir=dataset, plan=plan), encoding="utf-8")
    trainer = SdScriptsTrainingBackend(
        manager.layout.python,
        manager.layout.sd_scripts,
        binding,
        selected,
        timeout_seconds=3600,
    )
    result = trainer.train(
        TrainingRequest(
            run_id="live-smoke",
            attempt=1,
            output_name="lf_live",
            base_model=base_model,
            dataset_config=dataset_config,
            run_directory=tmp_path / "run",
            plan=plan,
            seed=42,
        ),
        CancellationToken(),
        lambda _progress: None,
    )
    assert result.completed_steps >= 1
    checkpoint = result.checkpoints[-1]
    sampler = SdScriptsSampler(
        python_executable=manager.layout.python,
        sd_scripts_root=manager.layout.sd_scripts,
        commit=str(manager.manifest["sd_scripts"]["commit"]),
        binding=binding,
        precision="bf16",
        timeout_seconds=1200,
    )
    sample = sampler.sample(
        SampleRequest(
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_path=checkpoint.path,
            base_model_path=base_model,
            prompt_id="live-smoke",
            prompt="lf_live, a simple portrait, neutral background",
            negative_prompt="low quality, text, watermark",
            seed=12345,
            weight=0.8,
            width=512,
            height=512,
            steps=4,
        ),
        tmp_path / "samples",
    )
    assert sample.success
    with Image.open(sample.image_path) as generated:
        assert generated.format == "PNG"
        assert generated.size == (512, 512)


@pytest.mark.live_gpu
@pytest.mark.live_sd_scripts
def test_character_validation_split_and_loss_on_pinned_sd_scripts(tmp_path: Path) -> None:
    started = time.monotonic()
    manager = _runtime()
    base_model = _base_model()
    device, devices = _selected_device()
    selected = tuple(item.uuid for item in devices)
    binding = bind_gpu_for_child(device.uuid, selected_uuids=selected, devices=devices)
    dataset = _make_dataset(tmp_path / "validation-dataset", count=24)
    asset_ids = tuple(path.stem for path in sorted(dataset.glob("*.png")))
    expected_training_ids, expected_validation_ids = sd_scripts_validation_partition(
        asset_ids,
        validation_count=2,
        seed=42,
    )
    verify_materialized_validation_partition(
        image_dir=dataset,
        expected_training_ids=expected_training_ids,
        expected_validation_ids=expected_validation_ids,
        seed=42,
    )
    plan = TrainingPlan(
        resolution=768,
        batch_size=1,
        gradient_accumulation=4,
        repeats=1,
        epochs=1,
        estimated_steps=6,
        network_dim=4,
        network_alpha=4,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=True,
        validation_image_count=2,
        validation_seed=42,
        training_gpu_uuid=device.uuid,
        estimated_disk_mb=512,
    )
    dataset_config = tmp_path / "validation-dataset.toml"
    dataset_config.write_text(
        render_dataset_toml(
            image_dir=dataset,
            plan=plan,
            validation_total_count=len(asset_ids),
        ),
        encoding="utf-8",
    )
    run_directory = tmp_path / "validation-run"
    expected_command = build_sd_scripts_command(
        python_executable=manager.layout.python,
        sd_scripts_root=manager.layout.sd_scripts,
        pretrained_model=base_model,
        dataset_config=dataset_config,
        output_dir=run_directory / "attempt-001" / "checkpoints",
        output_name="lf_validation_live",
        logging_dir=run_directory / "attempt-001" / "logs",
        plan=plan,
        binding=binding,
        selected_gpu_uuids=selected,
        seed=42,
    )
    upstream_partition = _probe_pinned_partition(
        manager=manager,
        binding_environment=binding.environment,
        command_argv=expected_command.argv,
        work_directory=tmp_path,
    )
    assert upstream_partition["training_ids"] == list(expected_training_ids)
    assert upstream_partition["validation_ids"] == list(expected_validation_ids)
    assert upstream_partition["training_count"] == 22
    assert upstream_partition["validation_count"] == 2

    observed = []
    trainer = SdScriptsTrainingBackend(
        manager.layout.python,
        manager.layout.sd_scripts,
        binding,
        selected,
        timeout_seconds=3600,
    )
    result = trainer.train(
        TrainingRequest(
            run_id="live-validation-smoke",
            attempt=1,
            output_name="lf_validation_live",
            base_model=base_model,
            dataset_config=dataset_config,
            run_directory=run_directory,
            plan=plan,
            seed=42,
        ),
        CancellationToken(),
        observed.append,
    )
    elapsed_seconds = time.monotonic() - started
    checkpoint = result.checkpoints[-1]
    stderr_path = result.log_path.with_name("training.stderr.log")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    logged_losses = [float(value) for value in _VALIDATION_LOSS.findall(stderr_text)]
    observed_losses = [
        item.validation_loss for item in observed if item.validation_loss is not None
    ]

    assert result.command_argv == expected_command.argv
    assert "--max_train_epochs=1" in result.command_argv
    assert "--gradient_accumulation_steps=4" in result.command_argv
    assert "--validation_seed=42" in result.command_argv
    assert "--validate_every_n_epochs=1" in result.command_argv
    assert "--max_validation_steps=2" in result.command_argv
    assert result.completed_steps == 6
    assert checkpoint.epoch == 1
    assert checkpoint.step == 6
    assert checkpoint.path.is_file()
    assert result.state_path.is_dir()
    assert logged_losses
    assert observed_losses
    assert checkpoint.validation_loss == pytest.approx(observed_losses[-1])
    assert checkpoint.validation_loss == pytest.approx(logged_losses[-1])

    evidence = {
        "runtime_version": trainer.version,
        "gpu_uuid": device.uuid,
        "dataset_config": str(dataset_config),
        "expected_training_ids": list(expected_training_ids),
        "expected_validation_ids": list(expected_validation_ids),
        "upstream_partition": upstream_partition,
        "command_argv": list(result.command_argv),
        "completed_steps": result.completed_steps,
        "checkpoint": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
        "state_path": str(result.state_path),
        "validation_loss": checkpoint.validation_loss,
        "elapsed_seconds": elapsed_seconds,
    }
    (tmp_path / "validation-smoke-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
