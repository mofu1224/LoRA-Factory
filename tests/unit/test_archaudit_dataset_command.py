from __future__ import annotations

from pathlib import Path

import pytest
import tomlkit

from lora_factory.config.models import TrainingPlan
from lora_factory.gpu.models import GpuBinding
from lora_factory.training import (
    build_sd_scripts_command,
    render_dataset_toml,
    write_dataset_configs,
)
from lora_factory.training.dataset_toml import (
    sd_scripts_validation_partition,
    verify_materialized_validation_partition,
)

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _plan(*, validation: bool = True) -> TrainingPlan:
    return TrainingPlan(
        resolution=896,
        batch_size=2,
        gradient_accumulation=2,
        repeats=4,
        epochs=8,
        estimated_steps=80,
        network_dim=32,
        network_alpha=16,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=2,
        validation_enabled=validation,
        training_gpu_uuid=GPU_UUID,
        estimated_disk_mb=512,
    )


def test_archaudit_dataset_toml_is_structured_and_separates_validation(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "教師 画像" / "train"
    validation_dir = tmp_path / "教師 画像" / "validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    paths = write_dataset_configs(
        output_dir=tmp_path / "config",
        training_image_dir=train_dir,
        validation_image_dir=validation_dir,
        plan=_plan(),
    )
    train = tomlkit.parse(paths.training.read_text(encoding="utf-8"))
    assert train["general"]["keep_tokens"] == 2
    assert train["datasets"][0]["resolution"] == 896
    assert train["datasets"][0]["batch_size"] == 2
    assert train["datasets"][0]["subsets"][0]["num_repeats"] == 4
    assert "validation_split" not in paths.training.read_text(encoding="utf-8")
    assert paths.validation is not None
    validation = tomlkit.parse(paths.validation.read_text(encoding="utf-8"))
    assert validation["general"]["shuffle_caption"] is False
    assert validation["datasets"][0]["batch_size"] == 1
    assert validation["datasets"][0]["subsets"][0]["num_repeats"] == 1


def test_archaudit_render_toml_round_trips_unicode_path(tmp_path: Path) -> None:
    image_dir = tmp_path / "日本語 (4)"
    rendered = render_dataset_toml(image_dir=image_dir, plan=_plan(validation=False))
    parsed = tomlkit.parse(rendered)
    assert parsed["datasets"][0]["subsets"][0]["image_dir"].endswith("日本語 (4)")


def test_archaudit_training_toml_places_validation_at_dataset_level(tmp_path: Path) -> None:
    image_dir = tmp_path / "all"
    image_dir.mkdir()
    plan = _plan().model_copy(update={"validation_image_count": 4})
    rendered = render_dataset_toml(
        image_dir=image_dir,
        plan=plan,
        validation_total_count=40,
    )
    parsed = tomlkit.parse(rendered)
    assert parsed["datasets"][0]["validation_split"] == 0.1
    assert parsed["datasets"][0]["validation_seed"] == 42
    assert "validation_split" not in parsed["datasets"][0]["subsets"][0]


def test_archaudit_recorded_validation_ids_match_pinned_sd_scripts_partition(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "all accepted"
    image_dir.mkdir()
    asset_ids = tuple(f"{index:064x}" for index in range(24))
    for asset_id in asset_ids:
        (image_dir / f"{asset_id}.png").write_bytes(b"fixture")
    training_ids, validation_ids = sd_scripts_validation_partition(
        asset_ids,
        validation_count=2,
        seed=42,
    )

    verify_materialized_validation_partition(
        image_dir=image_dir,
        expected_training_ids=training_ids,
        expected_validation_ids=validation_ids,
        seed=42,
    )
    with pytest.raises(ValueError, match="validation partition differs"):
        verify_materialized_validation_partition(
            image_dir=image_dir,
            expected_training_ids=training_ids,
            expected_validation_ids=training_ids[:2],
            seed=42,
        )


def test_archaudit_dataset_config_validation_failure_leaves_no_partial_train_file(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    output_dir = tmp_path / "config"
    with pytest.raises(ValueError, match="validation image directory"):
        write_dataset_configs(
            output_dir=output_dir,
            training_image_dir=train_dir,
            validation_image_dir=None,
            plan=_plan(validation=True),
        )
    assert not (output_dir / "dataset.toml").exists()


def test_archaudit_command_is_array_gpu_isolated_and_has_single_dataset_source(
    tmp_path: Path,
) -> None:
    python = tmp_path / "runtime" / "python.exe"
    sd_root = tmp_path / "sd scripts"
    script = sd_root / "sdxl_train_network.py"
    model = tmp_path / "モデル.safetensors"
    dataset = tmp_path / "dataset.toml"
    python.parent.mkdir()
    sd_root.mkdir()
    for path in (python, script, model, dataset):
        path.write_text("fixture", encoding="utf-8")
    binding = GpuBinding(
        uuid=GPU_UUID,
        physical_index=2,
        environment={"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    command = build_sd_scripts_command(
        python_executable=python,
        sd_scripts_root=sd_root,
        pretrained_model=model,
        dataset_config=dataset,
        output_dir=tmp_path / "output",
        output_name="日本語LoRA",
        logging_dir=tmp_path / "logs",
        plan=_plan(),
        binding=binding,
        selected_gpu_uuids=(GPU_UUID,),
        seed=123,
        extra_environment={"HF_HOME": str(tmp_path / "hf")},
    )
    assert isinstance(command.argv, tuple)
    assert command.argv[0] == str(python)
    assert "--num_processes=1" in command.argv
    assert "--save_model_as=safetensors" in command.argv
    assert command.environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert command.environment["HF_HOME"] == str(tmp_path / "hf")
    joined = "\n".join(command.argv)
    assert "--dataset_config=" in joined
    assert "--resolution=" not in joined
    assert "--train_batch_size=" not in joined
    assert "--num_repeats=" not in joined
    assert "--validate_every_n_epochs=1" in command.argv
    assert "--validation_seed=42" in command.argv


def test_archaudit_command_rejects_gpu_environment_override(tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    sd_root = tmp_path / "sd"
    script = sd_root / "sdxl_train_network.py"
    model = tmp_path / "base.safetensors"
    dataset = tmp_path / "dataset.toml"
    sd_root.mkdir()
    for path in (python, script, model, dataset):
        path.write_text("fixture", encoding="utf-8")
    binding = GpuBinding(
        uuid=GPU_UUID,
        physical_index=0,
        environment={
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
    )
    with pytest.raises(ValueError, match="cannot be overridden"):
        build_sd_scripts_command(
            python_executable=python,
            sd_scripts_root=sd_root,
            pretrained_model=model,
            dataset_config=dataset,
            output_dir=tmp_path / "out",
            output_name="safe",
            logging_dir=tmp_path / "logs",
            plan=_plan(),
            binding=binding,
            selected_gpu_uuids=(GPU_UUID,),
            seed=1,
            extra_environment={"CUDA_VISIBLE_DEVICES": "9"},
        )
