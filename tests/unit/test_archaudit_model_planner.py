from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from lora_factory.config.models import (
    AdvancedOverrides,
    DatasetStatistics,
    GpuCapability,
    PresetKind,
    ProjectConfig,
)
from lora_factory.model import ModelCompatibility, inspect_sdxl_safetensors
from lora_factory.training import (
    TrainingPlanningError,
    load_preset_profile,
    plan_training,
    plan_validation_split,
)

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _project(
    tmp_path: Path,
    *,
    preset: PresetKind = PresetKind.CHARACTER,
    advanced: AdvancedOverrides | None = None,
    locked_fields: frozenset[str] = frozenset(),
) -> ProjectConfig:
    return ProjectConfig(
        project_id="project",
        lora_name="日本語LoRA",
        preset=preset,
        trigger_token="lftrigger",  # noqa: S106 - domain trigger, not a password
        base_model=tmp_path / "base.safetensors",
        input_paths=(tmp_path / "images",),
        selected_gpu_uuids=(GPU_UUID,),
        output_root=tmp_path / "output",
        advanced=advanced or AdvancedOverrides(),
        locked_fields=locked_fields,
    )


def _gpu(*, free: int = 24 * 1024, bf16: bool = True) -> GpuCapability:
    return GpuCapability(
        uuid=GPU_UUID,
        index=0,
        name="RTX",
        total_vram_mb=24 * 1024,
        free_vram_mb=free,
        capability_major=12,
        capability_minor=0,
        bf16_supported=bf16,
    )


def test_archaudit_inspects_real_safetensors_header_without_loading_model(tmp_path: Path) -> None:
    model = tmp_path / "illustrious-test.safetensors"
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": np.zeros((1,), dtype=np.float32),
            "conditioner.embedders.1.model.transformer.resblocks.0.weight": np.zeros(
                (1,), dtype=np.float32
            ),
            "first_stage_model.encoder.weight": np.zeros((1,), dtype=np.float32),
        },
        model,
        metadata={"modelspec.title": "Illustrious XL", "prediction_type": "epsilon"},
    )
    inspection = inspect_sdxl_safetensors(model)
    assert inspection.sha256 and len(inspection.sha256) == 64
    assert inspection.is_sdxl
    assert inspection.possible_illustrious
    assert inspection.embedded_vae
    assert inspection.prediction_type == "epsilon"
    assert inspection.compatibility is ModelCompatibility.COMPATIBLE_WITH_WARNING


def test_archaudit_rejects_unknown_model_architecture(tmp_path: Path) -> None:
    model = tmp_path / "unknown.safetensors"
    save_file({"some.weight": np.zeros((1,), dtype=np.float32)}, model)
    inspection = inspect_sdxl_safetensors(model)
    assert not inspection.is_sdxl
    assert inspection.compatibility is ModelCompatibility.UNSUPPORTED


def test_archaudit_adaptive_character_plan_and_conditional_validation(tmp_path: Path) -> None:
    profile = load_preset_profile(PresetKind.CHARACTER)
    statistics = DatasetStatistics(
        accepted_count=24,
        short_side_p10=1000,
        short_side_median=1200,
        area_median=1_500_000,
        diversity_score=0.8,
    )
    plan = plan_training(_project(tmp_path), statistics, (_gpu(),), profile, has_class_token=True)
    assert plan.resolution == 1024
    assert plan.validation_enabled
    assert plan.keep_tokens == 2
    assert plan.precision == "bf16"
    assert plan.bucket_no_upscale and not plan.random_crop
    assert plan.estimated_steps == (
        (22 * plan.repeats * plan.epochs + plan.batch_size * plan.gradient_accumulation - 1)
        // (plan.batch_size * plan.gradient_accumulation)
    )

    disabled = plan_validation_split(tuple(f"asset-{index}" for index in range(23)), profile)
    enabled = plan_validation_split(tuple(f"asset-{index}" for index in range(24)), profile)
    assert not disabled.enabled
    assert len(enabled.validation_asset_ids) == 2
    assert enabled == plan_validation_split(tuple(f"asset-{index}" for index in range(24)), profile)


def test_archaudit_planner_honors_locked_override_and_low_vram_adaptation(
    tmp_path: Path,
) -> None:
    profile = load_preset_profile(PresetKind.CHARACTER)
    statistics = DatasetStatistics(
        accepted_count=20,
        short_side_p10=400,
        short_side_median=700,
        area_median=400_000,
    )
    automatic = plan_training(_project(tmp_path), statistics, (_gpu(free=9_000),), profile)
    assert automatic.resolution == 768
    assert automatic.precision == "bf16"

    advanced = AdvancedOverrides(resolution=1024, epochs=6, precision="fp16")
    locked = plan_training(
        _project(
            tmp_path,
            advanced=advanced,
            locked_fields=frozenset({"resolution", "epochs", "precision"}),
        ),
        statistics,
        (_gpu(free=9_000),),
        profile,
    )
    assert locked.resolution == 1024
    assert locked.epochs == 6
    assert locked.precision == "fp16"
    assert locked.no_half_vae
    assert locked.provenance["resolution"] == "user_lock"


def test_archaudit_style_low_diversity_does_not_increase_steps(tmp_path: Path) -> None:
    profile = load_preset_profile(PresetKind.STYLE)
    low = DatasetStatistics(accepted_count=40, short_side_p10=900, diversity_score=0.1)
    high = DatasetStatistics(accepted_count=40, short_side_p10=900, diversity_score=0.9)
    project = _project(tmp_path, preset=PresetKind.STYLE)
    low_plan = plan_training(project, low, (_gpu(),), profile)
    high_plan = plan_training(project, high, (_gpu(),), profile)
    assert low_plan.estimated_steps <= high_plan.estimated_steps
    assert low_plan.keep_tokens == 1


def test_archaudit_planner_blocks_hard_minimum_and_unsupported_bf16(tmp_path: Path) -> None:
    profile = load_preset_profile(PresetKind.CHARACTER)
    with pytest.raises(TrainingPlanningError, match="at least 8"):
        plan_training(
            _project(tmp_path),
            DatasetStatistics(accepted_count=7),
            (_gpu(),),
            profile,
        )
    with pytest.raises(TrainingPlanningError, match="bf16"):
        plan_training(
            _project(tmp_path, advanced=AdvancedOverrides(precision="bf16")),
            DatasetStatistics(accepted_count=20),
            (_gpu(bf16=False),),
            profile,
        )


def test_archaudit_planner_rejects_validation_below_conditional_threshold(
    tmp_path: Path,
) -> None:
    profile = load_preset_profile(PresetKind.CHARACTER)
    with pytest.raises(TrainingPlanningError, match="below the preset threshold"):
        plan_training(
            _project(tmp_path),
            DatasetStatistics(accepted_count=20, validation_count=2),
            (_gpu(),),
            profile,
        )
