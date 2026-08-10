"""Training presets, planning, dataset TOML, and command construction."""

from lora_factory.training.batch_probe import (
    BatchProbeAttempt,
    BatchProbeError,
    BatchProbeRequest,
    BatchProbeResult,
    FakeBatchProbe,
    TorchCudaBatchProbe,
    descending_batch_candidates,
    estimate_training_vram_mb,
)
from lora_factory.training.command_builder import TrainingCommand, build_sd_scripts_command
from lora_factory.training.dataset_toml import (
    DatasetConfigPaths,
    render_dataset_toml,
    write_dataset_configs,
)
from lora_factory.training.planner import (
    TrainingPlanningError,
    ValidationSplitPlan,
    plan_training,
    plan_validation_split,
    validation_count,
)
from lora_factory.training.profiles import PresetProfile, load_preset_profile

__all__ = [
    "BatchProbeAttempt",
    "BatchProbeError",
    "BatchProbeRequest",
    "BatchProbeResult",
    "DatasetConfigPaths",
    "FakeBatchProbe",
    "PresetProfile",
    "TorchCudaBatchProbe",
    "TrainingCommand",
    "TrainingPlanningError",
    "ValidationSplitPlan",
    "build_sd_scripts_command",
    "descending_batch_candidates",
    "estimate_training_vram_mb",
    "load_preset_profile",
    "plan_training",
    "plan_validation_split",
    "render_dataset_toml",
    "validation_count",
    "write_dataset_configs",
]
