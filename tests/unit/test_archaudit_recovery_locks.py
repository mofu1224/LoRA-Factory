from __future__ import annotations

from lora_factory.config.models import GpuCapability, TrainingPlan
from lora_factory.core.exceptions import ErrorClassification
from lora_factory.training.recovery import next_recovery

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_archaudit_recovery_never_changes_coupled_locked_fields() -> None:
    plan = TrainingPlan(
        resolution=896,
        batch_size=2,
        gradient_accumulation=4,
        repeats=2,
        epochs=4,
        estimated_steps=32,
        network_dim=32,
        network_alpha=32,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=False,
        training_gpu_uuid=GPU_UUID,
        estimated_disk_mb=256,
    )
    gpu = GpuCapability(
        uuid=GPU_UUID,
        index=0,
        name="GPU",
        total_vram_mb=16_384,
        free_vram_mb=8_000,
        capability_major=12,
        capability_minor=0,
        bf16_supported=True,
    )

    decision = next_recovery(
        classification=ErrorClassification.CUDA_OOM,
        plan=plan,
        selected_gpus=(gpu,),
        locked_fields=frozenset({"gradient_accumulation", "network_alpha"}),
        previous_changes=(),
        attempt=1,
    )

    assert decision.change == "resolution"
    assert decision.plan.gradient_accumulation == plan.gradient_accumulation
    assert decision.plan.network_alpha == plan.network_alpha
    assert decision.plan.resolution == 768
