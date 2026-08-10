"""Plain-text usage guide bundled with each final LoRA."""

from __future__ import annotations

from typing import Any


def build_output_readme(info: dict[str, Any]) -> str:
    warnings = info.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) or "- None"
    return f"""LoRA Factory output
===================

LoRA name: {info["lora_name"]}
Trigger: {info["trigger_token"]}
Preset: {info["preset"]}
Recommended weight: {info["recommended_weight"]:.2f}
Recommended range: {info["recommended_range"]}
Base model: {info["base_model"]}
Training resolution: {info["resolution"]}
Network rank / alpha: {info["network_dim"]} / {info["network_alpha"]}
Best checkpoint: {info["best_checkpoint"]}
Training images: {info["training_image_count"]}

Basic prompt example
--------------------
{info["trigger_token"]}, subject and scene tags...

Warnings
--------
{warning_text}

Install examples
----------------
AUTOMATIC1111 / Forge: models\\Lora\\{info["lora_name"]}.safetensors
ComfyUI: models\\loras\\{info["lora_name"]}.safetensors

Automatic metrics are decision aids, not an absolute aesthetic or identity judgment.
Use comparison.png and alternatives/ for manual review.
"""
