"""Versioned benchmark prompt loading."""

from __future__ import annotations

from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict

from lora_factory.config.models import PresetKind


class BenchmarkPrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    prompt: str
    negative_prompt: str = ""


def load_benchmark_prompts(preset: PresetKind) -> tuple[BenchmarkPrompt, ...]:
    resource = files("lora_factory.resources.prompts").joinpath(f"{preset.value}.yaml")
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), list):
        raise ValueError(f"Invalid benchmark prompt resource for {preset.value}")
    prompts = tuple(BenchmarkPrompt.model_validate(item) for item in payload["prompts"])
    if len(prompts) < 8:
        raise ValueError(f"{preset.value} requires at least eight benchmark prompts")
    return prompts
