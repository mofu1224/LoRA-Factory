"""Mutable, run-scoped context passed only through the headless pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lora_factory.config.models import ProjectConfig
from lora_factory.core.cancellation import CancellationToken
from lora_factory.core.events import EventBus
from lora_factory.project.layout import ProjectLayout


@dataclass(slots=True)
class PipelineContext:
    run_id: str
    config: ProjectConfig
    layout: ProjectLayout
    events: EventBus
    cancellation: CancellationToken
    artifacts: dict[str, Any] = field(default_factory=dict)
