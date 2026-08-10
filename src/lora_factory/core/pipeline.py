"""Persisted, idempotent stage orchestration independent of Qt."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from lora_factory.core.cancellation import CancelledError
from lora_factory.core.context import PipelineContext
from lora_factory.core.events import PipelineEvent
from lora_factory.core.exceptions import ErrorClassification, PipelineError
from lora_factory.core.fingerprint import stage_fingerprint
from lora_factory.core.stage import PipelineStage

JsonMapping = Mapping[str, Any]
StageRunner = Callable[[PipelineContext], dict[str, Any]]
InputResolver = Callable[[PipelineContext], JsonMapping]
StageSuccessHook = Callable[[PipelineContext, PipelineStage, dict[str, Any]], None]


class StagePersistence(Protocol):
    def successful_output(
        self, run_id: str, stage: str, fingerprint: str
    ) -> dict[str, Any] | None: ...

    def start(
        self,
        *,
        run_id: str,
        stage: str,
        fingerprint: str,
        stage_version: str,
        inputs: dict[str, Any],
    ) -> int: ...

    def finish(self, row_id: int, *, output: dict[str, Any]) -> None: ...

    def fail(self, row_id: int, *, error: dict[str, Any], cancelled: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class StageDefinition:
    stage: PipelineStage
    version: str
    resolve_inputs: InputResolver
    run: StageRunner
    backend_versions: JsonMapping


@dataclass(frozen=True, slots=True)
class PipelineResult:
    outputs: dict[str, dict[str, Any]]
    cache_hits: tuple[str, ...]


class PipelineEngine:
    def __init__(
        self,
        persistence: StagePersistence,
        *,
        on_stage_success: StageSuccessHook | None = None,
    ) -> None:
        self.persistence = persistence
        self.on_stage_success = on_stage_success

    def execute(
        self,
        context: PipelineContext,
        stages: Sequence[StageDefinition],
    ) -> PipelineResult:
        outputs: dict[str, dict[str, Any]] = {}
        cache_hits: list[str] = []
        total = max(len(stages), 1)

        for index, definition in enumerate(stages):
            context.cancellation.raise_if_cancelled()
            stage_name = definition.stage.value
            inputs = dict(definition.resolve_inputs(context))
            fingerprint = stage_fingerprint(
                stage=stage_name,
                stage_version=definition.version,
                inputs=inputs,
                config=context.config,
                backends=definition.backend_versions,
            )
            cached = self.persistence.successful_output(context.run_id, stage_name, fingerprint)
            if cached is not None:
                context.artifacts[stage_name] = cached
                outputs[stage_name] = cached
                if self.on_stage_success is not None:
                    self.on_stage_success(context, definition.stage, cached)
                cache_hits.append(stage_name)
                context.events.publish(
                    PipelineEvent(
                        event_type="stage_skipped",
                        stage=stage_name,
                        message=f"{stage_name} reused a verified matching result",
                        progress=(index + 1) / total,
                        details={"fingerprint": fingerprint},
                    )
                )
                continue

            row_id = self.persistence.start(
                run_id=context.run_id,
                stage=stage_name,
                fingerprint=fingerprint,
                stage_version=definition.version,
                inputs=inputs,
            )
            context.events.publish(
                PipelineEvent(
                    event_type="stage_started",
                    stage=stage_name,
                    message=f"{stage_name} started",
                    progress=index / total,
                )
            )
            try:
                output = definition.run(context)
                context.cancellation.raise_if_cancelled()
                if self.on_stage_success is not None:
                    self.on_stage_success(context, definition.stage, output)
                self.persistence.finish(row_id, output=output)
            except CancelledError as exc:
                self.persistence.fail(
                    row_id,
                    error={
                        "classification": ErrorClassification.CANCELLED.value,
                        "message": str(exc),
                        "recoverable": True,
                    },
                    cancelled=True,
                )
                context.events.publish(
                    PipelineEvent(
                        event_type="stage_cancelled",
                        stage=stage_name,
                        message=str(exc),
                        progress=index / total,
                    )
                )
                raise
            except PipelineError as exc:
                self.persistence.fail(
                    row_id,
                    error={
                        "classification": exc.classification.value,
                        "message": str(exc),
                        "recoverable": exc.recoverable,
                    },
                )
                context.events.publish(
                    PipelineEvent(
                        event_type="stage_failed",
                        stage=stage_name,
                        message=str(exc),
                        progress=index / total,
                        details={
                            "classification": exc.classification.value,
                            "recoverable": exc.recoverable,
                        },
                    )
                )
                raise
            except Exception as exc:
                self.persistence.fail(
                    row_id,
                    error={
                        "classification": ErrorClassification.UNKNOWN.value,
                        "message": str(exc),
                        "recoverable": False,
                        "exception_type": type(exc).__name__,
                    },
                )
                context.events.publish(
                    PipelineEvent(
                        event_type="stage_failed",
                        stage=stage_name,
                        message=str(exc),
                        progress=index / total,
                        details={"classification": ErrorClassification.UNKNOWN.value},
                    )
                )
                raise

            context.artifacts[stage_name] = output
            outputs[stage_name] = output
            context.events.publish(
                PipelineEvent(
                    event_type="stage_completed",
                    stage=stage_name,
                    message=f"{stage_name} completed",
                    progress=(index + 1) / total,
                    details={"fingerprint": fingerprint},
                )
            )

        return PipelineResult(outputs=outputs, cache_hits=tuple(cache_hits))
