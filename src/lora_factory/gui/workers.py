"""Qt worker objects that keep blocking pipeline calls off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from PySide6.QtCore import QObject, Signal, Slot

from lora_factory.config.models import ProjectConfig
from lora_factory.gui.contracts import ApplicationController, as_mapping


class PipelineWorker(QObject):
    """Invoke one blocking application pipeline operation inside a QThread."""

    event_received = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        controller: ApplicationController,
        mode: Literal["run", "resume"],
        *,
        config: ProjectConfig | None = None,
        project_id: str = "",
    ) -> None:
        super().__init__()
        self._controller = controller
        self._mode = mode
        self._config = config
        self._project_id = project_id

    @Slot()
    def run(self) -> None:
        try:
            if self._mode == "run":
                if self._config is None:
                    raise RuntimeError("Pipeline worker was started without a project config")
                result = self._controller.run_pipeline(self._config, self._emit_event)
            else:
                if not self._project_id:
                    raise RuntimeError("Resume worker was started without a project id")
                result = self._controller.resume_project(self._project_id, self._emit_event)
            self.completed.emit(as_mapping(result))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            self.finished.emit()

    def _emit_event(self, value: dict[str, Any]) -> None:
        self.event_received.emit(value)


class ApplicationActionWorker(QObject):
    """Run a potentially slow application-service action outside the Qt main thread."""

    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, action: Callable[[], object]) -> None:
        super().__init__()
        self._action = action

    @Slot()
    def run(self) -> None:
        try:
            self.completed.emit(self._action())
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            self.finished.emit()
