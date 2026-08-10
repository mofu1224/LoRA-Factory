"""Live pipeline progress, telemetry, logs, cancellation, and resume controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lora_factory.gui.contracts import PipelineUpdate


def _percent(value: float | None) -> int:
    if value is None:
        return 0
    normalized = value * 100 if 0 <= value <= 1 else value
    return max(0, min(100, round(normalized)))


class ProgressView(QWidget):
    """Render immutable pipeline events delivered through Qt queued signals."""

    cancel_requested = Signal()
    resume_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_id = ""
        layout = QVBoxLayout(self)
        title = QLabel("Pipeline Progress")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        self.status = QLabel("Waiting to start")
        self.status.setObjectName("progressStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.overall = QProgressBar()
        self.overall.setObjectName("overallProgress")
        self.overall.setFormat("Overall %p%")
        layout.addWidget(self.overall)
        self.stage = QLabel("Stage: —")
        self.stage.setObjectName("currentStage")
        layout.addWidget(self.stage)
        self.stage_progress = QProgressBar()
        self.stage_progress.setObjectName("stageProgress")
        self.stage_progress.setFormat("Stage %p%")
        layout.addWidget(self.stage_progress)

        form = QFormLayout()
        self.images = QLabel("—")
        self.epoch = QLabel("—")
        self.step = QLabel("—")
        self.training_loss = QLabel("—")
        self.validation_loss = QLabel("—")
        self.checkpoint = QLabel("—")
        form.addRow("Images", self.images)
        form.addRow("Epoch", self.epoch)
        form.addRow("Step", self.step)
        form.addRow("Training loss", self.training_loss)
        form.addRow("Validation loss", self.validation_loss)
        form.addRow("Current checkpoint", self.checkpoint)
        layout.addLayout(form)

        gpu_title = QLabel("Selected GPU tasks and telemetry")
        gpu_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(gpu_title)
        self.gpu_table = QTableWidget(0, 5)
        self.gpu_table.setObjectName("gpuProgress")
        self.gpu_table.setHorizontalHeaderLabels(
            ("GPU UUID", "Task", "Utilization", "VRAM used", "VRAM free")
        )
        self.gpu_table.verticalHeader().setVisible(False)
        self.gpu_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.gpu_table)

        layout.addWidget(QLabel("Summary log"))
        self.summary_log = QPlainTextEdit()
        self.summary_log.setObjectName("summaryLog")
        self.summary_log.setReadOnly(True)
        self.summary_log.setMaximumBlockCount(1000)
        layout.addWidget(self.summary_log)
        self.detailed_toggle = QCheckBox("Show detailed log")
        self.detailed_toggle.setObjectName("detailedLogToggle")
        self.detailed_toggle.toggled.connect(self._toggle_detailed)
        layout.addWidget(self.detailed_toggle)
        self.detailed_log = QPlainTextEdit()
        self.detailed_log.setObjectName("detailedLog")
        self.detailed_log.setReadOnly(True)
        self.detailed_log.setMaximumBlockCount(5000)
        self.detailed_log.hide()
        layout.addWidget(self.detailed_log)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelPipeline")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)
        self.resume_button = QPushButton("Resume from saved state")
        self.resume_button.setObjectName("resumePipeline")
        self.resume_button.clicked.connect(lambda: self.resume_requested.emit(self.project_id))
        self.resume_button.setEnabled(False)
        layout.addWidget(self.resume_button)

    def begin(self, project_id: str, *, resumed: bool = False) -> None:
        self.project_id = project_id
        self.overall.setValue(0)
        self.stage_progress.setValue(0)
        self.status.setText("Resuming pipeline…" if resumed else "Starting pipeline…")
        self.stage.setText("Stage: —")
        self.summary_log.clear()
        self.detailed_log.clear()
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        self.resume_button.setEnabled(False)

    def apply_event(self, value: object) -> None:
        event = PipelineUpdate.from_value(value)
        if event.message:
            self.status.setText(event.message)
            self.summary_log.appendPlainText(event.message)
        if event.stage:
            self.stage.setText(f"Stage: {event.stage}")
        stage_detail_event = event.event_type.casefold().endswith("_progress")
        if event.overall_progress is not None and not stage_detail_event:
            self.overall.setValue(_percent(event.overall_progress))
        if event.stage_progress is not None:
            self.stage_progress.setValue(_percent(event.stage_progress))
        elif stage_detail_event and event.overall_progress is not None:
            self.stage_progress.setValue(_percent(event.overall_progress))
        details = event.details
        self._set_pair(self.images, details, "images_current", "images_total")
        self._set_pair(self.epoch, details, "epoch", "epochs")
        total_step_key = "steps" if "steps" in details else "total_steps"
        self._set_pair(self.step, details, "step", total_step_key)
        self._set_value(self.training_loss, details, "training_loss", "train_loss", "loss")
        self._set_value(self.validation_loss, details, "validation_loss", "val_loss")
        self._set_value(self.checkpoint, details, "checkpoint", "current_checkpoint")
        detailed = details.get("detailed_log", details.get("log"))
        if detailed:
            self.detailed_log.appendPlainText(str(detailed))
        telemetry_warning = details.get("telemetry_warning")
        if telemetry_warning:
            self.detailed_log.appendPlainText(f"GPU telemetry warning: {telemetry_warning}")
        gpu_values = details.get("gpus", details.get("gpu_telemetry"))
        if isinstance(gpu_values, Sequence) and not isinstance(gpu_values, (str, bytes)):
            self._set_gpu_rows(gpu_values)
        event_key = event.event_type.casefold()
        if event_key in {"cancelled", "canceled", "failed_recoverable"}:
            self.mark_cancelled(event.message or "Pipeline stopped with a resumable state.")

    def mark_cancel_requested(self) -> None:
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")
        self.status.setText("Cancellation requested. Finishing the current safe boundary…")

    def mark_cancelled(self, message: str = "Cancelled safely.") -> None:
        self.status.setText(message)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancel")
        self.resume_button.setEnabled(bool(self.project_id))

    def mark_failed(self, message: str, *, recoverable: bool = True) -> None:
        self.status.setText(message)
        self.summary_log.appendPlainText(message)
        self.cancel_button.setEnabled(False)
        self.resume_button.setEnabled(recoverable and bool(self.project_id))

    def mark_completed(self) -> None:
        self.overall.setValue(100)
        self.stage_progress.setValue(100)
        self.status.setText("Completed")
        self.cancel_button.setEnabled(False)
        self.resume_button.setEnabled(False)

    @staticmethod
    def _set_pair(
        label: QLabel, details: Mapping[str, Any], current_key: str, total_key: str
    ) -> None:
        current = details.get(current_key)
        total = details.get(total_key)
        if current is not None or total is not None:
            current_text = current if current is not None else "—"
            total_text = total if total is not None else "—"
            label.setText(f"{current_text} / {total_text}")

    @staticmethod
    def _set_value(label: QLabel, details: Mapping[str, Any], *keys: str) -> None:
        for key in keys:
            if key in details and details[key] is not None:
                label.setText(str(details[key]))
                return

    def _set_gpu_rows(self, values: Sequence[object]) -> None:
        self.gpu_table.setRowCount(len(values))
        for row, raw in enumerate(values):
            data = dict(raw) if isinstance(raw, Mapping) else vars(raw)
            uuid = str(data.get("uuid", ""))
            display_uuid = uuid if len(uuid) <= 18 else f"{uuid[:14]}…{uuid[-4:]}"
            cells = (
                display_uuid,
                str(data.get("task", data.get("task_kind", "Idle"))),
                f"{float(data.get('utilization_percent', 0)):.0f}%",
                f"{int(data.get('memory_used_mb', data.get('vram_used_mb', 0)))} MB",
                f"{int(data.get('memory_free_mb', data.get('vram_free_mb', 0)))} MB",
            )
            for column, text in enumerate(cells):
                self.gpu_table.setItem(row, column, QTableWidgetItem(text))

    def _toggle_detailed(self, visible: bool) -> None:
        self.detailed_log.setVisible(visible)
