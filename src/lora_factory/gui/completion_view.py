"""Final artifact summary and manual checkpoint promotion controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class CompletionView(QWidget):
    """Expose the immediately usable final LoRA and top alternatives."""

    open_output_requested = Signal(str)
    copy_requested = Signal(str, str)
    promote_requested = Signal(str, str)
    duplicate_requested = Signal(str)
    new_project_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_id = ""
        self._model_path = ""
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        layout = QVBoxLayout(contents)
        self.title = QLabel("Completed")
        self.title.setObjectName("completionTitle")
        self.title.setStyleSheet("font-size: 24px; font-weight: 650; color: #067647;")
        layout.addWidget(self.title)
        self.action_status = QLabel()
        self.action_status.setObjectName("completionActionStatus")
        self.action_status.setWordWrap(True)
        layout.addWidget(self.action_status)
        form = QFormLayout()
        self.final_path = QLabel("—")
        self.final_path.setObjectName("finalModelPath")
        self.final_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.final_path.setWordWrap(True)
        self.trigger = QLabel("—")
        self.trigger.setObjectName("completionTrigger")
        self.trigger.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.weight = QLabel("—")
        self.weight.setObjectName("recommendedWeight")
        self.weight_range = QLabel("—")
        self.preset = QLabel("—")
        self.base_model = QLabel("—")
        self.base_model.setWordWrap(True)
        self.score = QLabel("—")
        self.score.setWordWrap(True)
        form.addRow("Final model", self.final_path)
        form.addRow("Trigger", self.trigger)
        form.addRow("Recommended weight", self.weight)
        form.addRow("Recommended range", self.weight_range)
        form.addRow("Preset", self.preset)
        form.addRow("Base model", self.base_model)
        form.addRow("Score summary", self.score)
        layout.addLayout(form)

        previews = QHBoxLayout()
        preview_column = QVBoxLayout()
        preview_column.addWidget(QLabel("Preview"))
        self.preview = QLabel("Preview not available")
        self.preview.setObjectName("previewImage")
        self.preview.setMinimumSize(260, 220)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("border: 1px solid #999;")
        preview_column.addWidget(self.preview)
        comparison_column = QVBoxLayout()
        comparison_column.addWidget(QLabel("Checkpoint / weight comparison"))
        self.comparison = QLabel("Comparison grid not available")
        self.comparison.setObjectName("comparisonGrid")
        self.comparison.setMinimumSize(360, 220)
        self.comparison.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.comparison.setStyleSheet("border: 1px solid #999;")
        comparison_column.addWidget(self.comparison)
        previews.addLayout(preview_column)
        previews.addLayout(comparison_column)
        layout.addLayout(previews)

        alternatives_title = QLabel("Top alternatives (manual override)")
        alternatives_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(alternatives_title)
        self.alternatives = QListWidget()
        self.alternatives.setObjectName("alternatives")
        self.alternatives.setMinimumHeight(110)
        layout.addWidget(self.alternatives)
        self.promote_button = QPushButton("Set selected alternative as Final")
        self.promote_button.setObjectName("promoteAlternative")
        self.promote_button.clicked.connect(self._request_promotion)
        layout.addWidget(self.promote_button)

        actions = QHBoxLayout()
        self.copy_buttons: list[QPushButton] = []
        open_button = QPushButton("Open output folder")
        open_button.setObjectName("openOutput")
        open_button.clicked.connect(lambda: self.open_output_requested.emit(self.project_id))
        actions.addWidget(open_button)
        for label, kind in (
            ("Copy to A1111", "a1111"),
            ("Copy to Forge", "forge"),
            ("Copy to ComfyUI", "comfyui"),
        ):
            button = QPushButton(label)
            button.setObjectName(f"copy_{kind}")
            button.clicked.connect(
                lambda checked=False, destination=kind: self.copy_requested.emit(
                    self.project_id, destination
                )
            )
            actions.addWidget(button)
            self.copy_buttons.append(button)
        layout.addLayout(actions)
        project_actions = QHBoxLayout()
        duplicate = QPushButton("Duplicate project settings")
        duplicate.setObjectName("duplicateProject")
        duplicate.clicked.connect(lambda: self.duplicate_requested.emit(self.project_id))
        new_project = QPushButton("Create another LoRA")
        new_project.setObjectName("newProjectFromCompletion")
        new_project.clicked.connect(self.new_project_requested)
        project_actions.addWidget(duplicate)
        project_actions.addWidget(new_project)
        project_actions.addStretch()
        layout.addLayout(project_actions)
        layout.addStretch()
        scroll.setWidget(contents)
        outer.addWidget(scroll)

    def set_result(self, value: Mapping[str, Any] | object) -> None:
        if isinstance(value, Mapping):
            data = dict(value)
        else:
            dump = getattr(value, "model_dump", None)
            data = dict(dump(mode="python")) if callable(dump) else dict(vars(value))
        self.project_id = str(data.get("project_id", self.project_id))
        self._model_path = str(
            data.get(
                "final_model_path",
                data.get("final_model", data.get("model_path", data.get("path", ""))),
            )
        )
        self.final_path.setText(self._model_path or "—")
        self.trigger.setText(str(data.get("trigger", data.get("trigger_token", "—"))))
        self.weight.setText(str(data.get("recommended_weight", "—")))
        recommended_range = data.get("recommended_range", data.get("weight_range", "—"))
        if isinstance(recommended_range, Sequence) and not isinstance(
            recommended_range, (str, bytes)
        ):
            recommended_range = " - ".join(str(item) for item in recommended_range)
        self.weight_range.setText(str(recommended_range))
        self.preset.setText(str(data.get("preset", "—")))
        self.base_model.setText(str(data.get("base_model", "—")))
        score = data.get("score_summary", data.get("score", "—"))
        if isinstance(score, Mapping):
            score = ", ".join(f"{key}: {item}" for key, item in score.items())
        self.score.setText(str(score))
        self._set_image(
            self.preview,
            data.get("preview_path", data.get("preview")),
            "Preview not available",
        )
        self._set_image(
            self.comparison,
            data.get("comparison_grid_path", data.get("comparison_path", data.get("comparison"))),
            "Comparison grid not available",
        )
        alternatives = data.get("alternatives", data.get("top_alternatives", ()))
        self.alternatives.clear()
        if isinstance(alternatives, Mapping):
            alternative_values: Sequence[object] = tuple(
                {"checkpoint_id": checkpoint_id, "path": path}
                for checkpoint_id, path in alternatives.items()
            )
        elif isinstance(alternatives, Sequence) and not isinstance(alternatives, (str, bytes)):
            alternative_values = alternatives
        else:
            alternative_values = ()
        for raw in alternative_values:
            alternative = dict(raw) if isinstance(raw, Mapping) else vars(raw)
            checkpoint_id = str(alternative.get("checkpoint_id", alternative.get("id", "")))
            label = str(alternative.get("label", checkpoint_id))
            score_value = alternative.get("score")
            if score_value is not None:
                label = f"{label} — score {score_value}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, checkpoint_id)
            self.alternatives.addItem(item)
        self.action_status.clear()

    def promotion_succeeded(self, value: Mapping[str, Any] | object) -> None:
        data = dict(value) if isinstance(value, Mapping) else dict(vars(value))
        new_path = data.get("final_model_path", data.get("final_model", data.get("model_path")))
        if new_path:
            self._model_path = str(new_path)
            self.final_path.setText(self._model_path)
        self.action_status.setText("Selected alternative is now the Final LoRA.")

    def action_succeeded(self, message: str) -> None:
        self.action_status.setStyleSheet("color: #067647;")
        self.action_status.setText(message)

    def action_failed(self, message: str) -> None:
        self.action_status.setText(message)
        self.action_status.setStyleSheet("color: #b42318;")

    def set_action_busy(self, busy: bool, message: str = "") -> None:
        self.promote_button.setEnabled(not busy)
        for button in self.copy_buttons:
            button.setEnabled(not busy)
        if busy and message:
            self.action_status.setStyleSheet("")
            self.action_status.setText(message)

    def _request_promotion(self) -> None:
        item = self.alternatives.currentItem()
        if item is None:
            self.action_failed("Select an alternative checkpoint first.")
            return
        checkpoint_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if checkpoint_id:
            self.promote_requested.emit(self.project_id, checkpoint_id)

    @staticmethod
    def _set_image(label: QLabel, raw_path: object, fallback: str) -> None:
        path = Path(str(raw_path)) if raw_path else None
        if path and path.is_file():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                label.setPixmap(
                    pixmap.scaled(
                        label.minimumSize(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                return
        label.setPixmap(QPixmap())
        label.setText(fallback)
