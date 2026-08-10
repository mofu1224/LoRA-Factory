"""Stable Diffusion destination settings page."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DESTINATIONS = (
    ("a1111", "AUTOMATIC1111 LoRA folder"),
    ("forge", "Forge LoRA folder"),
    ("comfyui", "ComfyUI LoRA folder"),
)


class SettingsView(QWidget):
    """Manage explicit, allowlisted output copy destinations."""

    save_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._paths: dict[str, QLineEdit] = {}
        self._enabled: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        explanation = QLabel(
            "Register one or more Stable Diffusion LoRA folders. Copy actions use Application "
            "Services and never overwrite a different file silently."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        for kind, label in DESTINATIONS:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            enabled = QCheckBox("Enabled")
            enabled.setObjectName(f"destinationEnabled_{kind}")
            path = QLineEdit()
            path.setObjectName(f"destinationPath_{kind}")
            browse = QPushButton("Browse…")
            browse.clicked.connect(
                lambda checked=False, destination=kind: self._choose(destination)
            )
            row_layout.addWidget(enabled)
            row_layout.addWidget(path)
            row_layout.addWidget(browse)
            form.addRow(label, row)
            self._enabled[kind] = enabled
            self._paths[kind] = path
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setObjectName("settingsStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        save = QPushButton("Save destinations")
        save.setObjectName("saveDestinations")
        save.clicked.connect(lambda: self.save_requested.emit(self.destinations()))
        layout.addWidget(save)
        layout.addStretch()

    def set_destinations(self, values: Iterable[Mapping[str, object] | object]) -> None:
        for raw in values:
            if isinstance(raw, Mapping):
                kind = str(raw.get("kind", ""))
                root = raw.get("root", raw.get("path", ""))
                enabled = bool(raw.get("enabled", True))
            else:
                kind = str(getattr(raw, "kind", ""))
                root = getattr(raw, "root", getattr(raw, "path", ""))
                enabled = bool(getattr(raw, "enabled", True))
            kind = kind.rsplit(".", maxsplit=1)[-1].casefold()
            if kind in self._paths:
                self._paths[kind].setText(str(root))
                self._enabled[kind].setChecked(enabled)

    def destinations(self) -> list[dict[str, object]]:
        values: list[dict[str, object]] = []
        for kind, _ in DESTINATIONS:
            raw_path = self._paths[kind].text().strip()
            if not raw_path:
                continue
            values.append(
                {
                    "kind": kind,
                    "root": Path(raw_path),
                    "enabled": self._enabled[kind].isChecked(),
                }
            )
        return values

    def saved(self) -> None:
        self.status.setStyleSheet("color: #067647;")
        self.status.setText("Destinations saved.")

    def save_failed(self, message: str) -> None:
        self.status.setStyleSheet("color: #b42318;")
        self.status.setText(message)

    def _choose(self, kind: str) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose LoRA destination", self._paths[kind].text()
        )
        if selected:
            self._paths[kind].setText(selected)
