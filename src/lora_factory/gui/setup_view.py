"""First-run environment status page."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lora_factory.gui.contracts import SetupCheckView


class SetupView(QWidget):
    """Display actionable runtime checks without asking users to use PowerShell."""

    refresh_requested = Signal()
    repair_requested = Signal(str)
    continue_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checks: list[SetupCheckView] = []
        layout = QVBoxLayout(self)
        title = QLabel("First-run Setup")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        description = QLabel(
            "LoRA Factory checks Codex, Git, the managed Python runtime, NVIDIA/CUDA, "
            "PyTorch, ONNX Runtime, sd-scripts and available disk space. Repairable items "
            "can be handled here without a command prompt."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.summary = QLabel()
        self.summary.setObjectName("setupSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("setupChecks")
        self.table.setHorizontalHeaderLabels(("Check", "Status", "Details", "Action"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        self.refresh_button = QPushButton("Run checks again")
        self.refresh_button.setObjectName("refreshSetup")
        self.refresh_button.clicked.connect(self.refresh_requested)
        layout.addWidget(self.refresh_button)
        self.continue_button = QPushButton("Continue to project editor")
        self.continue_button.setObjectName("continueFromSetup")
        self.continue_button.clicked.connect(self.continue_requested)
        layout.addWidget(self.continue_button)

    def set_checks(self, values: Iterable[object]) -> None:
        self._checks = [SetupCheckView.from_value(value) for value in values]
        self.table.setRowCount(len(self._checks))
        for row, check in enumerate(self._checks):
            name = QTableWidgetItem(check.name)
            status = QTableWidgetItem(check.status.upper())
            detail = QTableWidgetItem(check.detail)
            detail.setToolTip(check.detail)
            color = QColor("#067647" if check.healthy else "#b42318")
            status.setForeground(color)
            status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, status)
            self.table.setItem(row, 2, detail)
            if check.repairable:
                repair = QPushButton("Repair")
                repair.setProperty("checkName", check.name)
                repair.clicked.connect(
                    lambda checked=False, name=check.name: self.repair_requested.emit(name)
                )
                self.table.setCellWidget(row, 3, repair)
            else:
                self.table.setItem(row, 3, QTableWidgetItem("—"))
        failed = [check for check in self._checks if not check.healthy]
        if not self._checks:
            self.summary.setText("No setup checks were returned by Application Services.")
        elif failed:
            self.summary.setText(
                f"{len(failed)} check(s) need attention. You can inspect the project editor, "
                "but creation remains blocked until required capabilities are ready."
            )
        else:
            self.summary.setText("Setup is ready. All reported checks passed.")

    @property
    def ready(self) -> bool:
        return bool(self._checks) and all(check.healthy for check in self._checks)

    def set_repair_busy(self, busy: bool, check_name: str = "") -> None:
        self.table.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.continue_button.setEnabled(not busy)
        if busy:
            self.summary.setText(
                f"Repairing {check_name} in the background. This can take several minutes…"
            )
