"""Dataset review model/view surface with pre-run user overrides."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lora_factory.gui.contracts import DatasetItemView

CATEGORIES = ("Accepted", "Warning", "Rejected", "Duplicate", "Validation")


class DatasetReviewView(QWidget):
    """Inspect thumbnails, metadata, captions, and include/exclude choices."""

    override_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[DatasetItemView] = []
        self._run_locked = False
        self.project_id = ""
        layout = QVBoxLayout(self)
        title = QLabel("Dataset Review")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        self.help_text = QLabel(
            "Review is optional. Include/exclude and caption overrides are available only "
            "before the run snapshot is created; changes during a run apply to the next run."
        )
        self.help_text.setWordWrap(True)
        layout.addWidget(self.help_text)
        self.category = QComboBox()
        self.category.setObjectName("datasetCategory")
        self.category.currentIndexChanged.connect(self._render)
        layout.addWidget(self.category)
        self.table = QTableWidget(0, 8)
        self.table.setObjectName("datasetTable")
        self.table.setHorizontalHeaderLabels(
            (
                "Include",
                "Thumbnail / file",
                "Dimensions",
                "Bucket",
                "Quality reasons",
                "Raw tags",
                "Final caption",
                "Category",
            )
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setIconSize(QSize(96, 96))
        self.table.verticalHeader().setDefaultSectionSize(104)
        header = self.table.horizontalHeader()
        for column in (0, 2, 3, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (1, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._include_changed)
        layout.addWidget(self.table)

    def set_items(self, values: Iterable[object]) -> None:
        self._items = [DatasetItemView.from_value(value) for value in values]
        counts = Counter(
            category.casefold() for item in self._items for category in item.categories
        )
        current = self.category.currentData()
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem(f"All ({len(self._items)})", "all")
        for category in CATEGORIES:
            self.category.addItem(
                f"{category} ({counts[category.casefold()]})", category.casefold()
            )
        index = self.category.findData(current)
        self.category.setCurrentIndex(max(0, index))
        self.category.blockSignals(False)
        self._render()

    def set_run_locked(self, locked: bool) -> None:
        self._run_locked = locked
        self.help_text.setText(
            "This run snapshot is locked. Overrides made now are disabled until the current "
            "run ends."
            if locked
            else "Review is optional. Include/exclude and caption overrides are available only "
            "before the run snapshot is created; changes during a run apply to the next run."
        )
        self._render()

    def _filtered_items(self) -> list[DatasetItemView]:
        selected = str(self.category.currentData() or "all")
        if selected == "all":
            return list(self._items)
        return [
            item
            for item in self._items
            if selected in {category.casefold() for category in item.categories}
        ]

    def _render(self) -> None:
        items = self._filtered_items()
        self.table.blockSignals(True)
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            include = QTableWidgetItem()
            include.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | (
                    Qt.ItemFlag.ItemIsUserCheckable
                    if not self._run_locked
                    else Qt.ItemFlag.NoItemFlags
                )
            )
            include.setCheckState(
                Qt.CheckState.Checked if item.included else Qt.CheckState.Unchecked
            )
            include.setData(Qt.ItemDataRole.UserRole, item.asset_id)
            self.table.setItem(row, 0, include)
            filename = QTableWidgetItem(item.original_filename)
            if item.thumbnail_path and item.thumbnail_path.is_file():
                pixmap = QPixmap(str(item.thumbnail_path))
                if not pixmap.isNull():
                    filename.setIcon(QIcon(pixmap))
            filename.setToolTip(item.original_filename)
            self.table.setItem(row, 1, filename)
            self.table.setItem(row, 2, QTableWidgetItem(f"{item.width} x {item.height}"))
            self.table.setItem(row, 3, QTableWidgetItem(item.bucket))
            reasons = "; ".join(item.reasons) or "—"
            self.table.setItem(row, 4, QTableWidgetItem(reasons))
            self.table.setItem(row, 5, QTableWidgetItem(", ".join(item.raw_tags)))
            caption = QLineEdit(item.final_caption)
            caption.setEnabled(not self._run_locked)
            caption.setProperty("assetId", item.asset_id)
            caption.editingFinished.connect(
                lambda widget=caption, asset_id=item.asset_id: self._caption_changed(
                    asset_id, widget.text()
                )
            )
            self.table.setCellWidget(row, 6, caption)
            self.table.setItem(row, 7, QTableWidgetItem(", ".join(item.categories)))
        self.table.blockSignals(False)

    def _include_changed(self, item: QTableWidgetItem) -> None:
        if self._run_locked or item.column() != 0:
            return
        asset_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not asset_id:
            return
        included = item.checkState() == Qt.CheckState.Checked
        self._replace_included(asset_id, included)
        self.override_requested.emit(
            {
                "asset_id": asset_id,
                "included": included,
            }
        )

    def _caption_changed(self, asset_id: str, caption: str) -> None:
        for index, item in enumerate(self._items):
            if item.asset_id == asset_id:
                self._items[index] = replace(item, final_caption=caption)
                break
        self.override_requested.emit({"asset_id": asset_id, "final_caption": caption})

    def _replace_included(self, asset_id: str, included: bool) -> None:
        for index, item in enumerate(self._items):
            if item.asset_id == asset_id:
                self._items[index] = replace(item, included=included)
                return
