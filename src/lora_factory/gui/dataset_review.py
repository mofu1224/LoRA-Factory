"""Dataset review model/view surface with pre-run user overrides."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lora_factory.caption.audit import QaSeverity, audit_captions
from lora_factory.caption.character_policy import CHARACTER_FORBIDDEN
from lora_factory.caption.style_policy import STYLE_FORBIDDEN
from lora_factory.config.models import PresetKind, validate_trigger_word
from lora_factory.dataset.ontology import DEFAULT_ONTOLOGY
from lora_factory.gui.contracts import DatasetItemView

CATEGORIES = ("Accepted", "Warning", "Rejected", "Duplicate", "Validation")

_REFINEMENT_COLUMN_COUNT = 9


class DatasetReviewView(QWidget):
    """Inspect thumbnails, metadata, captions, and include/exclude choices."""

    override_requested = Signal(object)
    refinement_approval_requested = Signal(object)

    DECISION_COLUMN = 0
    ASSET_COLUMN = 1
    ORIGINAL_TAGS_COLUMN = 2
    CHANGE_COLUMN = 3
    PROPOSED_TAGS_COLUMN = 4
    DRAFT_CAPTION_COLUMN = 5
    PROPOSED_CAPTION_COLUMN = 6
    REASON_COLUMN = 7
    CONFIDENCE_COLUMN = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[DatasetItemView] = []
        self._run_locked = False
        self._refinement_state: dict[str, Any] | None = None
        self._refinement_rows: dict[str, tuple[QComboBox, QLineEdit, QLineEdit]] = {}
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
        self.refinement_controls = QWidget()
        refinement_layout = QVBoxLayout(self.refinement_controls)
        self.trigger_candidates = QComboBox()
        self.trigger_candidates.setObjectName("triggerCandidates")
        self.trigger_candidates.currentIndexChanged.connect(self._candidate_selected)
        refinement_layout.addWidget(self.trigger_candidates)
        self.trigger_word = QLineEdit()
        self.trigger_word.setObjectName("reviewTriggerWord")
        self.trigger_word.setPlaceholderText("Select or enter a Trigger Word")
        self.trigger_word.textChanged.connect(self._update_refinement_validity)
        refinement_layout.addWidget(self.trigger_word)
        self.bulk_accept_button = QPushButton("Accept all safe proposals")
        self.bulk_accept_button.setObjectName("bulkAcceptRefinement")
        self.bulk_accept_button.clicked.connect(self._bulk_accept)
        refinement_layout.addWidget(self.bulk_accept_button)
        self.refinement_error = QLabel()
        self.refinement_error.setStyleSheet("color: #b42318;")
        self.refinement_error.setWordWrap(True)
        refinement_layout.addWidget(self.refinement_error)
        self.approve_refinement_button = QPushButton("Approve and continue training")
        self.approve_refinement_button.setObjectName("approveRefinement")
        self.approve_refinement_button.clicked.connect(self._request_refinement_approval)
        refinement_layout.addWidget(self.approve_refinement_button)
        self.refinement_controls.setVisible(False)
        layout.addWidget(self.refinement_controls)
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
        self._refinement_state = None
        self.refinement_controls.setVisible(False)
        self.category.setVisible(True)
        self.table.clear()
        self.table.setColumnCount(8)
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

    def set_refinement_review(self, value: Mapping[str, Any]) -> None:
        """Display a persisted all-asset before/after review without Raw paths."""

        self._refinement_state = dict(value)
        self.project_id = str(value.get("project_id", self.project_id))
        self.refinement_controls.setVisible(True)
        self.category.setVisible(False)
        self.trigger_candidates.blockSignals(True)
        self.trigger_candidates.clear()
        for candidate in value.get("trigger_candidates", ()):
            if isinstance(candidate, Mapping):
                label = f"{candidate.get('value', '')} — {candidate.get('reason', '')}"
                self.trigger_candidates.addItem(label, str(candidate.get("value", "")))
        self.trigger_candidates.blockSignals(False)
        configured = str(value.get("trigger_word", ""))
        if not configured and self.trigger_candidates.count():
            configured = str(self.trigger_candidates.itemData(0))
        self.trigger_word.setText(configured)
        self.help_text.setText(
            "Runtime Codex reviewed every accepted image. Compare the immutable source tags "
            "and draft with the validated proposal. Training has not started."
        )
        self._render_refinement()
        self._update_refinement_validity()

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
        if self._refinement_state is not None:
            self._render_refinement()
            return
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

    def _render_refinement(self) -> None:
        if self._refinement_state is None:
            return
        raw_items = self._refinement_state.get("items", ())
        items = [dict(item) for item in raw_items if isinstance(item, Mapping)]
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(_REFINEMENT_COLUMN_COUNT)
        self.table.setHorizontalHeaderLabels(
            (
                "Decision",
                "Asset",
                "Original tags",
                "Validated changes",
                "Proposed tags",
                "Draft caption",
                "Proposed / edited caption",
                "Reason",
                "Confidence",
            )
        )
        self.table.setRowCount(len(items))
        self._refinement_rows.clear()
        for row, item in enumerate(items):
            asset_id = str(item.get("asset_id", ""))
            decision = QComboBox()
            decision.addItem("Accept", "accept")
            decision.addItem("Reject", "reject")
            decision.addItem("Edit", "edit")
            decision.currentIndexChanged.connect(self._update_refinement_validity)
            self.table.setCellWidget(row, self.DECISION_COLUMN, decision)
            self.table.setItem(row, self.ASSET_COLUMN, QTableWidgetItem(asset_id))
            self.table.setItem(
                row,
                self.ORIGINAL_TAGS_COLUMN,
                QTableWidgetItem(", ".join(str(tag) for tag in item.get("original_tags", ()))),
            )
            changes = [
                *(f"+ {str(tag).replace('_', ' ')}" for tag in item.get("added_tags", ())),
                *(f"- {str(tag).replace('_', ' ')}" for tag in item.get("removed_tags", ())),
            ]
            self.table.setItem(
                row,
                self.CHANGE_COLUMN,
                QTableWidgetItem("\n".join(changes) or "—"),
            )
            tags = QLineEdit(", ".join(str(tag) for tag in item.get("proposed_tags", ())))
            tags.textChanged.connect(self._update_refinement_validity)
            self.table.setCellWidget(row, self.PROPOSED_TAGS_COLUMN, tags)
            self.table.setItem(
                row, self.DRAFT_CAPTION_COLUMN, QTableWidgetItem(str(item.get("draft_caption", "")))
            )
            caption = QLineEdit(str(item.get("proposed_caption", "")))
            caption.textChanged.connect(self._update_refinement_validity)
            self.table.setCellWidget(row, self.PROPOSED_CAPTION_COLUMN, caption)
            reason = str(item.get("reason", ""))
            rejected = item.get("rejection_reason")
            if rejected:
                reason = f"{reason}; Factory fallback: {rejected}"
            self.table.setItem(row, self.REASON_COLUMN, QTableWidgetItem(reason))
            self.table.setItem(
                row,
                self.CONFIDENCE_COLUMN,
                QTableWidgetItem(f"{float(item.get('confidence', 0)):.2f}"),
            )
            self._refinement_rows[asset_id] = (decision, tags, caption)
        self.table.blockSignals(False)

    def _candidate_selected(self) -> None:
        value = self.trigger_candidates.currentData()
        if value:
            self.trigger_word.setText(str(value))

    def _bulk_accept(self) -> None:
        for decision, _tags, _caption in self._refinement_rows.values():
            decision.setCurrentIndex(decision.findData("accept"))
        self._update_refinement_validity()

    def _approval_payload(self) -> dict[str, Any]:
        if self._refinement_state is None:
            raise ValueError("No refinement review is loaded")
        trigger = validate_trigger_word(self.trigger_word.text())
        items: list[dict[str, Any]] = []
        if bool(self._refinement_state.get("requires_refinement_review", False)):
            original_by_id = {
                str(item.get("asset_id", "")): item
                for item in self._refinement_state.get("items", ())
                if isinstance(item, Mapping)
            }
            for asset_id, (decision, tags, caption) in self._refinement_rows.items():
                action = str(decision.currentData())
                item: dict[str, Any] = {"asset_id": asset_id, "decision": action}
                if action == "edit":
                    raw_parts = tuple(tags.text().split(","))
                    if not raw_parts or any(not part.strip() for part in raw_parts):
                        raise ValueError(f"{asset_id}.effective_tags: tags cannot be empty")
                    parsed_tags = tuple(
                        DEFAULT_ONTOLOGY.canonicalize(part.strip()) for part in raw_parts
                    )
                    if len(parsed_tags) != len(set(parsed_tags)):
                        raise ValueError(
                            f"{asset_id}.effective_tags: duplicate tags are not allowed"
                        )
                    max_tags = int(self._refinement_state.get("max_effective_tags", 75))
                    if len(parsed_tags) > max_tags:
                        raise ValueError(f"{asset_id}.effective_tags: maximum is {max_tags} tags")
                    vocabulary = {
                        DEFAULT_ONTOLOGY.canonicalize(str(value))
                        for value in self._refinement_state.get("pinned_tag_vocabulary", ())
                    }
                    vocabulary.update(
                        DEFAULT_ONTOLOGY.canonicalize(str(value))
                        for value in original_by_id.get(asset_id, {}).get("baseline_tags", ())
                    )
                    categories = {
                        DEFAULT_ONTOLOGY.canonicalize(str(key)): str(value)
                        for key, value in dict(
                            self._refinement_state.get("pinned_tag_categories", {})
                        ).items()
                    }
                    preset = PresetKind(str(self._refinement_state.get("preset", "character")))
                    forbidden = (
                        CHARACTER_FORBIDDEN if preset is PresetKind.CHARACTER else STYLE_FORBIDDEN
                    )
                    for tag in parsed_tags:
                        if vocabulary and tag not in vocabulary:
                            raise ValueError(
                                f"{asset_id}.effective_tags: {tag!r} is absent from the pinned "
                                "WD14 vocabulary"
                            )
                        category = DEFAULT_ONTOLOGY.classify(
                            tag,
                            model_category=categories.get(tag),
                        )
                        if category in forbidden:
                            raise ValueError(
                                f"{asset_id}.effective_tags: {tag!r} uses a forbidden category: "
                                f"{category.value}"
                            )
                    item["effective_tags"] = parsed_tags
                    edited_caption = caption.text().strip()
                    original_caption = str(
                        original_by_id.get(asset_id, {}).get("proposed_caption", "")
                    )
                    if edited_caption != original_caption:
                        if not edited_caption:
                            raise ValueError(f"{asset_id}.caption: edited caption cannot be empty")
                        audit = audit_captions(
                            {asset_id: edited_caption},
                            trigger,
                            preset,
                            class_token=(
                                str(self._refinement_state["class_token"])
                                if self._refinement_state.get("class_token")
                                else None
                            ),
                            invariant_tags=tuple(
                                str(value) for value in self._refinement_state.get("invariants", ())
                            ),
                        )
                        errors = [
                            issue.message
                            for issue in audit.issues
                            if issue.severity is QaSeverity.ERROR
                        ]
                        if errors:
                            raise ValueError(f"{asset_id}.caption: {errors[0]}")
                        if audit.semantic_content_coverage < 1.0:
                            raise ValueError(
                                f"{asset_id}.caption: caption must contain at least one "
                                "semantic tag"
                            )
                        item["caption"] = edited_caption
                items.append(item)
        return {
            "upstream_fingerprint": str(self._refinement_state["upstream_fingerprint"]),
            "trigger_word": trigger,
            "items": items,
        }

    def _update_refinement_validity(self) -> None:
        if self._refinement_state is None:
            return
        try:
            self._approval_payload()
        except ValueError as error:
            self.refinement_error.setText(str(error))
            self.approve_refinement_button.setEnabled(False)
        else:
            self.refinement_error.clear()
            self.approve_refinement_button.setEnabled(True)

    def _request_refinement_approval(self) -> None:
        try:
            payload = self._approval_payload()
        except ValueError as error:
            self.refinement_error.setText(str(error))
            return
        self.approve_refinement_button.setEnabled(False)
        self.refinement_approval_requested.emit(payload)

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
