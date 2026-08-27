"""Beginner-facing project editor and its advanced settings dialog."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lora_factory.config.models import (
    AdvancedOverrides,
    BackendMode,
    CodexRefinementMode,
    PresetKind,
    ProjectConfig,
    ProjectDraft,
    TriggerWordMode,
)
from lora_factory.config.validation import trigger_token_collision_warning
from lora_factory.gui.contracts import GpuView


class AdvancedSettingsDialog(QDialog):
    """Keeps expert controls out of the standard project form."""

    def __init__(
        self, overrides: AdvancedOverrides | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.resize(520, 560)
        current = overrides or AdvancedOverrides()

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Automatic planning is recommended. Set a field only when you need to lock an "
            "expert override. Raw images are never cropped or overwritten."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()

        self.resolution = QComboBox()
        self.resolution.setObjectName("advancedResolution")
        self.resolution.addItem("Auto", None)
        for value in (768, 896, 1024):
            self.resolution.addItem(str(value), value)
        if current.resolution is not None:
            self.resolution.setCurrentText(str(current.resolution))
        form.addRow("Resolution", self.resolution)

        self.network_dim = self._auto_int(4, 256, current.network_dim, step=4)
        self.network_dim.setObjectName("advancedRank")
        form.addRow("Network rank", self.network_dim)
        self.network_alpha = self._auto_int(1, 256, current.network_alpha)
        self.network_alpha.setObjectName("advancedAlpha")
        form.addRow("Network alpha", self.network_alpha)
        self.batch_size = self._auto_int(1, 32, current.batch_size)
        self.batch_size.setObjectName("advancedBatch")
        form.addRow("Batch size", self.batch_size)
        self.gradient_accumulation = self._auto_int(1, 64, current.gradient_accumulation)
        form.addRow("Gradient accumulation", self.gradient_accumulation)
        self.repeats = self._auto_int(1, 100, current.repeats)
        self.repeats.setObjectName("advancedRepeats")
        form.addRow("Repeats", self.repeats)
        self.epochs = self._auto_int(1, 100, current.epochs)
        self.epochs.setObjectName("advancedEpochs")
        form.addRow("Epochs", self.epochs)
        self.unet_lr = self._auto_float(current.unet_lr)
        form.addRow("UNet learning rate", self.unet_lr)
        self.text_encoder_lr = self._auto_float(current.text_encoder_lr)
        form.addRow("Text encoder learning rate", self.text_encoder_lr)

        self.optimizer = QLineEdit(current.optimizer or "")
        self.optimizer.setPlaceholderText("Auto")
        form.addRow("Optimizer", self.optimizer)
        self.precision = QComboBox()
        self.precision.addItem("Auto", None)
        for precision_value in ("bf16", "fp16", "fp32"):
            self.precision.addItem(precision_value, precision_value)
        if current.precision is not None:
            self.precision.setCurrentText(current.precision)
        form.addRow("Precision", self.precision)

        self.recursive_import = QCheckBox("Include subfolders")
        self.recursive_import.setTristate(True)
        self.recursive_import.setCheckState(
            Qt.CheckState.PartiallyChecked
            if current.recursive_import is None
            else (Qt.CheckState.Checked if current.recursive_import else Qt.CheckState.Unchecked)
        )
        form.addRow("Folder import", self.recursive_import)
        layout.addLayout(form)

        note = QLabel(
            "Bucket training stays enabled. Automatic crop, upscale, flip and color "
            "augmentation remain off by default."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _auto_int(minimum: int, maximum: int, value: int | None, *, step: int = 1) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(0, maximum)
        widget.setSpecialValueText("Auto")
        widget.setSingleStep(step)
        widget.setValue(value if value is not None else 0)
        widget.setProperty("realMinimum", minimum)
        return widget

    @staticmethod
    def _auto_float(value: float | None) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 0.1)
        widget.setDecimals(8)
        widget.setSingleStep(0.00001)
        widget.setSpecialValueText("Auto")
        widget.setValue(value if value is not None else 0.0)
        return widget

    def overrides(self) -> AdvancedOverrides:
        recursive_state = self.recursive_import.checkState()
        recursive: bool | None
        if recursive_state == Qt.CheckState.PartiallyChecked:
            recursive = None
        else:
            recursive = recursive_state == Qt.CheckState.Checked

        def optional_int(widget: QSpinBox) -> int | None:
            value = widget.value()
            if value == 0:
                return None
            minimum = int(widget.property("realMinimum") or 1)
            return max(minimum, value)

        def optional_float(widget: QDoubleSpinBox) -> float | None:
            return widget.value() or None

        return AdvancedOverrides(
            resolution=self.resolution.currentData(),
            network_dim=optional_int(self.network_dim),
            network_alpha=optional_int(self.network_alpha),
            batch_size=optional_int(self.batch_size),
            gradient_accumulation=optional_int(self.gradient_accumulation),
            repeats=optional_int(self.repeats),
            epochs=optional_int(self.epochs),
            unet_lr=optional_float(self.unet_lr),
            text_encoder_lr=optional_float(self.text_encoder_lr),
            optimizer=self.optimizer.text().strip() or None,
            precision=self.precision.currentData(),
            recursive_import=recursive,
        )


class ProjectEditor(QWidget):
    """Collect the seven standard inputs and emit one validated ProjectConfig."""

    start_requested = Signal(object)
    project_creation_requested = Signal(str)
    dataset_review_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._advanced = AdvancedOverrides()
        self._backend_mode = BackendMode.REAL
        self._gpu_checkboxes: dict[str, QCheckBox] = {}
        self._gpus: list[GpuView] = []

        outer = QVBoxLayout(self)
        header = QLabel("Create a LoRA")
        header.setObjectName("projectEditorTitle")
        header.setStyleSheet("font-size: 22px; font-weight: 600;")
        outer.addWidget(header)
        intro = QLabel(
            "Choose the model, original images and allowed GPUs. LoRA Factory copies source "
            "images into an immutable project snapshot and plans the technical settings."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        content_layout = QVBoxLayout(contents)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.lora_name = QLineEdit()
        self.lora_name.setObjectName("loraName")
        self.lora_name.setPlaceholderText("Example: My Character")
        name_row = QWidget()
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.addWidget(self.lora_name, 1)
        self.add_project_button = QPushButton("Add Project")
        self.add_project_button.setObjectName("addProject")
        self.add_project_button.setToolTip(
            "Create Project/<name>, input-Image, output-model, base-model and the "
            "internal project folders now."
        )
        self.add_project_button.clicked.connect(self.request_project_creation)
        name_layout.addWidget(self.add_project_button)
        form.addRow("LoRA name *", name_row)

        self.project_creation_status = QLabel()
        self.project_creation_status.setObjectName("projectCreationStatus")
        self.project_creation_status.setWordWrap(True)
        self.lora_name.textEdited.connect(self.project_creation_status.clear)
        form.addRow("Project", self.project_creation_status)

        self.preset = QComboBox()
        self.preset.setObjectName("preset")
        self.preset.addItem("Character LoRA", PresetKind.CHARACTER)
        self.preset.addItem("Style LoRA", PresetKind.STYLE)
        form.addRow("Preset *", self.preset)

        self.codex_refinement_mode = QComboBox()
        self.codex_refinement_mode.setObjectName("codexRefinementMode")
        self.codex_refinement_mode.addItem(
            "Apply safe changes automatically", CodexRefinementMode.AUTO
        )
        self.codex_refinement_mode.addItem(
            "Review changes before training", CodexRefinementMode.REVIEW
        )
        form.addRow("Caption / Tag refinement", self.codex_refinement_mode)
        self.codex_image_upload_notice = QLabel(
            "Runtime Codex sends metadata-free, resized copies (maximum edge 2048 px) of "
            "every accepted image to OpenAI in batches of up to 8. Originals and project "
            "paths are not sent."
        )
        self.codex_image_upload_notice.setObjectName("codexImageUploadNotice")
        self.codex_image_upload_notice.setWordWrap(True)
        form.addRow("", self.codex_image_upload_notice)

        self.trigger_word_mode = QComboBox()
        self.trigger_word_mode.setObjectName("triggerWordMode")
        self.trigger_word_mode.addItem("Enter manually", TriggerWordMode.MANUAL)
        self.trigger_word_mode.addItem(
            "Use entered value; otherwise let Codex choose",
            TriggerWordMode.CODEX_SUGGEST,
        )
        self.trigger_word_mode.setCurrentIndex(
            self.trigger_word_mode.findData(TriggerWordMode.CODEX_SUGGEST)
        )
        form.addRow("Trigger Word", self.trigger_word_mode)

        self.trigger_token = QLineEdit()
        self.trigger_token.setObjectName("triggerToken")
        self.trigger_token.setPlaceholderText("A unique token, without commas")
        form.addRow("LoRA tag (trigger)", self.trigger_token)
        self.trigger_warning = QLabel()
        self.trigger_warning.setObjectName("triggerWarning")
        self.trigger_warning.setWordWrap(True)
        self.trigger_warning.setStyleSheet("color: #b42318;")
        form.addRow("", self.trigger_warning)
        self.trigger_token.textChanged.connect(self._update_trigger_warning)
        self.trigger_word_mode.currentIndexChanged.connect(self._update_trigger_mode)
        self._update_trigger_mode()

        self.base_model = QLineEdit()
        self.base_model.setObjectName("baseModel")
        base_row = QWidget()
        base_layout = QHBoxLayout(base_row)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.addWidget(self.base_model)
        base_button = QPushButton("Browse…")
        base_button.setObjectName("browseBaseModel")
        base_button.clicked.connect(self._choose_base_model)
        base_layout.addWidget(base_button)
        form.addRow("SDXL / Illustrious model *", base_row)
        content_layout.addLayout(form)

        images_box = QGroupBox("Training images *")
        images_layout = QVBoxLayout(images_box)
        immutable_note = QLabel(
            "Folders and multiple PNG/JPEG/WEBP files are accepted. Originals are copied and "
            "never renamed, edited, cropped or deleted."
        )
        immutable_note.setWordWrap(True)
        images_layout.addWidget(immutable_note)
        image_actions = QHBoxLayout()
        folder_button = QPushButton("Add folder…")
        folder_button.setObjectName("addFolder")
        folder_button.clicked.connect(self._choose_folder)
        files_button = QPushButton("Add image files…")
        files_button.setObjectName("addFiles")
        files_button.clicked.connect(self._choose_files)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_selected_paths)
        image_actions.addWidget(folder_button)
        image_actions.addWidget(files_button)
        image_actions.addWidget(remove_button)
        image_actions.addStretch()
        images_layout.addLayout(image_actions)
        self.training_paths = QListWidget()
        self.training_paths.setObjectName("trainingPaths")
        self.training_paths.setMinimumHeight(100)
        images_layout.addWidget(self.training_paths)
        content_layout.addWidget(images_box)

        self.gpu_box = QGroupBox("CUDA devices *")
        self.gpu_layout = QVBoxLayout(self.gpu_box)
        self.gpu_note = QLabel(
            "Only checked GPU UUIDs are made visible to pipeline processes. At least one "
            "compatible GPU is required."
        )
        self.gpu_note.setWordWrap(True)
        self.gpu_layout.addWidget(self.gpu_note)
        self.auto_select_gpu = QPushButton("Auto Select")
        self.auto_select_gpu.setObjectName("autoSelectGpu")
        self.auto_select_gpu.clicked.connect(self.select_gpu_automatically)
        self.gpu_layout.addWidget(self.auto_select_gpu)
        content_layout.addWidget(self.gpu_box)

        output_form = QFormLayout()
        self.output_folder = QLineEdit()
        self.output_folder.setObjectName("outputFolder")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_folder)
        output_button = QPushButton("Browse…")
        output_button.setObjectName("browseOutputFolder")
        output_button.clicked.connect(self._choose_output_folder)
        output_layout.addWidget(output_button)
        output_form.addRow("Output folder *", output_row)
        self.quality_mode = QCheckBox(
            "Quality mode (more checkpoint samples and evaluation; takes longer)"
        )
        self.quality_mode.setObjectName("qualityMode")
        output_form.addRow("Optional", self.quality_mode)
        content_layout.addLayout(output_form)

        advanced_row = QHBoxLayout()
        advanced_button = QPushButton("Advanced Settings…")
        advanced_button.setObjectName("advancedSettings")
        advanced_button.clicked.connect(self._show_advanced)
        review_button = QPushButton("Dataset Review")
        review_button.clicked.connect(self.dataset_review_requested)
        advanced_row.addWidget(advanced_button)
        advanced_row.addWidget(review_button)
        advanced_row.addStretch()
        content_layout.addLayout(advanced_row)

        self.validation_message = QLabel()
        self.validation_message.setObjectName("validationMessage")
        self.validation_message.setWordWrap(True)
        self.validation_message.setStyleSheet("color: #b42318;")
        content_layout.addWidget(self.validation_message)
        self.start_button = QPushButton("Create LoRA")
        self.start_button.setObjectName("startPipeline")
        self.start_button.setMinimumHeight(42)
        self.start_button.clicked.connect(self.validate_and_start)
        content_layout.addWidget(self.start_button)
        content_layout.addStretch()
        scroll.setWidget(contents)
        outer.addWidget(scroll)

    def _update_trigger_warning(self, value: str) -> None:
        self.trigger_warning.setText(trigger_token_collision_warning(value) or "")

    def _update_trigger_mode(self) -> None:
        suggested = self.trigger_word_mode.currentData() == TriggerWordMode.CODEX_SUGGEST
        self.trigger_token.setPlaceholderText(
            "Optional: your value is kept; blank lets Codex choose"
            if suggested
            else "A unique token, without commas"
        )

    def set_gpus(self, values: Iterable[object]) -> None:
        for checkbox in self._gpu_checkboxes.values():
            self.gpu_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self._gpu_checkboxes.clear()
        self._gpus = [GpuView.from_value(value) for value in values]
        for gpu in self._gpus:
            capability = f"sm_{gpu.capability_major}{gpu.capability_minor}"
            state = "compatible" if gpu.compatible else "incompatible"
            label = (
                f"CUDA {gpu.index}: {gpu.name} — {gpu.short_uuid} — "
                f"{gpu.free_vram_mb / 1024:.1f}/{gpu.total_vram_mb / 1024:.1f} GB free — "
                f"{gpu.utilization_percent:.0f}% — {capability} — {state}"
            )
            checkbox = QCheckBox(label)
            checkbox.setObjectName(f"gpu_{gpu.index}")
            checkbox.setProperty("gpuUuid", gpu.uuid)
            checkbox.setEnabled(gpu.compatible)
            if not gpu.compatible:
                checkbox.setToolTip(gpu.compatibility_reason or "Capability test failed")
            self._gpu_checkboxes[gpu.uuid] = checkbox
            self.gpu_layout.addWidget(checkbox)
        if not self._gpus:
            self.gpu_note.setText(
                "No CUDA GPU was reported. Run Setup checks or install a compatible managed "
                "runtime before starting."
            )
        else:
            self.gpu_note.setText(
                "Only checked GPU UUIDs are made visible to pipeline processes. At least one "
                "compatible GPU is required."
            )

    def select_gpu_automatically(self) -> None:
        compatible = [gpu for gpu in self._gpus if gpu.compatible]
        if not compatible:
            self.validation_message.setText("No compatible CUDA GPU is available.")
            return
        selected = max(compatible, key=lambda gpu: (gpu.free_vram_mb, gpu.total_vram_mb))
        for uuid, checkbox in self._gpu_checkboxes.items():
            checkbox.setChecked(uuid == selected.uuid)
        self.validation_message.clear()

    def selected_gpu_uuids(self) -> tuple[str, ...]:
        return tuple(
            uuid for uuid, checkbox in self._gpu_checkboxes.items() if checkbox.isChecked()
        )

    def add_training_paths(self, paths: Iterable[str | Path]) -> None:
        existing = {
            str(self.training_paths.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.training_paths.count())
        }
        for raw_path in paths:
            path = Path(raw_path)
            canonical = str(path)
            if canonical in existing:
                continue
            item = QListWidgetItem(canonical)
            item.setData(Qt.ItemDataRole.UserRole, canonical)
            item.setToolTip(canonical)
            self.training_paths.addItem(item)
            existing.add(canonical)

    def input_paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(str(self.training_paths.item(index).data(Qt.ItemDataRole.UserRole)))
            for index in range(self.training_paths.count())
        )

    def validate_and_start(self) -> None:
        self.validation_message.clear()
        try:
            if not self.base_model.text().strip():
                raise ValueError("Choose an SDXL or Illustrious base model.")
            if not self.output_folder.text().strip():
                raise ValueError("Choose an output folder.")
            config = ProjectConfig(
                project_id=self.lora_name.text().strip(),
                lora_name=self.lora_name.text().strip(),
                preset=self.preset.currentData(),
                trigger_token=self.trigger_token.text(),
                codex_refinement_mode=self.codex_refinement_mode.currentData(),
                trigger_word_mode=self.trigger_word_mode.currentData(),
                base_model=Path(self.base_model.text().strip()),
                input_paths=self.input_paths(),
                selected_gpu_uuids=self.selected_gpu_uuids(),
                output_root=Path(self.output_folder.text().strip()),
                backend_mode=self._backend_mode,
                quality_mode=self.quality_mode.isChecked(),
                advanced=self._advanced,
            )
        except (ValidationError, ValueError) as error:
            if isinstance(error, ValidationError):
                messages = [str(item["msg"]) for item in error.errors()]
                self.validation_message.setText("Please fix: " + "; ".join(messages))
            else:
                self.validation_message.setText(str(error))
            return
        self.start_requested.emit(config)

    def request_project_creation(self) -> None:
        """Validate only the name and request creation of the durable draft tree."""

        self.validation_message.clear()
        try:
            draft = ProjectDraft(
                project_id=self.lora_name.text(),
                lora_name=self.lora_name.text(),
            )
        except ValidationError as error:
            messages = [str(item["msg"]) for item in error.errors()]
            self.validation_message.setText("Please fix: " + "; ".join(messages))
            return
        self.project_creation_requested.emit(draft.lora_name)

    def set_project_creation_busy(self, busy: bool) -> None:
        self.add_project_button.setEnabled(not busy)
        self.add_project_button.setText("Adding…" if busy else "Add Project")

    def project_creation_succeeded(self, path: str, *, created: bool) -> None:
        action = "Created" if created else "Already exists"
        self.project_creation_status.setStyleSheet("color: #067647;")
        self.project_creation_status.setText(f"{action}: {path}")

    def project_creation_failed(self, message: str) -> None:
        self.project_creation_status.setStyleSheet("color: #b42318;")
        self.project_creation_status.setText(f"Could not add project: {message}")

    def set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.add_project_button.setEnabled(not busy)
        self.start_button.setText("Starting…" if busy else "Create LoRA")

    def set_backend_mode(self, backend_mode: BackendMode) -> None:
        """Select the application backend without exposing a beginner-facing control."""

        self._backend_mode = backend_mode

    def load_project(self, value: object) -> None:
        def read(name: str, default: Any = "") -> Any:
            if isinstance(value, dict):
                return value.get(name, default)
            return getattr(value, name, default)

        self.lora_name.setText(str(read("lora_name", read("name", ""))))
        preset = str(read("preset", "character"))
        index = self.preset.findData(
            PresetKind.STYLE if preset.casefold() == "style" else PresetKind.CHARACTER
        )
        self.preset.setCurrentIndex(max(index, 0))
        refinement_mode = str(read("codex_refinement_mode", CodexRefinementMode.AUTO.value))
        refinement_index = self.codex_refinement_mode.findData(CodexRefinementMode(refinement_mode))
        self.codex_refinement_mode.setCurrentIndex(max(refinement_index, 0))
        trigger_mode = str(read("trigger_word_mode", TriggerWordMode.CODEX_SUGGEST.value))
        trigger_mode_index = self.trigger_word_mode.findData(TriggerWordMode(trigger_mode))
        self.trigger_word_mode.setCurrentIndex(max(trigger_mode_index, 0))
        self.trigger_token.setText(str(read("trigger_token", "")))
        self.base_model.setText(str(read("base_model", "")))
        self.output_folder.setText(str(read("output_root", read("output_folder", ""))))
        self.training_paths.clear()
        raw_inputs = read("input_paths", ())
        if isinstance(raw_inputs, (list, tuple)):
            self.add_training_paths(str(path) for path in raw_inputs)
        selected = {str(uuid) for uuid in read("selected_gpu_uuids", ())}
        for uuid, checkbox in self._gpu_checkboxes.items():
            checkbox.setChecked(uuid in selected and checkbox.isEnabled())
        self.quality_mode.setChecked(bool(read("quality_mode", False)))
        raw_advanced = read("advanced", None)
        if raw_advanced is not None:
            self._advanced = AdvancedOverrides.model_validate(raw_advanced)
        raw_backend = str(read("backend_mode", self._backend_mode.value))
        self._backend_mode = BackendMode(raw_backend)
        project_root = str(read("project_root", ""))
        if project_root:
            self.project_creation_succeeded(project_root, created=False)
        else:
            self.project_creation_status.clear()
        self.validation_message.clear()

    def clear_form(self) -> None:
        """Reset visible project inputs while preserving the configured backend policy."""

        self.lora_name.clear()
        self.preset.setCurrentIndex(0)
        self.codex_refinement_mode.setCurrentIndex(0)
        self.trigger_word_mode.setCurrentIndex(
            self.trigger_word_mode.findData(TriggerWordMode.CODEX_SUGGEST)
        )
        self.trigger_token.clear()
        self.base_model.clear()
        self.training_paths.clear()
        for checkbox in self._gpu_checkboxes.values():
            checkbox.setChecked(False)
        self.output_folder.clear()
        self.quality_mode.setChecked(False)
        self._advanced = AdvancedOverrides()
        self.project_creation_status.clear()
        self.validation_message.clear()

    def _choose_base_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose base model",
            self.base_model.text(),
            "Stable Diffusion checkpoint (*.safetensors *.ckpt);;All files (*)",
        )
        if path:
            self.base_model.setText(path)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose training image folder")
        if path:
            self.add_training_paths((path,))

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose training images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp);;All files (*)",
        )
        self.add_training_paths(paths)

    def _choose_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_folder.text()
        )
        if path:
            self.output_folder.setText(path)

    def _remove_selected_paths(self) -> None:
        for item in self.training_paths.selectedItems():
            self.training_paths.takeItem(self.training_paths.row(item))

    def _show_advanced(self) -> None:
        dialog = AdvancedSettingsDialog(self._advanced, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._advanced = dialog.overrides()
