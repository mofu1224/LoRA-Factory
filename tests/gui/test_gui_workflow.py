"""Connected pytest-qt coverage for the complete beginner GUI workflow."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit

from lora_factory.config.models import ProjectConfig
from lora_factory.gui.main_window import MainWindow
from lora_factory.gui.project_editor import AdvancedSettingsDialog

GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"
GPU_UUID_BAD = "GPU-22222222-2222-2222-2222-222222222222"


def test_gui_suite_uses_offscreen_platform(qtbot: Any) -> None:
    del qtbot
    assert QApplication.platformName() == "offscreen"


class FakeController:
    """Deterministic application boundary used without replacing GUI behavior."""

    def __init__(
        self, *, delay: float = 0.0, action_delay: float = 0.0, cancel_mode: bool = False
    ) -> None:
        self.delay = delay
        self.action_delay = action_delay
        self.cancel_mode = cancel_mode
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.run_thread_id: int | None = None
        self.configs: list[ProjectConfig] = []
        self.resume_calls: list[str] = []
        self.promotions: list[tuple[str, str]] = []
        self.copies: list[tuple[str, str]] = []
        self.saved_destinations: list[Mapping[str, Any]] = []
        self.created_projects: list[str] = []
        self.overrides: list[tuple[str, Mapping[str, Any]]] = []
        self.opened: list[str] = []
        self.checks_ready = True

    def discover_gpus(self) -> Sequence[object]:
        return (
            {
                "uuid": GPU_UUID,
                "index": 0,
                "name": "RTX Test",
                "total_vram_mb": 16384,
                "free_vram_mb": 12288,
                "utilization_percent": 4,
                "capability_major": 12,
                "capability_minor": 0,
                "compatible": True,
            },
            {
                "uuid": GPU_UUID_BAD,
                "index": 1,
                "name": "Legacy Test",
                "total_vram_mb": 8192,
                "free_vram_mb": 8192,
                "utilization_percent": 0,
                "capability_major": 6,
                "capability_minor": 1,
                "compatible": False,
                "compatibility_reason": "Managed PyTorch runtime does not support this GPU",
            },
        )

    def setup_checks(self) -> Sequence[Mapping[str, Any]]:
        return (
            {
                "name": "Managed Python Runtime",
                "status": "ok" if self.checks_ready else "error",
                "detail": "Python 3.12 ready" if self.checks_ready else "Runtime is missing",
                "repairable": not self.checks_ready,
            },
        )

    def repair_setup(self, check_name: str) -> None:
        assert check_name == "Managed Python Runtime"
        if self.action_delay:
            time.sleep(self.action_delay)
        self.checks_ready = True

    def recent_projects(self) -> Sequence[Mapping[str, Any]]:
        return (
            {
                "project_id": "recover-me",
                "lora_name": "Recover Me",
                "status": "failed_recoverable",
                "trigger_token": "recover_token",
            },
        )

    def create_project(self, name: str) -> Mapping[str, Any]:
        self.created_projects.append(name)
        return {
            "project_id": name,
            "lora_name": name,
            "status": "draft",
            "created": True,
            "project_root": f"C:/LoRA Factory/Project/{name}",
        }

    def run_pipeline(
        self, config: ProjectConfig, emit: Callable[[dict[str, Any]], None]
    ) -> Mapping[str, Any]:
        self.run_thread_id = threading.get_ident()
        self.configs.append(config)
        self.started.set()
        emit(
            {
                "event_type": "progress",
                "message": "Analyzing immutable image copies",
                "stage": "ANALYZING",
                "overall_progress": 0.2,
                "stage_progress": 0.5,
                "details": {
                    "images_current": 2,
                    "images_total": 4,
                    "epoch": 1,
                    "epochs": 3,
                    "step": 5,
                    "steps": 20,
                    "training_loss": 0.42,
                    "validation_loss": 0.51,
                    "checkpoint": "epoch-1",
                    "detailed_log": "safe detailed line",
                    "gpus": [
                        {
                            "uuid": GPU_UUID,
                            "task": "TAGGING",
                            "utilization_percent": 38,
                            "memory_used_mb": 4096,
                            "memory_free_mb": 12288,
                        }
                    ],
                    "dataset_items": [
                        {
                            "asset_id": "asset-1",
                            "category": "Accepted",
                            "original_filename": "日本語 image (1).png",
                            "width": 1024,
                            "height": 768,
                            "bucket": "1024x768",
                            "raw_tags": ["1girl", "smile"],
                            "final_caption": "test_token, 1girl, smile",
                            "included": True,
                        }
                    ],
                },
            }
        )
        if self.cancel_mode:
            if not self.cancelled.wait(timeout=3):
                raise RuntimeError("test cancellation was not delivered")
            return {
                "project_id": config.project_id,
                "status": "cancelled",
                "message": "Cancelled at a safe stage boundary",
            }
        if self.delay:
            time.sleep(self.delay)
        return self._completion(config.project_id, config)

    def cancel_current(self) -> None:
        self.cancelled.set()

    def resume_project(
        self, project_id: str, emit: Callable[[dict[str, Any]], None]
    ) -> Mapping[str, Any]:
        self.run_thread_id = threading.get_ident()
        self.resume_calls.append(project_id)
        emit(
            {
                "event_type": "progress",
                "message": "Resumed from persisted stage",
                "stage": "TRAINING",
                "overall_progress": 0.75,
                "stage_progress": 0.4,
            }
        )
        return self._completion(project_id)

    def promote_alternative(self, project_id: str, checkpoint_id: str) -> Mapping[str, Any]:
        self.promotions.append((project_id, checkpoint_id))
        return {"final_model": f"C:/outputs/{checkpoint_id}.safetensors"}

    def copy_output(self, project_id: str, destination_kind: str) -> Mapping[str, Any]:
        self.copies.append((project_id, destination_kind))
        return {"destination": f"C:/destinations/{destination_kind}"}

    def save_destinations(self, destinations: Sequence[Mapping[str, Any]]) -> None:
        self.saved_destinations = list(destinations)

    def set_dataset_override(self, project_id: str, override: Mapping[str, Any]) -> None:
        self.overrides.append((project_id, override))

    def open_output_folder(self, project_id: str) -> None:
        self.opened.append(project_id)

    @staticmethod
    def _completion(project_id: str, config: ProjectConfig | None = None) -> Mapping[str, Any]:
        return {
            "project_id": project_id,
            "status": "completed",
            "final_model": f"C:/outputs/{project_id}.safetensors",
            "trigger": config.trigger_token if config else "test_token",
            "recommended_weight": 0.8,
            "recommended_range": [0.65, 0.9],
            "preset": "character",
            "base_model": "C:/models/base.safetensors",
            "score_summary": {"identity": 0.91, "quality": 0.87},
            "alternatives": {
                "epoch-2": "C:/outputs/epoch-2.safetensors",
                "epoch-1": "C:/outputs/epoch-1.safetensors",
                "epoch-3": "C:/outputs/epoch-3.safetensors",
            },
        }


def fill_valid_project(window: MainWindow, tmp_path: Path) -> None:
    editor = window.project_editor
    editor.lora_name.setText("Test LoRA")
    editor.trigger_token.setText("test_token")
    editor.base_model.setText(str(tmp_path / "base.safetensors"))
    editor.output_folder.setText(str(tmp_path / "output"))
    editor.add_training_paths((tmp_path / "images",))
    editor.select_gpu_automatically()


def test_name_only_add_project_creates_structure_without_starting_pipeline(qtbot: Any) -> None:
    controller = FakeController()
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()
    editor = window.project_editor
    editor.lora_name.setText("Name Only Project")

    qtbot.mouseClick(editor.add_project_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=3000)

    assert controller.created_projects == ["Name Only Project"]
    assert controller.configs == []
    assert "Created: C:/LoRA Factory/Project/Name Only Project" in (
        editor.project_creation_status.text()
    )
    assert editor.add_project_button.isEnabled()


def test_first_run_setup_repair_and_incompatible_gpu(qtbot: Any) -> None:
    controller = FakeController()
    controller.checks_ready = False
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()

    assert window.pages.currentWidget() is window.setup_view
    assert window.setup_view.table.rowCount() == 1
    repair = window.setup_view.table.cellWidget(0, 3)
    assert repair is not None
    qtbot.mouseClick(repair, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=3000)
    assert window.setup_view.ready
    assert window.project_editor._gpu_checkboxes[GPU_UUID].isEnabled()
    assert not window.project_editor._gpu_checkboxes[GPU_UUID_BAD].isEnabled()


def test_editor_dialog_selection_validation_progress_and_completion(
    qtbot: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    controller = FakeController()
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()
    editor = window.project_editor
    images_folder = tmp_path / "教師 画像 (folder)"
    image_a = tmp_path / "日本語 a.png"
    image_b = tmp_path / "long file name [b].webp"
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(images_folder)
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(image_a), str(image_b)], "Images"),
    )
    base_model = tmp_path / "Illustrious base.safetensors"
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(base_model), "Stable Diffusion checkpoint"),
    )
    qtbot.mouseClick(window.new_project_button, Qt.MouseButton.LeftButton)
    assert window.pages.currentWidget() is editor
    base_button = editor.findChild(type(editor.start_button), "browseBaseModel")
    assert base_button is not None
    qtbot.mouseClick(base_button, Qt.MouseButton.LeftButton)
    assert editor.base_model.text() == str(base_model)
    qtbot.mouseClick(
        editor.findChild(type(editor.start_button), "addFolder"), Qt.MouseButton.LeftButton
    )
    qtbot.mouseClick(
        editor.findChild(type(editor.start_button), "addFiles"), Qt.MouseButton.LeftButton
    )
    assert editor.training_paths.count() == 3

    editor.lora_name.setText("Test LoRA")
    editor.preset.setCurrentIndex(1)
    assert editor.preset.currentText() == "Style LoRA"
    editor.trigger_token.setText("bad,token")
    editor.output_folder.setText(str(tmp_path / "output"))
    editor.select_gpu_automatically()
    qtbot.mouseClick(editor.start_button, Qt.MouseButton.LeftButton)
    assert "separator" in editor.validation_message.text()
    assert not controller.configs

    editor.trigger_token.setText("1girl")
    assert "common Danbooru tag" in editor.trigger_warning.text()

    editor.trigger_token.setText("test_token")
    assert not editor.trigger_warning.text()
    gui_thread_id = threading.get_ident()
    with qtbot.waitSignal(window.pipeline_completed, timeout=3000):
        qtbot.mouseClick(editor.start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._thread is None, timeout=3000)

    assert controller.run_thread_id != gui_thread_id
    assert controller.configs[0].preset.value == "style"
    assert controller.configs[0].selected_gpu_uuids == (GPU_UUID,)
    assert len(controller.configs[0].input_paths) == 3
    assert window.progress_view.overall.value() == 100
    assert window.progress_view.images.text() == "2 / 4"
    assert window.progress_view.gpu_table.rowCount() == 1
    assert window.dataset_review.table.rowCount() == 1
    assert window.pages.currentWidget() is window.completion_view
    assert window.completion_view.final_path.text().endswith("Test LoRA.safetensors")


def test_worker_keeps_qt_heartbeat_responsive(qtbot: Any, tmp_path: Path) -> None:
    controller = FakeController(delay=0.25)
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()
    fill_valid_project(window, tmp_path)
    heartbeats: list[int] = []
    timer = QTimer(window)
    timer.setInterval(20)
    timer.timeout.connect(lambda: heartbeats.append(1))
    timer.start()

    with qtbot.waitSignal(window.pipeline_completed, timeout=3000):
        qtbot.mouseClick(window.project_editor.start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._thread is None, timeout=3000)
    timer.stop()
    assert len(heartbeats) >= 4


def test_setup_repair_keeps_qt_heartbeat_responsive(qtbot: Any) -> None:
    controller = FakeController(action_delay=0.2)
    controller.checks_ready = False
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()
    heartbeats: list[int] = []
    timer = QTimer(window)
    timer.setInterval(20)
    timer.timeout.connect(lambda: heartbeats.append(1))
    timer.start()
    repair = window.setup_view.table.cellWidget(0, 3)
    assert repair is not None
    qtbot.mouseClick(repair, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=3000)
    timer.stop()
    assert window.setup_view.ready
    assert len(heartbeats) >= 4


def test_cancel_resume_alternative_copy_open_and_settings(qtbot: Any, tmp_path: Path) -> None:
    controller = FakeController(cancel_mode=True)
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()
    fill_valid_project(window, tmp_path)
    qtbot.mouseClick(window.project_editor.start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(controller.started.is_set, timeout=2000)
    qtbot.mouseClick(window.progress_view.cancel_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._thread is None, timeout=3000)
    assert controller.cancelled.is_set()
    assert window.progress_view.resume_button.isEnabled()

    with qtbot.waitSignal(window.pipeline_completed, timeout=3000):
        qtbot.mouseClick(window.progress_view.resume_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._thread is None, timeout=3000)
    assert controller.resume_calls == ["Test LoRA"]

    completion = window.completion_view
    completion.alternatives.setCurrentRow(0)
    promote = completion.findChild(type(completion.alternatives), "missing")
    assert promote is None
    promote_button = completion.findChild(
        type(window.project_editor.start_button), "promoteAlternative"
    )
    assert promote_button is not None
    qtbot.mouseClick(promote_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=3000)
    assert controller.promotions == [("Test LoRA", "epoch-2")]
    assert completion.final_path.text().endswith("epoch-2.safetensors")

    copy_button = completion.findChild(type(window.project_editor.start_button), "copy_a1111")
    open_button = completion.findChild(type(window.project_editor.start_button), "openOutput")
    assert copy_button is not None and open_button is not None
    qtbot.mouseClick(copy_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=3000)
    qtbot.mouseClick(open_button, Qt.MouseButton.LeftButton)
    assert controller.copies == [("Test LoRA", "a1111")]
    assert controller.opened == ["Test LoRA"]

    settings = window.settings_view
    settings._paths["comfyui"].setText(str(tmp_path / "ComfyUI" / "models" / "loras"))
    settings._enabled["comfyui"].setChecked(True)
    save = settings.findChild(type(window.project_editor.start_button), "saveDestinations")
    assert save is not None
    qtbot.mouseClick(save, Qt.MouseButton.LeftButton)
    assert controller.saved_destinations[0]["kind"] == "comfyui"


def test_dataset_review_override_and_advanced_settings(qtbot: Any) -> None:
    controller = FakeController()
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.dataset_review.project_id = "review-project"
    window.dataset_review.set_items(
        (
            {
                "asset_id": "a1",
                "category": "Validation",
                "categories": ["Accepted", "Warning", "Validation"],
                "original_filename": "source.png",
                "width": 768,
                "height": 1024,
                "bucket": "768x1024",
                "quality_reasons": ["low detail"],
                "raw_tags": ["1girl"],
                "final_caption": "token, 1girl",
                "included": True,
            },
        )
    )
    warning_index = window.dataset_review.category.findData("warning")
    validation_index = window.dataset_review.category.findData("validation")
    assert "(1)" in window.dataset_review.category.itemText(warning_index)
    assert "(1)" in window.dataset_review.category.itemText(validation_index)
    window.dataset_review.category.setCurrentIndex(validation_index)
    assert window.dataset_review.table.rowCount() == 1
    assert window.dataset_review.table.item(0, 7).text() == "Accepted, Warning, Validation"
    window.dataset_review.category.setCurrentIndex(0)
    include = window.dataset_review.table.item(0, 0)
    assert include is not None
    include.setCheckState(Qt.CheckState.Unchecked)
    assert controller.overrides[-1] == (
        "review-project",
        {"asset_id": "a1", "included": False},
    )
    caption = window.dataset_review.table.cellWidget(0, 6)
    assert isinstance(caption, QLineEdit)
    caption.setText("token, 1girl, smiling")
    caption.editingFinished.emit()
    assert controller.overrides[-1] == (
        "review-project",
        {"asset_id": "a1", "final_caption": "token, 1girl, smiling"},
    )

    dialog = AdvancedSettingsDialog()
    qtbot.addWidget(dialog)
    dialog.network_dim.setValue(32)
    dialog.epochs.setValue(8)
    dialog.resolution.setCurrentText("896")
    values = dialog.overrides()
    assert values.network_dim == 32
    assert values.epochs == 8
    assert values.resolution == 896
