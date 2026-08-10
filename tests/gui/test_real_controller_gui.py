"""Offscreen GUI integration against the real application controller."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QPushButton
from safetensors.numpy import save_file

from lora_factory.application.service import LoRAFactoryController
from lora_factory.config.models import AppSettings, BackendMode
from lora_factory.gui.main_window import MainWindow

FAKE_GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"


def _fake_sdxl(path: Path) -> None:
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": np.zeros((1, 1), dtype=np.float32),
            "conditioner.embedders.1.model.text_projection": np.ones((1, 1), dtype=np.float32),
        },
        path,
        metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )


def _images(root: Path) -> None:
    root.mkdir(parents=True)
    for index in range(8):
        generator = np.random.default_rng(2200 + index)
        pixels = generator.integers(0, 256, size=(544, 640, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = np.clip(
            pixels[:, :, index % 3].astype(np.int16) + index * 5, 0, 255
        ).astype(np.uint8)
        Image.fromarray(pixels).save(root / f"実GUI 画像 ({index + 1}).png")


def _button(window: MainWindow, name: str) -> QPushButton:
    result = window.findChild(QPushButton, name)
    assert result is not None
    return result


def test_real_controller_offscreen_cancel_resume_override_copy_promote(
    qtbot: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    source = tmp_path / "入力 画像"
    _images(source)
    base_model = tmp_path / "tiny Illustrious SDXL.safetensors"
    _fake_sdxl(base_model)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    data_root = tmp_path / "LoRAFactory"
    settings = AppSettings(
        projects_root=data_root / "projects",
        managed_runtime_root=data_root / "runtime",
        codex_runtime_root=data_root / "codex",
    )
    controller = LoRAFactoryController(settings)
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.show()

    setup_names: set[str] = set()
    for row in range(window.setup_view.table.rowCount()):
        setup_item = window.setup_view.table.item(row, 0)
        if setup_item is not None:
            setup_names.add(setup_item.text())
    assert {"Factory Python", "Managed Training Runtime", "NVIDIA CUDA GPUs"} <= setup_names

    qtbot.mouseClick(window.new_project_button, Qt.MouseButton.LeftButton)
    editor = window.project_editor
    editor.set_backend_mode(BackendMode.FAKE)
    editor.set_gpus(
        (
            {
                "uuid": FAKE_GPU_UUID,
                "index": 0,
                "name": "Deterministic Fake CUDA Device",
                "total_vram_mb": 24 * 1024,
                "free_vram_mb": 24 * 1024,
                "capability_major": 12,
                "capability_minor": 0,
                "compatible": True,
            },
        )
    )
    editor.lora_name.setText("実 GUI 統合")
    qtbot.mouseClick(editor.add_project_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=5000)
    draft_root = settings.projects_root / "実 GUI 統合"
    assert draft_root.is_dir()
    assert (draft_root / "project.yaml").is_file()
    assert (draft_root / "state.sqlite3").is_file()
    assert (draft_root / "dataset" / "manifest.json").is_file()
    for relative in (
        "input-Image",
        "output-model",
        "base-model",
        "dataset/raw",
        "dataset/.import-staging",
        "dataset/working",
        "dataset/captions",
        "dataset/validation",
        "dataset/rejected",
        "configs",
        "runs",
        "final",
    ):
        assert (draft_root / relative).is_dir()
    assert "Created:" in editor.project_creation_status.text()
    editor.trigger_token.setText("real_gui_token")
    editor.base_model.setText(str(base_model))
    editor.add_training_paths((source,))
    editor.output_folder.setText(str(tmp_path / "output"))
    editor.select_gpu_automatically()

    with qtbot.waitSignal(window.pipeline_failed, timeout=10_000):
        qtbot.mouseClick(editor.start_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.progress_view.stage.text() != "Stage: —", timeout=5000)
        qtbot.mouseClick(window.progress_view.cancel_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)
    assert window.progress_view.resume_button.isEnabled()
    assert "CancelledError" in window.progress_view.status.text()

    with qtbot.waitSignal(window.pipeline_completed, timeout=30_000):
        qtbot.mouseClick(window.progress_view.resume_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)

    completion = window.completion_view
    assert window.pages.currentWidget() is completion
    final_model = Path(completion.final_path.text())
    assert final_model.is_file()
    assert completion.trigger.text() == "real_gui_token"
    assert completion.weight.text() != "—"
    assert not completion.preview.pixmap().isNull()
    assert not completion.comparison.pixmap().isNull()
    assert window.progress_view.overall.value() == 100
    assert "READY completed" in window.progress_view.summary_log.toPlainText()
    assert window.progress_view.images.text() != "—"
    assert window.progress_view.gpu_table.rowCount() == 1
    gpu_item = window.progress_view.gpu_table.item(0, 0)
    assert gpu_item is not None
    assert gpu_item.text().startswith("GPU-")

    review = window.dataset_review
    assert review.table.rowCount() == 8
    include = review.table.item(0, 0)
    assert include is not None
    asset_id = str(include.data(Qt.ItemDataRole.UserRole))
    include.setCheckState(Qt.CheckState.Unchecked)
    caption = review.table.cellWidget(0, 6)
    assert isinstance(caption, QLineEdit)
    caption.setText("real_gui_token, 1girl, custom smile")
    caption.editingFinished.emit()
    override_path = settings.projects_root / "実 GUI 統合" / "dataset" / "review-overrides.json"
    qtbot.waitUntil(override_path.is_file, timeout=2000)
    overrides = json.loads(override_path.read_text(encoding="utf-8"))
    assert overrides[asset_id] == {
        "included": False,
        "final_caption": "real_gui_token, 1girl, custom smile",
    }

    qtbot.mouseClick(window.settings_button, Qt.MouseButton.LeftButton)
    destination_root = tmp_path / "stable-diffusion-webui"
    window.settings_view._paths["a1111"].setText(str(destination_root))
    window.settings_view._enabled["a1111"].setChecked(True)
    qtbot.mouseClick(_button(window, "saveDestinations"), Qt.MouseButton.LeftButton)
    assert "saved" in window.settings_view.status.text().casefold()
    assert (data_root / "settings.json").is_file()

    restored_window = MainWindow(LoRAFactoryController())
    qtbot.addWidget(restored_window)
    assert restored_window.settings_view._paths["a1111"].text() == str(destination_root)
    assert restored_window.settings_view._enabled["a1111"].isChecked()

    window.pages.setCurrentWidget(completion)
    qtbot.mouseClick(_button(window, "copy_a1111"), Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=5000)
    copied = destination_root / "models" / "Lora" / final_model.name
    assert copied.is_file()
    assert "Copied safely" in completion.action_status.text()

    assert completion.alternatives.count() >= 1
    completion.alternatives.setCurrentRow(0)
    qtbot.mouseClick(completion.promote_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._action_thread is None, timeout=5000)
    history = final_model.parent / "history"
    assert history.is_dir() and any(history.glob("*.safetensors"))
    assert "now the Final" in completion.action_status.text()

    window.refresh_recent_projects()
    assert window.recent_list.count() == 1
    recent = window.recent_list.item(0)
    assert recent is not None
    recent_data = recent.data(Qt.ItemDataRole.UserRole)
    assert isinstance(recent_data, Mapping)
    assert str(recent_data["status"]).casefold() == "completed"
    window.recent_list.itemActivated.emit(recent)
    assert window.pages.currentWidget() is completion

    qtbot.mouseClick(_button(window, "duplicateProject"), Qt.MouseButton.LeftButton)
    assert window.pages.currentWidget() is editor
    assert editor.lora_name.text() == "実 GUI 統合 Copy"
    assert editor.trigger_token.text() == "real_gui_token"
    assert editor.training_paths.count() == 1
    assert editor.selected_gpu_uuids() == (FAKE_GPU_UUID,)
