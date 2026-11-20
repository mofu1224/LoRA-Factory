from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lora_factory import qt_runtime


def _fake_pyside_tree(tmp_path: Path) -> tuple[Path, Path]:
    pyside = tmp_path / "PySide6"
    shiboken = tmp_path / "shiboken6"
    (pyside / "plugins").mkdir(parents=True)
    (pyside / "qml").mkdir()
    shiboken.mkdir()
    return pyside, shiboken


def test_prepare_qt_runtime_pins_frozen_bundle_before_external_path(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    pyside, shiboken = _fake_pyside_tree(tmp_path)
    for filename in ("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"):
        (pyside / filename).write_bytes(b"test")
    handles: list[str] = []
    preloaded: list[str] = []
    monkeypatch.setattr(qt_runtime.sys, "platform", "win32")
    monkeypatch.setattr(qt_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(qt_runtime.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(
        qt_runtime.os,
        "add_dll_directory",
        lambda value: handles.append(value) or value,
    )
    monkeypatch.setattr(
        qt_runtime.ctypes,
        "WinDLL",
        lambda value: preloaded.append(value) or value,
    )
    monkeypatch.setenv("PATH", r"C:\external-qt")
    monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("QML2_IMPORT_PATH", raising=False)

    directories = qt_runtime.prepare_qt_runtime()

    assert directories == (pyside.resolve(), shiboken.resolve(), tmp_path.resolve())
    assert handles == [str(path) for path in directories]
    assert preloaded == [
        str(pyside / "Qt6Core.dll"),
        str(pyside / "Qt6Gui.dll"),
        str(pyside / "Qt6Widgets.dll"),
    ]
    assert qt_runtime.os.environ["PATH"].split(os.pathsep)[:3] == [
        str(path) for path in directories
    ]
    assert qt_runtime.os.environ["QT_PLUGIN_PATH"] == str(pyside / "plugins")
    assert qt_runtime.os.environ["QML2_IMPORT_PATH"] == str(pyside / "qml")


def test_prepare_qt_runtime_resolves_source_package_without_frozen_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    pyside, shiboken = _fake_pyside_tree(tmp_path)
    handles: list[str] = []
    monkeypatch.setattr(qt_runtime.sys, "platform", "win32")
    monkeypatch.setattr(qt_runtime.sys, "frozen", False, raising=False)
    monkeypatch.delattr(qt_runtime.sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(
        qt_runtime.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(pyside)]),
    )
    monkeypatch.setattr(
        qt_runtime.os,
        "add_dll_directory",
        lambda value: handles.append(value) or value,
    )
    monkeypatch.setenv("PATH", r"C:\external-qt")

    directories = qt_runtime.prepare_qt_runtime()

    assert directories == (pyside.resolve(), shiboken.resolve())
    assert handles == [str(path) for path in directories]
    assert qt_runtime.os.environ["PATH"].split(os.pathsep)[:2] == [
        str(path) for path in directories
    ]


def test_prepare_qt_runtime_is_noop_outside_windows(monkeypatch: Any) -> None:
    monkeypatch.setattr(qt_runtime.sys, "platform", "linux")
    monkeypatch.setenv("PATH", "unchanged")

    assert qt_runtime.prepare_qt_runtime() == ()
    assert qt_runtime.os.environ["PATH"] == "unchanged"
