"""Keep PySide6 native libraries isolated from unrelated Qt installations."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

_DLL_DIRECTORY_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()
_PRELOADED_DLL_HANDLES: list[Any] = []
_PRELOADED_DLLS: set[str] = set()


def _path_key(value: str) -> str:
    """Return a case-insensitive key suitable for Windows path de-duplication."""

    return os.path.normcase(os.path.normpath(value))


def _source_pyside_directory() -> Path | None:
    """Resolve the installed PySide6 package without importing any Qt module."""

    try:
        module_spec = importlib.util.find_spec("PySide6")
    except (AttributeError, ImportError, ValueError):
        return None
    if module_spec is None or module_spec.submodule_search_locations is None:
        return None
    locations = list(module_spec.submodule_search_locations)
    if not locations:
        return None
    return Path(locations[0]).resolve()


def _pyside_directory() -> tuple[Path | None, Path | None]:
    """Return the PySide6 and package root directories for the current launcher."""

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if not meipass:
            return None, None
        bundle_root = Path(os.fspath(meipass)).resolve()
        pyside_directory = bundle_root / "PySide6"
        return (pyside_directory if pyside_directory.is_dir() else None, bundle_root)

    source_pyside_directory = _source_pyside_directory()
    if source_pyside_directory is None or not source_pyside_directory.is_dir():
        return None, None
    return source_pyside_directory, None


def _prepend_path(directories: tuple[Path, ...]) -> None:
    """Prepend native library directories while preserving existing entries."""

    current = os.environ.get("PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    priority_directories = list(directories)
    system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
    if system_root:
        priority_directories.extend(
            path
            for path in (
                Path(system_root) / "System32",
                Path(system_root),
                Path(system_root) / "System32" / "Wbem",
            )
            if path.is_dir()
        )
    priority_values: list[str] = []
    priority_keys: set[str] = set()
    for directory in priority_directories:
        value = os.fspath(directory)
        key = _path_key(value)
        if key not in priority_keys:
            priority_values.append(value)
            priority_keys.add(key)
    remaining = [entry for entry in entries if _path_key(entry) not in priority_keys]
    os.environ["PATH"] = os.pathsep.join(priority_values + remaining)


def _register_dll_directory(directory: Path) -> None:
    """Keep an add_dll_directory handle alive for the whole process."""

    value = os.fspath(directory)
    key = _path_key(value)
    if key in _REGISTERED_DLL_DIRECTORIES:
        return
    try:
        handle = os.add_dll_directory(value)
    except (AttributeError, OSError):
        # PATH is still updated below. This keeps source execution compatible
        # with Python implementations that do not expose the Windows API.
        return
    _DLL_DIRECTORY_HANDLES.append(handle)
    _REGISTERED_DLL_DIRECTORIES.add(key)


def _preload_dll(directory: Path, filename: str) -> None:
    """Load one bundled DLL by absolute path before extension imports resolve it."""

    path = directory / filename
    if not path.is_file():
        return
    value = os.fspath(path)
    key = _path_key(value)
    if key in _PRELOADED_DLLS:
        return
    try:
        handle = ctypes.WinDLL(value)
    except (AttributeError, OSError):
        # The subsequent PySide6 import remains the source of truth for a
        # missing system prerequisite such as the official VC++ runtime.
        return
    _PRELOADED_DLL_HANDLES.append(handle)
    _PRELOADED_DLLS.add(key)


def _preload_native_qt_dlls(pyside_directory: Path) -> None:
    """Pin the Qt libraries whose same-named dependencies commonly collide."""

    for qt_filename in ("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"):
        _preload_dll(pyside_directory, qt_filename)


def prepare_qt_runtime() -> tuple[Path, ...]:
    """Pin PySide6's DLL and plugin search paths before importing Qt modules.

    PySide6's own initializer configures part of this state, but it runs only
    after the first PySide6 import. Preparing the paths here prevents a
    similarly named Qt DLL from an IDE, GPU tool, or another application from
    satisfying a PySide6 extension's dependency first.
    """

    if sys.platform != "win32":
        return ()

    pyside_directory, bundle_root = _pyside_directory()
    if pyside_directory is None:
        return ()

    candidates: list[Path] = [pyside_directory]
    shiboken_directory = pyside_directory.parent / "shiboken6"
    if shiboken_directory.is_dir():
        candidates.append(shiboken_directory)
    if bundle_root is not None:
        candidates.append(bundle_root)

    directories: list[Path] = []
    known: set[str] = set()
    for directory in candidates:
        if directory.is_dir() and _path_key(os.fspath(directory)) not in known:
            directories.append(directory)
            known.add(_path_key(os.fspath(directory)))

    for directory in directories:
        _register_dll_directory(directory)
    _prepend_path(tuple(directories))
    _preload_native_qt_dlls(pyside_directory)

    plugin_directory = pyside_directory / "plugins"
    if plugin_directory.is_dir():
        os.environ["QT_PLUGIN_PATH"] = os.fspath(plugin_directory)
    qml_directory = pyside_directory / "qml"
    if qml_directory.is_dir():
        os.environ["QML2_IMPORT_PATH"] = os.fspath(qml_directory)

    return tuple(directories)
