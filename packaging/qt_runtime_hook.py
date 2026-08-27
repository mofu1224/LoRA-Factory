"""PyInstaller hook that isolates bundled PySide6 DLLs before app startup."""

from lora_factory.qt_runtime import prepare_qt_runtime

prepare_qt_runtime()
