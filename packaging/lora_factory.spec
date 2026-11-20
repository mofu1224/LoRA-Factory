# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


ROOT = Path(SPECPATH).parent
PACKAGE_ROOT = ROOT / "src"

# The application imports only QtCore, QtGui, QtNetwork, and QtWidgets. Keep
# optional Qt modules out of the one-dir distribution so unused GPL-only
# modules and their third-party payloads are not accidentally redistributed.
UNUSED_QT_MODULES = [
    "PySide6.QtPdf",
    "PySide6.QtPdfQuick",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtVirtualKeyboardQml",
    "PySide6.QtVirtualKeyboardSettings",
]


def without_unused_qt_artifacts(toc):
    """Remove native modules/plugins that are not part of this application."""

    def keep(entry):
        filename = Path(str(entry[0]).replace("\\", "/")).name.lower()
        if filename in {"qpdf.dll", "qtvirtualkeyboardplugin.dll"}:
            return False
        # Qt's Windows build resolves its unversioned ICU imports from the
        # operating system. PyInstaller can otherwise collect Poppler's
        # versioned ICU DLLs from the build host, which causes WinError 127.
        if filename.startswith("icu") and filename.endswith(".dll"):
            return False
        return not (
            filename.startswith("qt6pdf")
            or filename.startswith("qt6qml")
            or filename.startswith("qt6quick")
            or filename.startswith("qt6virtualkeyboard")
            or filename.startswith("qtpdf")
            or filename.startswith("qtqml")
            or filename.startswith("qtquick")
            or filename.startswith("qtvirtualkeyboard")
        )

    return [entry for entry in toc if keep(entry)]

datas = [
    (str(ROOT / "backend-manifest.json"), "."),
    (str(ROOT / "runtime-lock.txt"), "."),
    (str(ROOT / "presets"), "presets"),
    (str(ROOT / "schemas"), "schemas"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (
        str(ROOT / "src" / "lora_factory" / "runtime_scripts" / "wd14_infer.py"),
        "src/lora_factory/runtime_scripts",
    ),
    (
        str(ROOT / "src" / "lora_factory" / "runtime_scripts" / "clip_embed.py"),
        "src/lora_factory/runtime_scripts",
    ),
]
datas += collect_data_files("lora_factory.resources")

a = Analysis(
    [str(PACKAGE_ROOT / "lora_factory" / "app.py")],
    pathex=[str(PACKAGE_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "lora_factory.application.service",
        "lora_factory.storage.migrations.env",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "qt_runtime_hook.py")],
    excludes=[
        "hypothesis",
        "mypy",
        "onnxruntime",
        "pytest",
        "ruff",
        "torch",
        "torchvision",
        *UNUSED_QT_MODULES,
    ],
    noarchive=False,
    optimize=1,
)
a.binaries = without_unused_qt_artifacts(a.binaries)
a.datas = without_unused_qt_artifacts(a.datas)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LoRA Factory",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LoRA Factory",
)
