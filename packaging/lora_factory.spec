# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


ROOT = Path(SPECPATH).parent
PACKAGE_ROOT = ROOT / "src"

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
    runtime_hooks=[],
    excludes=[
        "hypothesis",
        "mypy",
        "onnxruntime",
        "pytest",
        "ruff",
        "torch",
        "torchvision",
    ],
    noarchive=False,
    optimize=1,
)
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
