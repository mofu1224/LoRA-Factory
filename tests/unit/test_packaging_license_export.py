from __future__ import annotations

from pathlib import Path

import pytest

from lora_factory.application.service import repository_root
from lora_factory.packaging.license_export import export_license_bundle


def test_license_export_copies_python_and_distribution_material(tmp_path: Path) -> None:
    python_license = tmp_path / "python-license.txt"
    python_license.write_text("PSF test license", encoding="utf-8")
    output = tmp_path / "licenses"

    copied = export_license_bundle(
        output,
        distribution_names=("pydantic",),
        python_license=python_license,
    )

    assert copied >= 3
    assert next(output.glob("CPython-*/LICENSE.txt")).read_text(encoding="utf-8") == (
        "PSF test license"
    )
    pydantic_root = next(output.glob("pydantic-*"))
    assert (pydantic_root / "PACKAGE-METADATA.txt").is_file()
    assert any(path.name == "LICENSE" for path in pydantic_root.rglob("LICENSE"))


def test_license_export_refuses_to_overwrite_existing_bundle(tmp_path: Path) -> None:
    output = tmp_path / "licenses"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        export_license_bundle(output, distribution_names=())


def test_windows_build_carries_external_runtime_inputs_and_license_export() -> None:
    root = repository_root()
    spec = (root / "packaging" / "lora_factory.spec").read_text(encoding="utf-8")
    build_script = (root / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    assert 'ROOT / "runtime-lock.txt"' in spec
    assert '"src" / "lora_factory" / "runtime_scripts" / "wd14_infer.py"' in spec
    assert '"src" / "lora_factory" / "runtime_scripts" / "clip_embed.py"' in spec
    assert "lora_factory.packaging.license_export" in build_script
    assert "CPython-*" in build_script
    assert "Build work directory already exists; refusing overwrite" in build_script
