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


def test_license_export_skips_optional_missing_distributions(tmp_path, capsys, monkeypatch) -> None:
    output = tmp_path / "licenses"
    python_license = tmp_path / "python-license.txt"
    python_license.write_text("PSF test license", encoding="utf-8")

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import distribution as original_distribution

    def fake_distribution(name: str):
        if name == "setuptools":
            raise PackageNotFoundError(name)
        return original_distribution(name)

    monkeypatch.setattr("lora_factory.packaging.license_export.distribution", fake_distribution)

    copied = export_license_bundle(
        output,
        distribution_names=("setuptools", "pydantic"),
        python_license=python_license,
    )

    captured = capsys.readouterr()
    assert copied >= 2
    assert "Skipping optional build distribution: setuptools" in captured.err
    assert next(output.glob("CPython-*/LICENSE.txt")).read_text(encoding="utf-8") == (
        "PSF test license"
    )


def test_license_export_fails_when_required_distribution_missing(tmp_path, monkeypatch) -> None:
    python_license = tmp_path / "python-license.txt"
    python_license.write_text("PSF test license", encoding="utf-8")

    def fake_distribution(name: str):
        from importlib.metadata import PackageNotFoundError

        if name == "definitely-not-installed-dist":
            raise PackageNotFoundError(name)
        raise NotImplementedError

    monkeypatch.setattr("lora_factory.packaging.license_export.distribution", fake_distribution)

    with pytest.raises(RuntimeError, match="Required build distribution is not installed"):
        export_license_bundle(
            tmp_path / "licenses",
            distribution_names=("definitely-not-installed-dist",),
            python_license=python_license,
        )


def test_windows_build_carries_external_runtime_inputs_and_license_export() -> None:
    root = repository_root()
    spec = (root / "packaging" / "lora_factory.spec").read_text(encoding="utf-8")
    build_script = (root / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    vcredist_script = (root / "scripts" / "install_vcredist.ps1").read_text(encoding="utf-8")

    assert 'ROOT / "runtime-lock.txt"' in spec
    assert '"src" / "lora_factory" / "runtime_scripts" / "wd14_infer.py"' in spec
    assert '"src" / "lora_factory" / "runtime_scripts" / "clip_embed.py"' in spec
    assert "lora_factory.packaging.license_export" in build_script
    assert "SystemVcRuntime" in build_script
    assert "systemVcRuntimeNames" in build_script
    assert "Where-Object { $_.Name -in $systemVcRuntimeNames }" in build_script
    assert "install_vcredist.ps1" in build_script
    assert "install_vcredist.cmd" in build_script
    assert "Microsoft.VCRedist.2015+.x64" in vcredist_script
    assert "https://aka.ms/vc14/vc_redist.x64.exe" in vcredist_script
    assert "Get-AuthenticodeSignature" in vcredist_script
    assert "Start-Process" in vcredist_script
    assert "--accept-package-agreements" in vcredist_script
    assert "--accept-source-agreements" in vcredist_script
    assert "CPython-*" in build_script
    assert "licenses\\Qt" in build_script
    assert "licenses\\Microsoft-Visual-Cpp" in build_script
    assert "LGPL-3.0-only.txt" in build_script
    assert "Build work directory already exists; refusing overwrite" in build_script
    assert '"PySide6.QtVirtualKeyboard"' in spec
    assert '"PySide6.QtPdf"' in spec
    assert 'filename.startswith("qt6virtualkeyboard")' in spec
    assert 'filename.startswith("qt6pdf")' in spec
