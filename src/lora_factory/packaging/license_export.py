"""Export the license files that accompany the windowed Factory distribution."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

FACTORY_DISTRIBUTIONS = (
    "alembic",
    "Mako",
    "MarkupSafe",
    "SQLAlchemy",
    "greenlet",
    "typing-extensions",
    "ImageHash",
    "numpy",
    "pillow",
    "PyWavelets",
    "scipy",
    "psutil",
    "pydantic",
    "annotated-types",
    "pydantic-core",
    "typing-inspection",
    "PySide6",
    "PySide6-Addons",
    "PySide6-Essentials",
    "shiboken6",
    "PyYAML",
    "rich",
    "markdown-it-py",
    "mdurl",
    "Pygments",
    "safetensors",
    "structlog",
    "tomlkit",
    "typer",
    "annotated-doc",
    "colorama",
    "shellingham",
    "setuptools",
    "packaging",
    "PyInstaller",
)

_LICENSE_NAME = re.compile(
    r"^(?:licen[cs]e|copying|notice|copyright|authors?)(?:[._-].*)?$",
    re.IGNORECASE,
)
_SAFE_COMPONENT = re.compile(r"[^0-9A-Za-z._-]+")


def _is_license_file(parts: tuple[str, ...]) -> bool:
    lowered = tuple(part.casefold() for part in parts)
    distribution_license_directory = any(
        part.endswith(".dist-info")
        and index + 1 < len(lowered)
        and lowered[index + 1] == "licenses"
        for index, part in enumerate(lowered)
    )
    return distribution_license_directory or bool(_LICENSE_NAME.fullmatch(parts[-1]))


def _safe_component(value: str) -> str:
    component = _SAFE_COMPONENT.sub("_", value).strip("._")
    if not component:
        raise ValueError("Distribution metadata produced an empty path component")
    return component


def _python_license() -> Path:
    candidates = (Path(sys.base_prefix) / "LICENSE.txt", Path(sys.prefix) / "LICENSE.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("The CPython LICENSE.txt file was not found")


def export_license_bundle(
    output: Path,
    *,
    distribution_names: Sequence[str] = FACTORY_DISTRIBUTIONS,
    python_license: Path | None = None,
) -> int:
    """Copy CPython and installed-distribution license material into a fresh directory."""

    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"License destination already exists: {output}")
    output.mkdir(parents=True)

    python_source = (python_license or _python_license()).resolve(strict=True)
    python_root = output / f"CPython-{sys.version_info.major}.{sys.version_info.minor}"
    python_root.mkdir()
    shutil.copy2(python_source, python_root / "LICENSE.txt")

    copied = 1
    for requested_name in distribution_names:
        try:
            installed = distribution(requested_name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required build distribution is not installed: {requested_name}"
            ) from exc

        name = installed.metadata.get("Name") or requested_name
        version = installed.version
        destination_root = output / f"{_safe_component(name)}-{_safe_component(version)}"
        destination_root.mkdir()

        metadata_lines = [
            f"Name: {name}",
            f"Version: {version}",
            f"License-Expression: {installed.metadata.get('License-Expression', '')}",
            f"License: {installed.metadata.get('License', '')}",
            f"Project-URL: {installed.metadata.get('Project-URL', '')}",
            "",
        ]
        (destination_root / "PACKAGE-METADATA.txt").write_text(
            "\n".join(metadata_lines), encoding="utf-8"
        )
        copied += 1

        license_count = 0
        for package_file in installed.files or ():
            parts = tuple(str(part) for part in package_file.parts)
            if not parts or ".." in parts or not _is_license_file(parts):
                continue
            relative_path = Path(*parts)
            if relative_path.is_absolute():
                continue
            source = Path(str(installed.locate_file(package_file))).resolve(strict=False)
            if not source.is_file():
                continue
            destination = (destination_root / relative_path).resolve(strict=False)
            try:
                destination.relative_to(destination_root)
            except ValueError:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            license_count += 1
            copied += 1
        if license_count == 0:
            raise RuntimeError(f"No license file was found for {name} {version}")

    return copied


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    copied = export_license_bundle(arguments.output)
    print(f"Exported {copied} license and metadata files to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
