"""Provision the pinned auxiliary tools (``uv`` and ``git``) inside the app.

The portable distribution never asks the user to install a language runtime or
a command-line helper.  On first use the applications downloads a pinned,
hash-verified official archive into ``<application>/tools`` and resolves the
executable from there, so the tools travel with the folder and none of them are
looked up on ``PATH``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lora_factory.util.app_paths import toolchain_root

ToolchainProgress = Callable[[str, float, str], None]
_HTTPS_HOSTS = frozenset({"github.com", "objects.githubusercontent.com"})
_PIN_FILENAME = "toolchain.json"


class ToolchainError(RuntimeError):
    """Raised when a pinned tool cannot be provisioned or verified."""


@dataclass(frozen=True, slots=True)
class ToolPin:
    """Immutable description of one pinned auxiliary tool."""

    name: str
    version: str
    url: str
    sha256: str
    size_bytes: int
    executable: str


UV_PIN = ToolPin(
    name="uv",
    version="0.13.1",
    url=("https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-pc-windows-msvc.zip"),
    sha256="3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672",
    size_bytes=15_308_685,
    executable="uv.exe",
)

GIT_PIN = ToolPin(
    name="git",
    version="2.56.0.2",
    url=(
        "https://github.com/git-for-windows/git/releases/download/"
        "v2.56.0.windows.2/MinGit-2.56.0.2-64-bit.zip"
    ),
    sha256="da35e72aa21c005a5a0d298cfbae110bc1609a815730ea0dde84b01a1b3cd3be",
    size_bytes=39_806_486,
    executable="cmd/git.exe",
)

TOOL_PINS: tuple[ToolPin, ...] = (UV_PIN, GIT_PIN)

_UV_FALLBACK_NAME = "uv"
_GIT_FALLBACK_NAME = "git"


def _tool_dir(pin: ToolPin) -> Path:
    return toolchain_root() / pin.name


def _pin_record(pin: ToolPin) -> Path:
    return _tool_dir(pin) / _PIN_FILENAME


def _installed_executable(pin: ToolPin) -> Path | None:
    candidate = _tool_dir(pin) / pin.executable
    if not candidate.is_file():
        return None
    record = _pin_record(pin)
    if not record.is_file():
        return None
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != pin.version or payload.get("sha256") != pin.sha256:
        return None
    return candidate


def resolve_uv() -> Path | None:
    """Return the ``uv`` executable, preferring the bundled copy over ``PATH``."""

    bundled = _installed_executable(UV_PIN)
    if bundled is not None:
        return bundled
    found = shutil.which(_UV_FALLBACK_NAME)
    return Path(found) if found else None


def resolve_git() -> Path | None:
    """Return the ``git`` executable, preferring the bundled copy over ``PATH``."""

    bundled = _installed_executable(GIT_PIN)
    if bundled is not None:
        return bundled
    found = shutil.which(_GIT_FALLBACK_NAME)
    return Path(found) if found else None


def _download_verified(pin: ToolPin) -> Path:
    parsed = urllib.parse.urlparse(pin.url)
    if parsed.scheme != "https" or parsed.hostname not in _HTTPS_HOSTS:
        raise ToolchainError(f"{pin.name} download requires an official GitHub HTTPS URL")

    destination_root = _tool_dir(pin)
    destination_root.mkdir(parents=True, exist_ok=True)
    archive = destination_root / f".{pin.name}-{pin.version}.zip"
    temporary = archive.with_name(archive.name + ".partial")

    existing = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(  # noqa: S310 - scheme and host validated above.
        pin.url,
        headers={"User-Agent": "LoRAFactory/v0.3"},
    )
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - pinned HTTPS.
        append = existing > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with temporary.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)

    actual_size = temporary.stat().st_size
    if actual_size != pin.size_bytes:
        temporary.unlink(missing_ok=True)
        raise ToolchainError(
            f"{pin.name} archive has an unexpected size: {actual_size} != {pin.size_bytes}"
        )
    digest = _sha256(temporary)
    if digest != pin.sha256:
        temporary.unlink(missing_ok=True)
        raise ToolchainError(
            f"{pin.name} archive SHA-256 verification failed: {digest} != {pin.sha256}"
        )
    temporary.replace(archive)
    return archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, target: Path) -> None:
    resolved_target = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename
            if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts:
                raise ToolchainError(f"{archive.name} contains an unsafe path: {name}")
            destination = (resolved_target / name).resolve()
            try:
                destination.relative_to(resolved_target)
            except ValueError as exc:
                raise ToolchainError(f"{archive.name} escapes its target: {name}") from exc
        bundle.extractall(resolved_target)


def ensure_tool(pin: ToolPin, progress: ToolchainProgress | None = None) -> Path:
    """Ensure one pinned tool is present and return its executable path."""

    installed = _installed_executable(pin)
    if installed is not None:
        return installed

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(pin.name, fraction, message)

    target = _tool_dir(pin)
    remove_previous_tree(target)
    target.mkdir(parents=True, exist_ok=True)
    report(0.05, f"Downloading pinned {pin.name} {pin.version}")
    archive = _download_verified(pin)
    staging = target / ".extracting"
    remove_previous_tree(staging)
    try:
        report(0.85, f"Extracting {pin.name} {pin.version}")
        _safe_extract(archive, staging)
        executable = staging / pin.executable
        if not executable.is_file():
            raise ToolchainError(f"{pin.name} archive did not contain {pin.executable}")
        for item in list(staging.iterdir()):
            item.replace(target / item.name)
    finally:
        remove_previous_tree(staging)
        archive.unlink(missing_ok=True)

    _pin_record(pin).write_text(
        json.dumps({"name": pin.name, "version": pin.version, "sha256": pin.sha256}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    final = _installed_executable(pin)
    if final is None:
        raise ToolchainError(f"{pin.name} could not be provisioned")
    report(1.0, f"{pin.name} {pin.version} ready")
    return final


def ensure_toolchain(progress: ToolchainProgress | None = None) -> tuple[Path, Path]:
    """Ensure both pinned tools exist and return ``(uv, git)`` executable paths."""

    offset = 0.0
    span = 1.0 / len(TOOL_PINS)

    def scoped(name: str, fraction: float, message: str) -> None:
        if progress is not None:
            progress(name, offset + span * fraction, message)

    resolved: dict[str, Path] = {}
    for index, pin in enumerate(TOOL_PINS):
        offset = span * index
        resolved[pin.name] = ensure_tool(pin, scoped)
    return resolved[UV_PIN.name], resolved[GIT_PIN.name]


def remove_previous_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)
