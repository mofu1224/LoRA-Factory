"""Portable toolchain provisioning: resolution, verification, and extraction."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lora_factory.runtime import toolchain


def _pin(root: Path, **overrides: Any) -> toolchain.ToolPin:
    defaults = {
        "name": "uv",
        "version": "9.9.9",
        "url": "https://github.com/example/tool/releases/download/v9.9.9/tool.zip",
        "sha256": "a" * 64,
        "size_bytes": 3,
        "executable": "tool.exe",
    }
    defaults.update(overrides)
    return toolchain.ToolPin(**defaults)


def _bundle_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_resolve_prefers_bundled_verified_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    pin = toolchain.UV_PIN
    tool_dir = tmp_path / pin.name
    tool_dir.mkdir(parents=True)
    (tool_dir / pin.executable).write_bytes(b"binary")
    (tool_dir / toolchain._PIN_FILENAME).write_text(
        json.dumps({"name": pin.name, "version": pin.version, "sha256": pin.sha256}),
        encoding="utf-8",
    )
    monkeypatch.setattr(toolchain.shutil, "which", lambda _name: None)

    resolved = toolchain.resolve_uv()

    assert resolved is not None
    assert resolved == tool_dir / pin.executable


def test_resolve_ignores_stale_pin_and_falls_back_to_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    pin = toolchain.UV_PIN
    tool_dir = tmp_path / pin.name
    tool_dir.mkdir(parents=True)
    (tool_dir / pin.executable).write_bytes(b"binary")
    (tool_dir / toolchain._PIN_FILENAME).write_text(
        json.dumps({"name": pin.name, "version": "0.0.0", "sha256": "b" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: f"C:/tools/{name}.exe")

    assert toolchain.resolve_uv() == Path("C:/tools/uv.exe")
    assert toolchain.resolve_git() == Path("C:/tools/git.exe")


def test_ensure_tool_extracts_archive_and_records_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    pin = _pin(tmp_path)
    archive = tmp_path / "tool.zip"
    archive.write_bytes(_bundle_bytes({pin.executable: b"binary"}))
    monkeypatch.setattr(toolchain, "_download_verified", lambda _pin: archive)

    resolved = toolchain.ensure_tool(pin)

    assert resolved.read_bytes() == b"binary"
    record = json.loads((tmp_path / pin.name / toolchain._PIN_FILENAME).read_text(encoding="utf-8"))
    assert record == {"name": pin.name, "version": pin.version, "sha256": pin.sha256}


@pytest.mark.parametrize("record", [b"[]", b"\xff"])
def test_corrupt_pin_can_be_repaired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record: bytes
) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    pin = _pin(tmp_path)
    target = tmp_path / pin.name
    target.mkdir()
    (target / pin.executable).write_bytes(b"old")
    (target / toolchain._PIN_FILENAME).write_bytes(record)
    archive = tmp_path / "replacement.zip"
    archive.write_bytes(_bundle_bytes({pin.executable: b"new"}))
    monkeypatch.setattr(toolchain, "_download_verified", lambda _pin: archive)
    assert toolchain.ensure_tool(pin).read_bytes() == b"new"


def test_ensure_tool_reports_missing_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    pin = _pin(tmp_path)
    archive = tmp_path / "tool.zip"
    archive.write_bytes(_bundle_bytes({"other.exe": b"binary"}))
    monkeypatch.setattr(toolchain, "_download_verified", lambda _pin: archive)

    with pytest.raises(toolchain.ToolchainError, match="did not contain"):
        toolchain.ensure_tool(pin)


def test_download_rejects_hash_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    payload = b"pinned-archive-bytes"
    pin = _pin(tmp_path, size_bytes=len(payload), sha256=hashlib.sha256(b"other").hexdigest())

    class _Response(io.BytesIO):
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.close()

    monkeypatch.setattr(
        toolchain.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )

    with pytest.raises(toolchain.ToolchainError, match="SHA-256 verification failed"):
        toolchain._download_verified(pin)


def test_download_rejects_unexpected_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(toolchain, "toolchain_root", lambda: tmp_path)
    payload = b"short"
    pin = _pin(
        tmp_path,
        size_bytes=len(payload) + 5,
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    class _Response(io.BytesIO):
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.close()

    monkeypatch.setattr(
        toolchain.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )

    with pytest.raises(toolchain.ToolchainError, match="unexpected size"):
        toolchain._download_verified(pin)


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_bundle_bytes({"../escape.exe": b"x"}))

    with pytest.raises(toolchain.ToolchainError, match="unsafe path"):
        toolchain._safe_extract(archive, tmp_path / "target")
