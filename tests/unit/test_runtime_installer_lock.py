from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from lora_factory.application.service import repository_root
from lora_factory.core.cancellation import CancellationToken
from lora_factory.runtime.installer import ManagedRuntimeInstaller
from lora_factory.runtime.manager import RuntimeManager
from lora_factory.runtime.validation_record import ValidationStatus
from lora_factory.util.hashing import sha256_file


def _write_manifest(root: Path, lock_sha256: str) -> Path:
    manifest_path = root / "backend-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "test-runtime",
                "managed_python": "3.12.13",
                "runtime_lock": {
                    "path": "runtime-lock.txt",
                    "sha256": lock_sha256,
                    "platform": "win32",
                    "python": "3.12.13",
                    "package_count": 1,
                    "excluded": ["torch", "torchvision", "sd-scripts"],
                },
                "sd_scripts": {
                    "repository": "https://example.invalid/sd-scripts.git",
                    "release": "v1.0.0",
                    "commit": "a" * 40,
                },
                "pytorch": {
                    "torch": "2.13.0",
                    "torchvision": "0.28.0",
                    "index_url": "https://download.pytorch.org/whl/cu130",
                },
                "wd14": {},
                "image_embedding": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_repository_runtime_lock_matches_manifest_and_contains_only_exact_packages() -> None:
    root = repository_root()
    manifest = json.loads((root / "backend-manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["runtime_lock"]
    lock_path = root / descriptor["path"]
    lock_bytes = lock_path.read_bytes()
    requirements = [
        line.strip()
        for line in lock_bytes.decode("utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert hashlib.sha256(lock_bytes).hexdigest() == descriptor["sha256"]
    assert len(requirements) == descriptor["package_count"] == 57
    assert all("==" in requirement for requirement in requirements)
    assert not any(requirement.casefold().startswith("torch==") for requirement in requirements)
    assert not any(
        requirement.casefold().startswith("torchvision==") for requirement in requirements
    )
    assert not any(requirement.startswith("-e ") for requirement in requirements)
    assert f"onnxruntime-gpu=={manifest['onnxruntime_gpu']['version']}" in requirements
    embedding = manifest["image_embedding"]
    assert embedding["model_id"] == "openai/clip-vit-large-patch14"
    assert len(embedding["revision"]) == 40
    assert embedding["transformers_version"] == "4.54.1"
    assert embedding["projection_dimension"] == 768
    assert set(embedding["artifacts"]) == {"config.json", "model.safetensors"}
    assert all(len(item["sha256"]) == 64 for item in embedding["artifacts"].values())


def test_installer_rejects_runtime_lock_hash_mismatch_before_install(tmp_path: Path) -> None:
    lock_path = tmp_path / "runtime-lock.txt"
    lock_path.write_text("example==1.0\n", encoding="utf-8")
    manager = RuntimeManager(tmp_path / "managed", _write_manifest(tmp_path, "0" * 64))

    with pytest.raises(RuntimeError, match="lock SHA-256 mismatch"):
        ManagedRuntimeInstaller(manager)._verified_runtime_lock()


def test_installed_commit_allows_only_the_managed_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RuntimeManager(tmp_path / "managed", _write_manifest(tmp_path, "0" * 64))
    manager.layout.sd_scripts.joinpath(".git").mkdir(parents=True)
    git_executable = str(tmp_path / "git.exe")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("lora_factory.runtime.manager.shutil.which", lambda _name: git_executable)

    def fake_run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["arguments"] = arguments
        captured.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0, stdout="a" * 40 + "\n", stderr="")

    monkeypatch.setattr("lora_factory.runtime.manager.subprocess.run", fake_run)

    assert manager.installed_commit() == "a" * 40
    assert captured["arguments"] == [
        git_executable,
        "-c",
        f"safe.directory={manager.layout.sd_scripts.resolve(strict=False)}",
        "rev-parse",
        "HEAD",
    ]
    assert captured["cwd"] == manager.layout.sd_scripts


def test_installer_uses_lock_and_no_deps_editable_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "runtime-lock.txt"
    lock_path.write_text("onnxruntime-gpu==1.28.0\n", encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    manager = RuntimeManager(tmp_path / "managed", _write_manifest(tmp_path, lock_sha256))
    layout = manager.layout
    layout.python.parent.mkdir(parents=True)
    layout.python.write_bytes(b"test python")
    layout.sd_scripts.mkdir(parents=True)
    layout.wd14.mkdir(parents=True)
    (layout.wd14 / "model.onnx").write_bytes(b"model")
    layout.image_embedding.mkdir(parents=True)
    (layout.image_embedding / "config.json").write_bytes(b"config")
    (layout.image_embedding / "model.safetensors").write_bytes(b"embedding")

    installer = ManagedRuntimeInstaller(manager)
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(manager, "installed_commit", lambda: "a" * 40)
    monkeypatch.setattr(ManagedRuntimeInstaller, "_check_disk", staticmethod(lambda _root: None))
    monkeypatch.setattr(
        "lora_factory.runtime.installer.shutil.which",
        lambda name: str(tmp_path / f"{name}.exe"),
    )
    monkeypatch.setattr(installer, "_install_wd14", lambda _cancellation, _progress: None)
    monkeypatch.setattr(
        installer,
        "_install_image_embedding",
        lambda _cancellation, _progress: None,
    )

    def capture(arguments: list[str], *, cwd: Path) -> None:
        calls.append((arguments, cwd))

    monkeypatch.setattr(installer, "_run", capture)
    installer.install(CancellationToken(), lambda _stage, _progress, _message: None)

    commands = [arguments for arguments, _cwd in calls]
    lock_command = next(command for command in commands if "--requirement" in command)
    editable_command = next(command for command in commands if "--editable" in command)
    assert "--no-deps" in lock_command
    assert lock_command[lock_command.index("--requirement") + 1] == str(lock_path)
    assert "--no-deps" in editable_command
    assert editable_command[editable_command.index("--editable") + 1] == str(layout.sd_scripts)
    assert not any("requirements.txt" in argument for command in commands for argument in command)
    assert not any(
        argument.startswith("onnxruntime-gpu==") for command in commands for argument in command
    )
    assert any(command[1:3] == ["pip", "check"] for command in commands)

    record: dict[str, Any] = json.loads(layout.installation_record.read_text(encoding="utf-8"))
    assert record["runtime_lock_sha256"] == lock_sha256
    assert record["image_embedding_model_sha256"] == sha256_file(
        layout.image_embedding / "model.safetensors"
    )
    validation = manager.read_validation_record()
    assert validation is not None
    assert validation.profile_id == "test-runtime"
    assert validation.ready is False
    assert validation.checks["managed_python"].status is ValidationStatus.OK
    assert validation.checks["sd_scripts"].status is ValidationStatus.OK
    assert validation.checks["pytorch_gpu"].status is ValidationStatus.NOT_RUN
    assert validation.checks["onnx_cuda"].status is ValidationStatus.NOT_RUN


def test_embedding_installer_prefers_verified_local_hugging_face_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "runtime-lock.txt"
    lock_path.write_text("example==1.0\n", encoding="utf-8")
    manager = RuntimeManager(
        tmp_path / "managed",
        _write_manifest(tmp_path, hashlib.sha256(lock_path.read_bytes()).hexdigest()),
    )
    cached_root = tmp_path / "hub-cache"
    revision = "b" * 40
    snapshot = cached_root / "models--openai--clip-vit-large-patch14" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    cached_files = {
        "config.json": b"pinned config",
        "model.safetensors": b"pinned weights",
    }
    artifacts = {}
    for filename, content in cached_files.items():
        path = snapshot / filename
        path.write_bytes(content)
        artifacts[filename] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manager.manifest["image_embedding"] = {
        "model_id": "openai/clip-vit-large-patch14",
        "revision": revision,
        "artifacts": artifacts,
    }
    monkeypatch.setenv("HF_HUB_CACHE", str(cached_root))
    installer = ManagedRuntimeInstaller(manager)
    monkeypatch.setattr(
        installer,
        "_download",
        lambda *_args: (_ for _ in ()).throw(AssertionError("network download attempted")),
    )

    installer._install_image_embedding(
        CancellationToken(), lambda _stage, _fraction, _message: None
    )

    assert (manager.layout.image_embedding / "config.json").read_bytes() == b"pinned config"
    assert (manager.layout.image_embedding / "model.safetensors").read_bytes() == (
        b"pinned weights"
    )
