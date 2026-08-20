"""Resumable managed sd-scripts/PyTorch/WD14 installer for Windows."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict, Field

from lora_factory.core.cancellation import CancellationToken
from lora_factory.runtime.manager import RuntimeManager
from lora_factory.runtime.validation_record import RuntimeValidationRecord
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import write_json_atomic

InstallProgress = Callable[[str, float, str], None]
_EXACT_REQUIREMENT = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*==[^\s;]+$")


class RuntimeLockDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9A-Fa-f]{64}$")
    platform: str = Field(pattern=r"^win32$")
    python: str = Field(pattern=r"^3\.12\.\d+$")
    package_count: int = Field(gt=0)
    excluded: tuple[str, ...]


class ManagedRuntimeInstaller:
    def __init__(self, manager: RuntimeManager) -> None:
        self.manager = manager

    def install(
        self,
        cancellation: CancellationToken,
        progress: InstallProgress,
    ) -> None:
        manifest = self.manager.manifest
        runtime_lock = self._verified_runtime_lock()
        layout = self.manager.layout
        self._check_disk(layout.root)
        layout.root.mkdir(parents=True, exist_ok=True)
        layout.models.mkdir(exist_ok=True)
        uv_executable = shutil.which("uv")
        git_executable = shutil.which("git")
        if uv_executable is None or git_executable is None:
            raise RuntimeError("Both uv and Git are required to install the managed runtime")

        cancellation.raise_if_cancelled()
        progress("python", 0.05, "Creating the isolated managed Python environment")
        if not layout.python.is_file():
            self._run(
                [
                    uv_executable,
                    "venv",
                    "--python",
                    str(manifest["managed_python"]),
                    str(layout.environment),
                ],
                cwd=layout.root,
            )

        cancellation.raise_if_cancelled()
        progress("sd_scripts", 0.15, "Obtaining the pinned unmodified sd-scripts source")
        expected_commit = str(manifest["sd_scripts"]["commit"])
        if not layout.sd_scripts.exists():
            self._run(
                [
                    git_executable,
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--branch",
                    str(manifest["sd_scripts"]["release"]),
                    str(manifest["sd_scripts"]["repository"]),
                    str(layout.sd_scripts),
                ],
                cwd=layout.root,
            )
        installed_commit = self.manager.installed_commit()
        if installed_commit != expected_commit:
            raise RuntimeError(
                "Managed sd-scripts source does not match the pinned commit; "
                "choose a fresh versioned runtime instead of modifying it"
            )

        cancellation.raise_if_cancelled()
        progress("pytorch", 0.30, "Installing the official CUDA 13 PyTorch wheels")
        pytorch = manifest["pytorch"]
        self._run(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                str(layout.python),
                f"torch=={pytorch['torch']}",
                f"torchvision=={pytorch['torchvision']}",
                "--index-url",
                str(pytorch["index_url"]),
            ],
            cwd=layout.root,
        )

        cancellation.raise_if_cancelled()
        progress("dependencies", 0.50, "Installing the hash-verified exact dependency lock")
        self._run(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                str(layout.python),
                "--no-deps",
                "--requirement",
                str(runtime_lock),
            ],
            cwd=layout.root,
        )
        self._run(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                str(layout.python),
                "--no-deps",
                "--editable",
                str(layout.sd_scripts),
            ],
            cwd=layout.sd_scripts,
        )

        cancellation.raise_if_cancelled()
        progress("wd14", 0.70, "Downloading and verifying the pinned WD14 model")
        self._install_wd14(cancellation, progress)

        cancellation.raise_if_cancelled()
        progress("image_embedding", 0.91, "Preparing and verifying the pinned CLIP model")
        self._install_image_embedding(cancellation, progress)

        cancellation.raise_if_cancelled()
        progress("verify", 0.95, "Verifying managed imports and the exact sd-scripts commit")
        self._run(
            [
                uv_executable,
                "pip",
                "check",
                "--python",
                str(layout.python),
            ],
            cwd=layout.root,
        )
        self._run(
            [
                str(layout.python),
                "-c",
                (
                    "import torch, onnxruntime; "
                    "import library.train_util; "
                    "print(torch.__version__); "
                    "print(onnxruntime.__version__)"
                ),
            ],
            cwd=layout.sd_scripts,
        )
        write_json_atomic(
            layout.installation_record,
            {
                "profile_id": manifest["profile_id"],
                "installed_at": datetime.now(UTC).isoformat(),
                "manifest_path": str(self.manager.manifest_path),
                "runtime_lock_sha256": manifest["runtime_lock"]["sha256"],
                "sd_scripts_commit": expected_commit,
                "python": str(layout.python),
                "wd14_model_sha256": sha256_file(layout.wd14 / "model.onnx"),
                "image_embedding_model_sha256": sha256_file(
                    layout.image_embedding / "model.safetensors"
                ),
                "live_gpu_validation": False,
            },
        )
        self.manager.write_validation_record(
            RuntimeValidationRecord.after_install(str(manifest["profile_id"]), str(layout.python))
        )
        progress("complete", 1.0, "Managed runtime installation completed")

    def _verified_runtime_lock(self) -> Path:
        try:
            descriptor = RuntimeLockDescriptor.model_validate(
                self.manager.manifest.get("runtime_lock")
            )
        except ValueError as exc:
            raise ValueError("Backend manifest contains an invalid runtime_lock") from exc
        if descriptor.python != str(self.manager.manifest.get("managed_python")):
            raise ValueError("Backend manifest runtime lock Python version does not match")
        required_exclusions = {"torch", "torchvision", "sd-scripts"}
        if set(descriptor.excluded) != required_exclusions:
            raise ValueError(
                "Backend manifest runtime lock exclusions do not match installer policy"
            )

        manifest_root = self.manager.manifest_path.parent.resolve(strict=True)
        candidate = Path(descriptor.path)
        if candidate.is_absolute():
            raise ValueError("Backend manifest runtime_lock.path must be relative")
        lock_path = (manifest_root / candidate).resolve(strict=True)
        try:
            lock_path.relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError(
                "Backend manifest runtime_lock.path escapes the manifest root"
            ) from exc
        actual_sha256 = sha256_file(lock_path)
        if actual_sha256.casefold() != descriptor.sha256.casefold():
            raise RuntimeError(
                "Managed runtime lock SHA-256 mismatch: "
                f"expected {descriptor.sha256}, got {actual_sha256}"
            )
        requirements = [
            line.strip()
            for line in lock_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(requirements) != descriptor.package_count:
            raise ValueError("Managed runtime lock package count does not match the manifest")
        if any(_EXACT_REQUIREMENT.fullmatch(requirement) is None for requirement in requirements):
            raise ValueError(
                "Managed runtime lock must contain exact package==version entries only"
            )
        normalized_names = {
            requirement.split("==", maxsplit=1)[0].casefold().replace("_", "-")
            for requirement in requirements
        }
        if normalized_names & {"torch", "torchvision"}:
            raise ValueError("Managed runtime lock must not contain Torch packages")
        return lock_path

    @staticmethod
    def _check_disk(root: Path) -> None:
        root.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = psutil.disk_usage(str(root.parent)).free
        if free_bytes < 15 * 1024**3:
            raise OSError("At least 15 GiB free space is required for the managed runtime")

    @staticmethod
    def _run(arguments: list[str], *, cwd: Path) -> None:
        completed = subprocess.run(  # noqa: S603 - arguments are installer-owned and shell-free.
            arguments,
            cwd=cwd,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-30:])
            raise RuntimeError(f"Managed runtime command failed ({completed.returncode}):\n{tail}")

    def _install_wd14(
        self,
        cancellation: CancellationToken,
        progress: InstallProgress,
    ) -> None:
        wd14 = self.manager.manifest["wd14"]
        revision = str(wd14["revision"])
        model_id = str(wd14["model_id"])
        root = self.manager.layout.wd14
        root.mkdir(parents=True, exist_ok=True)
        filenames = ("model.onnx", "selected_tags.csv", "config.json")
        for index, filename in enumerate(filenames):
            cancellation.raise_if_cancelled()
            url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}?download=true"
            destination = root / filename
            self._download(url, destination)
            artifact = wd14["artifacts"][filename]
            expected_size = int(artifact["size_bytes"])
            if destination.stat().st_size != expected_size:
                raise OSError(f"WD14 {filename} has an unexpected size")
            expected_sha = artifact.get("sha256")
            if expected_sha is not None and sha256_file(destination) != expected_sha:
                raise OSError(f"WD14 {filename} SHA-256 verification failed")
            expected_blob = artifact.get("git_blob")
            if expected_blob is not None and self._git_blob_hash(destination) != expected_blob:
                raise OSError(f"WD14 {filename} Git blob verification failed")
            progress(
                "wd14",
                0.70 + 0.20 * (index + 1) / len(filenames),
                f"Verified {filename}",
            )

    def _install_image_embedding(
        self,
        cancellation: CancellationToken,
        progress: InstallProgress,
    ) -> None:
        embedding = self.manager.manifest["image_embedding"]
        revision = str(embedding["revision"])
        model_id = str(embedding["model_id"])
        root = self.manager.layout.image_embedding
        root.mkdir(parents=True, exist_ok=True)
        artifacts = embedding["artifacts"]
        filenames = ("config.json", "model.safetensors")
        for index, filename in enumerate(filenames):
            cancellation.raise_if_cancelled()
            destination = root / filename
            artifact = artifacts[filename]
            expected_size = int(artifact["size_bytes"])
            expected_sha256 = str(artifact["sha256"])
            if not destination.is_file():
                cached = self._cached_hugging_face_artifact(model_id, revision, filename)
                if (
                    cached is not None
                    and cached.stat().st_size == expected_size
                    and sha256_file(cached) == expected_sha256
                ):
                    temporary = destination.with_name(f".{destination.name}.partial")
                    shutil.copy2(cached, temporary)
                    temporary.replace(destination)
                else:
                    url = (
                        f"https://huggingface.co/{model_id}/resolve/{revision}/"
                        f"{filename}?download=true"
                    )
                    self._download(url, destination)
            if destination.stat().st_size != expected_size:
                raise OSError(f"CLIP embedding {filename} has an unexpected size")
            if sha256_file(destination) != expected_sha256:
                raise OSError(f"CLIP embedding {filename} SHA-256 verification failed")
            progress(
                "image_embedding",
                0.91 + 0.03 * (index + 1) / len(filenames),
                f"Verified CLIP {filename}",
            )

    @staticmethod
    def _cached_hugging_face_artifact(
        model_id: str,
        revision: str,
        filename: str,
    ) -> Path | None:
        configured_cache = os.environ.get("HF_HUB_CACHE")
        if configured_cache:
            hub = Path(configured_cache)
        else:
            hugging_face_home = Path(
                os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
            )
            hub = hugging_face_home / "hub"
        repository = "models--" + model_id.replace("/", "--")
        candidate = hub / repository / "snapshots" / revision / filename
        return candidate.resolve(strict=True) if candidate.is_file() else None

    @staticmethod
    def _download(url: str, destination: Path) -> None:
        if destination.is_file():
            return
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
            raise ValueError("Managed model downloads require an official Hugging Face HTTPS URL")
        temporary = destination.with_name(f".{destination.name}.partial")
        existing = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(  # noqa: S310 - scheme and host validated above.
            url,
            headers={"User-Agent": "LoRAFactory/v0.1"},
        )
        if existing:
            request.add_header("Range", f"bytes={existing}-")
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - pinned HTTPS URL.
            append = existing > 0 and response.status == 206
            mode = "ab" if append else "wb"
            with temporary.open(mode) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        temporary.replace(destination)

    @staticmethod
    def _git_blob_hash(path: Path) -> str:
        payload = path.read_bytes()
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {len(payload)}\0".encode())
        digest.update(payload)
        return digest.hexdigest()
