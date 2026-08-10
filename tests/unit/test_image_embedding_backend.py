from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from lora_factory.core.cancellation import CancellationToken
from lora_factory.evaluation.embedding import (
    FakeImageEmbeddingBackend,
    ManagedClipImageEmbeddingBackend,
)
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.process_manager import ManagedProcessResult, ProcessManager
from lora_factory.util.hashing import sha256_file


def _image(path: Path, seed: int) -> Path:
    pixels = np.random.default_rng(seed).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)
    return path


def test_fake_embedding_is_deterministic_normalized_and_explicitly_fake(tmp_path: Path) -> None:
    image = _image(tmp_path / "image.png", 10)
    backend = FakeImageEmbeddingBackend()

    first = backend.embed_many((image,))
    second = backend.embed_many((image,))

    assert first == second
    assert first.device == "fake"
    assert first.model_id == "lora-factory/fake-image-embedding"
    assert first.dimension == 192
    assert np.linalg.norm(first.items[0].values) == pytest.approx(1.0)
    assert first.items[0].source_sha256 == sha256_file(image)


def test_managed_embedding_is_offline_gpu_scoped_and_validates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "runtime" / "python.exe"
    helper = tmp_path / "clip_embed.py"
    model = tmp_path / "model"
    python.parent.mkdir()
    python.write_bytes(b"python")
    helper.write_text("# helper", encoding="utf-8")
    model.mkdir()
    (model / "config.json").write_bytes(b"config")
    (model / "model.safetensors").write_bytes(b"weights")
    image = _image(tmp_path / "image.png", 20)
    observed: dict[str, Any] = {}

    def run(
        _self: ProcessManager,
        arguments: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
        cancellation: CancellationToken,
        timeout_seconds: int | None,
        on_line: Any,
    ) -> ManagedProcessResult:
        del cancellation, on_line
        observed.update(
            arguments=arguments,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        output = Path(arguments[arguments.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "model_id": "openai/clip-vit-large-patch14",
                    "revision": "a" * 40,
                    "dimension": 3,
                    "device": "cuda:0",
                    "items": [
                        {
                            "source_path": str(image.resolve()),
                            "source_sha256": sha256_file(image),
                            "values": [1.0, 0.0, 0.0],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return ManagedProcessResult(0, False, False, 0.1, stdout_path, stderr_path)

    monkeypatch.setattr(ProcessManager, "run", run)
    backend = ManagedClipImageEmbeddingBackend(
        python_executable=python,
        helper_script=helper,
        model_directory=model,
        model_id="openai/clip-vit-large-patch14",
        revision="a" * 40,
        config_sha256=sha256_file(model / "config.json"),
        model_sha256=sha256_file(model / "model.safetensors"),
        binding=GpuBinding(
            uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            physical_index=2,
            environment={
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "2",
            },
        ),
        work_directory=tmp_path / "work",
        cancellation=CancellationToken(),
    )

    batch = backend.embed_many((image,))

    assert batch.items[0].values == (1.0, 0.0, 0.0)
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert observed["environment"]["HF_HUB_OFFLINE"] == "1"
    assert observed["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert observed["arguments"][0] == str(python.resolve())
    assert observed["arguments"][1] == str(helper.resolve())


def test_managed_embedding_rejects_unverified_model_before_child_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python.exe"
    helper = tmp_path / "clip_embed.py"
    model = tmp_path / "model"
    python.write_bytes(b"python")
    helper.write_text("# helper", encoding="utf-8")
    model.mkdir()
    (model / "config.json").write_bytes(b"config")
    (model / "model.safetensors").write_bytes(b"corrupt")
    image = _image(tmp_path / "image.png", 30)
    monkeypatch.setattr(
        ProcessManager,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("child launched")),
    )
    backend = ManagedClipImageEmbeddingBackend(
        python_executable=python,
        helper_script=helper,
        model_directory=model,
        model_id="openai/clip-vit-large-patch14",
        revision="b" * 40,
        config_sha256=sha256_file(model / "config.json"),
        model_sha256="0" * 64,
        binding=GpuBinding(
            uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            physical_index=0,
            environment={
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        ),
        work_directory=tmp_path / "work",
        cancellation=CancellationToken(),
    )

    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        backend.embed_many((image,))
