from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lora_factory.caption.managed_wd14 import ManagedWD14Tagger
from lora_factory.core.cancellation import CancellationToken
from lora_factory.gpu.models import GpuBinding
from lora_factory.runtime_scripts.wd14_infer import _run_batched
from lora_factory.training.process_manager import ManagedProcessResult


def test_managed_helper_halves_oom_batch_and_preserves_all_outputs() -> None:
    observed_sizes: list[int] = []

    def infer(batch: np.ndarray) -> np.ndarray:
        observed_sizes.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
        return np.ones((len(batch), 3), dtype=np.float32)

    probabilities, successful = _run_batched(
        tuple(Path(f"image-{index}.png") for index in range(5)),
        initial_batch_size=8,
        prepare=lambda _path: np.zeros((4, 4, 3), dtype=np.float32),
        infer=infer,
    )

    assert observed_sizes == [5, 2, 2, 1]
    assert successful == 2
    assert probabilities.shape == (5, 3)


def test_managed_tagger_persists_successful_batch_size_per_gpu(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    python = tmp_path / "runtime" / "Scripts" / "python.exe"
    helper = tmp_path / "helper" / "wd14_infer.py"
    model = tmp_path / "models" / "wd14" / "model.onnx"
    tags = model.parent / "selected_tags.csv"
    for path in (python, helper, model, tags):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    images: list[Path] = []
    for index in range(3):
        path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (16, 16), (index * 20, 40, 60)).save(path)
        images.append(path)

    calls: list[list[str]] = []

    class StubManager:
        def run(
            self,
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
            del cwd, environment, cancellation, timeout_seconds, on_line
            calls.append(arguments)
            input_path = Path(arguments[arguments.index("--input") + 1])
            output_path = Path(arguments[arguments.index("--output") + 1])
            inputs = json.loads(input_path.read_text(encoding="utf-8"))["paths"]
            output_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "tags": [
                                    {
                                        "name": "1girl",
                                        "score": 0.95,
                                        "model_category": "general",
                                        "selected": True,
                                    }
                                ]
                            }
                            for _path in inputs
                        ],
                        "successful_batch_size": 2,
                    }
                ),
                encoding="utf-8",
            )
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text("ok", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            return ManagedProcessResult(0, False, False, 0.01, stdout_path, stderr_path)

    monkeypatch.setattr("lora_factory.caption.managed_wd14.ProcessManager", StubManager)
    binding = GpuBinding(
        uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        physical_index=3,
        environment={"CUDA_VISIBLE_DEVICES": "3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )

    def tagger(run: str) -> ManagedWD14Tagger:
        return ManagedWD14Tagger(
            python_executable=python,
            helper_script=helper,
            model_path=model,
            tags_path=tags,
            revision="a" * 40,
            binding=binding,
            work_directory=tmp_path / run,
            cancellation=CancellationToken(),
            initial_batch_size=8,
        )

    first = tagger("first")
    assert len(first.tag_many(images)) == 3
    assert first.successful_batch_size == 2
    second = tagger("second")
    assert len(second.tag_many(images)) == 3

    assert calls[0][calls[0].index("--batch-size") + 1] == "8"
    assert calls[1][calls[1].index("--batch-size") + 1] == "2"
    cache = json.loads(first._batch_cache_path.read_text(encoding="utf-8"))
    assert cache == {
        "gpu_uuid": binding.uuid,
        "revision": "a" * 40,
        "successful_batch_size": 2,
    }
