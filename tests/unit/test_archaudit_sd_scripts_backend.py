from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from lora_factory.config.models import TrainingPlan
from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.core.exceptions import ErrorClassification, PipelineError
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.backend import TrainingProgress, TrainingRequest
from lora_factory.training.process_manager import (
    LineCallback,
    ManagedProcessResult,
    ProcessManager,
)
from lora_factory.training.sd_scripts_backend import SdScriptsTrainingBackend

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
PINNED_COMMIT = "6721028c79ee85a78b3a06dfd8954dae310a1cce"


class StubProcessManager(ProcessManager):
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.calls: list[tuple[list[str], Mapping[str, str], int | None]] = []

    def run(
        self,
        arguments: list[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
        cancellation: CancellationToken,
        timeout_seconds: int | None,
        on_line: LineCallback,
    ) -> ManagedProcessResult:
        assert cwd.is_dir()
        assert not cancellation.cancelled
        self.calls.append((arguments, dict(environment), timeout_seconds))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        training_step = 6 if self.mode == "validation" else 10
        training_line = f"epoch=1 step={training_step}/{training_step} loss=0.125"
        stdout_path.write_text(training_line + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        on_line("stdout", training_line)
        if self.mode == "validation":
            validation_line = (
                "epoch validation steps: 100%|##########| 8/8 val_epoch_avg_loss=0.275 timestep=800"
            )
            stderr_path.write_text(validation_line + "\n", encoding="utf-8")
            on_line("stderr", validation_line)

        if self.mode == "cancelled":
            return self._result(stdout_path, stderr_path, cancelled=True)
        if self.mode == "timeout":
            return self._result(stdout_path, stderr_path, timed_out=True)
        if self.mode == "oom":
            stderr_path.write_text("CUDA out of memory", encoding="utf-8")
            return self._result(stdout_path, stderr_path, return_code=1)

        output_dir = Path(
            next(
                item.split("=", maxsplit=1)[1]
                for item in arguments
                if item.startswith("--output_dir=")
            )
        )
        output_name = next(
            item.split("=", maxsplit=1)[1]
            for item in arguments
            if item.startswith("--output_name=")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / f"{output_name}-000001.safetensors"
        if self.mode == "corrupt":
            checkpoint.write_bytes(b"not safetensors")
        else:
            save_file(
                {"lora_unet.test": np.ones((2, 2), dtype=np.float32)},
                checkpoint,
                metadata={"ss_epoch": "1", "ss_steps": str(training_step)},
            )
        if self.mode != "missing_state":
            (output_dir / f"{output_name}-000001-state").mkdir()
        return self._result(stdout_path, stderr_path)

    @staticmethod
    def _result(
        stdout_path: Path,
        stderr_path: Path,
        *,
        return_code: int = 0,
        cancelled: bool = False,
        timed_out: bool = False,
    ) -> ManagedProcessResult:
        return ManagedProcessResult(
            return_code=return_code,
            cancelled=cancelled,
            timed_out=timed_out,
            duration_seconds=0.1,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


def _plan() -> TrainingPlan:
    return TrainingPlan(
        resolution=768,
        batch_size=1,
        gradient_accumulation=4,
        repeats=2,
        epochs=1,
        estimated_steps=10,
        network_dim=16,
        network_alpha=8,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=False,
        training_gpu_uuid=GPU_UUID,
        estimated_disk_mb=256,
    )


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    runtime_root = tmp_path / "runtime"
    python = runtime_root / "python.exe"
    sd_scripts = runtime_root / "sd-scripts"
    model = tmp_path / "base.safetensors"
    dataset = tmp_path / "dataset.toml"
    sd_scripts.mkdir(parents=True)
    for path in (python, sd_scripts / "sdxl_train_network.py", model, dataset):
        path.write_text("fixture", encoding="utf-8")
    (runtime_root / "installation.json").write_text(
        json.dumps({"sd_scripts_commit": PINNED_COMMIT}),
        encoding="utf-8",
    )
    return python, sd_scripts, model, dataset


def _backend(
    tmp_path: Path, manager: StubProcessManager
) -> tuple[SdScriptsTrainingBackend, Path, Path]:
    python, sd_scripts, model, dataset = _fixture_paths(tmp_path)
    binding = GpuBinding(
        uuid=GPU_UUID,
        physical_index=2,
        environment={
            "CUDA_VISIBLE_DEVICES": "2",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
    )
    backend = SdScriptsTrainingBackend(
        python,
        sd_scripts,
        binding,
        (GPU_UUID,),
        process_manager=manager,
        timeout_seconds=300,
    )
    return backend, model, dataset


def _request(tmp_path: Path, model: Path, dataset: Path, **updates: object) -> TrainingRequest:
    request = TrainingRequest(
        run_id="run",
        attempt=1,
        output_name="日本語LoRA",
        base_model=model,
        dataset_config=dataset,
        run_directory=tmp_path / "run",
        plan=_plan(),
        seed=123,
    )
    return request.model_copy(update=updates)


def test_archaudit_real_backend_runs_pinned_single_gpu_and_validates_outputs(
    tmp_path: Path,
) -> None:
    manager = StubProcessManager()
    backend, model, dataset = _backend(tmp_path, manager)
    observed: list[TrainingProgress] = []
    result = backend.train(
        _request(tmp_path, model, dataset),
        CancellationToken(),
        observed.append,
    )
    assert backend.version == f"sd-scripts/{PINNED_COMMIT}"
    assert len(result.checkpoints) == 1
    assert result.checkpoints[0].epoch == 1
    assert result.completed_steps == 10
    assert result.state_path.is_dir()
    assert result.log_path.name == "training.stdout.log"
    assert not result.resumed
    assert result.command_argv == tuple(manager.calls[0][0])
    assert observed[-1].message.startswith("stdout:")
    arguments, environment, timeout = manager.calls[0]
    assert "--num_processes=1" in arguments
    assert "--save_state" in arguments
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert timeout == 300


def test_archaudit_real_backend_passes_resume_state_without_overwriting_attempt(
    tmp_path: Path,
) -> None:
    manager = StubProcessManager()
    backend, model, dataset = _backend(tmp_path, manager)
    resume_state = tmp_path / "previous-state"
    resume_state.mkdir()
    result = backend.train(
        _request(
            tmp_path,
            model,
            dataset,
            attempt=2,
            resume_state=resume_state,
        ),
        CancellationToken(),
        lambda _value: None,
    )
    assert result.resumed
    assert any(item == f"--resume={resume_state}" for item in manager.calls[0][0])


def test_archaudit_real_backend_attaches_logged_validation_loss_to_checkpoint(
    tmp_path: Path,
) -> None:
    backend, model, dataset = _backend(tmp_path, StubProcessManager("validation"))

    result = backend.train(
        _request(tmp_path, model, dataset),
        CancellationToken(),
        lambda _value: None,
    )

    checkpoint = result.checkpoints[0]
    assert result.completed_steps == 6
    assert checkpoint.train_loss == pytest.approx(0.125)
    assert checkpoint.validation_loss == pytest.approx(0.275)


def test_archaudit_real_backend_classifies_nonzero_oom(tmp_path: Path) -> None:
    backend, model, dataset = _backend(tmp_path, StubProcessManager("oom"))
    with pytest.raises(PipelineError) as caught:
        backend.train(
            _request(tmp_path, model, dataset),
            CancellationToken(),
            lambda _value: None,
        )
    assert caught.value.classification is ErrorClassification.CUDA_OOM
    assert caught.value.recoverable


@pytest.mark.parametrize(
    ("mode", "error_type", "message"),
    [
        ("cancelled", CancelledError, "cancelled"),
        ("timeout", TimeoutError, "timeout"),
        ("corrupt", ValueError, "no valid safetensors"),
        ("missing_state", ValueError, "save_state"),
    ],
)
def test_archaudit_real_backend_handles_cancel_timeout_and_invalid_artifacts(
    tmp_path: Path,
    mode: str,
    error_type: type[Exception],
    message: str,
) -> None:
    backend, model, dataset = _backend(tmp_path, StubProcessManager(mode))
    with pytest.raises(error_type, match=message):
        backend.train(
            _request(tmp_path, model, dataset),
            CancellationToken(),
            lambda _value: None,
        )


def test_archaudit_real_backend_refuses_unverified_runtime(tmp_path: Path) -> None:
    manager = StubProcessManager()
    backend, model, dataset = _backend(tmp_path, manager)
    (backend.sd_scripts_root.parent / "installation.json").unlink()
    unverified = SdScriptsTrainingBackend(
        backend.python_executable,
        backend.sd_scripts_root,
        backend.binding,
        backend.selected_gpu_uuids,
        process_manager=manager,
    )
    with pytest.raises(RuntimeError, match="no pinned"):
        unverified.train(
            _request(tmp_path, model, dataset),
            CancellationToken(),
            lambda _value: None,
        )


def test_archaudit_checkpoint_name_containing_state_is_not_a_resume_artifact(
    tmp_path: Path,
) -> None:
    backend, model, dataset = _backend(tmp_path, StubProcessManager("missing_state"))
    with pytest.raises(ValueError, match="save_state"):
        backend.train(
            _request(tmp_path, model, dataset, output_name="statefulLoRA"),
            CancellationToken(),
            lambda _value: None,
        )
