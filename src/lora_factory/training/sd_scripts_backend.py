"""Production adapter for the pinned, unmodified sd-scripts trainer."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lora_factory.core.cancellation import CancellationToken, CancelledError
from lora_factory.core.exceptions import ErrorClassification, PipelineError, classify_process_output
from lora_factory.gpu.models import GpuBinding
from lora_factory.training.backend import (
    CheckpointArtifact,
    ProgressCallback,
    TrainingProgress,
    TrainingRequest,
    TrainingResult,
)
from lora_factory.training.checkpoints import inspect_checkpoint
from lora_factory.training.command_builder import build_sd_scripts_command
from lora_factory.training.log_parser import parse_progress_line
from lora_factory.training.process_manager import ProcessManager

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_TAIL_BYTES = 64 * 1024


class SdScriptsProcessError(PipelineError):
    """A classified trainer failure retaining its shell-free argument array."""

    def __init__(
        self,
        message: str,
        *,
        classification: ErrorClassification,
        recoverable: bool,
        command_argv: tuple[str, ...],
    ) -> None:
        super().__init__(message, classification=classification, recoverable=recoverable)
        self.command_argv = command_argv


class SdScriptsTrainingBackend:
    """Run one sd-scripts training process on exactly one selected GPU UUID."""

    def __init__(
        self,
        python_executable: Path,
        sd_scripts_root: Path,
        binding: GpuBinding,
        selected_gpu_uuids: tuple[str, ...],
        *,
        process_manager: ProcessManager | None = None,
        timeout_seconds: int | None = 24 * 60 * 60,
    ) -> None:
        if not selected_gpu_uuids:
            raise ValueError("At least one GPU UUID must be selected")
        if len(set(selected_gpu_uuids)) != len(selected_gpu_uuids):
            raise ValueError("Selected GPU UUIDs must be unique")
        if binding.uuid not in selected_gpu_uuids:
            raise ValueError("Training GPU binding is outside the selected pool")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        self.python_executable = python_executable.resolve(strict=False)
        self.sd_scripts_root = sd_scripts_root.resolve(strict=False)
        self.binding = binding
        self.selected_gpu_uuids = selected_gpu_uuids
        self.process_manager = process_manager or ProcessManager()
        self.timeout_seconds = timeout_seconds
        self._version = self._installed_version()

    @property
    def version(self) -> str:
        return self._version

    def train(
        self,
        request: TrainingRequest,
        cancellation: CancellationToken,
        progress: ProgressCallback,
    ) -> TrainingResult:
        """Execute training, stream progress, and accept only valid safetensors outputs."""

        cancellation.raise_if_cancelled()
        if self._version == "sd-scripts/unverified":
            raise RuntimeError(
                "Managed runtime installation record has no pinned sd-scripts commit"
            )
        if request.plan.training_gpu_uuid != self.binding.uuid:
            raise ValueError("Training request plan and selected GPU binding differ")

        attempt_root = request.run_directory / f"attempt-{request.attempt:03d}"
        checkpoints_root = attempt_root / "checkpoints"
        states_root = attempt_root / "states"
        logs_root = attempt_root / "logs"
        existing_checkpoints = tuple(checkpoints_root.glob("*.safetensors"))
        if existing_checkpoints and request.resume_state is None:
            raise FileExistsError(
                "Attempt already contains checkpoints; use a resume state or a new attempt"
            )
        for directory in (checkpoints_root, states_root, logs_root):
            directory.mkdir(parents=True, exist_ok=True)

        command = build_sd_scripts_command(
            python_executable=self.python_executable,
            sd_scripts_root=self.sd_scripts_root,
            pretrained_model=request.base_model,
            dataset_config=request.dataset_config,
            output_dir=checkpoints_root,
            output_name=request.output_name,
            logging_dir=logs_root,
            plan=request.plan,
            binding=self.binding,
            selected_gpu_uuids=self.selected_gpu_uuids,
            seed=request.seed,
            resume_state=request.resume_state,
        )
        stdout_path = logs_root / "training.stdout.log"
        stderr_path = logs_root / "training.stderr.log"
        latest_training_progress: TrainingProgress | None = None
        current_epoch = 0
        losses_by_epoch: dict[int, tuple[float | None, float | None]] = {}

        def on_line(channel: str, line: str) -> None:
            nonlocal current_epoch, latest_training_progress
            parsed = parse_progress_line(
                line,
                default_total_steps=request.plan.estimated_steps,
            )
            if parsed is None:
                return
            if parsed.epoch > 0:
                current_epoch = parsed.epoch
            if current_epoch > 0 and (
                parsed.train_loss is not None or parsed.validation_loss is not None
            ):
                previous_train, previous_validation = losses_by_epoch.get(
                    current_epoch, (None, None)
                )
                losses_by_epoch[current_epoch] = (
                    parsed.train_loss if parsed.train_loss is not None else previous_train,
                    parsed.validation_loss
                    if parsed.validation_loss is not None
                    else previous_validation,
                )
            observed = parsed.model_copy(update={"message": f"{channel}: {parsed.message}"})
            if observed.validation_loss is None:
                latest_training_progress = observed
            progress(observed)

        process_result = self.process_manager.run(
            list(command.argv),
            cwd=command.cwd,
            environment=command.environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cancellation=cancellation,
            timeout_seconds=self.timeout_seconds,
            on_line=on_line,
        )
        if process_result.cancelled:
            raise CancelledError("sd-scripts training was cancelled by the user")
        if process_result.timed_out:
            raise TimeoutError(
                f"sd-scripts training exceeded the {self.timeout_seconds}-second timeout"
            )
        if process_result.return_code != 0:
            diagnostics = "\n".join(
                part
                for part in (
                    _read_tail(process_result.stdout_path),
                    _read_tail(process_result.stderr_path),
                )
                if part
            )
            classification, recoverable = classify_process_output(
                process_result.return_code, diagnostics
            )
            raise SdScriptsProcessError(
                f"sd-scripts exited with code {process_result.return_code}: "
                f"{diagnostics[-2000:] or 'no diagnostics'}",
                classification=classification,
                recoverable=recoverable,
                command_argv=command.argv,
            )

        checkpoints, invalid = _inspect_outputs(checkpoints_root)
        if not checkpoints:
            suffix = f" Invalid files: {', '.join(invalid)}" if invalid else ""
            raise ValueError(f"sd-scripts produced no valid safetensors checkpoints.{suffix}")
        checkpoints = tuple(
            checkpoint.model_copy(
                update={
                    "train_loss": losses_by_epoch.get(checkpoint.epoch, (None, None))[0]
                    if checkpoint.train_loss is None
                    else checkpoint.train_loss,
                    "validation_loss": losses_by_epoch.get(checkpoint.epoch, (None, None))[1]
                    if checkpoint.validation_loss is None
                    else checkpoint.validation_loss,
                }
            )
            for checkpoint in checkpoints
        )
        state_path = _find_latest_state(checkpoints_root, states_root)
        if state_path is None:
            raise ValueError("sd-scripts completed without a resumable save_state artifact")
        completed_steps = max(
            [checkpoint.step for checkpoint in checkpoints]
            + ([latest_training_progress.step] if latest_training_progress is not None else [])
        )
        return TrainingResult(
            checkpoints=checkpoints,
            state_path=state_path,
            log_path=process_result.stdout_path,
            completed_steps=completed_steps,
            resumed=request.resume_state is not None,
            command_argv=command.argv,
        )

    def _installed_version(self) -> str:
        record_path = self.sd_scripts_root.parent / "installation.json"
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "sd-scripts/unverified"
        if not isinstance(payload, dict):
            return "sd-scripts/unverified"
        commit = payload.get("sd_scripts_commit")
        if not isinstance(commit, str) or not _COMMIT_SHA.fullmatch(commit):
            return "sd-scripts/unverified"
        return f"sd-scripts/{commit.lower()}"


def _inspect_outputs(directory: Path) -> tuple[tuple[CheckpointArtifact, ...], tuple[str, ...]]:
    valid: list[CheckpointArtifact] = []
    invalid: list[str] = []
    for path in sorted(directory.glob("*.safetensors")):
        try:
            valid.append(inspect_checkpoint(path))
        except ValueError:
            invalid.append(path.name)
    valid.sort(key=lambda item: (item.epoch, item.step, item.path.name))
    return tuple(valid), tuple(invalid)


def _find_latest_state(checkpoints_root: Path, states_root: Path) -> Path | None:
    candidates = [
        path
        for root in (checkpoints_root, states_root)
        for path in root.iterdir()
        if path.is_dir() and path.name.lower().endswith("-state")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _read_tail(path: Path, *, limit: int = _TAIL_BYTES) -> str:
    if limit <= 0:
        raise ValueError("Tail limit must be positive")
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", errors="replace").strip()
