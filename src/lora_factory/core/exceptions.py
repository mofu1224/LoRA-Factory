"""User-facing error classes and deterministic recovery classification."""

from __future__ import annotations

from enum import StrEnum


class ErrorClassification(StrEnum):
    CUDA_OOM = "CUDA_OOM"
    CUDA_INCOMPATIBLE_ARCH = "CUDA_INCOMPATIBLE_ARCH"
    CUDA_DRIVER = "CUDA_DRIVER"
    PYTORCH_UNSUPPORTED = "PYTORCH_UNSUPPORTED"
    ONNX_PROVIDER = "ONNX_PROVIDER"
    NAN_LOSS = "NAN_LOSS"
    INVALID_MODEL = "INVALID_MODEL"
    INVALID_DATASET = "INVALID_DATASET"
    MISSING_CAPTION = "MISSING_CAPTION"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    DISK_FULL = "DISK_FULL"
    PROCESS_CRASH = "PROCESS_CRASH"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class PipelineError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: ErrorClassification = ErrorClassification.UNKNOWN,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.recoverable = recoverable


def classify_process_output(return_code: int, text: str) -> tuple[ErrorClassification, bool]:
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return ErrorClassification.CUDA_OOM, True
    if "nan" in lowered and "loss" in lowered:
        return ErrorClassification.NAN_LOSS, True
    if "no space left" in lowered or "disk full" in lowered:
        return ErrorClassification.DISK_FULL, True
    if "no kernel image" in lowered or "not compatible with the current pytorch" in lowered:
        return ErrorClassification.CUDA_INCOMPATIBLE_ARCH, False
    if "onnxruntime" in lowered and "cudaexecutionprovider" in lowered:
        return ErrorClassification.ONNX_PROVIDER, True
    if "caption" in lowered and ("missing" in lowered or "not found" in lowered):
        return ErrorClassification.MISSING_CAPTION, True
    if return_code != 0:
        return ErrorClassification.PROCESS_CRASH, True
    return ErrorClassification.UNKNOWN, False
