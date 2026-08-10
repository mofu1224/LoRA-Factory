from __future__ import annotations

import pytest

from lora_factory.core.exceptions import ErrorClassification, classify_process_output


@pytest.mark.parametrize(
    ("return_code", "output", "classification", "recoverable"),
    [
        (1, "CUDA out of memory", ErrorClassification.CUDA_OOM, True),
        (1, "loss became NaN", ErrorClassification.NAN_LOSS, True),
        (1, "No space left on device", ErrorClassification.DISK_FULL, True),
        (
            1,
            "onnxruntime could not enable CUDAExecutionProvider",
            ErrorClassification.ONNX_PROVIDER,
            True,
        ),
        (1, "unexpected worker exit", ErrorClassification.PROCESS_CRASH, True),
    ],
)
def test_failure_injection_process_output_classification(
    return_code: int,
    output: str,
    classification: ErrorClassification,
    recoverable: bool,
) -> None:
    assert classify_process_output(return_code, output) == (classification, recoverable)
