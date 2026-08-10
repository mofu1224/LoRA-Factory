"""Durable results from an explicitly requested managed-runtime deep probe."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.runtime.doctor import RuntimeDoctorReport


class ValidationStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    NOT_RUN = "not_run"


class RuntimeValidationCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ValidationStatus
    detail: str


class RuntimeValidationRecord(BaseModel):
    """Cached deep-probe evidence; setup screens read this instead of probing again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    profile_id: str
    validated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gpu_uuid: str | None = None
    ready: bool = False
    checks: dict[str, RuntimeValidationCheck]

    @classmethod
    def after_install(cls, profile_id: str, python: str) -> RuntimeValidationRecord:
        return cls(
            profile_id=profile_id,
            checks={
                "managed_python": RuntimeValidationCheck(
                    status=ValidationStatus.OK,
                    detail=f"Managed Python import environment created at {python}",
                ),
                "pytorch_gpu": RuntimeValidationCheck(
                    status=ValidationStatus.NOT_RUN,
                    detail="Installed; selected-GPU tensor and bf16 probes have not run yet",
                ),
                "onnx_cuda": RuntimeValidationCheck(
                    status=ValidationStatus.NOT_RUN,
                    detail="Installed; CUDA provider and device-transfer probes have not run yet",
                ),
                "sd_scripts": RuntimeValidationCheck(
                    status=ValidationStatus.OK,
                    detail="Pinned sd-scripts import completed during installation",
                ),
            },
        )

    @classmethod
    def from_doctor_report(
        cls, profile_id: str, report: RuntimeDoctorReport
    ) -> RuntimeValidationRecord:
        compatibility_ok = bool(
            report.compatibility is not None and report.compatibility.compatible
        )
        torch_ok = report.torch.ok and compatibility_ok
        torch_detail = report.torch.error
        if report.compatibility is not None:
            torch_detail = "; ".join(report.compatibility.reasons)
        elif not torch_detail:
            torch_detail = "PyTorch probe did not produce a compatibility decision"

        onnx_ok = bool(
            report.onnx.ok
            and report.onnx.details.get("cuda_provider")
            and report.onnx.details.get("cuda_transfer_smoke_ok")
        )
        onnx_detail = report.onnx.error or (
            "CUDAExecutionProvider and CUDA device transfer passed"
            if onnx_ok
            else "ONNX CUDA provider or CUDA device transfer failed"
        )
        sd_scripts_detail = report.sd_scripts.error or (
            "Pinned sd-scripts training module import passed"
            if report.sd_scripts.ok
            else "Pinned sd-scripts training module import failed"
        )
        return cls(
            profile_id=profile_id,
            gpu_uuid=report.gpu_uuid,
            ready=report.ready,
            checks={
                "managed_python": RuntimeValidationCheck(
                    status=ValidationStatus.OK,
                    detail="Managed Python executed the deep runtime probes",
                ),
                "pytorch_gpu": RuntimeValidationCheck(
                    status=ValidationStatus.OK if torch_ok else ValidationStatus.ERROR,
                    detail=torch_detail,
                ),
                "onnx_cuda": RuntimeValidationCheck(
                    status=ValidationStatus.OK if onnx_ok else ValidationStatus.ERROR,
                    detail=onnx_detail,
                ),
                "sd_scripts": RuntimeValidationCheck(
                    status=(
                        ValidationStatus.OK if report.sd_scripts.ok else ValidationStatus.ERROR
                    ),
                    detail=sd_scripts_detail,
                ),
            },
        )
