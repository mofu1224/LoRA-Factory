"""Capability-based PyTorch/CUDA compatibility decisions."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_VERSION_NUMBER = re.compile(r"\d+")


class CompatibilityDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    compatible: bool
    compute_capability: str
    required_arch: str
    torch_version: str
    torch_cuda_version: str
    reasons: tuple[str, ...]
    remediation: tuple[str, ...] = ()


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = tuple(int(item) for item in _VERSION_NUMBER.findall(value))
    return numbers or (0,)


def evaluate_runtime_compatibility(
    *,
    capability_major: Annotated[int, Field(ge=0)],
    capability_minor: Annotated[int, Field(ge=0)],
    torch_version: str,
    torch_cuda_version: str,
    torch_arch_list: tuple[str, ...],
    tensor_smoke_ok: bool,
    bf16_smoke_ok: bool,
) -> CompatibilityDecision:
    """Evaluate the installed wheel, not the host CUDA toolkit string."""

    capability = f"{capability_major}.{capability_minor}"
    required_arch = f"sm_{capability_major}{capability_minor}"
    normalized_arches = {item.lower().replace(" ", "") for item in torch_arch_list}
    reasons: list[str] = []
    remediation: list[str] = []

    if required_arch not in normalized_arches:
        reasons.append(f"Installed PyTorch arch list does not contain {required_arch}")
        remediation.append("Install a PyTorch wheel that explicitly includes this GPU architecture")
    if not tensor_smoke_ok:
        reasons.append("CUDA tensor arithmetic smoke test failed")
        remediation.append("Reinstall the managed PyTorch runtime and verify the NVIDIA driver")
    if capability_major >= 8 and not bf16_smoke_ok:
        reasons.append("bf16 arithmetic smoke test failed on a bf16-capable GPU")
        remediation.append(
            "Use a compatible bf16 wheel or select fp16 after a successful fp16 probe"
        )
    if (capability_major, capability_minor) >= (12, 0) and _version_tuple(torch_cuda_version) < (
        12,
        8,
    ):
        reasons.append("Blackwell sm_120 requires a PyTorch wheel built with CUDA 12.8 or newer")
        remediation.append("Install the managed Blackwell profile with a CUDA 12.8+ PyTorch wheel")

    return CompatibilityDecision(
        compatible=not reasons,
        compute_capability=capability,
        required_arch=required_arch,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        reasons=tuple(reasons) if reasons else ("Installed runtime passed capability probes",),
        remediation=tuple(dict.fromkeys(remediation)),
    )
