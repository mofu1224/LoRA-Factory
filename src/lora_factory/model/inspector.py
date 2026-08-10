"""Bounded safetensors header inspection for SDXL-compatible base models."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.model.compatibility import ModelCompatibility
from lora_factory.model.hashes import sha256_file

_MAX_HEADER_BYTES = 128 * 1024 * 1024
_KEY_SAMPLE_LIMIT = 64


class ModelInspectionError(RuntimeError):
    """Raised when a model is unreadable or not a valid safetensors file."""


class ModelInspection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    sha256: str
    file_size: int = Field(ge=0)
    tensor_count: int = Field(ge=0)
    key_sample: tuple[str, ...]
    metadata: dict[str, str]
    architecture_family: str
    is_sdxl: bool
    possible_illustrious: bool
    embedded_vae: bool
    prediction_type: str | None
    compatibility: ModelCompatibility
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]


def _read_header(path: Path) -> tuple[dict[str, Any], int, int]:
    file_size = path.stat().st_size
    if file_size < 10:
        raise ModelInspectionError("Safetensors file is too small to contain a header")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ModelInspectionError("Unable to read safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > _MAX_HEADER_BYTES:
            raise ModelInspectionError(f"Invalid safetensors header size: {header_size}")
        if 8 + header_size > file_size:
            raise ModelInspectionError("Safetensors header extends beyond end of file")
        raw_header = stream.read(header_size)
    try:
        payload = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelInspectionError(f"Invalid safetensors JSON header: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelInspectionError("Safetensors header must be a JSON object")
    return payload, file_size, header_size


def _validate_tensor_entries(
    header: dict[str, Any], *, file_size: int, header_size: int
) -> tuple[tuple[str, ...], dict[str, str]]:
    metadata_raw = header.get("__metadata__", {})
    if not isinstance(metadata_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata_raw.items()
    ):
        raise ModelInspectionError("Safetensors metadata must map strings to strings")
    metadata = dict(metadata_raw)
    tensor_keys: list[str] = []
    data_size = file_size - 8 - header_size
    ranges: list[tuple[int, int, str]] = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ModelInspectionError("Safetensors tensor entries must be named objects")
        offsets = value.get("data_offsets")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
            or not isinstance(shape, list)
            or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
            or not isinstance(dtype, str)
        ):
            raise ModelInspectionError(f"Malformed tensor metadata for {key!r}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ModelInspectionError(f"Tensor {key!r} has out-of-bounds data offsets")
        ranges.append((start, end, key))
        tensor_keys.append(key)
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:], strict=False):
        if previous[1] > current[0]:
            raise ModelInspectionError(
                f"Tensor data ranges overlap: {previous[2]!r} and {current[2]!r}"
            )
    return tuple(sorted(tensor_keys)), metadata


def _prediction_type(metadata: dict[str, str]) -> str | None:
    for key, value in metadata.items():
        normalized_key = key.lower().replace("-", "_")
        if normalized_key in {"prediction_type", "modelspec.prediction_type", "ss_prediction_type"}:
            return value
    return None


def inspect_sdxl_safetensors(path: Path) -> ModelInspection:
    """Inspect metadata and key names without materializing any model tensor."""

    if path.suffix.lower() != ".safetensors":
        raise ModelInspectionError("v1 base-model inspection accepts .safetensors files only")
    if not path.is_file():
        raise ModelInspectionError(f"Model file does not exist: {path}")
    header, file_size, header_size = _read_header(path)
    tensor_keys, metadata = _validate_tensor_entries(
        header, file_size=file_size, header_size=header_size
    )
    lowered_keys = tuple(key.lower() for key in tensor_keys)
    has_unet = any(
        key.startswith(("model.diffusion_model.", "unet.", "diffusion_model."))
        for key in lowered_keys
    )
    has_second_text_encoder = any(
        marker in key
        for key in lowered_keys
        for marker in (
            "conditioner.embedders.1.",
            "text_encoder_2.",
            "text_encoders.clip_g.",
        )
    )
    is_sdxl = has_unet and has_second_text_encoder
    embedded_vae = any(key.startswith(("first_stage_model.", "vae.")) for key in lowered_keys)
    metadata_text = " ".join([path.stem, *metadata.keys(), *metadata.values()]).lower()
    possible_illustrious = "illustrious" in metadata_text
    evidence: list[str] = []
    warnings: list[str] = []
    if has_unet:
        evidence.append("UNet/diffusion model tensor namespace found")
    if has_second_text_encoder:
        evidence.append("SDXL second text encoder tensor namespace found")
    if possible_illustrious:
        evidence.append("Filename or safetensors metadata mentions Illustrious")
    if not metadata:
        warnings.append("Checkpoint has no safetensors metadata")

    if is_sdxl and possible_illustrious:
        compatibility = ModelCompatibility.COMPATIBLE_WITH_WARNING
        warnings.append("Illustrious lineage is heuristic; SDXL structure is the hard gate")
    elif is_sdxl:
        compatibility = ModelCompatibility.COMPATIBLE
    else:
        compatibility = ModelCompatibility.UNSUPPORTED
        warnings.append("Required SDXL UNet and dual text-encoder markers were not both found")

    return ModelInspection(
        path=path,
        sha256=sha256_file(path),
        file_size=file_size,
        tensor_count=len(tensor_keys),
        key_sample=tensor_keys[:_KEY_SAMPLE_LIMIT],
        metadata=metadata,
        architecture_family="sdxl" if is_sdxl else "unknown",
        is_sdxl=is_sdxl,
        possible_illustrious=possible_illustrious,
        embedded_vae=embedded_vae,
        prediction_type=_prediction_type(metadata),
        compatibility=compatibility,
        evidence=tuple(evidence),
        warnings=tuple(warnings),
    )
