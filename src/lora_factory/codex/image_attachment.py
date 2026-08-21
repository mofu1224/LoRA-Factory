"""Deterministic, metadata-free image derivatives for Runtime Codex calls."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from PIL import Image, ImageCms
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lora_factory.util.hashing import sha256_bytes, sha256_file

_SAFE_ASSET_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_ALLOWED_JPEG_QUALITIES = (95, 90, 85, 80)


class CodexImageProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_edge: Annotated[int, Field(ge=256, le=2048)] = 2048
    max_bytes: Annotated[int, Field(ge=1024, le=8 * 1024 * 1024)] = 8 * 1024 * 1024
    jpeg_qualities: tuple[int, ...] = (95, 90, 85, 80)
    resampling: Literal["lanczos"] = "lanczos"

    @field_validator("jpeg_qualities")
    @classmethod
    def validate_jpeg_qualities(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("jpeg_qualities must be a non-empty ordered subset")
        positions = [_ALLOWED_JPEG_QUALITIES.index(quality) for quality in value]
        if positions != sorted(positions):
            raise ValueError("jpeg_qualities must preserve the fixed quality order")
        return value


DEFAULT_CODEX_IMAGE_PROFILE = CodexImageProfile()


class PreparedCodexImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    path: Path
    relative_name: str
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    width: Annotated[int, Field(ge=1, le=2048)]
    height: Annotated[int, Field(ge=1, le=2048)]
    quality: Annotated[int, Field(ge=80, le=95)]
    byte_count: Annotated[int, Field(ge=1, le=8 * 1024 * 1024)]
    profile: CodexImageProfile

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str) -> str:
        _validate_asset_id(value)
        return value

    @field_validator("relative_name")
    @classmethod
    def validate_relative_name(cls, value: str) -> str:
        if (
            not value
            or "/" in value
            or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(
                "relative_name must be a filename without separators or control characters"
            )
        return value

    @model_validator(mode="after")
    def validate_filename_mapping(self) -> PreparedCodexImage:
        if self.relative_name != f"{self.asset_id}.jpg":
            raise ValueError("relative_name must exactly match the asset_id JPEG filename")
        if self.path.name != self.relative_name:
            raise ValueError("path.name must exactly match relative_name")
        if self.quality not in self.profile.jpeg_qualities:
            raise ValueError("quality must be present in the transformation profile")
        return self


class CodexImagePayloadRecord(BaseModel):
    """Path-free image mapping embedded in a sanitized refinement request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    relative_name: str
    working_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    width: Annotated[int, Field(ge=1, le=2048)]
    height: Annotated[int, Field(ge=1, le=2048)]
    quality: Annotated[int, Field(ge=80, le=95)]
    byte_count: Annotated[int, Field(ge=1, le=8 * 1024 * 1024)]
    profile: CodexImageProfile

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str) -> str:
        _validate_asset_id(value)
        return value

    @field_validator("relative_name")
    @classmethod
    def validate_relative_name(cls, value: str) -> str:
        if value != Path(value).name or "/" in value or "\\" in value:
            raise ValueError("relative_name must be a safe filename")
        return value

    @model_validator(mode="after")
    def validate_filename_mapping(self) -> CodexImagePayloadRecord:
        if self.relative_name != f"{self.asset_id}.jpg":
            raise ValueError("relative_name must exactly match the asset_id JPEG filename")
        if self.quality not in self.profile.jpeg_qualities:
            raise ValueError("quality must be present in the transformation profile")
        return self

    @classmethod
    def from_prepared(cls, image: PreparedCodexImage) -> CodexImagePayloadRecord:
        return cls(
            asset_id=image.asset_id,
            relative_name=image.relative_name,
            working_sha256=image.source_sha256,
            image_sha256=image.output_sha256,
            width=image.width,
            height=image.height,
            quality=image.quality,
            byte_count=image.byte_count,
            profile=image.profile,
        )


def _validate_asset_id(asset_id: str) -> None:
    if not _SAFE_ASSET_ID.fullmatch(asset_id):
        raise ValueError(f"Unsafe asset_id for Codex image filename: {asset_id!r}")


def _encode_jpeg(image: Image.Image, *, quality: int) -> bytes:
    encoded = BytesIO()
    image.save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    return encoded.getvalue()


def _atomic_write(destination: Path, payload: bytes) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _convert_to_srgb(opened: Image.Image) -> Image.Image:
    rgb = opened.convert("RGB")
    icc_profile = opened.info.get("icc_profile")
    if isinstance(icc_profile, bytes):
        try:
            source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            target_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(
                rgb,
                source_profile,
                target_profile,
                outputMode="RGB",
            )
            if converted is not None:
                rgb = converted
        except (ImageCms.PyCMSError, OSError, TypeError, ValueError):
            # Invalid embedded profiles are treated as already-sRGB pixels.
            rgb = opened.convert("RGB")
    rgb.info.clear()
    return rgb


def prepare_codex_image(
    asset_id: str,
    source: Path,
    destination_root: Path,
    *,
    profile: CodexImageProfile = DEFAULT_CODEX_IMAGE_PROFILE,
) -> PreparedCodexImage:
    """Create a deterministic sanitized JPEG derivative of a working image."""

    _validate_asset_id(asset_id)
    profile = CodexImageProfile.model_validate(profile.model_dump())
    source = source.resolve(strict=True)
    destination_root = destination_root.resolve(strict=False)
    relative_name = f"{asset_id}.jpg"
    destination = (destination_root / relative_name).resolve(strict=False)
    if not destination.is_relative_to(destination_root):
        raise ValueError("Codex image destination escapes its scratch root")
    if destination == source:
        raise ValueError("Codex image destination must not overwrite its source")

    source_sha256 = sha256_file(source)
    with Image.open(source) as opened:
        opened.load()
        image = _convert_to_srgb(opened)
    image.thumbnail((profile.max_edge, profile.max_edge), Image.Resampling.LANCZOS)
    width, height = image.size

    selected_payload: bytes | None = None
    selected_quality: int | None = None
    for quality in profile.jpeg_qualities:
        payload = _encode_jpeg(image, quality=quality)
        if len(payload) <= profile.max_bytes:
            selected_payload = payload
            selected_quality = quality
            break
    if selected_payload is None or selected_quality is None:
        raise ValueError(
            f"Codex image exceeds profile.max_bytes after all JPEG qualities: {profile.max_bytes}"
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, selected_payload)
    return PreparedCodexImage(
        asset_id=asset_id,
        path=destination,
        relative_name=relative_name,
        source_sha256=source_sha256,
        output_sha256=sha256_bytes(selected_payload),
        width=width,
        height=height,
        quality=selected_quality,
        byte_count=len(selected_payload),
        profile=profile,
    )


def remove_codex_images(
    images: Sequence[PreparedCodexImage],
    *,
    scratch_root: Path,
) -> None:
    """Remove prepared derivatives only after proving they stay in scratch_root."""

    resolved_root = scratch_root.resolve(strict=False)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for image in images:
        supplied_path = Path(image.path)
        try:
            resolved_path = supplied_path.resolve(strict=False)
        except OSError as exc:
            raise ValueError("Codex image cleanup path cannot be resolved") from exc
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError("Codex image cleanup path escapes its scratch root")
        if resolved_path.name != image.relative_name:
            raise ValueError("Codex image cleanup filename does not match its record")
        try:
            path_stat = supplied_path.lstat()
        except FileNotFoundError:
            continue
        file_attributes = int(getattr(path_stat, "st_file_attributes", 0))
        if supplied_path.is_symlink() or (reparse_flag and file_attributes & reparse_flag):
            raise ValueError("Codex image cleanup path must not be a symlink or reparse point")
        supplied_path.unlink()


def validate_codex_image_attachments(
    images: Sequence[PreparedCodexImage],
    payload_records: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> tuple[PreparedCodexImage, ...]:
    """Revalidate record/payload/file integrity immediately before Codex launch."""

    validated = tuple(
        PreparedCodexImage.model_validate(image.model_dump(mode="python")) for image in images
    )
    payload = tuple(CodexImagePayloadRecord.model_validate(item) for item in payload_records)
    if len(validated) != len(payload):
        raise ValueError("Prepared image and payload counts do not match")
    if not validated:
        raise ValueError("Dataset refinement requires at least one image")
    asset_ids = tuple(image.asset_id for image in validated)
    if asset_ids != tuple(sorted(asset_ids)):
        raise ValueError("Prepared images must use stable asset ID order")
    if tuple(item.asset_id for item in payload) != asset_ids:
        raise ValueError("Prepared image and payload mapping is out of stable order")

    resolved_root = scratch_root.resolve(strict=True)
    checked: list[PreparedCodexImage] = []
    for image, payload_record in zip(validated, payload, strict=True):
        expected_payload = CodexImagePayloadRecord.from_prepared(image)
        for field_name in CodexImagePayloadRecord.model_fields:
            if getattr(payload_record, field_name) != getattr(expected_payload, field_name):
                raise ValueError(f"Prepared image and payload {field_name} mapping does not match")
        if image.profile != DEFAULT_CODEX_IMAGE_PROFILE:
            raise ValueError("Prepared image transformation profile does not match the contract")

        supplied_path = Path(image.path)
        try:
            path_stat = supplied_path.lstat()
            resolved_path = supplied_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Prepared image must be a readable regular scratch file") from exc
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        file_attributes = int(getattr(path_stat, "st_file_attributes", 0))
        if supplied_path.is_symlink() or (reparse_flag and file_attributes & reparse_flag):
            raise ValueError("Prepared image must not be a symlink or reparse point")
        if not stat.S_ISREG(path_stat.st_mode) or not resolved_path.is_relative_to(resolved_root):
            raise ValueError("Prepared image must be a regular file inside the scratch repository")
        if resolved_path.name != image.relative_name:
            raise ValueError("Prepared image filename mapping does not match")

        before = resolved_path.stat()
        contents = resolved_path.read_bytes()
        after = resolved_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Prepared image changed while its integrity was checked")
        if after.st_size != image.byte_count:
            raise ValueError("Prepared image byte count does not match its record and payload")
        if sha256_bytes(contents) != image.output_sha256:
            raise ValueError("Prepared image SHA-256 does not match its record and payload")

        try:
            with Image.open(BytesIO(contents)) as opened:
                opened.load()
                actual_dimensions = opened.size
                actual_format = opened.format
                actual_mode = opened.mode
        except OSError as exc:
            raise ValueError("Prepared image is not a readable JPEG") from exc
        if actual_format != "JPEG" or actual_mode != "RGB":
            raise ValueError("Prepared image must be an RGB JPEG")
        if actual_dimensions != (image.width, image.height):
            raise ValueError("Prepared image dimensions do not match its record and payload")
        checked.append(image.model_copy(update={"path": resolved_path}))
    return tuple(checked)
