"""Immutable import and derived-asset manifest models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lora_factory.dataset.scanner import SUPPORTED_IMAGE_EXTENSIONS


class SourceReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    original_path: Path
    original_filename: str
    imported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RawAsset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]+$")
    stored_filename: str
    extension: str
    size_bytes: int
    sources: tuple[SourceReference, ...]

    @model_validator(mode="after")
    def validate_content_addressed_identity(self) -> Self:
        if self.asset_id != self.sha256:
            raise ValueError("asset_id must equal sha256")
        if self.extension not in SUPPORTED_IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(f"extension must be one of: {supported}")
        expected = f"{self.sha256}{self.extension}"
        if self.stored_filename != expected:
            raise ValueError(f"stored_filename must be exactly {expected!r}")
        return self


class DerivedAsset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    source_asset_id: str
    relative_path: Path
    sha256: str
    media_type: str
    operation: str
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    project_id: str
    raw_assets: list[RawAsset] = Field(default_factory=list)
    derived_assets: list[DerivedAsset] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def by_sha256(self) -> dict[str, RawAsset]:
        return {asset.sha256: asset for asset in self.raw_assets}
