"""Crash-resumable verified imports into the content-addressed immutable Raw store."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.dataset.image_safety import ImageSafetyLimits, validate_image_header
from lora_factory.dataset.scanner import ScanIssue, scan_image_inputs
from lora_factory.project.layout import ProjectLayout
from lora_factory.project.manifest import DatasetManifest, RawAsset, SourceReference
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json, write_json_atomic


class ImportFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Path
    code: str
    message: str


class ImportResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: DatasetManifest
    imported_asset_ids: tuple[str, ...] = Field(default_factory=tuple)
    reused_asset_ids: tuple[str, ...] = Field(default_factory=tuple)
    failures: tuple[ImportFailure, ...] = Field(default_factory=tuple)
    scan_issues: tuple[ScanIssue, ...] = Field(default_factory=tuple)


def _source_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False))).casefold()


def _make_raw_read_only(path: Path) -> None:
    """Remove write bits while retaining readable immutable project data."""

    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _load_manifest(layout: ProjectLayout, project_id: str | None) -> DatasetManifest:
    if layout.manifest.exists():
        return DatasetManifest.model_validate(read_json(layout.manifest))
    if not project_id:
        raise FileNotFoundError(
            f"Dataset manifest does not exist and no project_id was provided: {layout.manifest}"
        )
    return DatasetManifest(project_id=project_id)


def _copy_verified(source: Path, staging: Path, expected_sha256: str) -> None:
    """Copy bytes (never link), flush them, and verify the completed staging object."""

    with source.open("rb") as reader, staging.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    actual_sha256 = sha256_file(staging)
    if actual_sha256 != expected_sha256:
        raise OSError(
            f"SHA-256 changed while copying {source}: expected {expected_sha256}, "
            f"copied {actual_sha256}"
        )


class ImmutableImportService:
    """Import supported files into Raw using monotonic manifest updates.

    A failure after the verified raw object is installed but before the manifest write is safe:
    the next call verifies and adopts the same content-addressed object. Existing manifest data is
    never removed, and an already-seen source reference is not duplicated.
    """

    def __init__(
        self,
        layout: ProjectLayout,
        *,
        project_id: str | None = None,
        safety_limits: ImageSafetyLimits | None = None,
    ) -> None:
        self.layout = layout
        self.project_id = project_id
        self.safety_limits = safety_limits or ImageSafetyLimits()

    def import_paths(
        self,
        inputs: Iterable[Path | str],
        *,
        recursive: bool = False,
        check_cancelled: Callable[[], None] | None = None,
    ) -> ImportResult:
        self.layout.create()
        scan = scan_image_inputs(inputs, recursive=recursive, limits=self.safety_limits)
        manifest = _load_manifest(self.layout, self.project_id)
        original_dump = manifest.model_dump(mode="json")
        by_sha = manifest.by_sha256()
        imported: list[str] = []
        reused: list[str] = []
        failures: list[ImportFailure] = []

        for source in scan.files:
            if check_cancelled is not None:
                check_cancelled()
            try:
                asset, was_new = self._import_one(source, manifest, by_sha)
            except (OSError, ValueError) as exc:
                failures.append(
                    ImportFailure(source=source, code=type(exc).__name__, message=str(exc))
                )
                continue
            (imported if was_new else reused).append(asset.asset_id)

        self._assert_monotonic(original_dump, manifest)
        write_json_atomic(self.layout.manifest, manifest.model_dump(mode="json"))
        return ImportResult(
            manifest=manifest,
            imported_asset_ids=tuple(imported),
            reused_asset_ids=tuple(reused),
            failures=tuple(failures),
            scan_issues=scan.issues,
        )

    def _import_one(
        self,
        source: Path,
        manifest: DatasetManifest,
        by_sha: dict[str, RawAsset],
    ) -> tuple[RawAsset, bool]:
        source = source.resolve(strict=True)
        if source.is_relative_to(self.layout.raw.resolve(strict=False)):
            raise ValueError(f"Raw store cannot be used as an import source: {source}")
        validate_image_header(source, limits=self.safety_limits)
        source_sha256 = sha256_file(source)
        existing = by_sha.get(source_sha256)
        reference = SourceReference(
            original_path=source,
            original_filename=source.name,
        )

        if existing is not None:
            raw_path = self.layout.raw / existing.stored_filename
            if not raw_path.is_file() or sha256_file(raw_path) != source_sha256:
                raise OSError(f"Immutable Raw object is missing or corrupted: {raw_path}")
            known_sources = {_source_identity(item.original_path) for item in existing.sources}
            if _source_identity(source) not in known_sources:
                updated = existing.model_copy(update={"sources": (*existing.sources, reference)})
                index = manifest.raw_assets.index(existing)
                manifest.raw_assets[index] = updated
                by_sha[source_sha256] = updated
                existing = updated
            _make_raw_read_only(raw_path)
            return existing, False

        extension = source.suffix.lower()
        stored_filename = f"{source_sha256}{extension}"
        destination = self.layout.raw / stored_filename
        staging = self.layout.import_staging / f"{source_sha256}.{uuid4().hex}.partial"
        try:
            if destination.exists():
                if not destination.is_file() or sha256_file(destination) != source_sha256:
                    raise OSError(f"Conflicting object already exists in Raw store: {destination}")
            else:
                _copy_verified(source, staging, source_sha256)
                staging.replace(destination)
            if sha256_file(destination) != source_sha256:
                raise OSError(f"Raw verification failed after install: {destination}")
            _make_raw_read_only(destination)
        finally:
            if staging.exists():
                staging.unlink()

        asset = RawAsset(
            asset_id=source_sha256,
            sha256=source_sha256,
            stored_filename=stored_filename,
            extension=extension,
            size_bytes=source.stat().st_size,
            sources=(reference,),
        )
        manifest.raw_assets.append(asset)
        by_sha[source_sha256] = asset
        return asset, True

    @staticmethod
    def _assert_monotonic(before: dict[str, object], after: DatasetManifest) -> None:
        previous = DatasetManifest.model_validate(before)
        current = {asset.sha256: asset for asset in after.raw_assets}
        for old_asset in previous.raw_assets:
            new_asset = current.get(old_asset.sha256)
            if new_asset is None:
                raise RuntimeError("Import attempted to remove an existing Raw manifest entry")
            immutable_fields = ("asset_id", "sha256", "stored_filename", "extension", "size_bytes")
            if any(
                getattr(old_asset, field) != getattr(new_asset, field) for field in immutable_fields
            ):
                raise RuntimeError("Import attempted to rewrite an existing Raw manifest entry")
            new_source_ids = {_source_identity(item.original_path) for item in new_asset.sources}
            if any(
                _source_identity(item.original_path) not in new_source_ids
                for item in old_asset.sources
            ):
                raise RuntimeError("Import attempted to remove an existing source reference")


# A concise public name for application-service composition.
ImportService = ImmutableImportService
