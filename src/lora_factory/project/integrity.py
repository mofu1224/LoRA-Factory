"""Immutable Raw-store snapshots and end-to-end integrity verification."""

from __future__ import annotations

from lora_factory.project.layout import ProjectLayout
from lora_factory.project.manifest import DatasetManifest
from lora_factory.project.raw_store import RawStoreBoundaryError, validate_raw_object_path
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import read_json


class RawIntegrityError(RuntimeError):
    """Raised when the Raw manifest or a manifest-backed object fails integrity checks."""


def verify_raw_store(
    layout: ProjectLayout,
    manifest: DatasetManifest | None = None,
) -> dict[str, str]:
    """Hash every manifest object and return a deterministic asset-id snapshot."""

    if manifest is None:
        try:
            current = DatasetManifest.model_validate(read_json(layout.manifest))
        except (OSError, ValueError) as exc:
            raise RawIntegrityError(f"Raw manifest is invalid or unreadable: {exc}") from exc
    else:
        current = manifest
    snapshot: dict[str, str] = {}
    for asset in sorted(current.raw_assets, key=lambda item: item.asset_id):
        try:
            path = validate_raw_object_path(layout, asset.stored_filename)
        except RawStoreBoundaryError as exc:
            raise RawIntegrityError(str(exc)) from exc
        if not path.is_file():
            raise RawIntegrityError(f"Raw object is missing: {path}")
        if path.stat().st_size != asset.size_bytes:
            raise RawIntegrityError(f"Raw object size changed: {path}")
        digest = sha256_file(path)
        if digest != asset.sha256 or digest != asset.asset_id:
            raise RawIntegrityError(f"Raw object hash changed: {path}")
        snapshot[asset.asset_id] = digest
    return snapshot


def compare_raw_snapshots(before: dict[str, str], after: dict[str, str]) -> None:
    """Require the exact same Raw asset set and bytes at both pipeline boundaries."""

    if before == after:
        return
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
    details = []
    if missing:
        details.append(f"missing={missing[0]}")
    if added:
        details.append(f"added={added[0]}")
    if changed:
        details.append(f"changed={changed[0]}")
    raise RawIntegrityError("Raw snapshot changed during the pipeline: " + ", ".join(details))
