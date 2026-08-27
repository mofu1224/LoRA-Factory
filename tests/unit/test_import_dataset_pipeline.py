from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lora_factory.config.models import PresetKind
from lora_factory.dataset import scanner as scanner_module
from lora_factory.dataset.clustering import MAX_PAIRWISE_COMPARISONS, require_pairwise_budget
from lora_factory.dataset.diversity import analyze_diversity
from lora_factory.dataset.duplicates import DuplicateCandidate, DuplicateKind, detect_duplicates
from lora_factory.dataset.image_normalizer import (
    AnimatedImageError,
    NeutralBackgroundPolicy,
    normalize_image,
)
from lora_factory.dataset.image_safety import ImageSafetyLimits
from lora_factory.dataset.quality import (
    DatasetGateThresholds,
    ImageQualityMetrics,
    QualityAssessment,
    QualityDisposition,
    QualityThresholds,
    add_tagger_quality_signals,
    assess_image,
    evaluate_quality_gate,
)
from lora_factory.dataset.scanner import scan_image_inputs
from lora_factory.dataset.split import deterministic_validation_split
from lora_factory.project.import_service import ImmutableImportService
from lora_factory.project.integrity import RawIntegrityError, verify_raw_store
from lora_factory.project.layout import ProjectLayout
from lora_factory.project.manifest import DatasetManifest
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import write_json_atomic


def _pattern(path: Path, *, size: tuple[int, int] = (128, 96), offset: int = 0) -> None:
    y, x = np.indices((size[1], size[0]))
    array = np.stack(
        (
            (x * 7 + offset) % 256,
            (y * 11 + offset) % 256,
            ((x + y) * 13 + offset) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


_VALID_RAW_DIGEST = "a" * 64


def _manifest_payload(**raw_overrides: object) -> dict[str, object]:
    raw_asset: dict[str, object] = {
        "asset_id": _VALID_RAW_DIGEST,
        "sha256": _VALID_RAW_DIGEST,
        "stored_filename": f"{_VALID_RAW_DIGEST}.png",
        "extension": ".png",
        "size_bytes": 1,
        "sources": [],
    }
    raw_asset.update(raw_overrides)
    return {"project_id": "project", "raw_assets": [raw_asset]}


def test_scan_unicode_is_shallow_by_default_and_recursive_on_request(tmp_path: Path) -> None:
    source = tmp_path / "入力 画像"
    nested = source / "下位"
    nested.mkdir(parents=True)
    _pattern(source / "日本語.png")
    _pattern(source / "Screenshot 01.JPG")
    _pattern(nested / "test (4).webp")
    (source / "ignore.txt").write_text("not an image", encoding="utf-8")

    shallow = scan_image_inputs((source,))
    recursive = scan_image_inputs((source,), recursive=True)

    assert [path.name for path in shallow.files] == ["Screenshot 01.JPG", "日本語.png"]
    assert {path.name for path in recursive.files} == {
        "Screenshot 01.JPG",
        "日本語.png",
        "test (4).webp",
    }


def test_scan_rejects_file_symlink_even_when_target_is_outside_selected_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.png"
    _pattern(outside)
    selected = tmp_path / "selected"
    selected.mkdir()
    link = selected / "linked.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"filesystem symlinks are unavailable: {exc}")

    result = scan_image_inputs((selected,))

    assert result.files == ()
    assert len(result.issues) == 1
    assert result.issues[0].code == "filesystem_link"
    assert result.issues[0].path == link.absolute()


def test_scan_rejects_explicit_path_through_parent_filesystem_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_directory = tmp_path / "junction"
    linked_directory.mkdir()
    monkeypatch.setattr(
        scanner_module,
        "is_filesystem_link",
        lambda path: path == linked_directory,
    )

    result = scan_image_inputs((linked_directory / "outside.png",))

    assert result.files == ()
    assert len(result.issues) == 1
    assert result.issues[0].code == "filesystem_link"


def test_scan_rejects_resolution_outside_selected_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    candidate = selected / "input.png"
    _pattern(candidate)
    outside = tmp_path / "outside.png"
    _pattern(outside)
    original_resolve = Path.resolve

    def resolve_with_swap(self: Path, strict: bool = False) -> Path:
        if self == candidate:
            return outside
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_with_swap)

    result = scan_image_inputs((candidate,))

    assert result.files == ()
    assert len(result.issues) == 1
    assert result.issues[0].code == "filesystem_link"


def test_import_rejects_direct_file_symlink_before_copy(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    _pattern(outside)
    link = tmp_path / "linked.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"filesystem symlinks are unavailable: {exc}")

    layout = ProjectLayout(tmp_path / "project")
    result = ImmutableImportService(layout, project_id="project").import_paths((link,))

    assert result.imported_asset_ids == ()
    assert len(result.scan_issues) == 1
    assert result.scan_issues[0].code == "filesystem_link"
    assert not tuple(layout.raw.iterdir())


def test_scan_enforces_count_file_and_total_size_limits(tmp_path: Path) -> None:
    source = tmp_path / "bounded"
    source.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _pattern(source / name, size=(8, 8))

    count_limited = scan_image_inputs(
        (source,),
        limits=ImageSafetyLimits(max_image_files=2),
    )
    assert len(count_limited.files) == 2
    assert {issue.code for issue in count_limited.issues} == {"too_many_images"}

    first_size = (source / "a.png").stat().st_size
    file_limited = scan_image_inputs(
        (source / "a.png",),
        limits=ImageSafetyLimits(max_file_size_bytes=first_size - 1),
    )
    assert file_limited.files == ()
    assert file_limited.issues[0].code == "file_too_large"

    total_limited = scan_image_inputs(
        (source,),
        limits=ImageSafetyLimits(max_total_size_bytes=first_size),
    )
    assert len(total_limited.files) == 1
    assert total_limited.issues[-1].code == "total_size_exceeded"


def test_duplicate_detection_rejects_unbounded_pairwise_work() -> None:
    item_count = next(
        count for count in range(1, 100_000) if count * (count - 1) // 2 > MAX_PAIRWISE_COMPARISONS
    )
    candidates = [
        DuplicateCandidate(
            asset_id=f"asset-{index}",
            path=Path(f"asset-{index}.png"),
            sha256="0" * 64,
            area=1,
            sharpness=1,
            compression_blockiness=0,
            alpha_fraction=0,
        )
        for index in range(item_count)
    ]

    with pytest.raises(ValueError, match="pairwise safety limit"):
        detect_duplicates(candidates)


def test_pairwise_override_cannot_raise_global_safety_ceiling() -> None:
    with pytest.raises(ValueError, match="global safety limit"):
        require_pairwise_budget(
            1,
            operation="test",
            max_pairwise_comparisons=MAX_PAIRWISE_COMPARISONS + 1,
        )


def test_import_and_quality_reject_excessive_pixels_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "oversized.png"
    _pattern(source, size=(20, 20))
    limits = ImageSafetyLimits(max_pixels=100)
    layout = ProjectLayout(tmp_path / "project")

    result = ImmutableImportService(
        layout,
        project_id="project",
        safety_limits=limits,
    ).import_paths((source,))

    assert result.imported_asset_ids == ()
    assert len(result.failures) == 1
    assert "400 pixels" in result.failures[0].message
    assert not tuple(layout.raw.iterdir())

    assessment = assess_image(source, thresholds=QualityThresholds(maximum_pixels=100))
    assert assessment.disposition is QualityDisposition.REJECT
    assert assessment.reasons[0].code == "unsafe_image"


def test_import_uses_verified_copies_and_monotonic_source_references(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    first = source / "日本語 file.png"
    second = source / "duplicate (2).png"
    _pattern(first)
    shutil.copyfile(first, second)
    source_hash = sha256_file(first)
    layout = ProjectLayout(tmp_path / "project")
    importer = ImmutableImportService(layout, project_id="project")

    first_result = importer.import_paths((first,))
    second_result = importer.import_paths((second,))

    assert first_result.imported_asset_ids == (source_hash,)
    assert second_result.reused_asset_ids == (source_hash,)
    assert len(second_result.manifest.raw_assets) == 1
    asset = second_result.manifest.raw_assets[0]
    assert {reference.original_filename for reference in asset.sources} == {
        first.name,
        second.name,
    }
    raw = layout.raw / asset.stored_filename
    assert raw.read_bytes() == first.read_bytes()
    assert os.stat(raw).st_ino != os.stat(first).st_ino

    _pattern(first, offset=37)
    assert sha256_file(raw) == source_hash
    assert sha256_file(first) != source_hash


def test_import_manifest_size_comes_from_verified_raw_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _pattern(source)
    layout = ProjectLayout(tmp_path / "project")
    importer = ImmutableImportService(layout, project_id="project")
    layout.create()
    manifest = DatasetManifest(project_id="project")
    actual_source_stat = source.stat()
    original_stat = Path.stat

    def forged_source_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = original_stat(path, *args, **kwargs)
        if path == source:
            values = list(result)
            values[6] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", forged_source_stat)
    asset, is_new = importer._import_one(source, manifest, {})

    raw_path = layout.raw / asset.stored_filename
    assert is_new
    assert actual_source_stat.st_size == raw_path.stat().st_size
    assert asset.size_bytes == raw_path.stat().st_size


def test_dataset_manifest_rejects_raw_path_traversal_at_validation_boundary() -> None:
    with pytest.raises(ValueError, match="stored_filename"):
        DatasetManifest.model_validate(_manifest_payload(stored_filename="../outside.png"))


@pytest.mark.parametrize("invalid_digest", ["A" * 64, "a" * 63])
def test_dataset_manifest_requires_lowercase_sha256_identity(invalid_digest: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        DatasetManifest.model_validate(
            _manifest_payload(
                asset_id=invalid_digest,
                sha256=invalid_digest,
                stored_filename=f"{invalid_digest}.png",
            )
        )


def test_dataset_manifest_requires_asset_id_to_match_sha256() -> None:
    with pytest.raises(ValueError, match="asset_id"):
        DatasetManifest.model_validate(_manifest_payload(asset_id="b" * 64))


def test_dataset_manifest_rejects_unsupported_raw_extension() -> None:
    with pytest.raises(ValueError, match="extension"):
        DatasetManifest.model_validate(
            _manifest_payload(
                stored_filename=f"{_VALID_RAW_DIGEST}.gif",
                extension=".gif",
            )
        )


def test_import_rejects_tampered_manifest_before_asset_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _pattern(source)
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    write_json_atomic(
        layout.manifest,
        _manifest_payload(stored_filename="../outside.png"),
    )

    def unexpected_asset_io(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("asset bytes were accessed before manifest validation")

    monkeypatch.setattr(
        "lora_factory.project.import_service.validate_image_header", unexpected_asset_io
    )
    monkeypatch.setattr("lora_factory.project.import_service.sha256_file", unexpected_asset_io)

    with pytest.raises(ValueError, match="stored_filename"):
        ImmutableImportService(layout, project_id="project").import_paths((source,))


def test_integrity_normalizes_tampered_manifest_validation_failure(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    write_json_atomic(
        layout.manifest,
        _manifest_payload(stored_filename="../outside.png"),
    )

    with pytest.raises(RawIntegrityError, match="Raw manifest is invalid or unreadable"):
        verify_raw_store(layout)


def test_dataset_manifest_accepts_content_addressed_raw_asset() -> None:
    manifest = DatasetManifest.model_validate(
        _manifest_payload(
            stored_filename=f"{_VALID_RAW_DIGEST}.webp",
            extension=".webp",
        )
    )

    asset = manifest.raw_assets[0]
    assert asset.asset_id == asset.sha256 == _VALID_RAW_DIGEST
    assert asset.stored_filename == f"{_VALID_RAW_DIGEST}{asset.extension}"


@pytest.mark.parametrize("manifest_contains_asset", [False, True])
def test_import_rejects_linked_raw_object_before_hash_or_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_contains_asset: bool,
) -> None:
    source = tmp_path / "source.png"
    _pattern(source)
    digest = sha256_file(source)
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    raw_path = layout.raw / f"{digest}.png"
    shutil.copyfile(source, raw_path)
    if manifest_contains_asset:
        write_json_atomic(
            layout.manifest,
            _manifest_payload(
                asset_id=digest,
                sha256=digest,
                stored_filename=raw_path.name,
                size_bytes=source.stat().st_size,
            ),
        )

    original_is_symlink = Path.is_symlink
    hashed_paths: list[Path] = []

    def report_raw_object_link(path: Path) -> bool:
        return path == raw_path or original_is_symlink(path)

    def tracked_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return sha256_file(path)

    def unexpected_chmod(_path: Path) -> None:
        raise AssertionError("linked Raw object reached chmod")

    monkeypatch.setattr(Path, "is_symlink", report_raw_object_link)
    monkeypatch.setattr("lora_factory.project.import_service.sha256_file", tracked_sha256)
    monkeypatch.setattr("lora_factory.project.import_service._make_raw_read_only", unexpected_chmod)

    result = ImmutableImportService(layout, project_id="project").import_paths((source,))

    assert result.imported_asset_ids == ()
    assert result.reused_asset_ids == ()
    assert len(result.failures) == 1
    assert "link" in result.failures[0].message.casefold()
    assert hashed_paths == [source.resolve()]


def test_import_rejects_hardlinked_raw_object_before_hash_or_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _pattern(source)
    digest = sha256_file(source)
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    raw_path = layout.raw / f"{digest}.png"
    os.link(source, raw_path)
    hashed_paths: list[Path] = []

    def tracked_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return sha256_file(path)

    def unexpected_chmod(_path: Path) -> None:
        raise AssertionError("hardlinked Raw object reached chmod")

    monkeypatch.setattr("lora_factory.project.import_service.sha256_file", tracked_sha256)
    monkeypatch.setattr("lora_factory.project.import_service._make_raw_read_only", unexpected_chmod)

    result = ImmutableImportService(layout, project_id="project").import_paths((source,))

    assert result.imported_asset_ids == ()
    assert len(result.failures) == 1
    assert "exactly one filesystem link" in result.failures[0].message
    assert hashed_paths == [source.resolve()]


def test_integrity_rejects_raw_root_junction_before_object_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    original_is_junction = Path.is_junction

    def report_raw_root_junction(path: Path) -> bool:
        return path == layout.raw or original_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", report_raw_root_junction)

    with pytest.raises(RawIntegrityError, match="symbolic link or junction"):
        verify_raw_store(
            layout,
            DatasetManifest.model_validate(_manifest_payload()),
        )


def test_integrity_rejects_hardlinked_raw_object(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _pattern(source)
    digest = sha256_file(source)
    layout = ProjectLayout(tmp_path / "project")
    layout.create()
    raw_path = layout.raw / f"{digest}.png"
    os.link(source, raw_path)
    manifest = DatasetManifest.model_validate(
        _manifest_payload(
            asset_id=digest,
            sha256=digest,
            stored_filename=raw_path.name,
            size_bytes=source.stat().st_size,
        )
    )

    with pytest.raises(RawIntegrityError, match="exactly one filesystem link"):
        verify_raw_store(layout, manifest)


def test_raw_integrity_verification_rejects_one_byte_change(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _pattern(source)
    layout = ProjectLayout(tmp_path / "project")
    result = ImmutableImportService(layout, project_id="project").import_paths((source,))
    snapshot = verify_raw_store(layout, result.manifest)

    raw = layout.raw / result.manifest.raw_assets[0].stored_filename
    raw.chmod(0o600)
    payload = bytearray(raw.read_bytes())
    payload[-1] ^= 1
    raw.write_bytes(payload)

    assert snapshot == {
        result.manifest.raw_assets[0].asset_id: result.manifest.raw_assets[0].sha256
    }
    with pytest.raises(RawIntegrityError, match="Raw object hash changed"):
        verify_raw_store(layout, result.manifest)


def test_normalization_preserves_raw_applies_exif_and_neutral_alpha(tmp_path: Path) -> None:
    oriented = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (12, 6), (20, 40, 80))
    exif = image.getexif()
    exif[274] = 6
    image.save(oriented, exif=exif)
    raw_before = sha256_file(oriented)
    normalized_path = tmp_path / "working" / "oriented.png"

    result = normalize_image(oriented, normalized_path)

    assert (result.width, result.height) == (6, 12)
    assert result.exif_transposed
    assert result.assumed_srgb
    assert sha256_file(oriented) == raw_before
    with Image.open(normalized_path) as normalized:
        assert normalized.mode == "RGB"

    alpha_source = tmp_path / "alpha.png"
    alpha = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    alpha.putpixel((0, 0), (255, 0, 0, 255))
    alpha.save(alpha_source)
    alpha_result = normalize_image(
        alpha_source,
        tmp_path / "working" / "alpha.png",
        background=NeutralBackgroundPolicy(color=(100, 110, 120)),
    )
    with Image.open(alpha_result.output_path) as normalized:
        assert normalized.convert("RGB").getpixel((3, 3)) == (100, 110, 120)
    assert alpha_result.alpha_present
    assert alpha_result.alpha_fraction == pytest.approx(15 / 16)


def test_animated_webp_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "animated.webp"
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    try:
        frames[0].save(path, save_all=True, append_images=frames[1:], format="WEBP", duration=20)
    except OSError:
        pytest.skip("Pillow build has no animated WEBP encoder")
    with pytest.raises(AnimatedImageError):
        normalize_image(path, tmp_path / "working.png")
    assessment = assess_image(path)
    assert assessment.disposition is QualityDisposition.REJECT
    assert assessment.reasons[0].code == "animated_image"


def test_quality_metrics_tagger_reasons_and_style_gate(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (128, 128), "black").save(blank)
    assessment = assess_image(blank)
    assert assessment.disposition is QualityDisposition.REJECT
    assert "all_black" in {reason.code for reason in assessment.reasons}

    with_signals = add_tagger_quality_signals(
        assessment.model_copy(update={"disposition": "accept", "reasons": ()}),
        {"watermark": 0.91, "2girls": 0.82},
    )
    assert with_signals.disposition is QualityDisposition.WARN
    assert {reason.code for reason in with_signals.reasons} == {
        "text_watermark",
        "multi_subject",
    }

    accepted = QualityAssessment(
        path=tmp_path / "accepted.png",
        disposition="accept",
        metrics=ImageQualityMetrics(file_size_bytes=1, width=1024, height=1024),
    )
    gate = evaluate_quality_gate(
        [accepted] * 16,
        PresetKind.STYLE,
        thresholds=DatasetGateThresholds(hard_minimum=16, warning_below=30),
        content_diversity=0.1,
        dominant_character_ratio=0.9,
    )
    assert not gate.passed
    assert "low_style_content_diversity" in {issue.code for issue in gate.issues}
    assert "single_character_dominance" in {issue.code for issue in gate.issues}


def test_exact_and_phash_duplicates_select_explainable_representative(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    exact_copy = tmp_path / "b.png"
    near = tmp_path / "c.png"
    _pattern(first)
    shutil.copyfile(first, exact_copy)
    with Image.open(first) as image:
        changed = image.copy()
    changed.putpixel((0, 0), (255, 255, 255))
    changed.save(near)
    candidates = (
        DuplicateCandidate(
            asset_id="a",
            path=first,
            sha256=sha256_file(first),
            area=100,
            sharpness=10,
            compression_blockiness=0,
            alpha_fraction=0,
        ),
        DuplicateCandidate(
            asset_id="b",
            path=exact_copy,
            sha256=sha256_file(exact_copy),
            area=200,
            sharpness=20,
            compression_blockiness=0,
            alpha_fraction=0,
        ),
        DuplicateCandidate(
            asset_id="c",
            path=near,
            sha256=sha256_file(near),
            area=150,
            sharpness=15,
            compression_blockiness=0,
            alpha_fraction=0,
        ),
    )

    report = detect_duplicates(candidates, phash_max_distance=8)

    assert len(report.clusters) == 1
    cluster = report.clusters[0]
    assert cluster.kind is DuplicateKind.NEAR
    assert cluster.representative_id == "b"
    assert cluster.rejected_exact_ids == ("a",)
    assert cluster.rationale


@pytest.mark.parametrize(
    ("preset", "below", "boundary", "expected"),
    [
        (PresetKind.CHARACTER, 23, 24, 2),
        (PresetKind.STYLE, 39, 40, 4),
    ],
)
def test_validation_split_exact_boundaries_are_deterministic(
    preset: PresetKind, below: int, boundary: int, expected: int
) -> None:
    below_ids = [f"asset-{index}" for index in range(below)]
    assert not deterministic_validation_split(below_ids, preset).validation_enabled
    ids = [f"asset-{index}" for index in range(boundary)]
    first = deterministic_validation_split(ids, preset, seed=99)
    second = deterministic_validation_split(reversed(ids), preset, seed=99)
    assert first == second
    assert len(first.validation) == expected
    assert not set(first.training) & set(first.validation)


def test_diversity_reports_content_and_duplicate_dominance() -> None:
    tags = {
        "a": ["1girl", "character:alice", "white_dress", "standing", "outdoors"],
        "b": ["1boy", "character:bob", "school_uniform", "sitting", "indoors"],
        "c": ["1girl", "character:alice", "white_dress", "standing", "outdoors"],
    }
    report = analyze_diversity(tags, duplicate_cluster_ids={"a": "dup", "c": "dup"})
    assert report.content_diversity > 0
    assert report.dominant_character_ratio == pytest.approx(2 / 3)
    assert report.near_duplicate_cluster_dominance == pytest.approx(2 / 3)


def test_scan_keeps_discovered_files_when_directory_iteration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupted_directory = tmp_path / "interrupted"
    interrupted_directory.mkdir()
    discovered_before_error = interrupted_directory / "first.png"
    separate_input = tmp_path / "separate.png"
    _pattern(discovered_before_error)
    _pattern(separate_input, offset=17)
    original_iterdir = Path.iterdir

    def interrupted_iterdir(path: Path) -> Iterator[Path]:
        if path == interrupted_directory:
            yield discovered_before_error
            raise OSError("directory enumeration interrupted")
        yield from original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", interrupted_iterdir)

    result = scan_image_inputs((interrupted_directory, separate_input))

    assert set(result.files) == {
        discovered_before_error.resolve(),
        separate_input.resolve(),
    }
    assert len(result.issues) == 1
    assert result.issues[0].path == interrupted_directory.resolve()
    assert result.issues[0].code == "unreadable"
    assert "directory enumeration interrupted" in result.issues[0].message


def test_import_continues_after_pillow_decompression_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bomb = tmp_path / "bomb.png"
    valid = tmp_path / "valid.png"
    _pattern(bomb)
    _pattern(valid, offset=29)
    bomb_hash = sha256_file(bomb)
    valid_hash = sha256_file(valid)
    original_open = Image.open

    def open_with_bomb(path: Path, *args: object, **kwargs: object) -> Image.Image:
        if path == bomb:
            raise Image.DecompressionBombError("synthetic decompression bomb")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", open_with_bomb)
    layout = ProjectLayout(tmp_path / "project")

    result = ImmutableImportService(layout, project_id="project").import_paths((bomb, valid))

    assert result.imported_asset_ids == (valid_hash,)
    assert len(result.failures) == 1
    assert result.failures[0].source == bomb.resolve()
    assert result.failures[0].code == "ImageSafetyError"
    assert "synthetic decompression bomb" in result.failures[0].message
    assert sha256_file(bomb) == bomb_hash
    assert sha256_file(valid) == valid_hash
    assert {path.name for path in layout.raw.iterdir()} == {f"{valid_hash}.png"}
