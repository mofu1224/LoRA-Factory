from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lora_factory.config.models import PresetKind
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
from lora_factory.util.hashing import sha256_file


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
