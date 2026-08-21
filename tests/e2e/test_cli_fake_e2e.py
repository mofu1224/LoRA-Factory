from __future__ import annotations

from pathlib import Path

import pytest
import tomlkit
from PIL import Image

from lora_factory.application import service
from lora_factory.cli import run_fake_e2e
from lora_factory.config.models import PresetKind
from lora_factory.gpu.discovery import GpuDiscoveryError
from lora_factory.util.json import read_json


def test_cli_fake_e2e_uses_the_application_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_physical_gpu() -> None:
        raise GpuDiscoveryError("test host intentionally has no NVIDIA GPU")

    monkeypatch.setattr(service, "discover_nvidia_gpus", no_physical_gpu)
    workspace = tmp_path / "run"
    workspace.mkdir()

    report = run_fake_e2e(workspace, image_count=24)

    assert report.status == "READY"
    assert report.gpu_uuid == "GPU-00000000-0000-0000-0000-000000000001"
    assert report.source_hashes_unchanged
    assert report.final_model.is_file()
    assert report.preview.is_file()
    assert report.comparison.is_file()
    assert report.event_count > 10
    assert {"IMPORTING", "TRAINING", "PACKAGING", "READY"}.issubset(report.observed_stages)
    persisted = read_json(report.report_path)
    assert isinstance(persisted, dict)
    assert persisted["final_sha256"] == report.final_sha256
    expected_sizes = {
        "日本語.png": (1024, 1024),
        "Screenshot 01.png": (1920, 1080),
        "test (4).jpg": (1080, 1920),
        "1.webp": (1600, 1200),
        "very-long-unicode-name-これは長いUnicodeファイル名の安全性を確認するための画像です-001.png": (  # noqa: E501
            2048,
            1365,
        ),
        "extreme-aspect-warning.png": (2048, 512),
        "low-resolution-warning.png": (480, 360),
    }
    source_directory = report.fixture_root / "source images"
    for name, expected_size in expected_sizes.items():
        with Image.open(source_directory / name) as image:
            assert image.size == expected_size

    project_root = workspace / "app-data" / "projects" / report.project_id
    dataset_config = tomlkit.parse(
        (project_root / "configs" / "dataset.toml").read_text(encoding="utf-8")
    )
    dataset = dataset_config["datasets"][0]
    assert dataset["validation_split"] == pytest.approx(2 / 24)
    training_directory = Path(str(dataset["subsets"][0]["image_dir"]))
    assert len(tuple(training_directory.glob("*.png"))) == 24
    assert len(tuple(training_directory.glob("*.txt"))) == 24

    manifest = read_json(report.reproducibility_manifest)
    assert isinstance(manifest, dict)
    assert len(manifest["source_images"]) == 24
    assert all("original_filenames" not in source for source in manifest["source_images"])
    observed_categories = {
        category
        for decision in manifest["dataset_decisions"]
        for category in decision["categories"]
    }
    assert {"Accepted", "Warning", "Validation"} <= observed_categories
    assert (
        sum("Validation" in decision["categories"] for decision in manifest["dataset_decisions"])
        == 2
    )
    assert manifest["batch_probe"]["result"]["selected_batch_size"] == 1
    assert manifest["batch_probe"]["result"]["training_artifacts_created"] is False
    codex_images = manifest["codex_refinement"]
    assert codex_images["codex_image_profile"]["max_edge"] == 2048
    assert codex_images["codex_image_count"] == manifest["dataset_statistics"]["accepted_count"]
    assert len(codex_images["ordered_image_hashes"]) == codex_images["codex_image_count"]
    assert len(codex_images["batch_input_hashes"]) == 3
    assert codex_images["cleanup_status"] == "removed_after_each_call"
    serialized_manifest = report.reproducibility_manifest.read_text(encoding="utf-8")
    assert ".jpg" not in serialized_manifest.casefold()
    assert "c:/" not in serialized_manifest.casefold()
    assert "\\\\" not in serialized_manifest
    assert "users/" not in serialized_manifest.casefold()
    assert "home/" not in serialized_manifest.casefold()
    assert "raw/" not in serialized_manifest.casefold()
    assert "project/" not in serialized_manifest.casefold()
    assert "日本語.png" not in serialized_manifest


def test_cli_fake_e2e_records_and_uses_a_two_gpu_pool(tmp_path: Path) -> None:
    gpu_pool = (
        "GPU-aaaaaaaa-1111-2222-3333-444444444444",
        "GPU-bbbbbbbb-1111-2222-3333-444444444444",
    )
    workspace = tmp_path / "run"
    workspace.mkdir()

    report = run_fake_e2e(
        workspace,
        image_count=8,
        requested_gpu_uuids=gpu_pool,
    )

    assert report.status == "READY"
    assert report.gpu_uuid == gpu_pool[0]
    assert report.gpu_uuids == gpu_pool
    manifest = read_json(report.reproducibility_manifest)
    assert isinstance(manifest, dict)
    public_gpu_pool = ("gpu-1", "gpu-2")
    assert tuple(item["uuid"] for item in manifest["gpu_capabilities"]) == public_gpu_pool
    assert set(manifest["gpu_assignments"]["tagging"]) == set(public_gpu_pool)
    assert set(manifest["gpu_assignments"]["generated_tagging"]) == set(public_gpu_pool)


def test_style_fake_e2e_uses_reference_embedding_similarity_in_diversity(
    tmp_path: Path,
) -> None:
    gpu_pool = (
        "GPU-aaaaaaaa-1111-2222-3333-444444444444",
        "GPU-bbbbbbbb-1111-2222-3333-444444444444",
    )
    workspace = tmp_path / "style-run"
    workspace.mkdir()

    report = run_fake_e2e(
        workspace,
        preset=PresetKind.STYLE,
        image_count=18,
        requested_gpu_uuids=gpu_pool,
    )

    manifest = read_json(report.reproducibility_manifest)
    assert isinstance(manifest, dict)
    embedding = manifest["dataset_reference_embedding"]
    scores = tuple(embedding["centroid_similarity_by_asset"].values())
    assert len(scores) == 18
    assert manifest["dataset_diversity"]["style_consistency"] == pytest.approx(
        sum(scores) / len(scores)
    )
    assert set(manifest["gpu_assignments"]["reference_embedding"]) == {"gpu-1", "gpu-2"}
