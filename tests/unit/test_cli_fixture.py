from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import typer
from PIL import Image
from safetensors.numpy import save_file
from typer.testing import CliRunner

from lora_factory.application.service import repository_root
from lora_factory.cli import FakeE2EReport, _run_runtime_doctors, _select_gpu_uuids, app, main
from lora_factory.config.models import PresetKind
from lora_factory.dataset.quality import QualityDisposition, assess_image
from lora_factory.gpu.models import GpuBinding, GpuDevice
from lora_factory.model.inspector import inspect_sdxl_safetensors
from lora_factory.testing.fixture_factory import generate_fake_fixture
from lora_factory.training.profiles import default_preset_path


def test_cli_main_preserves_real_command_exit_status(tmp_path: Path) -> None:
    compatible = tmp_path / "compatible.safetensors"
    incompatible = tmp_path / "incompatible.safetensors"
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": np.zeros((1, 1), dtype=np.float32),
            "conditioner.embedders.1.model.text_projection": np.ones((1, 1), dtype=np.float32),
        },
        compatible,
        metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )
    save_file({"unrelated.weight": np.zeros((1,), dtype=np.float32)}, incompatible)

    assert main(["inspect-model", str(compatible), "--json"]) == 0
    assert main(["inspect-model", str(incompatible)]) == 2


def test_fake_fixture_is_valid_diverse_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "fixture"
    fixture = generate_fake_fixture(destination, image_count=18)

    assert fixture.image_count == 18
    assert len(fixture.image_paths) == 18
    assert len(set(fixture.source_sha256.values())) == 18
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
    assert set(expected_sizes) <= fixture.source_sha256.keys()
    for name, expected_size in expected_sizes.items():
        with Image.open(destination / "source images" / name) as image:
            assert image.size == expected_size
    assert len({assess_image(path).metrics.aspect_ratio for path in fixture.image_paths}) >= 5
    assert all(
        assess_image(path).disposition is not QualityDisposition.REJECT
        for path in fixture.image_paths
    )
    assert (
        assess_image(destination / "source images" / "extreme-aspect-warning.png").disposition
        is QualityDisposition.WARN
    )
    assert (
        assess_image(destination / "source images" / "low-resolution-warning.png").disposition
        is QualityDisposition.WARN
    )

    model = inspect_sdxl_safetensors(fixture.base_model)
    assert model.is_sdxl
    assert model.tensor_count == 3
    assert model.metadata["lora_factory_fixture"] == "true"

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        generate_fake_fixture(destination, image_count=8)


def test_inspect_model_cli_emits_machine_readable_json(tmp_path: Path) -> None:
    fixture = generate_fake_fixture(tmp_path / "fixture", image_count=8)
    result = CliRunner().invoke(app, ["inspect-model", str(fixture.base_model), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["is_sdxl"] is True
    assert payload["compatibility"] == "compatible"


def test_repository_root_uses_pyinstaller_bundle_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "_internal"
    bundle.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    assert repository_root() == bundle.resolve()
    assert default_preset_path(PresetKind.CHARACTER) == bundle / "presets" / "character.yaml"


def test_style_fake_e2e_rejects_too_few_images_before_creating_workspace(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "fake-e2e",
            "--workspace-root",
            str(workspace_root),
            "--preset",
            "style",
            "--image-count",
            "8",
        ],
    )

    assert result.exit_code == 2
    assert "must be at least 16" in result.output
    assert not workspace_root.exists()


def test_fake_e2e_cli_accepts_repeatable_synthetic_gpu_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_pool = (
        "GPU-aaaaaaaa-1111-2222-3333-444444444444",
        "GPU-bbbbbbbb-1111-2222-3333-444444444444",
    )
    captured: list[tuple[str, ...]] = []

    def fake_run(workspace: Path, **kwargs: Any) -> FakeE2EReport:
        selected = tuple(kwargs["requested_gpu_uuids"])
        captured.append(selected)
        return FakeE2EReport(
            status="READY",
            project_id="fake-project",
            run_id="fake-run",
            workspace=workspace,
            fixture_root=workspace / "fixture",
            output_directory=workspace / "output",
            final_model=workspace / "output" / "model.safetensors",
            final_sha256="0" * 64,
            preview=workspace / "output" / "preview.png",
            comparison=workspace / "output" / "comparison.png",
            event_count=1,
            observed_stages=("READY",),
            source_hashes_unchanged=True,
            gpu_uuid=selected[0],
            gpu_uuids=selected,
            reproducibility_manifest=workspace / "output" / "reproducibility_manifest.json",
            report_path=workspace / "fake-e2e-report.json",
        )

    monkeypatch.setattr("lora_factory.cli.run_fake_e2e", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "fake-e2e",
            "--workspace-root",
            str(tmp_path / "runs"),
            "--gpu-uuid",
            gpu_pool[0],
            "--gpu-uuid",
            gpu_pool[1],
            "--quiet",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [gpu_pool]
    payload = json.loads(result.stdout)
    assert payload["gpu_uuid"] == gpu_pool[0]
    assert payload["gpu_uuids"] == list(gpu_pool)


def test_live_smoke_deep_validates_every_requested_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_a = GpuDevice(
        uuid="GPU-aaaaaaaa-1111-2222-3333-444444444444",
        index=0,
        name="GPU A",
        total_vram_mb=16_384,
        free_vram_mb=12_000,
        capability_major=12,
        capability_minor=0,
    )
    gpu_b = gpu_a.model_copy(
        update={
            "uuid": "GPU-bbbbbbbb-1111-2222-3333-444444444444",
            "index": 1,
            "name": "GPU B",
            "free_vram_mb": 10_000,
        }
    )
    runtime = SimpleNamespace(
        installed=lambda: True,
        source_matches_manifest=lambda: True,
        layout=SimpleNamespace(
            python=tmp_path / "python.exe",
            sd_scripts=tmp_path / "sd-scripts",
        ),
    )
    controller = SimpleNamespace(runtime=runtime, discover_gpus=lambda: (gpu_a, gpu_b))
    calls: list[tuple[str, str]] = []

    def inspect(
        *, python_executable: Path, binding: GpuBinding, sd_scripts_root: Path
    ) -> SimpleNamespace:
        assert python_executable == runtime.layout.python
        assert sd_scripts_root == runtime.layout.sd_scripts
        calls.append((binding.uuid, binding.environment["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(gpu_uuid=binding.uuid, ready=True)

    monkeypatch.setattr("lora_factory.cli.inspect_runtime", inspect)
    reports = _run_runtime_doctors(cast(Any, controller), (gpu_b.uuid, gpu_a.uuid))

    assert tuple(report.gpu_uuid for report in reports) == (gpu_b.uuid, gpu_a.uuid)
    assert calls == [(gpu_b.uuid, "1"), (gpu_a.uuid, "0")]


def test_live_smoke_rejects_duplicate_gpu_pool() -> None:
    gpu = GpuDevice(
        uuid="GPU-aaaaaaaa-1111-2222-3333-444444444444",
        index=0,
        name="GPU A",
        total_vram_mb=16_384,
        free_vram_mb=12_000,
        capability_major=12,
        capability_minor=0,
    )
    controller = SimpleNamespace(discover_gpus=lambda: (gpu,))

    with pytest.raises(typer.BadParameter, match="must be unique"):
        _select_gpu_uuids(
            cast(Any, controller),
            (gpu.uuid, gpu.uuid),
            allow_fake_without_gpu=False,
        )


def test_live_smoke_rejects_requested_gpu_when_discovery_is_empty() -> None:
    controller = SimpleNamespace(discover_gpus=lambda: ())

    with pytest.raises(typer.BadParameter, match="No NVIDIA CUDA GPU"):
        _select_gpu_uuids(
            cast(Any, controller),
            ("GPU-aaaaaaaa-1111-2222-3333-444444444444",),
            allow_fake_without_gpu=False,
        )
