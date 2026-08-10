from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tomlkit
from PIL import Image
from safetensors.numpy import save_file
from sqlalchemy import func, select

from lora_factory.application.service import LoRAFactoryController
from lora_factory.config.models import (
    AppSettings,
    BackendMode,
    DestinationConfig,
    DestinationKind,
    PresetKind,
    ProjectConfig,
)
from lora_factory.sampling.prompts import load_benchmark_prompts
from lora_factory.storage.database import Database
from lora_factory.storage.orm import (
    AssetRow,
    CaptionRow,
    CheckpointRow,
    CodexCallRow,
    DatasetSplitRow,
    EvaluationRow,
    SampleRow,
    TagResultRow,
    TrainingAttemptRow,
    TrainingPlanRow,
)
from lora_factory.util.hashing import sha256_file


def _fake_sdxl(path: Path) -> None:
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": np.zeros((1, 1), dtype=np.float32),
            "conditioner.embedders.1.model.text_projection": np.ones((1, 1), dtype=np.float32),
        },
        path,
        metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )


def _images(root: Path) -> None:
    root.mkdir(parents=True)
    for index in range(8):
        generator = np.random.default_rng(1000 + index)
        pixels = generator.integers(0, 256, size=(544, 640, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = np.clip(
            pixels[:, :, index % 3].astype(np.int16) + index * 7, 0, 255
        ).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(root / f"画像 {index + 1}.png")


def test_full_fake_pipeline_uses_real_formats_and_stage_graph(tmp_path: Path) -> None:
    source = tmp_path / "入力 画像"
    _images(source)
    base_model = tmp_path / "tiny SDXL.safetensors"
    _fake_sdxl(base_model)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
        destinations=[
            DestinationConfig(
                kind=DestinationKind.A1111,
                root=tmp_path / "stable-diffusion-webui",
            )
        ],
    )
    controller = LoRAFactoryController(settings)
    events: list[dict[str, object]] = []
    result = controller.run_pipeline(
        ProjectConfig(
            project_id="Fake 統合",
            lora_name="Fake 統合",
            preset=PresetKind.CHARACTER,
            trigger_token="lfx_person",  # noqa: S106 - domain trigger, not a credential
            base_model=base_model,
            input_paths=(source,),
            selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
            output_root=tmp_path / "output",
            backend_mode=BackendMode.FAKE,
        ),
        events.append,
    )

    assert result["status"] == "READY"
    assert result["raw_integrity_verified"] is True
    final_model = Path(str(result["final_model"]))
    assert final_model.is_file() and final_model.suffix == ".safetensors"
    assert Path(str(result["preview"])).is_file()
    assert Path(str(result["comparison"])).is_file()
    with Image.open(result["preview"]) as preview:
        assert preview.format == "PNG"
        assert preview.size == (512, 512)
    manifest = json.loads(
        (final_model.parent / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["backend_manifest"]["sd_scripts"]["commit"]
    assert manifest["codex_pretrain_audit"]["fallback_used"] is True
    assert len(manifest["source_images"]) == 8
    assert all(len(item["sha256"]) == 64 for item in manifest["source_images"])
    assert len(manifest["dataset_decisions"]) == 8
    assert (
        manifest["dataset_statistics"]["diversity_score"]
        == manifest["dataset_diversity"]["content_diversity"]
    )
    assert manifest["validation_split"]["enabled"] is False
    assert "training_command_arguments" not in manifest
    assert manifest["gpu_capabilities"][0]["uuid"] == "gpu-1"
    assert set(manifest["gpu_assignments"]) == {
        "tagging",
        "reference_embedding",
        "training",
        "screening",
        "weight_sweep",
        "multi_seed_validation",
        "generated_tagging",
        "image_embedding",
    }
    assert {
        gpu_uuid for assignments in manifest["gpu_assignments"].values() for gpu_uuid in assignments
    } == {"gpu-1"}
    evaluation = json.loads((final_model.parent / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["image_embedding"]["model_id"] == ("lora-factory/fake-image-embedding")
    assert evaluation["image_embedding"]["dimension"] == 192
    assert all(item["evidence"]["semantic_embedding_used"] for item in evaluation["metrics"])
    reference_embedding = manifest["dataset_reference_embedding"]
    assert reference_embedding["status"] == "complete"
    assert reference_embedding["model_id"] == "lora-factory/fake-image-embedding"
    assert reference_embedding["dimension"] == 192
    assert len(reference_embedding["centroid_similarity_by_asset"]) == 8
    assert reference_embedding["path"] == "embeddings.json"
    assert set(reference_embedding["outlier_asset_ids"]) <= {
        item["asset_id"] for item in manifest["dataset_decisions"] if item["accepted"]
    }
    assert set(manifest["sampling"]["prompt_ids"]) == {
        prompt.id for prompt in load_benchmark_prompts(PresetKind.CHARACTER)
    }
    assert manifest["destination_copies"] == []

    copied = controller.copy_output("Fake 統合", DestinationKind.A1111.value)
    assert Path(str(copied["destination"])).is_file()
    manifest = json.loads(
        (final_model.parent / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["destination_copies"][-1]["sha256"] == copied["sha256"]
    alternatives = result["alternatives"]
    assert isinstance(alternatives, dict) and alternatives
    promoted_id = next(iter(alternatives))
    promoted = controller.promote_alternative("Fake 統合", promoted_id)
    assert promoted["sha256"]
    manifest = json.loads(
        (final_model.parent / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["manual_selected_checkpoint_id"] == promoted_id
    assert manifest["final_lora"]["sha256"] == promoted["sha256"]
    project_root = settings.projects_root / "Fake 統合"
    assert (project_root / "run_snapshot.yaml").is_file()
    assert (project_root / "runs" / str(result["run_id"]) / "run_snapshot.yaml").is_file()
    dataset_config = tomlkit.parse(
        (project_root / "configs" / "dataset.toml").read_text(encoding="utf-8")
    )
    assert "validation_split" not in dataset_config["datasets"][0]
    raw_files = list((project_root / "dataset" / "raw").glob("*"))
    assert len(raw_files) == 8
    assert all(path.name.startswith(path.stem[:16]) for path in raw_files)
    completed_stages = {
        str(event["stage"]) for event in events if event["event_type"] == "stage_completed"
    }
    assert {"IMPORTING", "TAGGING", "TRAINING", "PACKAGING", "READY"} <= completed_stages

    database = Database(project_root / "state.sqlite3")
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(AssetRow)) == 16
        assert session.scalar(select(func.count()).select_from(TagResultRow)) == 8
        assert session.scalar(select(func.count()).select_from(CaptionRow)) == 8
        assert session.scalar(select(func.count()).select_from(DatasetSplitRow)) == 8
        assert session.scalar(select(func.count()).select_from(TrainingPlanRow)) == 1
        assert session.scalar(select(func.count()).select_from(TrainingAttemptRow)) == 1
        assert session.scalar(select(func.count()).select_from(CheckpointRow)) >= 1
        assert session.scalar(select(func.count()).select_from(SampleRow)) >= 1
        observed_prompt_ids = set(session.scalars(select(SampleRow.prompt_id)))
        expected_prompt_ids = {prompt.id for prompt in load_benchmark_prompts(PresetKind.CHARACTER)}
        assert observed_prompt_ids == expected_prompt_ids
        assert session.scalar(select(func.count()).select_from(EvaluationRow)) >= 1
        assert session.scalar(select(func.count()).select_from(CodexCallRow)) == 4


def test_pre_snapshot_include_override_can_restore_a_quality_reject(tmp_path: Path) -> None:
    source = tmp_path / "input"
    _images(source)
    rejected_source = source / "manual include.png"
    Image.new("RGB", (512, 512), (0, 0, 0)).save(rejected_source)
    rejected_asset_id = sha256_file(rejected_source)
    base_model = tmp_path / "base.safetensors"
    _fake_sdxl(base_model)
    settings = AppSettings(
        projects_root=tmp_path / "projects",
        managed_runtime_root=tmp_path / "runtime",
        codex_runtime_root=tmp_path / "codex",
    )
    controller = LoRAFactoryController(settings)
    config = ProjectConfig(
        project_id="include-override",
        lora_name="include_override",
        preset=PresetKind.CHARACTER,
        trigger_token="include_override_token",  # noqa: S106 - domain trigger
        base_model=base_model,
        input_paths=(source,),
        selected_gpu_uuids=("GPU-00000000-0000-0000-0000-000000000001",),
        output_root=tmp_path / "output",
        backend_mode=BackendMode.FAKE,
    )
    controller.projects.create(config)
    controller.set_dataset_override(
        config.project_id,
        {"asset_id": rejected_asset_id, "included": True},
    )

    events: list[dict[str, object]] = []
    result = controller.run_pipeline(config, events.append)

    assert result["status"] == "READY"
    manifest = json.loads(
        (Path(str(result["final_model"])).parent / "reproducibility_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    restored = next(
        item for item in manifest["dataset_decisions"] if item["asset_id"] == rejected_asset_id
    )
    assert restored["accepted"] is True
    assert "Warning" in restored["categories"]
    assert manifest["dataset_statistics"]["accepted_count"] == 9
    review_event = next(event for event in events if event["event_type"] == "dataset_review")
    review_details = review_event["details"]
    assert isinstance(review_details, dict)
    restored_review = next(
        item for item in review_details["items"] if item["asset_id"] == rejected_asset_id
    )
    assert restored_review["bucket"] == "512x512"
    assert review_details["gate"]["accepted_count"] == 9
    database = Database(settings.projects_root / config.project_id / "state.sqlite3")
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(TagResultRow)) == 9
        assert session.scalar(select(func.count()).select_from(CaptionRow)) == 9
