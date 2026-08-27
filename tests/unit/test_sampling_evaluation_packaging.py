from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import save_file

from lora_factory.config.models import DestinationKind, PresetKind
from lora_factory.evaluation.metrics import (
    SampleObservation,
    aggregate_candidate_signals,
    inspect_generated_image,
    normalize_values,
)
from lora_factory.evaluation.models import CandidateMetrics
from lora_factory.evaluation.ranking import rank_candidates
from lora_factory.packaging.destinations import copy_without_overwrite
from lora_factory.packaging.finalizer import (
    CheckpointCandidate,
    FinalizationRequest,
    Finalizer,
)
from lora_factory.sampling.backend import SampleRequest
from lora_factory.sampling.fake_backend import FakeSampler
from lora_factory.sampling.grid import render_grid
from lora_factory.sampling.matrix import screening_matrix
from lora_factory.sampling.prompts import load_benchmark_prompts
from lora_factory.sampling.sd_scripts_backend import SdScriptsSampler
from lora_factory.util.hashing import sha256_file


def make_checkpoint(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"lora_unet_test.weight": np.full((2, 2), value, dtype=np.float32)},
        path,
        metadata={"ss_network_module": "networks.lora"},
    )
    return path


def test_versioned_benchmark_prompts_cover_both_presets() -> None:
    character = load_benchmark_prompts(PresetKind.CHARACTER)
    style = load_benchmark_prompts(PresetKind.STYLE)

    assert len(character) >= 8
    assert len(style) >= 8
    assert {item.id for item in character} >= {"identity_portrait", "outfit_change"}
    assert {item.id for item in style} >= {"landscape", "complex_composition"}


def test_fake_sampler_writes_decodable_png_and_sidecar(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoint.safetensors", 1.0)
    request = SampleRequest(
        checkpoint_id="epoch-1",
        checkpoint_path=checkpoint,
        base_model_path=tmp_path / "base.safetensors",
        prompt_id="portrait",
        prompt="trigger, portrait",
        seed=123,
        weight=0.8,
        width=256,
        height=320,
    )

    result = FakeSampler().sample(request, tmp_path / "samples")

    assert result.success
    with Image.open(result.image_path) as image:
        assert image.size == (256, 320)
        assert image.format == "PNG"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["seed"] == 123
    assert metadata["weight"] == 0.8
    assert metadata["backend"] == "fake-sampler/1"


def test_matrix_is_three_pass_and_hard_capped() -> None:
    matrix = screening_matrix(
        ["one", "two", "three", "four"],
        [f"prompt-{index}" for index in range(8)],
        max_images=80,
    )

    assert len(matrix) == 80
    assert {cell.pass_number for cell in matrix} == {1, 2}
    assert matrix[0].weight == 0.8


def test_real_sampler_arguments_match_pinned_sdxl_parser_and_keep_prompt_single_arg(
    tmp_path: Path,
) -> None:
    sampler = SdScriptsSampler(
        python_executable=tmp_path / "python.exe",
        sd_scripts_root=tmp_path / "sd-scripts",
        gpu_uuid="GPU-12345678-abcd",
        commit="6721028c79ee85a78b3a06dfd8954dae310a1cce",
    )
    request = SampleRequest(
        checkpoint_id="epoch-1",
        checkpoint_path=tmp_path / "lora.safetensors",
        base_model_path=tmp_path / "base.safetensors",
        prompt_id="portrait",
        prompt="trigger, portrait",
        negative_prompt="low quality, blurry",
        seed=99,
        weight=0.8,
    )

    arguments = sampler.build_arguments(request, tmp_path / "samples")

    assert arguments[:2] == [
        str((tmp_path / "python.exe").resolve()),
        str((tmp_path / "sd-scripts" / "sdxl_gen_img.py").resolve()),
    ]
    prompt = arguments[arguments.index("--prompt") + 1]
    assert prompt == "trigger, portrait --n low quality, blurry"
    assert arguments[arguments.index("--network_weights") + 1] == str(request.checkpoint_path)
    assert arguments[arguments.index("--network_mul") + 1] == "0.8"
    assert "--bf16" in arguments


def test_technical_metrics_detect_blank_and_normalize_constant_values(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (32, 32), "black").save(blank)

    technical = inspect_generated_image(blank)

    assert technical.valid
    assert technical.black_or_blank
    assert normalize_values({"a": 4.0, "b": 4.0}) == {"a": 0.5, "b": 0.5}


def test_candidate_signals_use_actual_pixels_tags_and_failure_rate(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"image-{index}.png"
        if index == 2:
            Image.new("RGB", (64, 64), "black").save(path)
        else:
            generator = np.random.default_rng(120 + index)
            Image.fromarray(
                generator.integers(0, 256, size=(64, 64, 3), dtype=np.uint8),
                mode="RGB",
            ).save(path)
        paths.append(path)

    signals = aggregate_candidate_signals(
        reference_paths=[paths[0]],
        observations=[
            SampleObservation(
                path=paths[0],
                prompt_id="portrait",
                prompt="demo_token, portrait, looking at viewer",
                seed=1,
                weight=0.8,
                success=True,
                tags=("portrait", "looking_at_viewer", "blue_hair"),
            ),
            SampleObservation(
                path=paths[1],
                prompt_id="outdoors",
                prompt="demo_token, outdoors",
                seed=2,
                weight=0.9,
                success=True,
                tags=("outdoors", "blue_hair"),
            ),
            SampleObservation(
                path=paths[2],
                prompt_id="portrait",
                prompt="demo_token, portrait",
                seed=3,
                weight=1.0,
                success=True,
            ),
        ],
        trigger_token="demo_token",  # noqa: S106 - domain trigger, not a credential.
        invariant_tags=("blue_hair",),
    )

    assert signals.generation_failure_rate == pytest.approx(1 / 3)
    assert signals.prompt_compliance == 1.0
    assert 0 <= signals.reference_similarity <= 1
    assert signals.evidence["reference_count"] == 1
    assert signals.evidence["usable_sample_count"] == 2


def test_candidate_signals_blend_pinned_learned_embeddings_with_pixel_evidence(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    generated = tmp_path / "generated.png"
    Image.new("RGB", (64, 64), "red").save(reference)
    generated_pixels = np.random.default_rng(99).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    generated_pixels[:, :, 2] = 255
    Image.fromarray(generated_pixels, mode="RGB").save(generated)
    learned = {
        reference.resolve(): np.asarray([1.0, 0.0], dtype=np.float32),
        generated.resolve(): np.asarray([1.0, 0.0], dtype=np.float32),
    }

    signals = aggregate_candidate_signals(
        reference_paths=[reference],
        observations=[
            SampleObservation(
                path=generated,
                prompt_id="portrait",
                prompt="demo_token, portrait",
                seed=1,
                weight=0.8,
                success=True,
                tags=("portrait",),
            )
        ],
        trigger_token="demo_token",  # noqa: S106 - domain trigger, not a credential.
        learned_embeddings=learned,
        embedding_model_id="openai/clip-vit-large-patch14",
        embedding_revision="a" * 40,
    )

    assert signals.evidence["semantic_embedding_used"] is True
    assert signals.evidence["semantic_embedding_model"] == "openai/clip-vit-large-patch14"
    assert signals.evidence["semantic_reference_similarity"] == pytest.approx(1.0)
    assert signals.reference_similarity == pytest.approx(
        0.5 * float(signals.evidence["pixel_reference_similarity"]) + 0.5
    )


def test_candidate_signals_reject_partial_learned_embedding_evidence(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    generated = tmp_path / "generated.png"
    Image.new("RGB", (64, 64), "red").save(reference)
    generated_pixels = np.random.default_rng(100).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    generated_pixels[:, :, 2] = 255
    Image.fromarray(generated_pixels, mode="RGB").save(generated)

    with pytest.raises(ValueError, match="missing for image"):
        aggregate_candidate_signals(
            reference_paths=[reference],
            observations=[
                SampleObservation(
                    path=generated,
                    prompt_id="portrait",
                    prompt="demo_token, portrait",
                    seed=1,
                    weight=0.8,
                    success=True,
                )
            ],
            trigger_token="demo_token",  # noqa: S106 - domain trigger, not a credential.
            learned_embeddings={reference.resolve(): np.asarray([1.0, 0.0], dtype=np.float32)},
            embedding_model_id="openai/clip-vit-large-patch14",
            embedding_revision="a" * 40,
        )


def test_ranking_uses_preset_signal_and_excludes_corrupted_candidate() -> None:
    good = CandidateMetrics(
        checkpoint_id="good",
        identity=0.9,
        style_similarity=0.2,
        prompt_compliance=0.8,
        flexibility=0.7,
        consistency=0.8,
        validation=0.7,
        technical=1.0,
        visual_heuristic=0.6,
    )
    corrupt = good.model_copy(update={"checkpoint_id": "corrupt", "corrupted": True})

    ranked = rank_candidates([corrupt, good], PresetKind.CHARACTER)

    assert ranked[0].checkpoint_id == "good"
    assert ranked[0].hard_gate_passed
    assert ranked[1].score == 0
    assert not ranked[1].hard_gate_passed


def test_finalizer_creates_complete_hash_verified_output_and_safe_destination(
    tmp_path: Path,
) -> None:
    selected_path = make_checkpoint(tmp_path / "checkpoints" / "epoch1.safetensors", 1.0)
    alternative_path = make_checkpoint(tmp_path / "checkpoints" / "epoch2.safetensors", 2.0)
    preview = tmp_path / "preview-source.png"
    comparison = tmp_path / "comparison-source.png"
    Image.new("RGB", (64, 64), "navy").save(preview)
    render_grid([(preview, "epoch 1")], comparison, columns=1, cell_size=(64, 64))
    base_model = tmp_path / "private-personal-model.safetensors"
    base_model.write_bytes(b"private base model")
    base_model_sha256 = sha256_file(base_model)
    output = tmp_path / "output" / "Snow"
    request = FinalizationRequest(
        output_directory=output,
        lora_name="Snow",
        trigger_token="snow_person",  # noqa: S106 - domain trigger token, not a password.
        preset=PresetKind.CHARACTER,
        base_model=base_model,
        base_model_sha256=base_model_sha256,
        selected=CheckpointCandidate(
            checkpoint_id="epoch1",
            path=selected_path,
            score=0.91,
            recommended_weight=0.8,
        ),
        alternatives=(
            CheckpointCandidate(
                checkpoint_id="epoch2",
                path=alternative_path,
                score=0.82,
                recommended_weight=0.7,
            ),
        ),
        preview=preview,
        comparison=comparison,
        resolved_config={
            "resolution": 1024,
            "base_model": r"E:\Private Models\base.safetensors",
            "input_paths": [r"C:\Users\Alice\Pictures\private-name.png"],
            "selected_gpu_uuids": ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
        },
        evaluation={
            "ranking": ["epoch1", "epoch2"],
            "path": r"C:\Users\Alice\evaluation.json",
            "gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        },
        training_info={
            "resolution": 1024,
            "network_dim": 32,
            "network_alpha": 16,
            "training_image_count": 24,
            "recommended_range": "0.70-0.90",
        },
        reproducibility={
            "source_image_hashes": ["a" * 64],
            "source_images": [
                {
                    "sha256": "a" * 64,
                    "original_filenames": ["private-name.png"],
                }
            ],
            "training_command_arguments": [
                "--train_data_dir",
                r"C:\Users\Alice\Pictures",
            ],
        },
        warnings=(r"Backend warning at C:\Users\Alice\runtime\stderr.log",),
    )

    result = Finalizer().finalize(request)

    expected = {
        "Snow.safetensors",
        "preview.png",
        "comparison.png",
        "README.txt",
        "training_info.json",
        "evaluation.json",
        "resolved_config.yaml",
        "reproducibility_manifest.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert result.sha256 == sha256_file(selected_path)
    assert result.alternative_models[0].name == "candidate_2.safetensors"
    assert "Trigger: snow_person" in (output / "README.txt").read_text(encoding="utf-8")
    manifest = json.loads((output / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_lora"]["sha256"] == result.sha256
    assert manifest["source_images"] == [{"sha256": "a" * 64}]
    public_metadata = "\n".join(
        (output / name).read_text(encoding="utf-8")
        for name in (
            "README.txt",
            "training_info.json",
            "evaluation.json",
            "resolved_config.yaml",
            "reproducibility_manifest.json",
        )
    )
    assert "C:\\Users\\Alice" not in public_metadata
    assert "E:\\Private Models" not in public_metadata
    assert "private-name.png" not in public_metadata
    assert "private-personal-model.safetensors" not in public_metadata
    assert "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" not in public_metadata
    assert "gpu-1" in public_metadata
    assert "Base model: sha256:" + base_model_sha256 in public_metadata
    assert "Backend warning at <local-path>" in public_metadata

    a1111_root = tmp_path / "a1111"
    first = copy_without_overwrite(
        result.final_model,
        configured_root=a1111_root,
        kind=DestinationKind.A1111,
    )
    second = copy_without_overwrite(
        result.final_model,
        configured_root=a1111_root,
        kind=DestinationKind.A1111,
    )
    assert first.destination.name == "Snow.safetensors"
    assert second.destination.name == "Snow_v2.safetensors"
    assert second.versioned
    assert first.sha256 == second.sha256 == result.sha256


def test_finalizer_refuses_to_overwrite_different_existing_artifact(tmp_path: Path) -> None:
    source = make_checkpoint(tmp_path / "source.safetensors", 1.0)
    destination = make_checkpoint(tmp_path / "destination.safetensors", 2.0)

    with pytest.raises(FileExistsError):
        Finalizer._atomic_verified_copy(source, destination)


def test_finalizer_rejects_invalid_or_mismatched_base_model_digest(tmp_path: Path) -> None:
    base_model = tmp_path / "base.safetensors"
    base_model.write_bytes(b"base model")
    request = FinalizationRequest.model_construct(
        base_model=base_model,
        base_model_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="does not match"):
        Finalizer._validate_base_model(request)

    malformed = request.model_copy(update={"base_model_sha256": "C:\\private\\model"})
    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        Finalizer._validate_base_model(malformed)
