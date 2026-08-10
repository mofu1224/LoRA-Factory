from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lora_factory.caption.audit import audit_captions
from lora_factory.caption.character_policy import build_character_captions
from lora_factory.caption.style_policy import build_style_captions
from lora_factory.caption.tagger import FakeTagger, RawTagStore, TagScore
from lora_factory.caption.wd14_backend import WD14BackendConfig, WD14OnnxTagger
from lora_factory.caption.writer import CaptionWriter
from lora_factory.config.models import PresetKind
from lora_factory.dataset.diversity import analyze_diversity
from lora_factory.util.hashing import sha256_file


def _tag_rows(count: int = 10) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for index in range(count):
        rows[f"asset-{index}"] = {
            "1girl": 0.96,
            "solo": 0.90,
            "white_hair": 0.91,
            "blue_eyes": 0.88,
            "white_dress" if index % 2 else "school_uniform": 0.82,
            "smile" if index % 2 else "closed_mouth": 0.75,
            "outdoors" if index % 3 else "indoors": 0.72,
            "masterpiece": 0.99,
            "character:alice": 0.93,
        }
    return rows


def test_character_policy_keeps_class_and_variables_but_binds_invariants() -> None:
    batch = build_character_captions("alice_token", _tag_rows())

    assert batch.keep_tokens == 2
    assert batch.class_token.token == "1girl"  # noqa: S105 - caption class, not a secret
    assert set(batch.invariants.selected_tags) == {"blue_eyes", "white_hair"}
    for caption in batch.captions.values():
        tags = caption.split(", ")
        assert tags[:2] == ["alice_token", "1girl"]
        assert "solo" in tags
        assert "white hair" not in tags
        assert "blue eyes" not in tags
        assert "masterpiece" not in tags
        assert "character:alice" not in tags
        assert "white dress" in tags or "school uniform" in tags

    audit = audit_captions(
        batch.captions,
        "alice_token",
        PresetKind.CHARACTER,
        class_token=batch.class_token.token,
        invariant_tags=batch.invariants.selected_tags,
    )
    assert audit.passed
    assert audit.class_token_consistency == 1.0
    assert audit.clothing_coverage == 1.0


def test_mixed_character_dataset_does_not_invent_fixed_class() -> None:
    tags = {
        f"asset-{index}": ({"1girl": 0.9} if index < 5 else {"1boy": 0.9}) for index in range(10)
    }
    batch = build_character_captions("mixed_token", tags)
    assert batch.class_token.token is None
    assert batch.keep_tokens == 1
    assert batch.captions["asset-0"] == "mixed_token, 1girl"
    assert batch.captions["asset-9"] == "mixed_token, 1boy"


def test_near_duplicate_frequency_does_not_bind_identity() -> None:
    tags = {
        **{f"duplicate-{index}": {"1girl": 0.9, "blue_hair": 0.9} for index in range(9)},
        "independent": {"1girl": 0.9, "brown_hair": 0.9},
    }
    cluster_ids = {f"duplicate-{index}": "same-shot" for index in range(9)}
    batch = build_character_captions("safe_token", tags, duplicate_cluster_ids=cluster_ids)
    assert "blue_hair" not in batch.invariants.selected_tags
    assert "blue hair" in batch.captions["duplicate-0"]
    assert any("near-duplicate" in warning for warning in batch.warnings)


def test_style_policy_retains_semantics_and_removes_style_and_source_names() -> None:
    tags = {
        "asset": {
            "1girl": 0.95,
            "blue_hair": 0.85,
            "school_uniform": 0.8,
            "sitting": 0.75,
            "classroom": 0.7,
            "watercolor": 0.9,
            "artist:someone": 0.9,
            "character:alice": 0.9,
            "copyright:series": 0.9,
            "best_quality": 0.9,
        }
    }
    batch = build_style_captions("example_style", tags)
    caption = batch.captions["asset"]
    assert caption.startswith("example_style, 1girl")
    for semantic in ("blue hair", "school uniform", "sitting", "classroom"):
        assert semantic in caption
    for forbidden in (
        "watercolor",
        "artist:someone",
        "character:alice",
        "copyright:series",
        "best quality",
    ):
        assert forbidden not in caption
    assert batch.keep_tokens == 1
    assert audit_captions(batch.captions, "example_style", PresetKind.STYLE).passed


def test_caption_qa_reports_trigger_order_duplicates_and_leaks() -> None:
    result = audit_captions(
        {"asset": "1girl, token, best quality, best quality"},
        "token",
        PresetKind.CHARACTER,
    )
    assert not result.passed
    assert {issue.code for issue in result.issues} == {
        "trigger_not_first",
        "duplicated_tags",
        "forbidden_tags",
    }


def test_wd14_model_category_removes_arbitrary_character_names_and_drives_diversity() -> None:
    image_tags = {
        f"asset-{index}": (
            TagScore(name="1girl", score=0.98),
            TagScore(
                name="hatsune_miku",
                score=0.97,
                model_category="character",
            ),
            TagScore(name="school_uniform", score=0.88),
        )
        for index in range(4)
    }

    character = build_character_captions("example_character", image_tags)
    style = build_style_captions("example_style", image_tags)
    diversity = analyze_diversity(image_tags)

    assert all("hatsune miku" not in caption for caption in character.captions.values())
    assert all("hatsune miku" not in caption for caption in style.captions.values())
    assert all("school uniform" in caption for caption in character.captions.values())
    assert all("school uniform" in caption for caption in style.captions.values())
    assert all("hatsune_miku" in removed for removed in character.removed_tags.values())
    assert all("hatsune_miku" in removed for removed in style.removed_tags.values())
    assert diversity.dominant_character_ratio == 1.0
    assert any("One character dominates" in warning for warning in diversity.warnings)


def test_fake_tagger_and_raw_json_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (16, 16), (20, 40, 60)).save(first)
    second.write_bytes(first.read_bytes())
    tagger = FakeTagger()
    first_result = tagger.tag(first, asset_id="first")
    second_result = tagger.tag(second, asset_id="second")
    assert first_result.tags == second_result.tags
    output = RawTagStore(tmp_path / "raw-tags").write(first_result)
    assert output.read_text(encoding="utf-8").startswith("{")


def test_real_wd14_adapter_is_lazy_and_requires_pinned_revision(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    tags = tmp_path / "selected_tags.csv"
    model.write_bytes(b"not loaded yet")
    tags.write_text("tag_id,name,category\n0,1girl,0\n", encoding="utf-8")
    backend = WD14OnnxTagger(
        WD14BackendConfig(
            revision="0123456789abcdef",
            model_path=model,
            tags_path=tags,
        )
    )
    assert backend.model_id == "SmilingWolf/wd-eva02-large-tagger-v3"
    assert backend._session is None


def test_real_wd14_adapter_halves_batch_on_onnx_oom(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (16, 16), (index, 20, 30)).save(path)
        paths.append(path)

    class OomSession:
        def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            batch = inputs["input"]
            if batch.shape[0] > 1:
                raise RuntimeError("CUDA out of memory")
            return [np.asarray([[0.95]], dtype=np.float32)]

    backend = WD14OnnxTagger(WD14BackendConfig(revision="0123456789abcdef"))
    backend._session = OomSession()
    backend._input_name = "input"
    backend._input_shape = (None, 16, 16, 3)
    backend._rows = (("1girl", "general"),)

    results = backend.tag_many(paths)

    assert len(results) == 3
    assert backend.successful_batch_size == 1
    assert all(result.tags[0].selected for result in results)


def test_real_wd14_adapter_fails_before_inference_without_configured_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    tags = tmp_path / "selected_tags.csv"
    model.write_bytes(b"provider check happens before model load")
    tags.write_text("tag_id,name,category\n0,1girl,0\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
    )
    backend = WD14OnnxTagger(
        WD14BackendConfig(
            revision="0123456789abcdef",
            model_path=model,
            tags_path=tags,
            providers=("CUDAExecutionProvider",),
        )
    )

    with pytest.raises(RuntimeError, match="None of the configured ONNX providers"):
        backend.tag_many((tmp_path / "unused.png",))


def test_caption_writer_never_changes_raw_and_is_idempotent(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset" / "raw"
    raw_root.mkdir(parents=True)
    raw = raw_root / "asset.png"
    Image.new("RGB", (8, 8), "red").save(raw)
    before = sha256_file(raw)
    writer = CaptionWriter(tmp_path / "dataset" / "captions", raw_root=raw_root)

    first = writer.write({"asset": "token, 1girl"}, raw_paths={"asset": raw})
    second = writer.write({"asset": "token, 1girl"}, raw_paths={"asset": raw})

    assert first.paths["asset"].read_text(encoding="utf-8") == "token, 1girl\n"
    assert second.unchanged == ("asset",)
    assert sha256_file(raw) == before
