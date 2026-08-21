from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_factory.application.refinement_review import RefinementReviewItem
from lora_factory.caption.refinement import (
    AssetRefinementProposal,
    CaptionDraftAsset,
    RefinementDecisionKind,
    TriggerCandidate,
    batch_refinement_assets,
    build_caption_drafts,
    chunk_refinement_assets,
    finalize_captions,
    normalize_trigger_candidates,
    validate_refinement_proposals,
)
from lora_factory.caption.tag_vocabulary import WD14TagVocabulary
from lora_factory.caption.tagger import TagScore
from lora_factory.config.models import PresetKind


def character_tags() -> dict[str, tuple[TagScore, ...]]:
    common = (
        TagScore(name="1girl", score=0.99),
        TagScore(name="blue_hair", score=0.95),
        TagScore(name="smile", score=0.90),
        TagScore(name="best_quality", score=0.98),
    )
    return {"asset-b": common, "asset-a": common}


def make_draft_assets(count: int, *, tag_count: int = 1) -> tuple[CaptionDraftAsset, ...]:
    return tuple(
        CaptionDraftAsset(
            asset_id=f"asset-{index:03d}",
            original_tags=(),
            tag_confidences={},
            effective_tags=tuple(f"tag_{tag_index}" for tag_index in range(tag_count)),
            draft_caption="draft",
        )
        for index in range(count)
    )


def proposal(
    asset_id: str,
    *,
    tags: tuple[str, ...],
    decision: RefinementDecisionKind = RefinementDecisionKind.REPLACE,
) -> AssetRefinementProposal:
    return AssetRefinementProposal(
        asset_id=asset_id,
        decision=decision,
        effective_tags=tags,
        reason="Visual review",
        confidence=0.9,
    )


def default_refinement_vocabulary() -> WD14TagVocabulary:
    return WD14TagVocabulary.from_names(
        (
            "smile",
            "blue_hair",
            "brown_hair",
            "green_hair",
            "red_hair",
            "looking_at_viewer",
        )
    )


def test_wd14_vocabulary_reads_utf8_bom_and_canonical_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("name,category\nblue_hair,0\nrating:safe,9\n", encoding="utf-8-sig")

    vocabulary = WD14TagVocabulary.from_csv(csv_path)

    assert vocabulary.contains("blue hair")
    assert vocabulary.model_category("rating:safe") == "rating"


@pytest.mark.parametrize(
    ("contents",),
    [
        ("name,category\n,0\n",),
        ("name,category\nblue_hair,0\nblue hair,4\n",),
    ],
)
def test_wd14_vocabulary_rejects_empty_or_duplicate_rows(tmp_path: Path, contents: str) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="empty or duplicate"):
        WD14TagVocabulary.from_csv(csv_path)


def test_wd14_vocabulary_maps_character_and_rating_categories(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "name,category\ncharacter:alice,4\nrating:safe,9\nsmile,0\n", encoding="utf-8"
    )

    vocabulary = WD14TagVocabulary.from_csv(csv_path)

    assert vocabulary.model_category("character:alice") == "character"
    assert vocabulary.model_category("rating:safe") == "rating"
    assert vocabulary.model_category("smile") == "general"


def test_image_proposal_keeps_known_addition_and_rejects_unknown_only() -> None:
    drafts = build_caption_drafts(
        PresetKind.STYLE, {"asset-a": (TagScore(name="smile", score=0.9),)}
    )

    validated = validate_refinement_proposals(
        drafts,
        (proposal("asset-a", tags=("smile", "blue_hair", "invented_fact")),),
        vocabulary=WD14TagVocabulary.from_names(("smile", "blue_hair")),
    )

    assert validated.effective_tags["asset-a"] == ("smile", "blue_hair")
    assert validated.decisions["asset-a"].accepted is True
    assert validated.decisions["asset-a"].rejected_tags == {
        "invented_fact": "tag is absent from the pinned WD14 vocabulary"
    }


def test_image_proposal_rejects_known_preset_forbidden_new_tag(tmp_path: Path) -> None:
    drafts = build_caption_drafts(
        PresetKind.STYLE, {"asset-a": (TagScore(name="smile", score=0.9),)}
    )
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("name,category\nsmile,0\ncharacter:alice,4\n", encoding="utf-8")

    validated = validate_refinement_proposals(
        drafts,
        (proposal("asset-a", tags=("smile", "character:alice")),),
        vocabulary=WD14TagVocabulary.from_csv(csv_path),
    )

    assert validated.effective_tags["asset-a"] == ("smile",)
    assert validated.decisions["asset-a"].rejected_tags == {
        "character:alice": "tag uses a forbidden category: character"
    }


def test_image_proposal_rejects_separator_duplicates_and_overflow_individually() -> None:
    drafts = build_caption_drafts(
        PresetKind.STYLE, {"asset-a": (TagScore(name="smile", score=0.9),)}
    )
    vocabulary = WD14TagVocabulary.from_names(
        ("smile", "blue_hair", "brown_hair", "green_hair", "red_hair")
    )

    validated = validate_refinement_proposals(
        drafts,
        (
            proposal(
                "asset-a",
                tags=("smile", "blue_hair", "blue_hair", "brown,hair", "green_hair", "red_hair"),
            ),
        ),
        vocabulary=vocabulary,
        max_tags=3,
    )

    assert validated.effective_tags["asset-a"] == ("smile", "blue_hair", "green_hair")
    assert validated.decisions["asset-a"].rejected_tags == {
        "blue_hair": "tag duplicates an earlier effective tag",
        "brown,hair": "tag contains a caption separator",
        "red_hair": "tag exceeds the maximum of 3 effective tags",
    }


def test_keep_tag_change_remains_a_response_contract_failure() -> None:
    drafts = build_caption_drafts(
        PresetKind.STYLE, {"asset-a": (TagScore(name="smile", score=0.9),)}
    )

    validated = validate_refinement_proposals(
        drafts,
        (proposal("asset-a", tags=("smile", "blue_hair"), decision=RefinementDecisionKind.KEEP),),
        vocabulary=WD14TagVocabulary.from_names(("smile", "blue_hair")),
    )

    assert validated.effective_tags["asset-a"] == ("smile",)
    assert validated.decisions["asset-a"].accepted is False
    assert validated.decisions["asset-a"].rejection_reason == "keep decision changed effective tags"


def test_refinement_requires_an_explicit_wd14_vocabulary() -> None:
    drafts = build_caption_drafts(
        PresetKind.STYLE, {"asset-a": (TagScore(name="smile", score=0.9),)}
    )

    with pytest.raises(TypeError, match="vocabulary"):
        validate_refinement_proposals(drafts, (proposal("asset-a", tags=("smile",)),))


def test_refinement_batches_stop_at_eight_assets() -> None:
    batches = batch_refinement_assets(make_draft_assets(17), max_items=8)

    assert [len(batch) for batch in batches] == [8, 8, 1]


def test_refinement_batches_are_stable_and_respect_item_and_byte_limits() -> None:
    assets = make_draft_assets(9, tag_count=20)

    batches = batch_refinement_assets(tuple(reversed(assets)), max_items=8, max_bytes=2_000)

    assert [asset.asset_id for batch in batches for asset in batch] == [
        asset.asset_id for asset in assets
    ]
    assert all(len(batch) <= 8 for batch in batches)
    assert all(
        len(
            json.dumps(
                {"assets": [asset.model_dump(mode="json") for asset in batch]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= 2_000
        for batch in batches
    )


def test_refinement_review_item_derives_added_and_removed_tags() -> None:
    item = RefinementReviewItem(
        asset_id="asset-a",
        original_tags=("smile", "blue_hair", "looking_at_viewer"),
        proposed_tags=("smile", "brown_hair"),
        effective_tags=("smile", "brown_hair"),
        draft_caption="smile, blue hair, looking at viewer",
        proposed_caption="smile, brown hair",
        reason="Visual review",
        confidence=0.9,
        factory_accepted=True,
        added_tags=("untrusted",),
        removed_tags=("untrusted",),
    )

    assert item.added_tags == ("brown_hair",)
    assert item.removed_tags == ("blue_hair", "looking_at_viewer")


def test_refinement_review_item_excludes_rejected_proposal_tags_from_diffs() -> None:
    item = RefinementReviewItem(
        asset_id="asset-a",
        original_tags=("smile",),
        proposed_tags=("smile", "blue_hair", "invented_fact"),
        effective_tags=("smile", "blue_hair"),
        draft_caption="smile",
        proposed_caption="smile, blue hair",
        reason="Visual review",
        confidence=0.9,
        factory_accepted=True,
    )

    assert item.added_tags == ("blue_hair",)
    assert item.removed_tags == ()


def test_refinement_validation_does_not_mutate_raw_tag_score_evidence() -> None:
    image_tags = character_tags()
    original = dict(image_tags)
    drafts = build_caption_drafts(PresetKind.CHARACTER, image_tags)

    validate_refinement_proposals(
        drafts,
        tuple(proposal(asset_id, tags=("smile", "blue_hair")) for asset_id in drafts.assets),
        vocabulary=WD14TagVocabulary.from_names(("smile", "blue_hair")),
    )

    assert image_tags == original


def test_build_character_drafts_has_no_trigger_and_preserves_fixed_class() -> None:
    batch = build_caption_drafts(PresetKind.CHARACTER, character_tags())

    assert batch.class_token == "1girl"  # noqa: S105 - Danbooru class tag, not a credential
    assert batch.keep_tokens == 2
    assert tuple(batch.assets) == ("asset-a", "asset-b")
    assert batch.assets["asset-a"].fixed_tokens == ("1girl",)
    assert not batch.assets["asset-a"].draft_caption.startswith(",")
    assert "best quality" not in batch.assets["asset-a"].draft_caption
    assert "blue_hair" in batch.invariants
    assert "blue hair" not in batch.assets["asset-a"].draft_caption


def test_build_style_drafts_removes_forbidden_style_and_artist_tags() -> None:
    batch = build_caption_drafts(
        PresetKind.STYLE,
        {
            "asset-a": (
                TagScore(name="smile", score=0.9),
                TagScore(name="artist:someone", score=0.9),
                TagScore(name="watercolor", score=0.9, model_category="style"),
            )
        },
    )

    assert batch.assets["asset-a"].effective_tags == ("smile",)


def test_refinement_rejects_only_invalid_new_tag_and_keeps_valid_asset() -> None:
    drafts = build_caption_drafts(PresetKind.CHARACTER, character_tags())
    proposals = (
        AssetRefinementProposal(
            asset_id="asset-a",
            decision=RefinementDecisionKind.REPLACE,
            effective_tags=("smile",),
            reason="Remove unstable appearance",
            confidence=0.8,
        ),
        AssetRefinementProposal(
            asset_id="asset-b",
            decision=RefinementDecisionKind.REPLACE,
            effective_tags=("invented_visual_fact",),
            reason="Invalid addition",
            confidence=0.9,
        ),
    )

    validated = validate_refinement_proposals(
        drafts, proposals, vocabulary=default_refinement_vocabulary()
    )

    assert validated.effective_tags["asset-a"] == ("smile",)
    assert validated.effective_tags["asset-b"] == ()
    assert validated.decisions["asset-a"].accepted is True
    assert validated.decisions["asset-b"].accepted is True
    assert validated.decisions["asset-b"].rejected_tags == {
        "invented_visual_fact": "tag is absent from the pinned WD14 vocabulary"
    }


@pytest.mark.parametrize(
    "proposals",
    [
        (),
        (
            AssetRefinementProposal(
                asset_id="asset-a",
                decision="keep",
                effective_tags=("blue_hair", "smile"),
                reason="same",
                confidence=1,
            ),
            AssetRefinementProposal(
                asset_id="asset-a",
                decision="keep",
                effective_tags=("blue_hair", "smile"),
                reason="duplicate",
                confidence=1,
            ),
        ),
        (
            AssetRefinementProposal(
                asset_id="unknown",
                decision="keep",
                effective_tags=(),
                reason="unknown",
                confidence=1,
            ),
        ),
    ],
)
def test_refinement_rejects_missing_duplicate_or_unknown_asset_contract(
    proposals: tuple[AssetRefinementProposal, ...],
) -> None:
    drafts = build_caption_drafts(PresetKind.CHARACTER, character_tags())

    with pytest.raises(ValueError, match="asset"):
        validate_refinement_proposals(drafts, proposals, vocabulary=default_refinement_vocabulary())


def test_trigger_candidates_are_deduplicated_and_common_tags_are_rejected() -> None:
    candidates = normalize_trigger_candidates(
        (
            TriggerCandidate(value="lfx_nova", reason="short"),
            TriggerCandidate(value="LFX_NOVA", reason="duplicate"),
            TriggerCandidate(value="1girl", reason="common tag"),
            TriggerCandidate(value="lfx_ember", reason="distinct"),
            TriggerCandidate(value="lfx_quartz", reason="distinct"),
            TriggerCandidate(value="lfx_cobalt", reason="distinct"),
        )
    )

    assert [item.value for item in candidates] == [
        "lfx_nova",
        "lfx_ember",
        "lfx_quartz",
        "lfx_cobalt",
    ]


def test_finalize_captions_puts_trigger_first_and_keeps_character_class_second() -> None:
    drafts = build_caption_drafts(PresetKind.CHARACTER, character_tags())
    proposals = tuple(
        AssetRefinementProposal(
            asset_id=asset_id,
            decision="replace",
            effective_tags=("smile",),
            reason="stable subset",
            confidence=0.8,
        )
        for asset_id in drafts.assets
    )
    validated = validate_refinement_proposals(
        drafts, proposals, vocabulary=default_refinement_vocabulary()
    )

    captions = finalize_captions(drafts, validated.effective_tags, "lfx_nova")

    assert captions == {
        "asset-a": "lfx_nova, 1girl, smile",
        "asset-b": "lfx_nova, 1girl, smile",
    }

    with pytest.raises(ValueError, match="collides with common Danbooru tag"):
        finalize_captions(drafts, validated.effective_tags, "1girl")


def test_chunking_is_stable_and_never_exceeds_serialized_limit() -> None:
    tags = {
        f"asset-{index:03d}": (
            TagScore(name=f"unknown_tag_{index}_{tag_index}", score=0.9) for tag_index in range(20)
        )
        for index in range(24)
    }
    drafts = build_caption_drafts(
        PresetKind.STYLE,
        {asset_id: tuple(values) for asset_id, values in tags.items()},
    )

    first = chunk_refinement_assets(tuple(drafts.assets.values()), max_bytes=3_000)
    second = chunk_refinement_assets(
        tuple(reversed(tuple(drafts.assets.values()))), max_bytes=3_000
    )

    assert first == second
    assert len(first) > 1
    assert [item.asset_id for chunk in first for item in chunk] == sorted(drafts.assets)
    assert all(
        len(
            json.dumps(
                {"assets": [item.model_dump(mode="json") for item in chunk]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= 3_000
        for chunk in first
    )
