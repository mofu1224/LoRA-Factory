"""Pure caption-refinement contracts between WD14, Runtime Codex, and training."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.caption.character_policy import (
    CHARACTER_FORBIDDEN,
    build_character_captions,
)
from lora_factory.caption.invariant import TagInput
from lora_factory.caption.parser import parse_caption_tags
from lora_factory.caption.style_policy import STYLE_FORBIDDEN, build_style_captions
from lora_factory.caption.tag_vocabulary import WD14TagVocabulary
from lora_factory.caption.tagger import TagScore
from lora_factory.config.models import PresetKind, validate_trigger_word
from lora_factory.config.validation import KNOWN_DANBOORU_TRIGGER_COLLISIONS
from lora_factory.dataset.ontology import (
    DEFAULT_ONTOLOGY,
    OntologyCategory,
    TagOntology,
)

_DRAFT_TRIGGER = "lfx_pending_trigger_7f3a"


class RefinementModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CaptionDraftAsset(RefinementModel):
    asset_id: str
    original_tags: tuple[str, ...]
    tag_confidences: dict[str, Annotated[float, Field(ge=0, le=1)]]
    fixed_tokens: tuple[str, ...] = ()
    effective_tags: tuple[str, ...]
    draft_caption: str


class CaptionDraftBatch(RefinementModel):
    preset: PresetKind
    assets: dict[str, CaptionDraftAsset]
    class_token: str | None = None
    invariants: tuple[str, ...] = ()
    keep_tokens: Annotated[int, Field(ge=1, le=2)]
    warnings: tuple[str, ...] = ()


class RefinementDecisionKind(StrEnum):
    KEEP = "keep"
    REPLACE = "replace"


class AssetRefinementProposal(RefinementModel):
    asset_id: str
    decision: RefinementDecisionKind
    effective_tags: tuple[str, ...]
    reason: Annotated[str, Field(min_length=1, max_length=500)]
    confidence: Annotated[float, Field(ge=0, le=1)]


class ValidatedAssetDecision(RefinementModel):
    proposal: AssetRefinementProposal
    accepted: bool
    rejection_reason: str | None = None
    rejected_tags: dict[str, str] = Field(default_factory=dict)


class ValidatedRefinement(RefinementModel):
    decisions: dict[str, ValidatedAssetDecision]
    effective_tags: dict[str, tuple[str, ...]]


class TriggerCandidate(RefinementModel):
    value: str
    reason: Annotated[str, Field(min_length=1, max_length=300)]


def _canonical_scores(
    values: TagInput,
    *,
    ontology: TagOntology,
) -> tuple[tuple[str, ...], dict[str, float]]:
    scores: dict[str, float] = {}
    categories: dict[str, str] = {}
    if isinstance(values, Mapping):
        items = tuple(TagScore(name=name, score=score) for name, score in values.items())
    else:
        items = tuple(values)
    for item in items:
        if not item.selected:
            continue
        canonical = ontology.canonicalize(item.name)
        previous = scores.get(canonical)
        if previous is None or item.score > previous:
            scores[canonical] = item.score
            categories[canonical] = item.model_category
    ordered = tuple(sorted(scores, key=lambda tag: (-scores[tag], tag)))
    return ordered, scores


def build_caption_drafts(
    preset: PresetKind,
    image_tags: Mapping[str, TagInput],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    duplicate_cluster_ids: Mapping[str, str] | None = None,
) -> CaptionDraftBatch:
    """Build policy-compliant caption drafts without a Trigger Word."""

    class_token: str | None = None
    invariants: tuple[str, ...] = ()
    if preset is PresetKind.CHARACTER:
        batch = build_character_captions(
            _DRAFT_TRIGGER,
            image_tags,
            ontology=ontology,
            duplicate_cluster_ids=duplicate_cluster_ids,
        )
        captions = batch.captions
        class_token = batch.class_token.token
        invariants = batch.invariants.selected_tags
        keep_tokens = batch.keep_tokens
        warnings = batch.warnings
    else:
        style_batch = build_style_captions(_DRAFT_TRIGGER, image_tags, ontology=ontology)
        captions = style_batch.captions
        keep_tokens = style_batch.keep_tokens
        warnings = style_batch.warnings

    assets: dict[str, CaptionDraftAsset] = {}
    for asset_id in sorted(image_tags):
        original_tags, confidences = _canonical_scores(image_tags[asset_id], ontology=ontology)
        caption_tokens = parse_caption_tags(captions[asset_id])
        if not caption_tokens or caption_tokens[0] != _DRAFT_TRIGGER:
            raise ValueError(f"Caption policy omitted the draft marker for asset {asset_id}")
        draft_tokens = caption_tokens[1:]
        fixed_tokens: tuple[str, ...] = ()
        if class_token is not None:
            class_display = ontology.display(class_token)
            if not draft_tokens or draft_tokens[0] != class_display:
                raise ValueError(f"Character class token is unstable for asset {asset_id}")
            fixed_tokens = (class_display,)
        variable_tokens = draft_tokens[len(fixed_tokens) :]
        effective_tags = tuple(ontology.canonicalize(tag) for tag in variable_tokens)
        assets[asset_id] = CaptionDraftAsset(
            asset_id=asset_id,
            original_tags=original_tags,
            tag_confidences=confidences,
            fixed_tokens=fixed_tokens,
            effective_tags=effective_tags,
            draft_caption=", ".join(draft_tokens),
        )
    return CaptionDraftBatch(
        preset=preset,
        assets=assets,
        class_token=class_token,
        invariants=invariants,
        keep_tokens=keep_tokens,
        warnings=warnings,
    )


def _validated_proposal_tags(
    draft: CaptionDraftAsset,
    proposal: AssetRefinementProposal,
    *,
    preset: PresetKind,
    ontology: TagOntology,
    vocabulary: WD14TagVocabulary,
    max_tags: int,
) -> tuple[tuple[str, ...], dict[str, str], str | None]:
    canonical = tuple(ontology.canonicalize(tag) for tag in proposal.effective_tags)
    if proposal.decision is RefinementDecisionKind.KEEP and canonical != draft.effective_tags:
        return draft.effective_tags, {}, "keep decision changed effective tags"

    original = set(draft.effective_tags)
    retained = set(canonical).intersection(original)
    effective: list[str] = []
    rejected: dict[str, str] = {}
    seen: set[str] = set()
    forbidden = set(CHARACTER_FORBIDDEN if preset is PresetKind.CHARACTER else STYLE_FORBIDDEN)
    for tag in canonical:
        if tag in seen:
            rejected[tag] = "tag duplicates an earlier effective tag"
            continue
        seen.add(tag)
        if not tag:
            rejected[tag] = "tag is empty"
            continue
        if any(separator in tag for separator in (",", ";", "|", "\n", "\r")):
            rejected[tag] = "tag contains a caption separator"
            continue
        if tag in original:
            effective.append(tag)
            continue
        if not vocabulary.contains(tag):
            rejected[tag] = "tag is absent from the pinned WD14 vocabulary"
            continue
        category = ontology.classify(tag, model_category=vocabulary.model_category(tag))
        if category in forbidden:
            rejected[tag] = f"tag uses a forbidden category: {category.value}"
            continue
        if len(effective) + len(retained - set(effective)) >= max_tags:
            rejected[tag] = f"tag exceeds the maximum of {max_tags} effective tags"
            continue
        effective.append(tag)
    return tuple(effective), rejected, None


def validate_refinement_proposals(
    drafts: CaptionDraftBatch,
    proposals: Sequence[AssetRefinementProposal],
    *,
    vocabulary: WD14TagVocabulary,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
    max_tags: int = 75,
) -> ValidatedRefinement:
    if max_tags < 1:
        raise ValueError("Maximum effective tag count must be positive")
    expected = set(drafts.assets)
    received = [proposal.asset_id for proposal in proposals]
    if len(received) != len(set(received)):
        raise ValueError("Codex response contains a duplicate asset")
    unknown = set(received) - expected
    if unknown:
        raise ValueError(f"Codex response contains an unknown asset: {sorted(unknown)[0]}")
    missing = expected - set(received)
    if missing:
        raise ValueError(f"Codex response is missing asset: {sorted(missing)[0]}")

    decisions: dict[str, ValidatedAssetDecision] = {}
    effective: dict[str, tuple[str, ...]] = {}
    by_id = {proposal.asset_id: proposal for proposal in proposals}
    for asset_id in sorted(expected):
        proposal = by_id[asset_id]
        tags, rejected_tags, error = _validated_proposal_tags(
            drafts.assets[asset_id],
            proposal,
            preset=drafts.preset,
            ontology=ontology,
            vocabulary=vocabulary,
            max_tags=max_tags,
        )
        decisions[asset_id] = ValidatedAssetDecision(
            proposal=proposal,
            accepted=error is None,
            rejection_reason=error,
            rejected_tags=rejected_tags,
        )
        effective[asset_id] = tags
    return ValidatedRefinement(decisions=decisions, effective_tags=effective)


def normalize_trigger_candidates(
    candidates: Sequence[TriggerCandidate],
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
) -> tuple[TriggerCandidate, ...]:
    normalized: list[TriggerCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            value = validate_trigger_word(candidate.value)
        except ValueError:
            continue
        key = value.casefold()
        if key in seen:
            continue
        canonical = ontology.canonicalize(value)
        if (
            canonical in KNOWN_DANBOORU_TRIGGER_COLLISIONS
            or ontology.classify(value) is not OntologyCategory.UNCLASSIFIED
        ):
            continue
        seen.add(key)
        normalized.append(candidate.model_copy(update={"value": value}))
        if len(normalized) == 5:
            break
    if len(normalized) < 3:
        raise ValueError("Codex must provide 3-5 unique valid Trigger Word candidates")
    return tuple(normalized)


def finalize_captions(
    drafts: CaptionDraftBatch,
    effective_tags: Mapping[str, Sequence[str]],
    trigger_word: str,
    *,
    ontology: TagOntology = DEFAULT_ONTOLOGY,
) -> dict[str, str]:
    trigger = validate_trigger_word(trigger_word)
    if set(effective_tags) != set(drafts.assets):
        raise ValueError("Effective tags must cover every draft asset exactly once")
    captions: dict[str, str] = {}
    for asset_id, draft in drafts.assets.items():
        tags = tuple(ontology.canonicalize(tag) for tag in effective_tags[asset_id])
        if len(tags) != len(set(tags)):
            raise ValueError(f"Effective tags contain duplicates for asset {asset_id}")
        captions[asset_id] = ", ".join(
            (trigger, *draft.fixed_tokens, *(ontology.display(tag) for tag in tags))
        )
    return captions


def _serialized_chunk_size(items: Sequence[CaptionDraftAsset]) -> int:
    payload = {"assets": [item.model_dump(mode="json") for item in items]}
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def chunk_refinement_assets(
    assets: Sequence[CaptionDraftAsset],
    *,
    max_bytes: int = 128 * 1024,
) -> tuple[tuple[CaptionDraftAsset, ...], ...]:
    if max_bytes < 256:
        raise ValueError("Refinement chunk limit is too small")
    chunks: list[tuple[CaptionDraftAsset, ...]] = []
    current: list[CaptionDraftAsset] = []
    for asset in sorted(assets, key=lambda item: item.asset_id):
        candidate = (*current, asset)
        if _serialized_chunk_size(candidate) <= max_bytes:
            current.append(asset)
            continue
        if not current:
            raise ValueError(f"Refinement asset exceeds chunk limit: {asset.asset_id}")
        chunks.append(tuple(current))
        current = [asset]
        if _serialized_chunk_size(current) > max_bytes:
            raise ValueError(f"Refinement asset exceeds chunk limit: {asset.asset_id}")
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def batch_refinement_assets(
    assets: Sequence[CaptionDraftAsset],
    *,
    max_items: int = 8,
    max_bytes: int = 128 * 1024,
) -> tuple[tuple[CaptionDraftAsset, ...], ...]:
    """Create stable Runtime Codex batches within its item and JSON-byte limits."""

    if not 1 <= max_items <= 8:
        raise ValueError("Refinement batch item limit must be between 1 and 8")
    if not 256 <= max_bytes <= 128 * 1024:
        raise ValueError("Refinement batch byte limit must be between 256 and 131072")

    batches: list[tuple[CaptionDraftAsset, ...]] = []
    current: list[CaptionDraftAsset] = []
    for asset in sorted(assets, key=lambda item: item.asset_id):
        candidate = (*current, asset)
        if len(candidate) <= max_items and _serialized_chunk_size(candidate) <= max_bytes:
            current.append(asset)
            continue
        if not current:
            raise ValueError(f"Refinement asset exceeds batch limit: {asset.asset_id}")
        batches.append(tuple(current))
        current = [asset]
        if _serialized_chunk_size(current) > max_bytes:
            raise ValueError(f"Refinement asset exceeds batch limit: {asset.asset_id}")
    if current:
        batches.append(tuple(current))
    return tuple(batches)
