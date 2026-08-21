"""Pinned WD14 tag vocabulary loading and membership checks."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from lora_factory.caption.tagger import FAKE_WD14_VOCABULARY
from lora_factory.dataset.ontology import normalize_tag_key


class WD14TagVocabulary(BaseModel):
    """Canonical tags and model categories from a pinned WD14 CSV."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: frozenset[str]
    model_categories: dict[str, Literal["general", "character", "rating"]]

    @classmethod
    def from_csv(cls, path: Path) -> WD14TagVocabulary:
        categories: dict[str, Literal["general", "character", "rating"]] = {}
        with path.resolve(strict=True).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = normalize_tag_key((row.get("name") or row.get("tag") or "").strip())
                if not name or name in categories:
                    raise ValueError("WD14 CSV contains an empty or duplicate tag")
                category_id = (row.get("category") or "0").strip()
                categories[name] = (
                    "rating"
                    if category_id == "9"
                    else "character"
                    if category_id == "4"
                    else "general"
                )
        if not categories:
            raise ValueError("WD14 CSV is empty")
        return cls(tags=frozenset(categories), model_categories=categories)

    @classmethod
    def from_names(cls, names: Iterable[str]) -> WD14TagVocabulary:
        canonical = tuple(normalize_tag_key(name) for name in names)
        if (
            not canonical
            or any(not name for name in canonical)
            or len(set(canonical)) != len(canonical)
        ):
            raise ValueError("WD14 vocabulary names must be non-empty and unique")
        model_categories: dict[str, Literal["general", "character", "rating"]] = dict.fromkeys(
            canonical, "general"
        )
        return cls(
            tags=frozenset(canonical),
            model_categories=model_categories,
        )

    @classmethod
    def fake(cls) -> WD14TagVocabulary:
        return cls.from_names(FAKE_WD14_VOCABULARY)

    def model_category(self, tag: str) -> Literal["general", "character", "rating"] | None:
        return self.model_categories.get(normalize_tag_key(tag))

    def contains(self, tag: str) -> bool:
        return normalize_tag_key(tag) in self.tags
