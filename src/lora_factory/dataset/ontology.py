"""Versioned WD tag ontology with conservative unknown-tag preservation."""

from __future__ import annotations

import re
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OntologyCategory(StrEnum):
    PERSON_CLASS = "person_class"
    COUNT = "count"
    APPEARANCE = "appearance"
    SPECIES = "species"
    PERMANENT_ACCESSORY = "permanent_accessory"
    CLOTHING = "clothing"
    TEMPORARY_ACCESSORY = "temporary_accessory"
    EXPRESSION = "expression"
    POSE = "pose"
    VIEW = "view"
    COMPOSITION = "composition"
    BACKGROUND = "background"
    LIGHTING = "lighting"
    OBJECT = "object"
    STYLE = "style"
    ARTIST = "artist"
    CHARACTER = "character"
    COPYRIGHT = "copyright"
    QUALITY = "quality"
    RATING = "rating"
    RESOLUTION = "resolution"
    TEXT_WATERMARK = "text_watermark"
    UNCLASSIFIED = "unclassified"


class OntologyData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    categories: dict[OntologyCategory, tuple[str, ...]]
    synonyms: dict[str, str] = Field(default_factory=dict)
    conflicts: dict[str, tuple[str, ...]] = Field(default_factory=dict)


def normalize_tag_key(tag: str) -> str:
    return re.sub(r"_+", "_", tag.strip().lower().replace(" ", "_"))


class TagOntology:
    def __init__(self, data: OntologyData) -> None:
        self.data = data
        self._category_by_tag = {
            normalize_tag_key(tag): category
            for category, tags in data.categories.items()
            for tag in tags
        }
        self._synonyms = {
            normalize_tag_key(alias): normalize_tag_key(canonical)
            for alias, canonical in data.synonyms.items()
        }

    @classmethod
    def bundled(cls, version: int = 1) -> TagOntology:
        resource = files("lora_factory.resources.ontology").joinpath(f"v{version}.yaml")
        payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
        return cls(OntologyData.model_validate(payload))

    @classmethod
    def from_yaml(cls, path: Path) -> TagOntology:
        return cls(OntologyData.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))

    def canonicalize(self, tag: str) -> str:
        key = normalize_tag_key(tag)
        seen: set[str] = set()
        while key in self._synonyms and key not in seen:
            seen.add(key)
            key = self._synonyms[key]
        return key

    def classify(self, tag: str, *, model_category: str | None = None) -> OntologyCategory:
        if model_category == "character":
            return OntologyCategory.CHARACTER
        if model_category == "rating":
            return OntologyCategory.RATING
        key = self.canonicalize(tag)
        category = self._category_by_tag.get(key)
        if category is not None:
            return category
        if key.startswith("artist:") or key.endswith("_(artist)"):
            return OntologyCategory.ARTIST
        if key.startswith("character:") or key.endswith("_(character)"):
            return OntologyCategory.CHARACTER
        if key.startswith("copyright:") or key.startswith("series:"):
            return OntologyCategory.COPYRIGHT
        if key.startswith("rating:"):
            return OntologyCategory.RATING
        if key.endswith("_quality"):
            return OntologyCategory.QUALITY
        return OntologyCategory.UNCLASSIFIED

    def conflicts_for(self, tag: str) -> frozenset[str]:
        key = self.canonicalize(tag)
        return frozenset(self.canonicalize(item) for item in self.data.conflicts.get(key, ()))

    @staticmethod
    def display(tag: str) -> str:
        return normalize_tag_key(tag).replace("_", " ")


DEFAULT_ONTOLOGY = TagOntology.bundled()
