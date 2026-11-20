"""Tagger protocol, durable raw results, and a deterministic fake backend."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import write_json_atomic


class TagScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    score: Annotated[float, Field(ge=0, le=1)]
    model_category: str = "general"
    selected: bool = True


class ImageTagResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    source_path: Path
    source_sha256: str
    model_id: str
    revision: str
    tags: tuple[TagScore, ...] = Field(default_factory=tuple)

    def score_map(self, *, selected_only: bool = False) -> dict[str, float]:
        return {tag.name: tag.score for tag in self.tags if not selected_only or tag.selected}


@runtime_checkable
class TaggerProtocol(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def tag(self, path: Path, *, asset_id: str | None = None) -> ImageTagResult: ...

    def tag_many(
        self, paths: Sequence[Path], *, asset_ids: Sequence[str] | None = None
    ) -> tuple[ImageTagResult, ...]: ...


# Name used by backend wiring and external adapters.
WD14TaggerProtocol = TaggerProtocol


FAKE_WD14_VOCABULARY = (
    "smile",
    "looking_at_viewer",
    "outdoors",
    "indoors",
    "standing",
    "sitting",
    "upper_body",
    "full_body",
    "white_dress",
    "school_uniform",
    "blue_hair",
    "brown_hair",
)


class RawTagStore:
    """Persist raw score JSON separately from captions."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)

    def write(self, result: ImageTagResult) -> Path:
        if not result.asset_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for char in result.asset_id
        ):
            raise ValueError("asset_id contains characters unsafe for a result filename")
        destination = (self.root / f"{result.asset_id}.json").resolve(strict=False)
        if not destination.is_relative_to(self.root):
            raise ValueError("Tag result path escapes its store")
        write_json_atomic(destination, result.model_dump(mode="json"))
        return destination


class FakeTagger:
    """Deterministic no-GPU backend suitable for integration and fake E2E tests."""

    model_id = "lora-factory/fake-wd14"
    revision = "1"

    def __init__(
        self,
        fixtures: Mapping[str, Mapping[str, float] | Iterable[TagScore]] | None = None,
        *,
        default_class: str = "1girl",
    ) -> None:
        self.fixtures = dict(fixtures or {})
        self.default_class = default_class

    @staticmethod
    def _fixture_tags(value: Mapping[str, float] | Iterable[TagScore]) -> tuple[TagScore, ...]:
        if isinstance(value, Mapping):
            return tuple(
                TagScore(name=name, score=score)
                for name, score in sorted(value.items(), key=lambda item: item[0])
            )
        return tuple(sorted(value, key=lambda tag: tag.name))

    def tag(self, path: Path, *, asset_id: str | None = None) -> ImageTagResult:
        path = path.resolve(strict=True)
        digest = sha256_file(path)
        result_id = asset_id or digest
        fixture = self.fixtures.get(result_id) or self.fixtures.get(path.name)
        if fixture is not None:
            tags = self._fixture_tags(fixture)
        else:
            seed = hashlib.sha256(bytes.fromhex(digest)).digest()
            selected = [
                FAKE_WD14_VOCABULARY[index]
                for index in range(len(FAKE_WD14_VOCABULARY))
                if seed[index] % 5 == 0
            ][:5]
            if len(selected) < 3:
                selected.extend(tag for tag in FAKE_WD14_VOCABULARY if tag not in selected)
                selected = selected[:3]
            tags = (
                TagScore(name=self.default_class, score=0.97, model_category="general"),
                TagScore(name="solo", score=0.92, model_category="general"),
                *(
                    TagScore(name=name, score=0.55 + seed[index] / 1024)
                    for index, name in enumerate(selected)
                ),
            )
        return ImageTagResult(
            asset_id=result_id,
            source_path=path,
            source_sha256=digest,
            model_id=self.model_id,
            revision=self.revision,
            tags=tuple(tags),
        )

    def tag_many(
        self, paths: Sequence[Path], *, asset_ids: Sequence[str] | None = None
    ) -> tuple[ImageTagResult, ...]:
        if asset_ids is not None and len(asset_ids) != len(paths):
            raise ValueError("asset_ids must align with paths")
        return tuple(
            self.tag(path, asset_id=asset_ids[index] if asset_ids is not None else None)
            for index, path in enumerate(paths)
        )
