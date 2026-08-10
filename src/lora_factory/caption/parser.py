"""Parsing helpers for tagger scores and comma-separated captions."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from io import StringIO

from lora_factory.caption.tagger import TagScore


def parse_caption_tags(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    reader = csv.reader(StringIO(text), skipinitialspace=True)
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError("Caption must contain exactly one logical line")
    return tuple(item.strip() for item in rows[0] if item.strip())


def parse_tag_scores(
    value: Mapping[str, float] | Iterable[TagScore],
) -> tuple[TagScore, ...]:
    if isinstance(value, Mapping):
        return tuple(
            TagScore(name=name, score=score)
            for name, score in sorted(value.items(), key=lambda item: item[0])
        )
    return tuple(value)


parse_tags = parse_caption_tags
