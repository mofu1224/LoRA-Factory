"""Tagger adapters and deterministic caption policies."""

from lora_factory.caption.tagger import FakeTagger, ImageTagResult, TaggerProtocol, TagScore

__all__ = ["FakeTagger", "ImageTagResult", "TagScore", "TaggerProtocol"]
