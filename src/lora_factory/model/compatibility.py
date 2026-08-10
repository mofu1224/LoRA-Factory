"""Model compatibility result types."""

from __future__ import annotations

from enum import StrEnum


class ModelCompatibility(StrEnum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_WARNING = "compatible_with_warning"
    UNSUPPORTED = "unsupported"
