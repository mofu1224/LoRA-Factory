"""Deterministic configuration layer resolution with user lock enforcement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lora_factory.config.models import ResolvedConfig


def resolve_layers(
    layers: list[tuple[str, Mapping[str, Any]]],
    *,
    locked_values: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Merge named layers from lowest to highest priority.

    Explicit locked values are applied last and cannot be changed by later automation.
    ``None`` in a layer means "no opinion" rather than clearing a prior value.
    """

    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for source, layer in layers:
        for key, value in layer.items():
            if value is not None:
                values[key] = value
                provenance[key] = source

    locked = dict(locked_values or {})
    for key, value in locked.items():
        if value is None:
            raise ValueError(f"Locked value {key!r} cannot be None")
        values[key] = value
        provenance[key] = "user_lock"

    return ResolvedConfig(
        values=values,
        provenance=provenance,
        locked_fields=frozenset(locked),
    )
