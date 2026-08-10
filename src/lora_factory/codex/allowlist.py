"""Revalidate Runtime Codex suggestions before applying deterministic changes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RECOVERY_ALLOWLIST = frozenset(
    {
        "batch_size",
        "gradient_accumulation",
        "epochs",
        "repeats",
        "target_steps",
        "network_dim",
        "network_alpha",
        "unet_lr",
        "text_encoder_lr",
        "optimizer",
        "precision",
        "no_half_vae",
        "cache_latents",
        "gradient_checkpointing",
    }
)

CAPTION_TAG_ALLOWLIST = frozenset({"remove_tags", "restore_tags"})

INTEGER_RANGES: dict[str, tuple[int, int]] = {
    "batch_size": (1, 32),
    "gradient_accumulation": (1, 64),
    "epochs": (1, 100),
    "repeats": (1, 100),
    "target_steps": (1, 1_000_000),
    "network_dim": (4, 256),
    "network_alpha": (1, 256),
}
FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "unet_lr": (1e-8, 0.1),
    "text_encoder_lr": (1e-8, 0.1),
}
BOOLEAN_FIELDS = frozenset({"no_half_vae", "cache_latents", "gradient_checkpointing"})
ENUM_FIELDS: dict[str, frozenset[str]] = {
    "precision": frozenset({"bf16", "fp16", "fp32"}),
    "optimizer": frozenset(
        {
            "AdamW",
            "AdamW8bit",
            "adafactor",
            "Prodigy",
            "DAdaptAdam",
        }
    ),
}


def validate_recovery_changes(
    proposed: Mapping[str, Any],
    *,
    locked_fields: frozenset[str] = frozenset(),
) -> dict[str, str | int | float | bool]:
    """Return only type/range-safe, unlocked allowlisted changes.

    A disallowed field is treated as a security boundary violation rather than ignored.
    """

    disallowed = set(proposed) - RECOVERY_ALLOWLIST
    if disallowed:
        raise ValueError(f"Codex proposed forbidden setting: {sorted(disallowed)[0]}")
    conflicts = set(proposed) & set(locked_fields)
    if conflicts:
        raise ValueError(f"Codex attempted to change locked setting: {sorted(conflicts)[0]}")

    validated: dict[str, str | int | float | bool] = {}
    for key, value in proposed.items():
        normalized: str | int | float | bool
        if key in INTEGER_RANGES:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            integer_minimum, integer_maximum = INTEGER_RANGES[key]
            if not integer_minimum <= value <= integer_maximum:
                raise ValueError(f"{key} is outside the safe range")
            normalized = value
        elif key in FLOAT_RANGES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            float_minimum, float_maximum = FLOAT_RANGES[key]
            numeric = float(value)
            if not float_minimum <= numeric <= float_maximum:
                raise ValueError(f"{key} is outside the safe range")
            normalized = numeric
        elif key in BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean")
            normalized = value
        elif key in ENUM_FIELDS:
            if not isinstance(value, str) or value not in ENUM_FIELDS[key]:
                raise ValueError(f"{key} is not an allowed value")
            normalized = value
        else:
            raise ValueError(f"No validator is defined for {key}")
        validated[key] = normalized
    return validated
