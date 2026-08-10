"""Safe YAML configuration persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def load_yaml_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    """Load a UTF-8 YAML mapping and validate it as ``model_type``."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return model_type.model_validate(payload)


def dump_yaml_model(path: Path, model: BaseModel) -> None:
    """Atomically persist a Pydantic model as readable UTF-8 YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = model.model_dump(mode="json", exclude_none=False)
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a generic UTF-8 YAML mapping used by immutable run snapshots."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return {str(key): value for key, value in payload.items()}


def dump_yaml_mapping(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a JSON-compatible mapping as readable UTF-8 YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)
