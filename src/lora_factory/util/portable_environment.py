"""Point tool caches inside the portable application folder.

The values are applied to the running process only (``os.environ``), never to
the machine's permanent user or system settings.  Existing values are kept so an
explicit override still wins.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from lora_factory.util.app_paths import cache_root, python_root


def default_environment() -> dict[str, str]:
    """Return the process-scoped cache locations for the current app folder."""

    cache = cache_root()
    hugging_face = cache / "huggingface"
    return {
        "UV_CACHE_DIR": str(cache / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(python_root()),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "HF_HOME": str(hugging_face),
        "HF_HUB_CACHE": str(hugging_face / "hub"),
        "TRANSFORMERS_CACHE": str(hugging_face / "hub"),
        "TORCH_HOME": str(cache / "torch"),
    }


def configure_process_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply the portable cache locations without overwriting explicit values."""

    target: MutableMapping[str, str] = os.environ if environment is None else environment
    applied: dict[str, str] = {}
    for key, value in default_environment().items():
        if not target.get(key):
            target[key] = value
            applied[key] = value
    return applied


def cache_directories() -> tuple[Path, ...]:
    """Return the cache directories the assets are written to."""

    return tuple(Path(value) for value in default_environment().values())


def child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child-process environment redirected into the app folder."""

    source = dict(os.environ if base is None else base)
    for key, value in default_environment().items():
        source.setdefault(key, value)
    return source
