"""Location-independent application paths for the portable distribution.

Every writable location is derived from :func:`application_root`, so a copied or
renamed distribution keeps all of its state inside its own folder.  The paths
never consult ``LOCALAPPDATA``, ``APPDATA`` or the user profile.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repository_root() -> Path:
    """Return the directory that holds the packaged data files."""

    if bool(getattr(sys, "frozen", False)):
        bundle_root = getattr(sys, "_MEIPASS", None)
        if isinstance(bundle_root, str):
            return Path(bundle_root).resolve(strict=True)
    return Path(__file__).resolve().parents[3]


def application_root() -> Path:
    """Return the folder that owns the portable runtime and user data.

    In a frozen build this is the directory that contains the executable, which
    is the folder a user copies to move the application.  From a source
    checkout it is the repository root.
    """

    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return repository_root()


def data_root_in(root: Path) -> Path:
    return root / "data"


def settings_path_in(root: Path) -> Path:
    return data_root_in(root) / "settings.json"


def codex_scratch_root_in(root: Path) -> Path:
    return data_root_in(root) / "codex-scratch"


def managed_runtime_root_in(root: Path) -> Path:
    return root / "runtime"


def toolchain_root_in(root: Path) -> Path:
    return root / "tools"


def cache_root_in(root: Path) -> Path:
    return root / "cache"


def python_root_in(root: Path) -> Path:
    """Return the directory that holds uv-managed interpreters."""

    return root / "python"


def data_root() -> Path:
    return data_root_in(application_root())


def settings_path() -> Path:
    return settings_path_in(application_root())


def codex_scratch_root() -> Path:
    return codex_scratch_root_in(application_root())


def managed_runtime_root() -> Path:
    return managed_runtime_root_in(application_root())


def toolchain_root() -> Path:
    return toolchain_root_in(application_root())


def cache_root() -> Path:
    return cache_root_in(application_root())


def python_root() -> Path:
    return python_root_in(application_root())


def same_location(left: Path, right: Path) -> bool:
    """Compare two paths the way Windows resolves them (case-insensitive)."""

    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )
