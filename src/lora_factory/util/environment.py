"""Environment construction for managed child processes."""

from __future__ import annotations

import re
from collections.abc import Mapping

_CUDA_VERSIONED_PATH = re.compile(r"^CUDA_PATH_V\d+(?:_\d+)*$", re.IGNORECASE)

_MANAGED_ENVIRONMENT_KEYS = frozenset(
    {
        "ALLUSERSPROFILE",
        "APPDATA",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMSPEC",
        "CUDA_CACHE_PATH",
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "HF_DATASETS_OFFLINE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_HUB_CACHE",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OMP_NUM_THREADS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PUBLIC",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)


def build_managed_child_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Keep only runtime/system variables needed by pinned local helpers.

    The parent environment is never merged implicitly.  In particular, credentials and
    proxy settings are excluded unless a future key is deliberately added to this list.
    CUDA toolkit versioned variables are retained because Windows installations commonly
    expose them as ``CUDA_PATH_<version>``.
    """

    return {
        key: value
        for key, value in source.items()
        if key.upper() in _MANAGED_ENVIRONMENT_KEYS
        or _CUDA_VERSIONED_PATH.fullmatch(key) is not None
    }
