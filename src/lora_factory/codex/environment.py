"""Minimal non-secret environment shared by every Codex CLI subprocess."""

from __future__ import annotations

from collections.abc import Mapping

_CODEX_ENVIRONMENT_KEYS = frozenset(
    {
        "APPDATA",
        "CODEX_CA_CERTIFICATE",
        "CODEX_HOME",
        "CODEX_SQLITE_HOME",
        "COMSPEC",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
)


def build_codex_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Keep only OS paths, locale, and non-secret Codex state configuration."""

    return {key: value for key, value in source.items() if key.upper() in _CODEX_ENVIRONMENT_KEYS}
