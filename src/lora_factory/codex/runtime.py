"""Environment-adaptive profiles for bounded Codex CLI execution."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from lora_factory.codex.environment import build_codex_environment

_CONFIG_MAX_BYTES = 256 * 1024
_MCP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ROUTER_MARKERS = (
    "codex-router",
    "_codex-router",
    "router",
    "127.0.0.1:4202",
    "localhost:4202",
)

# These are deliberately conservative.  A successful help probe narrows the
# set to what the installed CLI actually advertises.
_KNOWN_CLI_FLAGS = frozenset(
    {
        "--cd",
        "--config",
        "--disable",
        "--ephemeral",
        "--ignore-rules",
        "--ignore-user-config",
        "--image",
        "--json",
        "--output-last-message",
        "--output-schema",
        "--sandbox",
    }
)
_REQUIRED_CLI_FLAGS = frozenset(
    {
        "--cd",
        "--ephemeral",
        "--json",
        "--output-last-message",
        "--output-schema",
        "--sandbox",
    }
)
_RUNTIME_MCP_SERVERS = ("blender", "node_repl", "unityMCP")


class CodexEnvironmentKind(StrEnum):
    """Transport/configuration family used by the installed Codex CLI."""

    DIRECT = "direct"
    ROUTER = "router"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CodexRuntimeProfile:
    """One safe invocation strategy for a Codex CLI installation."""

    executable: str
    environment: CodexEnvironmentKind
    version: str | None = None
    supported_flags: frozenset[str] = field(default_factory=frozenset)
    configured_mcp_servers: tuple[str, ...] = ()
    config_path: Path | None = None
    ignore_user_config: bool = False
    capabilities_known: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        suffix = "-isolated" if self.ignore_user_config else "-configured"
        return f"{self.environment.value}{suffix}"

    def supports(self, option: str) -> bool:
        """Return whether an option is safe to use for this profile."""

        # An unavailable help probe must not turn a harmless CLI upgrade into a
        # false negative.  The process layer still treats every timeout as
        # bounded and the gateway can fall back or try the next profile.
        return not self.capabilities_known or option in self.supported_flags

    @property
    def usable(self) -> bool:
        """Whether this profile can preserve the runtime safety contract."""

        if not all(self.supports(option) for option in _REQUIRED_CLI_FLAGS):
            return False
        if self.ignore_user_config:
            return self.supports("--ignore-user-config") and self.supports("--disable")
        if self.configured_mcp_servers and not self.supports("--config"):
            return False
        return self.supports("--disable")

    def usable_for_call(self, *, has_images: bool) -> bool:
        """Return whether the profile also supports this call's input shape."""

        return self.usable and (not has_images or self.supports("--image"))


def _env_value(source: Mapping[str, str], name: str) -> str | None:
    """Read an environment key case-insensitively on Windows."""

    direct = source.get(name)
    if direct:
        return direct
    folded = name.casefold()
    for key, value in source.items():
        if key.casefold() == folded and value:
            return value
    return None


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


def _dedupe_mcp_servers(names: Mapping[str, object] | None) -> tuple[str, ...]:
    values: set[str] = set(_RUNTIME_MCP_SERVERS)
    if names is not None:
        values.update(name for name in names if _MCP_NAME_PATTERN.fullmatch(name))
    return tuple(sorted(values, key=str.casefold))


def _config_path(source: Mapping[str, str]) -> Path:
    configured_home = _env_value(source, "CODEX_HOME")
    if configured_home:
        candidate = Path(configured_home)
        if candidate.is_absolute():
            return candidate / "config.toml"
    home = _env_value(source, "USERPROFILE") or _env_value(source, "HOME")
    if home:
        candidate = Path(home)
        if candidate.is_absolute():
            return candidate / ".codex" / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _load_config(path: Path) -> tuple[dict[str, object], bool, tuple[str, ...]]:
    if not _path_is_file(path):
        return {}, False, ()
    try:
        if path.stat().st_size > _CONFIG_MAX_BYTES:
            return {}, True, ("Codex config is too large to inspect safely",)
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError):
        return {}, True, ("Codex config could not be parsed safely",)
    if not isinstance(loaded, dict):
        return {}, True, ("Codex config did not contain a mapping",)
    return loaded, True, ()


def _text_marker(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    return any(marker in lowered for marker in _ROUTER_MARKERS)


def _detect_environment(
    config: Mapping[str, object],
    *,
    config_exists: bool,
    source: Mapping[str, str],
) -> CodexEnvironmentKind:
    for key in ("CODEX_ROUTER_URL", "CODEX_ROUTER_BASE_URL"):
        if _env_value(source, key):
            return CodexEnvironmentKind.ROUTER
    for key in (
        "base_url",
        "model_catalog_json",
        "provider",
        "provider_name",
        "model_provider",
    ):
        if _text_marker(config.get(key)):
            return CodexEnvironmentKind.ROUTER
    providers = config.get("model_providers")
    if isinstance(providers, Mapping) and any(
        _text_marker(name) or _text_marker(value) for name, value in providers.items()
    ):
        return CodexEnvironmentKind.ROUTER
    if config_exists:
        return CodexEnvironmentKind.DIRECT
    return CodexEnvironmentKind.DIRECT


def _configured_mcp_servers(config: Mapping[str, object]) -> tuple[str, ...]:
    raw_servers = config.get("mcp_servers")
    if not isinstance(raw_servers, Mapping):
        return _dedupe_mcp_servers(None)
    return _dedupe_mcp_servers(raw_servers)


def _parse_help_flags(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", text))


class CodexRuntimeAdapter:
    """Resolve and cache safe direct/Router profiles for one application run."""

    def __init__(
        self,
        executable: str = "codex",
        *,
        source_environment: Mapping[str, str] | None = None,
        probe: bool = False,
        probe_timeout_seconds: float = 5.0,
    ) -> None:
        self.executable = executable
        self._source_environment = (
            dict(source_environment) if source_environment is not None else None
        )
        self.probe = probe
        self.probe_timeout_seconds = probe_timeout_seconds
        self._profiles_cache: dict[str | None, tuple[CodexRuntimeProfile, ...]] = {}

    def _environment_source(self) -> Mapping[str, str]:
        return self._source_environment if self._source_environment is not None else os.environ

    def safe_environment(self) -> dict[str, str]:
        return build_codex_environment(self._environment_source())

    def resolve_executable(self) -> str | None:
        source = self._environment_source()
        requested = self.executable
        if os.path.isabs(requested) or "/" in requested or "\\" in requested:
            candidate = Path(requested)
            if not _path_is_file(candidate):
                return None
            try:
                return str(candidate.resolve(strict=True))
            except (OSError, ValueError):
                return None
        path_value = _env_value(source, "PATH")
        try:
            resolved = shutil.which(requested, path=path_value)
        except TypeError:
            # Test doubles and older Python shims may expose the one-argument
            # shape of shutil.which.
            resolved = shutil.which(requested)
        if resolved:
            return resolved
        if requested.casefold() not in {"codex", "codex.exe", "codex.cmd"}:
            return None

        local_app_data = _env_value(source, "LOCALAPPDATA")
        app_data = _env_value(source, "APPDATA")
        home = _env_value(source, "USERPROFILE") or _env_value(source, "HOME")
        candidates: list[Path] = []
        if local_app_data:
            candidates.append(
                Path(local_app_data) / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
            )
        if app_data:
            candidates.extend(
                (
                    Path(app_data) / "npm" / "codex.cmd",
                    Path(app_data) / "npm" / "codex.exe",
                )
            )
        if home:
            candidates.append(Path(home) / ".local" / "bin" / "codex")
        candidates.extend((Path("/usr/local/bin/codex"), Path("/usr/bin/codex")))
        for candidate in candidates:
            if _path_is_file(candidate):
                return str(candidate)
        return None

    def _probe_capabilities(self, executable: str) -> tuple[frozenset[str] | None, str | None]:
        try:
            completed = subprocess.run(  # noqa: S603 - executable resolved or explicitly configured.
                [executable, "exec", "--help"],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.probe_timeout_seconds,
                check=False,
                env=self.safe_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError, TypeError):
            return None, "Codex CLI capability probe failed"
        flags = _parse_help_flags(f"{completed.stdout}\n{completed.stderr}")
        if completed.returncode != 0 or not flags:
            return None, "Codex CLI capability probe returned no usable flags"
        return flags, None

    def profiles(self, *, version_hint: str | None = None) -> tuple[CodexRuntimeProfile, ...]:
        if version_hint in self._profiles_cache:
            return self._profiles_cache[version_hint]
        resolved = self.resolve_executable()
        warnings: list[str] = []
        if resolved is None:
            if version_hint is None:
                self._profiles_cache[version_hint] = ()
                return ()
            resolved = self.executable
            warnings.append(
                "Codex executable could not be resolved; using its configured command name"
            )
        config_path = _config_path(self._environment_source())
        config, config_exists, config_warnings = _load_config(config_path)
        warnings.extend(config_warnings)
        kind = (
            CodexEnvironmentKind.UNKNOWN
            if config_warnings
            else _detect_environment(
                config,
                config_exists=config_exists,
                source=self._environment_source(),
            )
        )
        configured_mcp = _configured_mcp_servers(config)
        supported_flags = _KNOWN_CLI_FLAGS
        capabilities_known = False
        if self.probe:
            probed_flags, probe_warning = self._probe_capabilities(resolved)
            if probed_flags is not None:
                supported_flags = probed_flags
                capabilities_known = True
            elif probe_warning:
                warnings.append(probe_warning)
        primary = CodexRuntimeProfile(
            executable=resolved,
            environment=kind,
            version=version_hint,
            supported_flags=supported_flags,
            configured_mcp_servers=configured_mcp,
            config_path=config_path if config_exists else None,
            capabilities_known=capabilities_known,
            warnings=tuple(warnings),
        )
        profiles: list[CodexRuntimeProfile] = [primary]
        if kind in (CodexEnvironmentKind.ROUTER, CodexEnvironmentKind.UNKNOWN) and primary.supports(
            "--ignore-user-config"
        ):
            profiles.append(
                CodexRuntimeProfile(
                    executable=resolved,
                    environment=CodexEnvironmentKind.DIRECT,
                    version=version_hint,
                    supported_flags=supported_flags,
                    config_path=None,
                    ignore_user_config=True,
                    capabilities_known=capabilities_known,
                    warnings=("Router profile failed; retrying with isolated user configuration",),
                )
            )
        result = tuple(profile for profile in profiles if profile.usable)
        self._profiles_cache[version_hint] = result
        return result


__all__ = [
    "CodexEnvironmentKind",
    "CodexRuntimeAdapter",
    "CodexRuntimeProfile",
]
