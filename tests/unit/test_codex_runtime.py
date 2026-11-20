from __future__ import annotations

from pathlib import Path
from typing import Any

from lora_factory.codex.runtime import (
    CodexEnvironmentKind,
    CodexRuntimeAdapter,
)


def _environment(tmp_path: Path) -> dict[str, str]:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    return {
        "CODEX_HOME": str(codex_home),
        "PATH": str(tmp_path / "bin"),
        "USERPROFILE": str(tmp_path / "user"),
        "LOCALAPPDATA": str(tmp_path / "local"),
        "APPDATA": str(tmp_path / "roaming"),
    }


def test_router_profile_collects_configured_mcp_names_without_values(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    environment = _environment(tmp_path)
    config_path = Path(environment["CODEX_HOME"]) / "config.toml"
    config_path.write_text(
        """
base_url = "http://127.0.0.1:4202/_codex-router/session/v1"
model_catalog_json = "merged-models.json"

[mcp_servers.custom-safe]
command = "never-read"

[mcp_servers."not safe"]
command = "ignored"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lora_factory.codex.runtime.shutil.which",
        lambda _name, **_kwargs: "C:/tools/codex.exe",
    )

    profiles = CodexRuntimeAdapter(source_environment=environment).profiles(
        version_hint="codex-cli 1.0"
    )

    assert len(profiles) == 2
    assert profiles[0].environment is CodexEnvironmentKind.ROUTER
    assert profiles[0].ignore_user_config is False
    assert "custom-safe" in profiles[0].configured_mcp_servers
    assert "not safe" not in profiles[0].configured_mcp_servers
    assert profiles[1].environment is CodexEnvironmentKind.DIRECT
    assert profiles[1].ignore_user_config is True
    assert "4202" not in repr(profiles)


def test_malformed_config_is_unknown_and_still_has_isolated_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    environment = _environment(tmp_path)
    (Path(environment["CODEX_HOME"]) / "config.toml").write_text(
        "base_url = [",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lora_factory.codex.runtime.shutil.which",
        lambda _name, **_kwargs: "C:/tools/codex.exe",
    )

    profiles = CodexRuntimeAdapter(source_environment=environment).profiles(
        version_hint="codex-cli 1.0"
    )

    assert profiles[0].environment is CodexEnvironmentKind.UNKNOWN
    assert any("could not be parsed" in warning for warning in profiles[0].warnings)
    assert profiles[1].ignore_user_config is True


def test_executable_resolver_uses_standard_windows_install_location(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    executable = (
        Path(environment["LOCALAPPDATA"]) / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")

    adapter = CodexRuntimeAdapter(source_environment=environment)

    assert adapter.resolve_executable() == str(executable)


def test_help_probe_narrows_profile_capabilities(tmp_path: Path, monkeypatch: Any) -> None:
    environment = _environment(tmp_path)
    (Path(environment["CODEX_HOME"]) / "config.toml").write_text(
        'base_url = "https://api.openai.com/v1"',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lora_factory.codex.runtime.shutil.which",
        lambda _name, **_kwargs: "C:/tools/codex.exe",
    )

    class Completed:
        returncode = 0
        stdout = (
            "Usage: codex exec [--json] [--sandbox] [--cd] [--ephemeral] "
            "[--disable] [--config] [--output-schema] [--output-last-message] "
            "[--ignore-user-config]"
        )
        stderr = ""

    monkeypatch.setattr(
        "lora_factory.codex.runtime.subprocess.run",
        lambda *_args, **_kwargs: Completed(),
    )

    profile = CodexRuntimeAdapter(source_environment=environment, probe=True).profiles(
        version_hint="codex-cli 1.0"
    )[0]

    assert profile.capabilities_known is True
    assert profile.supports("--json") is True
    assert profile.supports("--image") is False
    assert profile.usable is True
