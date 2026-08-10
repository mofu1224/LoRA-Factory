from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sqlalchemy import inspect

from lora_factory.application import service as application_service
from lora_factory.application.service import LoRAFactoryController, default_app_settings
from lora_factory.storage.database import Database


def test_default_controller_adds_name_only_project_under_portable_project_root(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app_root = tmp_path / "portable-app"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(application_service, "application_root", lambda: app_root)
    settings = default_app_settings()
    assert settings.projects_root == app_root / "Project"
    controller = LoRAFactoryController(settings)

    result = controller.create_project("新しい LoRA")
    project_root = app_root / "Project" / "新しい LoRA"

    assert result == {
        "schema_version": 1,
        "status": "draft",
        "project_id": "新しい LoRA",
        "lora_name": "新しい LoRA",
        "created": True,
        "project_root": str(project_root.resolve(strict=False)),
    }
    assert project_root.is_dir()
    assert (project_root / "project.yaml").is_file()
    assert (project_root / "dataset" / "manifest.json").is_file()
    assert (project_root / "input-Image").is_dir()
    assert (project_root / "output-model").is_dir()
    assert (project_root / "base-model").is_dir()
    database = Database(project_root / "state.sqlite3")
    assert "projects" in inspect(database.engine).get_table_names()
    recent = controller.recent_projects()
    assert len(recent) == 1
    assert recent[0]["project_id"] == "新しい LoRA"
    assert recent[0]["status"] == "draft"
    assert recent[0]["project_root"] == str(project_root.resolve(strict=False))

    again = controller.create_project("新しい LoRA")
    assert again["created"] is False


def test_frozen_application_root_is_the_executable_directory(
    monkeypatch: Any, tmp_path: Path
) -> None:
    executable = tmp_path / "LoRA Factory" / "LoRA Factory.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert application_service.application_root() == executable.parent.resolve(strict=False)
