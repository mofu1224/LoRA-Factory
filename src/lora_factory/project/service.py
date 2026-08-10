"""Project lifecycle operations with atomic configuration persistence."""

from __future__ import annotations

from pathlib import Path

from lora_factory.config.loader import dump_yaml_model, load_yaml_model
from lora_factory.config.models import ProjectConfig, ProjectDraft
from lora_factory.project.layout import ProjectLayout
from lora_factory.project.manifest import DatasetManifest
from lora_factory.util.json import read_json, write_json_atomic


class ProjectService:
    def __init__(self, projects_root: Path) -> None:
        self.projects_root = projects_root.resolve(strict=False)

    def layout_for(self, project_id: str) -> ProjectLayout:
        root = (self.projects_root / project_id).resolve(strict=False)
        if not root.is_relative_to(self.projects_root):
            raise ValueError("Project identifier escapes the projects root")
        return ProjectLayout(root)

    def create(self, config: ProjectConfig) -> ProjectLayout:
        layout = self.layout_for(config.project_id)
        if layout.config.exists():
            raise FileExistsError(f"Project already exists: {config.project_id}")
        layout.create()
        dump_yaml_model(layout.config, config)
        write_json_atomic(
            layout.manifest,
            DatasetManifest(project_id=config.project_id).model_dump(mode="json"),
        )
        return layout

    def create_draft(self, draft: ProjectDraft) -> tuple[ProjectLayout, bool]:
        """Create the complete base tree from only a user-facing project name.

        The operation is idempotent.  A later full ProjectConfig atomically replaces the
        draft record when the user starts the LoRA pipeline.
        """

        layout = self.layout_for(draft.project_id)
        created = not layout.config.exists()
        if layout.config.exists():
            try:
                existing_id = self.load_config(draft.project_id).project_id
            except ValueError:
                existing_id = self.load_draft(draft.project_id).project_id
            if existing_id != draft.project_id:
                raise ValueError("Existing project metadata does not match its directory")
        layout.create()
        if created:
            dump_yaml_model(layout.config, draft)
        if not layout.manifest.exists():
            write_json_atomic(
                layout.manifest,
                DatasetManifest(project_id=draft.project_id).model_dump(mode="json"),
            )
        return layout, created

    def load_config(self, project_id: str) -> ProjectConfig:
        return load_yaml_model(self.layout_for(project_id).config, ProjectConfig)

    def load_draft(self, project_id: str) -> ProjectDraft:
        return load_yaml_model(self.layout_for(project_id).config, ProjectDraft)

    def save_config(self, config: ProjectConfig) -> None:
        layout = self.layout_for(config.project_id)
        if not layout.root.exists():
            raise FileNotFoundError(f"Project does not exist: {config.project_id}")
        dump_yaml_model(layout.config, config)

    def load_manifest(self, project_id: str) -> DatasetManifest:
        payload = read_json(self.layout_for(project_id).manifest)
        return DatasetManifest.model_validate(payload)

    def save_manifest(self, manifest: DatasetManifest) -> None:
        layout = self.layout_for(manifest.project_id)
        write_json_atomic(layout.manifest, manifest.model_dump(mode="json"))
