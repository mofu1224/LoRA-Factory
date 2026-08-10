"""Canonical on-disk project layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "project.yaml"

    @property
    def database(self) -> Path:
        return self.root / "state.sqlite3"

    @property
    def run_snapshot(self) -> Path:
        return self.root / "run_snapshot.yaml"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def input_images(self) -> Path:
        return self.root / "input-Image"

    @property
    def output_model(self) -> Path:
        return self.root / "output-model"

    @property
    def base_model(self) -> Path:
        return self.root / "base-model"

    @property
    def raw(self) -> Path:
        return self.dataset / "raw"

    @property
    def import_staging(self) -> Path:
        return self.dataset / ".import-staging"

    @property
    def working(self) -> Path:
        return self.dataset / "working"

    @property
    def captions(self) -> Path:
        return self.dataset / "captions"

    @property
    def validation(self) -> Path:
        return self.dataset / "validation"

    @property
    def rejected(self) -> Path:
        return self.dataset / "rejected"

    @property
    def manifest(self) -> Path:
        return self.dataset / "manifest.json"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def resolved_config(self) -> Path:
        return self.configs / "resolved.yaml"

    @property
    def dataset_toml(self) -> Path:
        return self.configs / "dataset.toml"

    @property
    def training_config(self) -> Path:
        return self.configs / "training.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def final(self) -> Path:
        return self.root / "final"

    def run(self, run_id: str) -> RunLayout:
        return RunLayout(self.runs / run_id)

    def create(self) -> None:
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"Project root is not a directory: {self.root}")
        directories = (
            self.root,
            self.input_images,
            self.output_model,
            self.base_model,
            self.raw,
            self.import_staging,
            self.working,
            self.captions,
            self.validation,
            self.rejected,
            self.configs,
            self.runs,
            self.final,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def states(self) -> Path:
        return self.root / "states"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def codex(self) -> Path:
        return self.root / "codex"

    def create(self) -> None:
        for directory in (
            self.root,
            self.logs,
            self.checkpoints,
            self.states,
            self.samples,
            self.evaluation,
            self.codex,
        ):
            directory.mkdir(parents=True, exist_ok=True)
