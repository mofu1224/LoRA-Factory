from __future__ import annotations

import shutil
import subprocess

import pytest

from lora_factory.application.service import repository_root


def _is_ignored(path: str) -> bool:
    git = shutil.which("git")
    if git is None:
        raise AssertionError("git is required to verify the publication boundary")
    result = subprocess.run(  # noqa: S603 - resolved Git, fixed read-only arguments.
        [git, "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise AssertionError(f"git check-ignore failed for {path}: {result.stderr}")
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        "production.env",
        "secrets.yaml",
        ".config/gcloud/application_default_credentials.json",
        ".config/gh/hosts.yml",
        ".git-credentials",
        "private/vault.kdbx",
        ".agent/local.json",
        ".agents/cache.json",
        ".ipynb_checkpoints/notebook.ipynb",
        ".uv-cache-publication/archive-v0/package.whl",
        "data/train.png",
        "private-session/source.webp",
        "wandb/run-1/config.yaml",
        "release/MyLoRA/training_info.json",
        "release/MyLoRA/evaluation.json",
        "release/MyLoRA/resolved_config.yaml",
        "release/MyLoRA/reproducibility_manifest.json",
        "release/MyLoRA/README.txt",
        "release/MyLoRA/.preview.png.copying",
        "relocated/project/dataset/review-overrides.json",
        "relocated/project/configs/validation.toml",
        "relocated/project/dataset/training/image.png",
        "models/private.safetensors",
        "logs/training.log",
        "docs/execplans/local-validation.md",
    ],
)
def test_private_publication_artifacts_are_ignored(path: str) -> None:
    assert _is_ignored(path), path


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        ".agent/PLANS.md",
        "AGENTS.md",
        "backend-manifest.json",
        "runtime-lock.txt",
        "uv.lock",
        "packaging/lora_factory.spec",
        "presets/character.yaml",
        "src/lora_factory/core/context.py",
        "src/lora_factory/assets/model.index",
        "src/lora_factory/resources/tokenizer.json",
        "locale/messages.local.json",
        "docs/assets/workflow.png",
        "src/lora_factory/resources/public/icon.webp",
        "tests/fixtures/public/synthetic.jpg",
    ],
)
def test_required_publication_sources_remain_visible(path: str) -> None:
    assert not _is_ignored(path), path
