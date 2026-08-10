"""Assemble an immediately usable final LoRA plus alternatives and evidence."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lora_factory.config.models import PresetKind
from lora_factory.packaging.metadata import (
    PublicMetadataSanitizer,
    artifact_record,
    inspect_safetensors,
)
from lora_factory.packaging.readme import build_output_readme
from lora_factory.util.hashing import sha256_file
from lora_factory.util.json import write_json_atomic


class CheckpointCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_id: str
    path: Path
    score: float = Field(ge=0, le=1)
    recommended_weight: float = Field(ge=0, le=2)


class FinalizationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    output_directory: Path
    lora_name: str
    trigger_token: str
    preset: PresetKind
    base_model: Path
    base_model_sha256: str
    selected: CheckpointCandidate
    alternatives: tuple[CheckpointCandidate, ...] = ()
    preview: Path
    comparison: Path
    resolved_config: dict[str, Any]
    evaluation: dict[str, Any]
    training_info: dict[str, Any]
    reproducibility: dict[str, Any]
    warnings: tuple[str, ...] = ()


class FinalizationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path
    final_model: Path
    alternative_models: tuple[Path, ...]
    preview: Path
    comparison: Path
    sha256: str


class Finalizer:
    def finalize(self, request: FinalizationRequest) -> FinalizationResult:
        sanitizer = PublicMetadataSanitizer()
        selected_info = inspect_safetensors(request.selected.path)
        for candidate in request.alternatives[:3]:
            inspect_safetensors(candidate.path)
        if not request.preview.is_file() or not request.comparison.is_file():
            raise FileNotFoundError("Preview and comparison images are required")

        root = request.output_directory
        root.mkdir(parents=True, exist_ok=True)
        alternatives_dir = root / "alternatives"
        alternatives_dir.mkdir(exist_ok=True)
        final_model = root / f"{request.lora_name}.safetensors"
        self._atomic_verified_copy(request.selected.path, final_model)

        alternative_paths: list[Path] = []
        for index, candidate in enumerate(request.alternatives[:3], start=2):
            destination = alternatives_dir / f"candidate_{index}.safetensors"
            self._atomic_verified_copy(candidate.path, destination)
            alternative_paths.append(destination)

        preview_path = root / "preview.png"
        comparison_path = root / "comparison.png"
        self._atomic_verified_copy(request.preview, preview_path)
        self._atomic_verified_copy(request.comparison, comparison_path)

        training_info = sanitizer.sanitize(
            {
                **request.training_info,
                "lora_name": request.lora_name,
                "trigger_token": request.trigger_token,
                "preset": request.preset.value,
                "base_model": request.base_model.name,
                "base_model_sha256": request.base_model_sha256,
                "best_checkpoint": request.selected.checkpoint_id,
                "recommended_weight": request.selected.recommended_weight,
                "final_model": artifact_record(final_model),
            }
        )
        write_json_atomic(root / "training_info.json", training_info)
        write_json_atomic(root / "evaluation.json", sanitizer.sanitize(request.evaluation))
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                sanitizer.sanitize(request.resolved_config),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        manifest = sanitizer.sanitize(
            {
                **request.reproducibility,
                "generated_at": datetime.now(UTC).isoformat(),
                "base_model_sha256": request.base_model_sha256,
                "final_lora": artifact_record(final_model),
                "selected_checkpoint": selected_info,
                "alternatives": [artifact_record(path) for path in alternative_paths],
                "preview": artifact_record(preview_path),
                "comparison": artifact_record(comparison_path),
            }
        )
        write_json_atomic(root / "reproducibility_manifest.json", manifest)
        readme_info = {
            **training_info,
            "recommended_range": request.training_info.get("recommended_range", "0.60-1.00"),
            "resolution": request.training_info["resolution"],
            "network_dim": request.training_info["network_dim"],
            "network_alpha": request.training_info["network_alpha"],
            "training_image_count": request.training_info["training_image_count"],
            "warnings": sanitizer.sanitize(request.warnings),
        }
        (root / "README.txt").write_text(build_output_readme(readme_info), encoding="utf-8")

        return FinalizationResult(
            directory=root,
            final_model=final_model,
            alternative_models=tuple(alternative_paths),
            preview=preview_path,
            comparison=comparison_path,
            sha256=sha256_file(final_model),
        )

    def promote_alternative(
        self,
        *,
        final_model: Path,
        alternative: Path,
        history_directory: Path,
    ) -> str:
        inspect_safetensors(final_model)
        inspect_safetensors(alternative)
        history_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = history_directory / f"{final_model.stem}-{timestamp}{final_model.suffix}"
        self._atomic_verified_copy(final_model, backup)
        self._atomic_verified_copy(alternative, final_model, replace=True)
        return sha256_file(final_model)

    @staticmethod
    def _atomic_verified_copy(source: Path, destination: Path, *, replace: bool = False) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256_file(source)
        if destination.exists():
            if sha256_file(destination) == source_hash:
                return
            if not replace:
                raise FileExistsError(f"Refusing to overwrite existing artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.copying")
        shutil.copy2(source, temporary)
        if source_hash != sha256_file(temporary):
            temporary.unlink(missing_ok=True)
            raise OSError(f"Checksum mismatch while copying {source.name}")
        temporary.replace(destination)
