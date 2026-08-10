"""Typed application boundary and view-data normalizers used by the GUI."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lora_factory.config.models import ProjectConfig

EventCallback = Callable[[dict[str, Any]], None]


@runtime_checkable
class ApplicationController(Protocol):
    """Only application operations that widgets are allowed to invoke."""

    def discover_gpus(self) -> Sequence[object]: ...

    def setup_checks(self) -> Sequence[Mapping[str, Any]]: ...

    def recent_projects(self) -> Sequence[Mapping[str, Any]]: ...

    def create_project(self, name: str) -> Mapping[str, Any]: ...

    def run_pipeline(self, config: ProjectConfig, emit: EventCallback) -> Mapping[str, Any]: ...

    def cancel_current(self) -> None: ...

    def resume_project(self, project_id: str, emit: EventCallback) -> Mapping[str, Any]: ...

    def promote_alternative(self, project_id: str, checkpoint_id: str) -> Mapping[str, Any]: ...

    def copy_output(self, project_id: str, destination_kind: str) -> Mapping[str, Any]: ...

    def save_destinations(self, destinations: Sequence[Mapping[str, Any]]) -> None: ...


def field_value(value: object, name: str, default: Any = None) -> Any:
    """Read a field from a mapping, dataclass, Pydantic model, or plain object."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def as_mapping(value: object) -> dict[str, Any]:
    """Convert supported application boundary values into a mutable mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return dict(fields)
    raise TypeError(f"Unsupported GUI boundary value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class SetupCheckView:
    name: str
    status: str
    detail: str = ""
    repairable: bool = False
    required: bool = True

    @classmethod
    def from_value(cls, value: object) -> SetupCheckView:
        return cls(
            name=str(field_value(value, "name", "Unknown check")),
            status=str(field_value(value, "status", "unknown")),
            detail=str(field_value(value, "detail", "")),
            repairable=bool(field_value(value, "repairable", False)),
            required=bool(field_value(value, "required", True)),
        )

    @property
    def healthy(self) -> bool:
        return not self.required or self.status.casefold() in {
            "ok",
            "ready",
            "pass",
            "passed",
            "healthy",
        }


@dataclass(frozen=True, slots=True)
class GpuView:
    uuid: str
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    utilization_percent: float
    capability_major: int
    capability_minor: int
    compatible: bool
    compatibility_reason: str = ""

    @classmethod
    def from_value(cls, value: object) -> GpuView:
        return cls(
            uuid=str(field_value(value, "uuid", "")),
            index=int(field_value(value, "index", 0)),
            name=str(field_value(value, "name", "Unknown NVIDIA GPU")),
            total_vram_mb=int(field_value(value, "total_vram_mb", 0)),
            free_vram_mb=int(field_value(value, "free_vram_mb", 0)),
            utilization_percent=float(field_value(value, "utilization_percent", 0.0)),
            capability_major=int(field_value(value, "capability_major", 0)),
            capability_minor=int(field_value(value, "capability_minor", 0)),
            compatible=bool(field_value(value, "compatible", False)),
            compatibility_reason=str(field_value(value, "compatibility_reason", "")),
        )

    @property
    def short_uuid(self) -> str:
        return self.uuid if len(self.uuid) <= 18 else f"{self.uuid[:14]}…{self.uuid[-4:]}"


@dataclass(frozen=True, slots=True)
class DatasetItemView:
    asset_id: str
    category: str
    original_filename: str
    width: int
    height: int
    bucket: str
    reasons: tuple[str, ...] = ()
    raw_tags: tuple[str, ...] = ()
    final_caption: str = ""
    included: bool = True
    thumbnail_path: Path | None = None
    categories: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> DatasetItemView:
        reasons = field_value(value, "reasons", field_value(value, "quality_reasons", ()))
        tags = field_value(value, "raw_tags", ())
        thumb = field_value(value, "thumbnail_path")
        category = str(field_value(value, "category", "Accepted"))
        raw_categories = field_value(value, "categories", (category,))
        categories = tuple(dict.fromkeys(str(item) for item in raw_categories or (category,)))
        return cls(
            asset_id=str(field_value(value, "asset_id", field_value(value, "id", ""))),
            category=category,
            original_filename=str(field_value(value, "original_filename", "")),
            width=int(field_value(value, "width", 0)),
            height=int(field_value(value, "height", 0)),
            bucket=str(field_value(value, "bucket", field_value(value, "assigned_bucket", "—"))),
            reasons=tuple(str(item) for item in reasons or ()),
            raw_tags=tuple(str(item) for item in tags or ()),
            final_caption=str(field_value(value, "final_caption", "")),
            included=bool(field_value(value, "included", True)),
            thumbnail_path=Path(thumb) if thumb else None,
            categories=categories,
        )


@dataclass(frozen=True, slots=True)
class PipelineUpdate:
    event_type: str
    message: str
    stage: str = ""
    overall_progress: float | None = None
    stage_progress: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: object) -> PipelineUpdate:
        data = as_mapping(value)
        details_value = data.get("details", {})
        details = dict(details_value) if isinstance(details_value, Mapping) else {}
        overall = data.get(
            "overall_progress", data.get("progress", details.get("overall_progress"))
        )
        stage_progress = data.get("stage_progress", details.get("stage_progress"))
        return cls(
            event_type=str(data.get("event_type", data.get("type", "progress"))),
            message=str(data.get("message", "")),
            stage=str(data.get("stage", details.get("stage", "")) or ""),
            overall_progress=float(overall) if overall is not None else None,
            stage_progress=float(stage_progress) if stage_progress is not None else None,
            details={
                **details,
                **{
                    key: val
                    for key, val in data.items()
                    if key
                    not in {
                        "event_type",
                        "type",
                        "message",
                        "stage",
                        "progress",
                        "overall_progress",
                        "stage_progress",
                        "details",
                    }
                },
            },
        )


class UnavailableController:
    """Safe launch fallback when application service construction fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def discover_gpus(self) -> Sequence[object]:
        return ()

    def setup_checks(self) -> Sequence[Mapping[str, Any]]:
        return (
            {
                "name": "Application Services",
                "status": "error",
                "detail": self.reason,
                "repairable": False,
            },
        )

    def recent_projects(self) -> Sequence[Mapping[str, Any]]:
        return ()

    def create_project(self, name: str) -> Mapping[str, Any]:
        del name
        raise RuntimeError(self.reason)

    def run_pipeline(self, config: ProjectConfig, emit: EventCallback) -> Mapping[str, Any]:
        del config, emit
        raise RuntimeError(self.reason)

    def cancel_current(self) -> None:
        return None

    def resume_project(self, project_id: str, emit: EventCallback) -> Mapping[str, Any]:
        del project_id, emit
        raise RuntimeError(self.reason)

    def promote_alternative(self, project_id: str, checkpoint_id: str) -> Mapping[str, Any]:
        del project_id, checkpoint_id
        raise RuntimeError(self.reason)

    def copy_output(self, project_id: str, destination_kind: str) -> Mapping[str, Any]:
        del project_id, destination_kind
        raise RuntimeError(self.reason)

    def save_destinations(self, destinations: Sequence[Mapping[str, Any]]) -> None:
        del destinations
        raise RuntimeError(self.reason)
