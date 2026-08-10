from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import func, select

from lora_factory.application.service import LoRAFactoryController
from lora_factory.caption.tagger import FakeTagger
from lora_factory.config.models import (
    AppSettings,
    BackendMode,
    PresetKind,
    ProjectConfig,
    TrainingPlan,
)
from lora_factory.core.cancellation import CancellationToken
from lora_factory.core.context import PipelineContext
from lora_factory.core.events import EventBus
from lora_factory.core.stage import PipelineStage, RunStatus
from lora_factory.evaluation.embedding import FakeImageEmbeddingBackend
from lora_factory.gpu.models import GpuBinding, GpuDevice, GpuTaskKind
from lora_factory.project.layout import ProjectLayout
from lora_factory.sampling.fake_backend import FakeSampler
from lora_factory.storage.database import Database
from lora_factory.storage.orm import GpuLeaseRow, ProjectRow, RunRow

SELECTED_GPU = "GPU-00000000-0000-0000-0000-000000000001"
UNSELECTED_GPU = "GPU-00000000-0000-0000-0000-000000000002"
OUTSIDE_GPU = "GPU-00000000-0000-0000-0000-000000000003"


class _Backend:
    version = "captured-real-backend/1"


def _device(uuid: str, *, index: int, free_vram_mb: int) -> GpuDevice:
    return GpuDevice(
        uuid=uuid,
        index=index,
        name=f"GPU at current index {index}",
        total_vram_mb=24 * 1024,
        free_vram_mb=free_vram_mb,
        capability_major=12,
        capability_minor=0,
    )


def _plan() -> TrainingPlan:
    return TrainingPlan(
        resolution=768,
        batch_size=1,
        gradient_accumulation=4,
        repeats=1,
        epochs=1,
        estimated_steps=10,
        network_dim=32,
        network_alpha=16,
        unet_lr=1e-4,
        text_encoder_lr=5e-5,
        optimizer="AdamW",
        precision="bf16",
        keep_tokens=1,
        validation_enabled=False,
        training_gpu_uuid=SELECTED_GPU,
        estimated_disk_mb=256,
    )


def _context(
    tmp_path: Path,
    *,
    mode: BackendMode,
    selected_gpu_uuids: tuple[str, ...] = (SELECTED_GPU,),
) -> PipelineContext:
    layout = ProjectLayout(tmp_path / f"project-{mode.value}")
    layout.create()
    config = ProjectConfig(
        project_id=f"gpu-lease-{mode.value}",
        lora_name=f"gpu_lease_{mode.value}",
        preset=PresetKind.CHARACTER,
        trigger_token="gpu_lease_token",  # noqa: S106 - domain trigger, not a credential
        base_model=tmp_path / "base.safetensors",
        input_paths=(tmp_path / "inputs",),
        selected_gpu_uuids=selected_gpu_uuids,
        output_root=tmp_path / "output",
        backend_mode=mode,
    )
    return PipelineContext(
        run_id="run-gpu-lease",
        config=config,
        layout=layout,
        events=EventBus(),
        cancellation=CancellationToken(),
    )


def _controller(tmp_path: Path) -> LoRAFactoryController:
    return LoRAFactoryController(
        AppSettings(
            projects_root=tmp_path / "projects",
            managed_runtime_root=tmp_path / "runtime",
            codex_runtime_root=tmp_path / "codex",
        )
    )


def test_archaudit_real_training_attempt_leases_selected_uuid_at_fresh_index(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    context = _context(tmp_path, mode=BackendMode.REAL)
    database = Database(context.layout.database)
    database.initialize()
    with database.session() as session:
        session.add(
            ProjectRow(
                id=context.config.project_id,
                name=context.config.lora_name,
                root_path=str(context.layout.root),
                config_json={},
            )
        )
        session.add(
            RunRow(
                id=context.run_id,
                project_id=context.config.project_id,
                status=RunStatus.ACTIVE.value,
            )
        )

    monkeypatch.setattr(
        "lora_factory.gpu.execution.discover_nvidia_gpus",
        lambda: (
            _device(UNSELECTED_GPU, index=0, free_vram_mb=24 * 1024),
            _device(SELECTED_GPU, index=2, free_vram_mb=12 * 1024),
        ),
    )
    captured: dict[str, object] = {}
    controller = _controller(tmp_path)

    def backend_factory(
        _context: PipelineContext,
        effective_plan: TrainingPlan,
        *,
        binding: GpuBinding | None = None,
    ) -> _Backend:
        captured["plan"] = effective_plan
        captured["binding"] = binding
        return _Backend()

    controller._training_backend = backend_factory  # type: ignore[method-assign]
    with controller._training_attempt_backend(context, _plan()) as (_, plan, lease):
        binding = captured["binding"]
        assert isinstance(binding, GpuBinding)
        assert binding.uuid == SELECTED_GPU
        assert binding.physical_index == 2
        assert binding.environment["CUDA_VISIBLE_DEVICES"] == "2"
        assert plan.training_gpu_uuid == SELECTED_GPU
        assert lease is not None
        assert lease["gpu_uuid"] == SELECTED_GPU
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(GpuLeaseRow)) == 1
            active = session.scalar(select(GpuLeaseRow))
            assert active is not None
            assert active.gpu_uuid != UNSELECTED_GPU

    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(GpuLeaseRow)) == 0


def test_archaudit_fake_training_attempt_does_not_discover_nvidia(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "lora_factory.gpu.execution.discover_nvidia_gpus",
        lambda: (_ for _ in ()).throw(AssertionError("Fake attempted NVIDIA discovery")),
    )
    controller = _controller(tmp_path)
    context = _context(tmp_path, mode=BackendMode.FAKE)

    with controller._training_attempt_backend(context, _plan()) as (backend, plan, lease):
        assert backend.version == "fake-trainer/1"
        assert plan == _plan()
        assert lease is None


def test_archaudit_real_tagger_and_sampler_sessions_use_durable_selected_gpu_leases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    context = _context(tmp_path, mode=BackendMode.REAL)
    database = Database(context.layout.database)
    database.initialize()
    with database.session() as session:
        session.add(
            ProjectRow(
                id=context.config.project_id,
                name=context.config.lora_name,
                root_path=str(context.layout.root),
                config_json={},
            )
        )
        session.add(
            RunRow(
                id=context.run_id,
                project_id=context.config.project_id,
                status=RunStatus.ACTIVE.value,
            )
        )
    context.artifacts[PipelineStage.TRAINING.value] = {"plan": _plan().model_dump(mode="json")}
    monkeypatch.setattr(
        "lora_factory.gpu.execution.discover_nvidia_gpus",
        lambda: (
            _device(UNSELECTED_GPU, index=0, free_vram_mb=24 * 1024),
            _device(SELECTED_GPU, index=3, free_vram_mb=12 * 1024),
        ),
    )
    controller = _controller(tmp_path)
    bindings: list[GpuBinding] = []

    def tagger_factory(
        _context: PipelineContext,
        *,
        binding: GpuBinding | None = None,
        work_directory: Path | None = None,
    ) -> FakeTagger:
        del work_directory
        assert binding is not None
        bindings.append(binding)
        return FakeTagger()

    def sampler_factory(
        _context: PipelineContext,
        *,
        binding: GpuBinding | None = None,
    ) -> FakeSampler:
        assert binding is not None
        bindings.append(binding)
        return FakeSampler()

    controller._tagger = tagger_factory  # type: ignore[method-assign]
    controller._sampler = sampler_factory  # type: ignore[method-assign]

    with controller._tagger_session(context, task=GpuTaskKind.WD14_TAG) as (_, uuid):
        assert uuid == SELECTED_GPU
        with database.session() as session:
            lease = session.scalar(select(GpuLeaseRow))
            assert lease is not None and lease.task == GpuTaskKind.WD14_TAG.value
    with controller._sampler_session(context, pass_number=1) as (_, uuid):
        assert uuid == SELECTED_GPU
        with database.session() as session:
            lease = session.scalar(select(GpuLeaseRow))
            assert lease is not None and lease.task == GpuTaskKind.SAMPLE.value

    assert [binding.physical_index for binding in bindings] == [3, 3]
    assert all(binding.environment["CUDA_VISIBLE_DEVICES"] == "3" for binding in bindings)
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(GpuLeaseRow)) == 0


def test_archaudit_independent_gpu_tasks_rotate_over_selected_pool_without_discovery(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "lora_factory.gpu.execution.discover_nvidia_gpus",
        lambda: (_ for _ in ()).throw(AssertionError("Fake attempted NVIDIA discovery")),
    )
    controller = _controller(tmp_path)
    context = _context(
        tmp_path,
        mode=BackendMode.FAKE,
        selected_gpu_uuids=(SELECTED_GPU, UNSELECTED_GPU),
    )

    with controller._tagger_session(context, task=GpuTaskKind.WD14_TAG) as (_, uuid):
        assert uuid == SELECTED_GPU
    with controller._tagger_session(context, task=GpuTaskKind.GENERATED_TAG) as (_, uuid):
        assert uuid == UNSELECTED_GPU
    with controller._sampler_session(context, pass_number=1) as (_, uuid):
        assert uuid == SELECTED_GPU
    with controller._sampler_session(context, pass_number=2) as (_, uuid):
        assert uuid == UNSELECTED_GPU


def test_archaudit_fake_wd14_shards_preserve_input_order_and_selected_pool(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "lora_factory.application.service.discover_nvidia_gpus",
        lambda: (_ for _ in ()).throw(AssertionError("Fake attempted NVIDIA discovery")),
    )
    controller = _controller(tmp_path)
    context = _context(
        tmp_path,
        mode=BackendMode.FAKE,
        selected_gpu_uuids=(SELECTED_GPU, UNSELECTED_GPU),
    )
    paths: list[Path] = []
    for index in range(5):
        path = tmp_path / f"shard-{index}.png"
        Image.new("RGB", (32, 32), (index * 20, 40, 80)).save(path)
        paths.append(path)
    asset_ids = [f"asset-{index}" for index in range(len(paths))]

    batch = controller._tag_many_sharded(
        context,
        task=GpuTaskKind.WD14_TAG,
        paths=paths,
        asset_ids=asset_ids,
        work_directory=tmp_path / "tagging",
    )

    assert [result.asset_id for result in batch.results] == asset_ids
    assert batch.gpu_uuids == (SELECTED_GPU, UNSELECTED_GPU)
    assert batch.model_id == FakeTagger.model_id


def test_archaudit_fake_embedding_shards_preserve_unique_order_without_discovery(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "lora_factory.application.service.discover_nvidia_gpus",
        lambda: (_ for _ in ()).throw(AssertionError("Fake attempted NVIDIA discovery")),
    )
    controller = _controller(tmp_path)
    context = _context(
        tmp_path,
        mode=BackendMode.FAKE,
        selected_gpu_uuids=(SELECTED_GPU, UNSELECTED_GPU),
    )
    paths: list[Path] = []
    for index in range(5):
        path = tmp_path / f"embedding-shard-{index}.png"
        Image.new("RGB", (32, 32), (index * 20, 40, 80)).save(path)
        paths.append(path)

    sharded = controller._embed_many_sharded(
        context,
        task=GpuTaskKind.REFERENCE_EMBED,
        paths=[*paths, paths[0]],
        work_directory=tmp_path / "embedding",
    )

    assert [item.source_path for item in sharded.result.items] == [path.resolve() for path in paths]
    assert sharded.gpu_uuids == (SELECTED_GPU, UNSELECTED_GPU)
    assert sharded.result.model_id == FakeImageEmbeddingBackend.model_id
    assert sharded.result.dimension == 192
    assert {
        str(definition.backend_versions["image_embedding"])
        for definition in controller._stages(context)
    } == {"fake-image-embedding/1"}


def test_archaudit_real_embedding_shards_only_selected_fresh_gpu_bindings(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    context = _context(
        tmp_path,
        mode=BackendMode.REAL,
        selected_gpu_uuids=(SELECTED_GPU, UNSELECTED_GPU),
    )
    database = Database(context.layout.database)
    database.initialize()
    with database.session() as session:
        session.add(
            ProjectRow(
                id=context.config.project_id,
                name=context.config.lora_name,
                root_path=str(context.layout.root),
                config_json={},
            )
        )
        session.add(
            RunRow(
                id=context.run_id,
                project_id=context.config.project_id,
                status=RunStatus.ACTIVE.value,
            )
        )
    devices = (
        _device(OUTSIDE_GPU, index=0, free_vram_mb=23 * 1024),
        _device(SELECTED_GPU, index=2, free_vram_mb=12 * 1024),
        _device(UNSELECTED_GPU, index=3, free_vram_mb=16 * 1024),
    )
    monkeypatch.setattr("lora_factory.application.service.discover_nvidia_gpus", lambda: devices)
    monkeypatch.setattr("lora_factory.gpu.execution.discover_nvidia_gpus", lambda: devices)
    controller = _controller(tmp_path)
    bindings: list[GpuBinding] = []

    def embedding_factory(
        _context: PipelineContext,
        *,
        binding: GpuBinding | None = None,
        work_directory: Path,
    ) -> FakeImageEmbeddingBackend:
        del work_directory
        assert binding is not None
        bindings.append(binding)
        return FakeImageEmbeddingBackend()

    controller._embedding_backend = embedding_factory  # type: ignore[method-assign]
    paths: list[Path] = []
    for index in range(6):
        path = tmp_path / f"real-embedding-shard-{index}.png"
        Image.new("RGB", (32, 32), (index * 20, 40, 80)).save(path)
        paths.append(path)

    sharded = controller._embed_many_sharded(
        context,
        task=GpuTaskKind.REFERENCE_EMBED,
        paths=paths,
        work_directory=tmp_path / "embedding",
    )

    assert [item.source_path for item in sharded.result.items] == [path.resolve() for path in paths]
    assert sharded.gpu_uuids == (SELECTED_GPU, UNSELECTED_GPU)
    assert {binding.uuid for binding in bindings} == {SELECTED_GPU, UNSELECTED_GPU}
    assert {binding.physical_index for binding in bindings} == {2, 3}
    assert all(binding.uuid != OUTSIDE_GPU for binding in bindings)
    embedding_version = str(controller._stages(context)[0].backend_versions["image_embedding"])
    assert str(controller.runtime.manifest["image_embedding"]["revision"]) in embedding_version
    assert (
        str(
            controller.runtime.manifest["image_embedding"]["artifacts"]["model.safetensors"][
                "sha256"
            ]
        )
        in embedding_version
    )
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(GpuLeaseRow)) == 0
