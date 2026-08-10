from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lora_factory.runtime_scripts.clip_embed import _encode


class _FakePixels:
    def __init__(self, owner: _FakeTorch, values: list[int]) -> None:
        self.owner = owner
        self.values = values
        owner.live_pixels += 1

    def to(self, **_kwargs: Any) -> _FakePixels:
        return self

    def __del__(self) -> None:
        self.owner.live_pixels -= 1


class _FakeProjected:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def float(self) -> _FakeProjected:
        return self


class _FakeNormalized:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def cpu(self) -> _FakeNormalized:
        return self

    def tolist(self) -> list[list[float]]:
        return [[float(value)] for value in self.values]


class _FakeCuda:
    def __init__(self, owner: _FakeTorch) -> None:
        self.owner = owner
        self.empty_cache_live_counts: list[int] = []

    def empty_cache(self) -> None:
        self.empty_cache_live_counts.append(self.owner.live_pixels)


class _FakeTorch:
    class OutOfMemoryError(RuntimeError):
        pass

    float16 = object()

    def __init__(self) -> None:
        self.live_pixels = 0
        self.cuda = _FakeCuda(self)

    def stack(self, values: list[int]) -> _FakePixels:
        return _FakePixels(self, values)

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class _OomOnceModel:
    def __init__(self, torch_module: _FakeTorch) -> None:
        self.torch_module = torch_module
        self.batch_sizes: list[int] = []

    def __call__(self, *, pixel_values: _FakePixels) -> SimpleNamespace:
        self.batch_sizes.append(len(pixel_values.values))
        if len(self.batch_sizes) == 1:
            raise self.torch_module.OutOfMemoryError("injected CLIP OOM")
        return SimpleNamespace(image_embeds=_FakeProjected(pixel_values.values))


def test_clip_helper_releases_failed_cuda_batch_before_halving_and_preserves_order() -> None:
    torch_module = _FakeTorch()
    model = _OomOnceModel(torch_module)
    paths = tuple(Path(f"{index}.png") for index in range(4))

    values = _encode(
        model,
        paths,
        4,
        torch_module=torch_module,
        normalize_fn=lambda projected, *, dim: _FakeNormalized(projected.values),
        prepare_fn=lambda path: int(path.stem),
    )

    assert values == [[0.0], [1.0], [2.0], [3.0]]
    assert model.batch_sizes == [4, 2, 2]
    assert torch_module.cuda.empty_cache_live_counts == [0]
    assert torch_module.live_pixels == 0
