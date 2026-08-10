"""Deterministic sampler that emits real PNGs and metadata sidecars."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from lora_factory.sampling.backend import SampleRequest, SampleResult
from lora_factory.util.json import write_json_atomic


class FakeSampler:
    @property
    def version(self) -> str:
        return "fake-sampler/1"

    def sample(self, request: SampleRequest, output_directory: Path) -> SampleResult:
        output_directory.mkdir(parents=True, exist_ok=True)
        identifier = (
            f"{request.checkpoint_id}-{request.prompt_id}-{request.seed}-{request.weight:.2f}"
        )
        digest = hashlib.sha256(identifier.encode("utf-8")).digest()
        color = tuple(48 + component % 160 for component in digest[:3])
        image = Image.new("RGB", (request.width, request.height), color=color)
        draw = ImageDraw.Draw(image)
        margin = max(min(request.width, request.height) // 24, 8)
        draw.rectangle(
            (margin, margin, request.width - margin, request.height - margin),
            outline="white",
            width=max(margin // 4, 2),
        )
        draw.text(
            (margin * 2, margin * 2),
            f"{request.prompt_id}\nseed {request.seed}\nweight {request.weight:.2f}",
            fill="white",
        )
        image_path = output_directory / f"{identifier}.png"
        metadata_path = image_path.with_suffix(".json")
        image.save(image_path, format="PNG")
        write_json_atomic(
            metadata_path,
            {
                **request.model_dump(mode="json"),
                "backend": self.version,
                "image_filename": image_path.name,
            },
        )
        return SampleResult(
            request=request,
            image_path=image_path,
            metadata_path=metadata_path,
            success=True,
        )
