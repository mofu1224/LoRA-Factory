"""Bounded three-pass checkpoint/weight/seed sampling plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatrixCell:
    checkpoint_id: str
    prompt_id: str
    weight: float
    seed: int
    pass_number: int


def screening_matrix(
    checkpoint_ids: list[str],
    prompt_ids: list[str],
    *,
    max_images: int = 256,
) -> tuple[MatrixCell, ...]:
    if max_images < 1:
        raise ValueError("max_images must be positive")
    if not checkpoint_ids or not prompt_ids:
        return ()

    cells: list[MatrixCell] = []
    for checkpoint_id in checkpoint_ids:
        for prompt_id in prompt_ids:
            cells.append(MatrixCell(checkpoint_id, prompt_id, 0.8, 101, 1))
            if len(cells) >= max_images:
                return tuple(cells)

    finalists = checkpoint_ids[: min(5, len(checkpoint_ids))]
    for checkpoint_id in finalists:
        for weight in (0.6, 0.7, 0.8, 0.9, 1.0):
            for prompt_id in prompt_ids[:4]:
                cells.append(MatrixCell(checkpoint_id, prompt_id, weight, 202, 2))
                if len(cells) >= max_images:
                    return tuple(cells)

    final_ids = checkpoint_ids[: min(3, len(checkpoint_ids))]
    for checkpoint_id in final_ids:
        for weight in (0.7, 0.8, 0.9):
            for seed in (303, 404, 505):
                for prompt_id in prompt_ids[:4]:
                    cells.append(MatrixCell(checkpoint_id, prompt_id, weight, seed, 3))
                    if len(cells) >= max_images:
                        return tuple(cells)
    return tuple(cells)
