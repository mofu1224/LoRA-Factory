"""Version-tolerant extraction of common sd-scripts progress fields."""

from __future__ import annotations

import re

from lora_factory.training.backend import TrainingProgress

EPOCH_PATTERN = re.compile(r"epoch\s*[=:]?\s*(\d+)", re.IGNORECASE)
TQDM_STEP_PATTERN = re.compile(
    r"(?:step|steps)\s*:\s*\d+%\|[^|]*\|\s*(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
STEP_PATTERN = re.compile(
    r"(?:step|steps)\s*[=:]?\s*(\d+)(?![\d%])(?:\s*/\s*(\d+))?", re.IGNORECASE
)
LOSS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:train_loss|avr_loss|loss)\s*[=:]\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
VALIDATION_PATTERN = re.compile(
    r"(?:val_epoch_avg_loss|val_avg_loss|val_loss|validation_loss|"
    r"loss/validation/(?:epoch|step)_average)\s*[=:]\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)


def parse_progress_line(line: str, *, default_total_steps: int) -> TrainingProgress | None:
    epoch_match = EPOCH_PATTERN.search(line)
    step_match = TQDM_STEP_PATTERN.search(line) or STEP_PATTERN.search(line)
    loss_match = LOSS_PATTERN.search(line)
    validation_match = VALIDATION_PATTERN.search(line)
    if not any((epoch_match, step_match, loss_match, validation_match)):
        return None
    step = int(step_match.group(1)) if step_match else 0
    total = int(step_match.group(2)) if step_match and step_match.group(2) else default_total_steps
    return TrainingProgress(
        epoch=int(epoch_match.group(1)) if epoch_match else 0,
        step=step,
        total_steps=max(1, total),
        train_loss=float(loss_match.group(1)) if loss_match else None,
        validation_loss=float(validation_match.group(1)) if validation_match else None,
        message=line.strip(),
    )
