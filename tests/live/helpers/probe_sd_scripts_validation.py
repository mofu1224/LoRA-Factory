"""Inspect dataset groups built by the exact pinned sd-scripts trainer command."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

MARKER = "LORA_FACTORY_VALIDATION_PROBE="


class DatasetGroupView(Protocol):
    image_data: Mapping[str, object]
    num_train_images: int


def _trainer_entrypoint(command: list[str]) -> tuple[Path, list[str]]:
    for index, argument in enumerate(command):
        if Path(argument).name.casefold() == "sdxl_train_network.py":
            return Path(argument).resolve(strict=True), command[index + 1 :]
    raise ValueError("sdxl_train_network.py is missing from the command")


def _asset_ids(group: DatasetGroupView) -> list[str]:
    return sorted(Path(path).stem for path in group.image_data)


def main() -> int:
    if len(sys.argv) != 2:
        raise ValueError("Expected one command JSON path")
    command_path = Path(sys.argv[1]).resolve(strict=True)
    command = json.loads(command_path.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("Command JSON must be a string array")

    trainer_entrypoint, trainer_arguments = _trainer_entrypoint(command)
    sys.path.insert(0, str(trainer_entrypoint.parent))
    from library import config_util
    from library.config_util import BlueprintGenerator, ConfigSanitizer
    from sdxl_train_network import setup_parser

    args = setup_parser().parse_args(trainer_arguments)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
    user_config = config_util.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    training_group, validation_group = config_util.generate_dataset_group_by_blueprint(
        blueprint.dataset_group
    )
    if validation_group is None:
        raise ValueError("Pinned sd-scripts did not construct a validation dataset")
    payload = {
        "training_ids": _asset_ids(training_group),
        "validation_ids": _asset_ids(validation_group),
        "training_count": training_group.num_train_images,
        "validation_count": validation_group.num_train_images,
    }
    print(MARKER + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
