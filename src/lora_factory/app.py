"""Desktop entry point for LoRA Factory."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib import import_module
from typing import cast

from PySide6.QtWidgets import QApplication

from lora_factory.gui.contracts import ApplicationController, UnavailableController
from lora_factory.gui.main_window import MainWindow


def create_default_controller() -> ApplicationController:
    """Load Application Services lazily so setup failures never crash the GUI launcher."""

    try:
        services = import_module("lora_factory.application.service")
        controller_type = services.LoRAFactoryController
        return cast(ApplicationController, controller_type())
    except Exception as error:
        return UnavailableController(
            f"Application Services could not be initialized. {type(error).__name__}: {error}"
        )


def create_window(controller: ApplicationController | None = None) -> MainWindow:
    """Construct a window with an injectable controller for tests and packaging."""

    return MainWindow(controller or create_default_controller())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native desktop application."""

    arguments = list(argv) if argv is not None else sys.argv
    application = QApplication.instance() or QApplication(arguments)
    application.setApplicationName("LoRA Factory")
    application.setOrganizationName("LoRA Factory")
    window = create_window()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
