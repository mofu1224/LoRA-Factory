"""Deterministic Qt configuration for the GUI test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def qapp_args() -> list[str]:
    """Keep pytest-qt independent from the interactive Windows desktop."""

    return ["lora-factory-gui-tests", "-platform", "offscreen"]
