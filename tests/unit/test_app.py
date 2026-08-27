from __future__ import annotations

from types import SimpleNamespace

import lora_factory.app as app_module


def test_create_default_controller_does_not_evaluate_protocol_at_runtime(monkeypatch) -> None:
    class FakeController:
        pass

    monkeypatch.setattr(
        app_module,
        "import_module",
        lambda _name: SimpleNamespace(LoRAFactoryController=FakeController),
    )

    controller = app_module.create_default_controller()

    assert isinstance(controller, FakeController)
