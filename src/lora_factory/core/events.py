"""Thread-safe event bus for headless and Qt consumers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    event_type: str
    message: str
    stage: str | None = None
    progress: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


EventHandler = Callable[[PipelineEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        with self._lock:
            self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers[event_type]:
                    self._handlers[event_type].remove(handler)

        return unsubscribe

    def publish(self, event: PipelineEvent) -> None:
        with self._lock:
            handlers = [*self._handlers.get(event.event_type, []), *self._handlers.get("*", [])]
        for handler in handlers:
            handler(event)
