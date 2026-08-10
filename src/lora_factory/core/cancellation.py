"""Cooperative cancellation safe for worker threads."""

from __future__ import annotations

from threading import Event


class CancelledError(RuntimeError):
    """Raised at a safe stage boundary after user cancellation."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("Operation cancelled by the user")
