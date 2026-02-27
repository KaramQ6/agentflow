"""Event system for pipeline streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from .types import Event


class EventEmitter:
    """Async event emitter for pipeline progress."""

    def __init__(self):
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()

    def emit(self, event_type: str, agent: str = "", **data: Any) -> None:
        """Emit an event (non-blocking)."""
        event = Event(type=event_type, agent=agent, data=data)
        self._queue.put_nowait(event)

    def done(self) -> None:
        """Signal that no more events will be emitted."""
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncGenerator[Event, None]:
        """Async generator that yields events until done."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
