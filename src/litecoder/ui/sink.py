"""UI event sinks and dispatch helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Protocol

from litecoder.ui.events import RuntimeUIEvent


class RuntimeUISink(Protocol):
    """Protocol describing the runtime ui sink behavior."""
    def emit(self, event: RuntimeUIEvent) -> object: ...
    def flush(self) -> object: ...


@dataclass(slots=True)
class RecordingUISink:
    """Data model representing the recording ui sink."""
    events: list[RuntimeUIEvent] = field(default_factory=list)

    def emit(self, event: RuntimeUIEvent) -> None:
        """Emit the supplied event."""
        self.events.append(event)

    def flush(self) -> None:
        """Flush pending output."""
        return None


class CompositeUISink:
    """Component responsible for the composite ui sink."""
    def __init__(self, *sinks: RuntimeUISink) -> None:
        self.sinks = tuple(sinks)

    async def emit(self, event: RuntimeUIEvent) -> None:
        """Emit the supplied event."""
        for sink in self.sinks:
            try:
                outcome = sink.emit(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                continue

    async def flush(self) -> None:
        """Flush pending output."""
        for sink in self.sinks:
            try:
                outcome = sink.flush()
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                continue


class NullUISink:
    """Component responsible for the null ui sink."""
    def emit(self, event: RuntimeUIEvent) -> None:
        """Emit the supplied event."""
        return None

    def flush(self) -> None:
        """Flush pending output."""
        return None


async def emit_ui(sink: RuntimeUISink | None, event: RuntimeUIEvent) -> None:
    """Emit the ui."""
    if sink is None:
        return
    try:
        outcome = sink.emit(event)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        return


async def flush_ui(sink: RuntimeUISink | None) -> None:
    """Flush the ui."""
    if sink is None:
        return
    try:
        outcome = sink.flush()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        return
