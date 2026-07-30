"""Tracing context propagation helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol


class TraceSink(Protocol):
    """Protocol describing the trace sink behavior."""
    async def record(self, payload: Mapping[str, object]) -> None: ...


_current: ContextVar[TraceContext | None] = ContextVar(
    "litecoder_trace", default=None
)


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Data model representing the trace context."""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    root_session_id: str
    session_id: str
    agent_id: str
    recorder: TraceSink

    @classmethod
    def root(
        cls,
        trace_id: str,
        session_id: str,
        agent_id: str,
        recorder: TraceSink,
    ) -> TraceContext:
        """Create the root context."""
        return cls(
            trace_id=trace_id,
            span_id="root",
            parent_span_id=None,
            root_session_id=session_id,
            session_id=session_id,
            agent_id=agent_id,
            recorder=recorder,
        )

    @classmethod
    def current(cls) -> TraceContext:
        """Return the active context."""
        value = _current.get()
        if value is None:
            raise RuntimeError("No active TraceContext")
        return value

    @contextmanager
    def bind(self) -> Iterator[TraceContext]:
        """Temporarily bind this context for nested operations."""
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)