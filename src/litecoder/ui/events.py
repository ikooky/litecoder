"""Trace event data models and event names."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from litecoder.providers._json import JsonValue, snapshot_mapping


class UIEventType(StrEnum):
    """Well-known event names emitted by the runtime UI layer."""
    TURN_STARTED = "turn.started"
    TURN_FINISHED = "turn.finished"
    MODEL_REQUESTED = "model.requested"
    MODEL_REQUEST_ID = "model.request_id"
    MODEL_COMPLETED = "model.completed"
    ASSISTANT_DELTA = "assistant.delta"
    ASSISTANT_COMPLETED = "assistant.completed"
    THINKING_STARTED = "thinking.started"
    THINKING_DELTA = "thinking.delta"
    THINKING_COMPLETED = "thinking.completed"
    TOOL_CALL_STARTED = "tool_call.started"
    TOOL_CALL_INPUT_DELTA = "tool_call.input_delta"
    TOOL_CALL_COMPLETED = "tool_call.completed"
    TOOL_EXECUTION_STARTED = "tool_execution.started"
    TOOL_EXECUTION_FINISHED = "tool_execution.finished"
    TOOL_EXECUTION_FAILED = "tool_execution.failed"
    TOOL_EXECUTION_DENIED = "tool_execution.denied"
    TODO_UPDATED = "todo.updated"
    PROVIDER_ERROR = "provider.error"
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_RESOLVED = "permission.resolved"
    NOTICE_RAISED = "notice.raised"
    DIAGNOSTIC = "diagnostic"
    USAGE_UPDATED = "usage.updated"


@dataclass(frozen=True, slots=True)
class RuntimeUIEvent:
    """Validated immutable event carrying one UI lifecycle notification."""
    type: UIEventType | str
    sequence: int
    timestamp: float
    session_id: str | None = None
    root_session_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    request_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    payload: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            event_type = UIEventType(self.type)
        except ValueError as error:
            raise ValueError(f"unknown UI event type: {self.type!r}") from error
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be a positive integer")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or self.timestamp < 0
        ):
            raise ValueError("timestamp must be a non-negative number")
        for field_name in (
            "session_id",
            "root_session_id",
            "trace_id",
            "span_id",
            "request_id",
            "tool_call_id",
            "tool_name",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string")
        payload = snapshot_mapping(self.payload, "UI event payload")
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "payload", MappingProxyType(payload))


class UIEventFactory:
    """Create sequential events with shared session and trace identifiers."""
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        session_id: str | None = None,
        root_session_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        self._clock = clock
        self._next_sequence = 1
        self.session_id = session_id
        self.root_session_id = root_session_id
        self.trace_id = trace_id
        self.span_id = span_id

    def next(
        self,
        event_type: UIEventType | str,
        *,
        request_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> RuntimeUIEvent:
        """Create the next event in the factory's sequence."""
        event = RuntimeUIEvent(
            event_type,
            sequence=self._next_sequence,
            timestamp=float(self._clock()),
            session_id=self.session_id,
            root_session_id=self.root_session_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            request_id=request_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            payload=payload or {},
        )
        self._next_sequence += 1
        return event
