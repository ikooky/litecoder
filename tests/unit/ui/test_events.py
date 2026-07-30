from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from litecoder.ui.events import RuntimeUIEvent, UIEventFactory, UIEventType


def test_runtime_ui_event_snapshots_payload() -> None:
    source = {"text": "hello", "nested": {"count": 1}}

    event = RuntimeUIEvent(
        UIEventType.ASSISTANT_DELTA,
        sequence=1,
        timestamp=12.5,
        session_id="session-1",
        payload=source,
    )
    source["text"] = "changed"
    source["nested"]["count"] = 2  # type: ignore[index]

    assert event.type is UIEventType.ASSISTANT_DELTA
    assert event.payload == {"text": "hello", "nested": {"count": 1}}
    assert isinstance(event.payload, MappingProxyType)
    with pytest.raises(TypeError):
        event.payload["text"] = "mutated"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        event.sequence = 9  # type: ignore[misc]


def test_runtime_ui_event_rejects_invalid_payload_values() -> None:
    with pytest.raises(ValueError, match="payload"):
        RuntimeUIEvent(
            UIEventType.DIAGNOSTIC,
            sequence=1,
            timestamp=1.0,
            payload={"bad": object()},
        )


def test_ui_event_factory_assigns_ordered_sequences_and_context() -> None:
    factory = UIEventFactory(
        clock=lambda: 100.0,
        session_id="session-1",
        root_session_id="root-1",
        trace_id="trace-1",
        span_id="root",
    )

    first = factory.next(UIEventType.TURN_STARTED, payload={"prompt": "hello"})
    second = factory.next("assistant.delta", payload={"text": "world"})

    assert first.sequence == 1
    assert second.sequence == 2
    assert first.timestamp == 100.0
    assert second.type is UIEventType.ASSISTANT_DELTA
    assert second.session_id == "session-1"
    assert second.root_session_id == "root-1"
    assert second.trace_id == "trace-1"
    assert second.span_id == "root"
