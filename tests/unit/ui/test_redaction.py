from __future__ import annotations

import pytest

from litecoder.common.trace import SecretRedactor
from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.redaction import StreamingEventTextRedactor, redact_event


def test_redact_event_redacts_payload_and_top_level_ids() -> None:
    redactor = SecretRedactor.with_values(("top-secret",))
    event = RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=1,
        timestamp=1.0,
        tool_call_id="top-secret",
        payload={"content": "Bearer abc123 and top-secret"},
    )

    rendered = redact_event(event, redactor)

    assert rendered.tool_call_id == "[REDACTED]"
    assert rendered.payload == {"content": "[REDACTED] and [REDACTED]"}


def test_streaming_event_text_redactor_handles_split_secrets() -> None:
    redactor = SecretRedactor.with_values(("split-secret",))
    emitted: list[str] = []
    stream = StreamingEventTextRedactor(redactor, emitted.append)

    stream.write("before split-")
    stream.write("secret after")
    stream.flush()

    assert "".join(emitted) == "before [REDACTED] after"


def test_redacting_sink_redacts_events_before_delegating() -> None:
    from litecoder.ui.redaction import RedactingUISink
    from litecoder.ui.sink import RecordingUISink

    secret = "runtime-secret"
    delegate = RecordingUISink()
    sink = RedactingUISink(delegate, SecretRedactor.with_values((secret,)))
    event = RuntimeUIEvent(
        UIEventType.ASSISTANT_COMPLETED,
        sequence=1,
        timestamp=1.0,
        session_id="session-runtime-secret",
        payload={"text": f"answer {secret} Bearer abc123"},
    )

    sink.emit(event)

    assert len(delegate.events) == 1
    rendered = delegate.events[0]
    assert secret not in rendered.session_id
    assert secret not in str(rendered.payload)
    assert "abc123" not in str(rendered.payload)
    assert "[REDACTED]" in str(rendered.payload)


@pytest.mark.asyncio
async def test_redacting_sink_redacts_before_composite_fan_out() -> None:
    from litecoder.ui.redaction import RedactingUISink
    from litecoder.ui.sink import CompositeUISink, RecordingUISink, emit_ui

    secret = "runtime-secret"
    first = RecordingUISink()
    second = RecordingUISink()
    sink = RedactingUISink(
        CompositeUISink(first, second),
        SecretRedactor.with_values((secret,)),
    )
    raw_event = RuntimeUIEvent(
        UIEventType.ASSISTANT_COMPLETED,
        sequence=1,
        timestamp=1.0,
        session_id="session-runtime-secret",
        payload={"text": f"answer {secret} Bearer abc123"},
    )

    await emit_ui(sink, raw_event)

    assert secret in str(raw_event.payload)
    assert len(first.events) == 1
    assert len(second.events) == 1
    first_rendered = first.events[0]
    second_rendered = second.events[0]
    assert first_rendered is second_rendered
    for rendered in (first_rendered, second_rendered):
        assert secret not in (rendered.session_id or "")
        assert secret not in str(rendered.payload)
        assert "abc123" not in str(rendered.payload)
        assert "[REDACTED]" in str(rendered.payload)
