from __future__ import annotations

import inspect

import pytest

from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.sink import (
    CompositeUISink,
    NullUISink,
    RecordingUISink,
    emit_ui,
    flush_ui,
)


def event(text: str = "hello") -> RuntimeUIEvent:
    return RuntimeUIEvent(
        UIEventType.ASSISTANT_DELTA,
        sequence=1,
        timestamp=1.0,
        payload={"text": text},
    )


@pytest.mark.asyncio
async def test_recording_sink_keeps_events_in_order() -> None:
    sink = RecordingUISink()
    first = event("one")
    second = RuntimeUIEvent(
        UIEventType.DIAGNOSTIC,
        sequence=2,
        timestamp=2.0,
        payload={"message": "two"},
    )

    await emit_ui(sink, first)
    await emit_ui(sink, second)

    assert sink.events == [first, second]


@pytest.mark.asyncio
async def test_emit_ui_supports_async_emitters() -> None:
    seen: list[RuntimeUIEvent] = []

    class AsyncSink:
        async def emit(self, item: RuntimeUIEvent) -> None:
            seen.append(item)

        async def flush(self) -> None:
            return None

    await emit_ui(AsyncSink(), event())

    assert seen == [event()]


@pytest.mark.asyncio
async def test_emit_ui_does_not_raise_renderer_failures() -> None:
    class BrokenSink:
        def emit(self, item: RuntimeUIEvent) -> None:
            raise RuntimeError("render failed")

        def flush(self) -> None:
            raise RuntimeError("flush failed")

    await emit_ui(BrokenSink(), event())
    await emit_ui(None, event())
    assert NullUISink().emit(event()) is None
    pending = emit_ui(BrokenSink(), event())
    try:
        assert inspect.isawaitable(pending) is True
    finally:
        pending.close()


@pytest.mark.asyncio
async def test_composite_sink_forwards_to_sync_and_async_children_in_order() -> None:
    calls: list[str] = []

    class SyncSink:
        def emit(self, item: RuntimeUIEvent) -> None:
            calls.append(f"sync emit {item.payload['text']}")

        def flush(self) -> None:
            calls.append("sync flush")

    class AsyncSink:
        async def emit(self, item: RuntimeUIEvent) -> None:
            calls.append(f"async emit {item.payload['text']}")

        async def flush(self) -> None:
            calls.append("async flush")

    sink = CompositeUISink(SyncSink(), AsyncSink())

    await emit_ui(sink, event("fan out"))
    await flush_ui(sink)

    assert isinstance(sink.sinks, tuple)
    assert calls == [
        "sync emit fan out",
        "async emit fan out",
        "sync flush",
        "async flush",
    ]


@pytest.mark.asyncio
async def test_composite_sink_continues_after_a_broken_child() -> None:
    class BrokenSink:
        def emit(self, item: RuntimeUIEvent) -> None:
            raise RuntimeError("render failed")

        def flush(self) -> None:
            raise RuntimeError("flush failed")

    class FlushingRecordingUISink(RecordingUISink):
        flushed = False

        def flush(self) -> None:
            self.flushed = True

    recording = FlushingRecordingUISink()
    item = event()
    sink = CompositeUISink(BrokenSink(), recording)

    await emit_ui(sink, item)
    await flush_ui(sink)

    assert recording.events == [item]
    assert recording.flushed is True
