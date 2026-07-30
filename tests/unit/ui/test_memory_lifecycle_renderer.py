from __future__ import annotations

from io import StringIO

from rich.console import Console

from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.renderers.terminal import TerminalRenderer


def _event(memory: dict[str, object]) -> RuntimeUIEvent:
    return RuntimeUIEvent(
        UIEventType.DIAGNOSTIC,
        sequence=1,
        timestamp=1.0,
        payload={"memory": memory},
    )


def test_terminal_renderer_suppresses_background_memory_lifecycle() -> None:
    stream = StringIO()
    renderer = TerminalRenderer(Console(file=stream, width=120))

    renderer.emit(_event({"operation": "extract", "status": "timeout"}))
    renderer.emit(_event({"operation": "dream", "status": "timeout"}))
    renderer.emit(_event({"operation": "dream", "status": "skipped"}))
    renderer.emit(
        _event({
            "operation": "memory",
            "status": "timeout",
            "prompt": "private memory prompt",
        })
    )

    assert stream.getvalue() == ""


def test_terminal_renderer_shows_explicit_memory_extraction_failure() -> None:
    stream = StringIO()
    renderer = TerminalRenderer(Console(file=stream, width=120))

    renderer.emit(
        _event({
            "operation": "extract",
            "status": "empty",
            "accepted": 0,
            "written": 0,
            "visible": True,
        })
    )

    assert "Memory extract: empty" in stream.getvalue()
