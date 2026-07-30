"""Secret detection and redaction for diagnostic output."""

from __future__ import annotations

import re
from collections.abc import Callable

from litecoder.common.trace import SecretRedactor
from litecoder.ui.events import RuntimeUIEvent
from litecoder.ui.sink import RuntimeUISink


_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)


def redact_event(event: RuntimeUIEvent, redactor: SecretRedactor) -> RuntimeUIEvent:
    """Handle the redact event operation."""
    payload = redactor.redact_data(dict(event.payload))
    if not isinstance(payload, dict):
        payload = {"redaction_error": True}
    return RuntimeUIEvent(
        event.type,
        sequence=event.sequence,
        timestamp=event.timestamp,
        session_id=_redact_optional(event.session_id, redactor),
        root_session_id=_redact_optional(event.root_session_id, redactor),
        trace_id=_redact_optional(event.trace_id, redactor),
        span_id=_redact_optional(event.span_id, redactor),
        request_id=_redact_optional(event.request_id, redactor),
        tool_call_id=_redact_optional(event.tool_call_id, redactor),
        tool_name=_redact_optional(event.tool_name, redactor),
        payload=payload,  # type: ignore[arg-type]
    )


class RedactingUISink:
    """Component responsible for the redacting ui sink."""
    def __init__(self, sink: RuntimeUISink, redactor: SecretRedactor) -> None:
        self.sink = sink
        self.redactor = redactor

    def emit(self, event: RuntimeUIEvent) -> object:
        """Emit the supplied event."""
        return self.sink.emit(redact_event(event, self.redactor))

    def flush(self) -> object:
        """Flush pending output."""
        return self.sink.flush()


def _redact_optional(value: str | None, redactor: SecretRedactor) -> str | None:
    return None if value is None else redactor.redact_text(value)


class StreamingEventTextRedactor:
    """Component responsible for the streaming event text redactor."""
    def __init__(self, redactor: SecretRedactor, emit: Callable[[str], object]) -> None:
        self.redactor = redactor
        self.emit = emit
        self._buffer = ""
        self._holdback = max((len(value) for value in redactor.values), default=1) - 1

    def write(self, text: str) -> object:
        """Write the supplied data."""
        if not text:
            return None
        self._buffer += text
        cut = max(0, len(self._buffer) - self._holdback)
        cut = min(cut, _bearer_hold_start(self._buffer))
        for match in _BEARER.finditer(self._buffer):
            if match.start() < cut < match.end():
                cut = match.start()
        for secret in self.redactor.values:
            start = self._buffer.find(secret)
            while start >= 0:
                end = start + len(secret)
                if start < cut < end:
                    cut = start
                start = self._buffer.find(secret, start + 1)
        if cut <= 0:
            return None
        ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        rendered = self.redactor.redact_text(ready)
        if rendered:
            return self.emit(rendered)
        return None

    def flush(self) -> object:
        """Flush pending output."""
        if not self._buffer:
            return None
        value, self._buffer = self._buffer, ""
        rendered = self.redactor.redact_text(value)
        if rendered:
            return self.emit(rendered)
        return None


def _bearer_hold_start(value: str) -> int:
    lowered = value.casefold()
    marker = "bearer"
    hold = len(value)
    for size in range(1, len(marker)):
        if lowered.endswith(marker[:size]):
            hold = min(hold, len(value) - size)
    start = lowered.rfind(marker)
    if start < 0:
        return hold
    tail = value[start + len(marker) :]
    if not tail or tail.isspace():
        return min(hold, start)
    if tail[0].isspace():
        token = tail.lstrip()
        if token and not any(character.isspace() for character in token):
            return min(hold, start)
    return hold
