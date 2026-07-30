"""Output safety and size guards."""

from __future__ import annotations

import io
import logging
import sys
from types import TracebackType
from typing import TextIO


class _DiscardingBytes(io.RawIOBase):
    """Internal helper for the discarding bytes."""
    def writable(self) -> bool:
        """Return whether the stream is writable."""
        return True

    def write(self, value: bytes | bytearray) -> int:
        """Write the supplied data."""
        return len(value)


class _DiscardingText(io.TextIOBase):
    """Internal helper for the discarding text."""
    def __init__(self) -> None:
        super().__init__()
        self._buffer = _DiscardingBytes()

    @property
    def encoding(self) -> str:
        """Return the stream encoding."""
        return "utf-8"

    @property
    def errors(self) -> str:
        """Return the stream error policy."""
        return "replace"

    @property
    def buffer(self) -> _DiscardingBytes:
        """Return the underlying byte buffer."""
        return self._buffer

    def writable(self) -> bool:
        """Return whether the stream is writable."""
        return True

    def isatty(self) -> bool:
        """Return whether the stream is attached to a terminal."""
        return False

    def write(self, value: str) -> int:
        """Write the supplied data."""
        return len(value)

    def flush(self) -> None:
        """Flush pending output."""
        return None

    def reconfigure(self, **kwargs: object) -> None:
        """Handle the reconfigure operation."""
        del kwargs


class TUIOutputGuard:
    """Keep unstructured process output out of the inline Textual surface."""

    def __init__(self) -> None:
        self._stdout_sink = _DiscardingText()
        self._stderr_sink = _DiscardingText()
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._redirected_handlers: list[tuple[logging.StreamHandler, TextIO]] = []

    def __enter__(self) -> TUIOutputGuard:
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        for handler in _stream_handlers():
            stream = getattr(handler, "stream", None)
            if stream is self._stdout:
                sink = self._stdout_sink
            elif stream is self._stderr:
                sink = self._stderr_sink
            else:
                continue
            if _set_handler_stream(handler, sink):
                self._redirected_handlers.append((handler, stream))
        sys.stdout = self._stdout_sink
        sys.stderr = self._stderr_sink
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        stdout = self._stdout
        stderr = self._stderr
        if stdout is None or stderr is None:
            return
        sys.stdout = stdout
        sys.stderr = stderr

        redirected = {id(handler) for handler, _ in self._redirected_handlers}
        for handler, stream in reversed(self._redirected_handlers):
            _set_handler_stream(handler, stream)
        for handler in _stream_handlers():
            if id(handler) in redirected:
                continue
            stream = getattr(handler, "stream", None)
            if stream is self._stdout_sink:
                _set_handler_stream(handler, stdout)
            elif stream is self._stderr_sink:
                _set_handler_stream(handler, stderr)
        self._redirected_handlers.clear()
        self._stdout = None
        self._stderr = None


def _stream_handlers() -> tuple[logging.StreamHandler, ...]:
    handlers: list[logging.StreamHandler] = []
    seen: set[int] = set()
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers.extend(
        value
        for value in logging.Logger.manager.loggerDict.values()
        if isinstance(value, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            if isinstance(handler, logging.FileHandler):
                continue
            identity = id(handler)
            if identity in seen:
                continue
            seen.add(identity)
            handlers.append(handler)
    return tuple(handlers)


def _set_handler_stream(handler: logging.StreamHandler, stream: TextIO) -> bool:
    descriptor = getattr(type(handler), "stream", None)
    if isinstance(descriptor, property) and descriptor.fset is None:
        return False
    setter = getattr(handler, "setStream", None)
    if callable(setter):
        setter(stream)
    else:
        handler.stream = stream
    return True
