"""Public interfaces for the trace package."""

from litecoder.common.trace.context import TraceContext, TraceSink
from litecoder.common.trace.emit import trace_annotation
from litecoder.common.trace.events import TraceRecord
from litecoder.common.trace.recorder import TraceRecorder
from litecoder.common.trace.redaction import (
    SecretRedactor,
    bind_secret_redactor,
    current_secret_redactor,
)

__all__ = [
    "SecretRedactor",
    "bind_secret_redactor",
    "current_secret_redactor",
    "TraceContext",
    "TraceRecord",
    "TraceSink",
    "TraceRecorder",
    "trace_annotation",
]