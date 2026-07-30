"""Public interfaces for the ui package."""

from litecoder.ui.events import RuntimeUIEvent, UIEventFactory, UIEventType
from litecoder.ui.input import TerminalInput
from litecoder.ui.permissions import select_permission_choice
from litecoder.ui.redaction import RedactingUISink
from litecoder.ui.sink import CompositeUISink, NullUISink, RecordingUISink, RuntimeUISink, emit_ui

__all__ = [
    "CompositeUISink",
    "NullUISink",
    "RecordingUISink",
    "RuntimeUIEvent",
    "RuntimeUISink",
    "RedactingUISink",
    "TerminalInput",
    "UIEventFactory",
    "UIEventType",
    "emit_ui",
    "select_permission_choice",
]
