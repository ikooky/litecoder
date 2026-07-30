"""Helpers for emitting trace records."""

from __future__ import annotations

import math
from collections.abc import Mapping

from litecoder.common.trace.context import TraceContext
from litecoder.common.trace.redaction import current_secret_redactor


_LIFECYCLE_FACTS = frozenset(
    {"start", "end", "status", "duration", "result", "error"}
)
_JSON_ERROR = "attributes must be JSON-compatible"


async def trace_annotation(
    *,
    intent: str | None,
    reason: str | None,
    attributes: Mapping[str, object],
) -> None:
    """Handle the trace annotation operation."""
    if intent is None and reason is None:
        raise ValueError("intent or reason is required")

    normalized = _json_mapping(attributes, set())
    if _LIFECYCLE_FACTS.intersection(normalized):
        raise ValueError("lifecycle facts are not allowed in trace annotations")

    context = TraceContext.current()
    payload = {
        "event": "trace.annotation",
        "trace_id": context.trace_id,
        "span_id": context.span_id,
        "parent_span_id": context.parent_span_id,
        "root_session_id": context.root_session_id,
        "session_id": context.session_id,
        "agent_id": context.agent_id,
        "intent": intent,
        "reason": reason,
        "attributes": normalized,
    }
    redactor = current_secret_redactor()
    redacted = redactor.redact_data(payload)
    assert isinstance(redacted, dict)
    await context.recorder.record(redacted)


def _json_mapping(
    value: Mapping[object, object], active: set[int]
) -> dict[str, object]:
    identity = id(value)
    if identity in active:
        raise ValueError(_JSON_ERROR)
    active.add(identity)
    try:
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(_JSON_ERROR)
            _validate_text(key)
            result[key] = _json_value(item, active)
        return result
    finally:
        active.remove(identity)


def _json_value(value: object, active: set[int]) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        _validate_text(value)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(_JSON_ERROR)
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value, active)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(_JSON_ERROR)
        active.add(identity)
        try:
            return [_json_value(item, active) for item in value]
        finally:
            active.remove(identity)
    raise ValueError(_JSON_ERROR)


def _validate_text(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(_JSON_ERROR) from None