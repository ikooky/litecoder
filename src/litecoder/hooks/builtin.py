"""Built-in hook implementations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from itertools import islice

from litecoder.common.trace.context import TraceContext
from litecoder.common.trace.redaction import (
    SecretRedactor,
    current_secret_redactor,
)
from litecoder.common.text import truncate_utf8_text


_TRACE_TEXT_BYTES = 1_000
_TRACE_KEY_BYTES = 256
_TRACE_COLLECTION_ITEMS = 50
_TRACE_PROJECTED_LEAVES = 64
_TRACE_MAX_DEPTH = 12
_TRACE_PAYLOAD_BYTES = 60_000
_TRACE_CONTEXT_KEYS = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "root_session_id",
    "session_id",
    "agent_id",
)
_TRACE_ESSENTIAL_FACT_KEYS = (
    "event",
    "point",
    "phase",
    "dispatch_id",
    "hook_id",
    "tool_call_id",
    "tool_name",
    "stage",
    "status",
    "error",
    "error_code",
    "failure_origin",
    "failure_code",
    "recovery_strategy",
    "attempt",
    "max_attempts",
    "delay_seconds",
    "blocked",
    "allowed",
    "mutated",
    "hard_invariant",
    "automatic_retry",
    "changed_workspace",
    "workspace_version",
    "diagnostic_count",
)


class _TraceBudget:
    """Internal helper for the trace budget."""
    def __init__(self) -> None:
        self.remaining_leaves = _TRACE_PROJECTED_LEAVES

    @property
    def exhausted(self) -> bool:
        """Return whether the resource is exhausted."""
        return self.remaining_leaves <= 0

    def claim_leaf(self) -> bool:
        """Handle the claim leaf operation."""
        if self.exhausted:
            return False
        self.remaining_leaves -= 1
        return True


class TraceHook:
    """Mandatory runtime-fact recorder used by :class:`HookManager`."""

    async def record(self, fact: Mapping[str, object]) -> None:
        """Record the supplied event or payload."""
        context = TraceContext.current()
        redactor = current_secret_redactor()
        budget = _TraceBudget()
        essential = {
            key: _trace_identity_value(fact[key], redactor)
            for key in _TRACE_ESSENTIAL_FACT_KEYS
            if key in fact
        }
        details = {
            key: _trace_value(value, set(), redactor, budget, 0)
            for key, value in fact.items()
            if key not in essential
        }
        payload = {
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "parent_span_id": context.parent_span_id,
            "root_session_id": context.root_session_id,
            "session_id": context.session_id,
            "agent_id": context.agent_id,
            **essential,
            **details,
        }
        redacted = redactor.redact_data(payload)
        assert isinstance(redacted, dict)
        rendered_bytes = len(
            json.dumps(
                redacted, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        if rendered_bytes > _TRACE_PAYLOAD_BYTES:
            redacted = _compact_trace_payload(redacted, rendered_bytes)
        await context.recorder.record(redacted)


def _trace_identity_value(value: object, redactor: SecretRedactor) -> object:
    if isinstance(value, str):
        redacted = redactor.redact_text(value)
        byte_count = len(redacted.encode("utf-8"))
        if byte_count <= _TRACE_TEXT_BYTES:
            return redacted
        return {
            "type": "text",
            "preview": truncate_utf8_text(redacted, _TRACE_TEXT_BYTES),
            "bytes": byte_count,
            "truncated": True,
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _trace_identity_value(value.value, redactor)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return {"type": _type_name(value), "truncated": True}

def _trace_value(
    value: object,
    active: set[int],
    redactor: SecretRedactor,
    budget: _TraceBudget,
    depth: int,
) -> object:
    if depth >= _TRACE_MAX_DEPTH:
        return _trace_marker(budget, "max_depth")
    if isinstance(value, str):
        if not budget.claim_leaf():
            return _truncated_marker("shared_budget")
        redacted = redactor.redact_text(value)
        byte_count = len(redacted.encode("utf-8"))
        if byte_count <= _TRACE_TEXT_BYTES:
            return redacted
        return {
            "type": "text",
            "preview": truncate_utf8_text(redacted, _TRACE_TEXT_BYTES),
            "bytes": byte_count,
            "truncated": True,
        }
    if value is None or isinstance(value, (bool, int, float)):
        if not budget.claim_leaf():
            return _truncated_marker("shared_budget")
        return value
    if isinstance(value, Enum):
        return _trace_value(value.value, active, redactor, budget, depth)
    if isinstance(value, bytes):
        if not budget.claim_leaf():
            return _truncated_marker("shared_budget")
        return {"type": "bytes", "size": len(value)}

    identity = id(value)
    if identity in active:
        if not budget.claim_leaf():
            return _truncated_marker("shared_budget")
        return {"type": _type_name(value), "cycle": True}

    if is_dataclass(value) and not isinstance(value, type):
        active.add(identity)
        try:
            preview: dict[str, object] = {}
            data_fields = [item for item in fields(value) if item.repr]
            for data_field in data_fields:
                if budget.exhausted:
                    break
                preview[data_field.name] = _trace_value(
                    getattr(value, data_field.name),
                    active,
                    redactor,
                    budget,
                    depth + 1,
                )
            if len(preview) == len(data_fields):
                return preview
            return {
                "type": _type_name(value),
                "preview": preview,
                "truncated": True,
            }
        finally:
            active.remove(identity)

    if isinstance(value, Mapping):
        active.add(identity)
        try:
            size = len(value)
            preview: dict[str, object] = {}
            for key, item in islice(value.items(), _TRACE_COLLECTION_ITEMS):
                if budget.exhausted:
                    break
                safe_key = _unique_trace_key(key, preview, redactor)
                preview[safe_key] = _trace_value(
                    item, active, redactor, budget, depth + 1
                )
            if len(preview) == size and size <= _TRACE_COLLECTION_ITEMS:
                return preview
            return {
                "type": "mapping",
                "size": size,
                "preview": preview,
                "truncated": True,
            }
        finally:
            active.remove(identity)

    if isinstance(value, Sequence) and not isinstance(value, str):
        active.add(identity)
        try:
            size = len(value)
            preview: list[object] = []
            for item in islice(value, _TRACE_COLLECTION_ITEMS):
                if budget.exhausted:
                    break
                preview.append(
                    _trace_value(
                        item, active, redactor, budget, depth + 1
                    )
                )
            if len(preview) == size and size <= _TRACE_COLLECTION_ITEMS:
                return preview
            return {
                "type": "sequence",
                "size": size,
                "preview": preview,
                "truncated": True,
            }
        finally:
            active.remove(identity)

    if not budget.claim_leaf():
        return _truncated_marker("shared_budget")
    return {"type": _type_name(value)}


def _trace_marker(budget: _TraceBudget, reason: str) -> dict[str, object]:
    budget.claim_leaf()
    return _truncated_marker(reason)


def _truncated_marker(reason: str) -> dict[str, object]:
    return {"type": "trace_projection", "reason": reason, "truncated": True}


def _compact_trace_payload(
    payload: Mapping[str, object], original_bytes: int
) -> dict[str, object]:
    compacted = {
        key: payload[key]
        for key in _TRACE_CONTEXT_KEYS
        if key in payload
    }
    compacted.update(
        {
            key: payload[key]
            for key in _TRACE_ESSENTIAL_FACT_KEYS
            if key in payload
        }
    )
    compacted["trace_projection"] = {
        "type": "trace_payload",
        "bytes": original_bytes,
        "truncated": True,
    }
    return compacted


def _unique_trace_key(
    key: object, occupied: Mapping[str, object], redactor: SecretRedactor
) -> str:
    base = truncate_utf8_text(
        redactor.redact_text(str(key)), _TRACE_KEY_BYTES
    )
    candidate = base
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}#{suffix}"
        suffix += 1
    return candidate


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
