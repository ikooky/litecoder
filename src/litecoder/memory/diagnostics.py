"""Bounded memory diagnostics."""

from __future__ import annotations

from litecoder.common.errors import ErrorCode


MAX_DIAGNOSTIC_COUNT = 1_000_000

_ALLOWED_STATUSES = {
    "load": frozenset({"recalled"}),
    "extract": frozenset({
        "completed",
        "empty",
        "provider_failed",
        "truncated",
        "malformed",
        "partial_rejected",
        "failed",
        "timeout",
    }),
    "dream": frozenset({
        "completed",
        "rejected",
        "conflict",
        "failed",
        "timeout",
    }),
}
_COUNT_FIELDS = {
    "load": ("count",),
    "extract": ("accepted", "rejected", "written"),
    "dream": ("before", "after"),
}
_PROVIDER_CODES = frozenset(code.value for code in ErrorCode)


def memory_diagnostic(
    operation: object,
    status: object,
    **fields: object,
) -> dict[str, object]:
    """Return a bounded diagnostic containing only trusted allowlisted fields."""
    if (
        not isinstance(operation, str)
        or not isinstance(status, str)
        or status not in _ALLOWED_STATUSES.get(operation, ())
    ):
        return {"operation": "memory", "status": "failed"}

    event: dict[str, object] = {
        "operation": operation,
        "status": status,
    }
    for name in _COUNT_FIELDS.get(operation, ()):
        value = fields.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            event[name] = min(MAX_DIAGNOSTIC_COUNT, max(0, value))

    if operation == "extract" and status == "provider_failed":
        code = fields.get("code")
        if isinstance(code, str) and code in _PROVIDER_CODES:
            event["code"] = code
    if operation == "extract" and status == "truncated":
        limit = fields.get("limit")
        if isinstance(limit, int) and not isinstance(limit, bool):
            event["limit"] = min(MAX_DIAGNOSTIC_COUNT, max(0, limit))
    return event
