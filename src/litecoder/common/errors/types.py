"""Shared error types and error codes."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable categories used to classify runtime failures."""
    PROVIDER_TRANSIENT = "provider_transient"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_INVALID_RESPONSE = "provider_invalid_response"
    CONTEXT_OVERFLOW = "context_overflow"
    TOOL_FAILED = "tool_failed"
    PERMISSION_DENIED = "permission_denied"
    TASK_GRAPH_INVALID = "task_graph_invalid"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class LiteCoderError(Exception):
    """Structured runtime error with a stable code and retry metadata."""
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}
