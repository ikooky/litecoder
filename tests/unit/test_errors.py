from __future__ import annotations

from litecoder.common.errors import ErrorCode, LiteCoderError


def test_error_codes_are_stable_and_provider_agnostic() -> None:
    assert {code.name: code.value for code in ErrorCode} == {
        "PROVIDER_TRANSIENT": "provider_transient",
        "PROVIDER_RATE_LIMIT": "provider_rate_limit",
        "PROVIDER_INVALID_RESPONSE": "provider_invalid_response",
        "CONTEXT_OVERFLOW": "context_overflow",
        "TOOL_FAILED": "tool_failed",
        "PERMISSION_DENIED": "permission_denied",
        "TASK_GRAPH_INVALID": "task_graph_invalid",
        "CANCELLED": "cancelled",
        "INTERNAL": "internal",
    }


def test_litecoder_error_exposes_stable_fields() -> None:
    details: dict[str, object] = {"attempt": 2}

    error = LiteCoderError(
        ErrorCode.PROVIDER_TRANSIENT,
        "request temporarily unavailable",
        retryable=True,
        details=details,
    )

    assert str(error) == "request temporarily unavailable"
    assert error.code is ErrorCode.PROVIDER_TRANSIENT
    assert error.retryable is True
    assert error.details == {"attempt": 2}


def test_litecoder_error_defaults_are_not_shared() -> None:
    first = LiteCoderError(ErrorCode.INTERNAL, "first")
    second = LiteCoderError(ErrorCode.CANCELLED, "second")

    first.details["state"] = "changed"

    assert first.retryable is False
    assert second.details == {}

