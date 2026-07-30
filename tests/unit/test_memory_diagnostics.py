from __future__ import annotations

from litecoder.common.errors import ErrorCode
from litecoder.memory.diagnostics import (
    MAX_DIAGNOSTIC_COUNT,
    memory_diagnostic,
)


def test_memory_diagnostic_allows_only_bounded_trusted_fields() -> None:
    event = memory_diagnostic(
        "extract",
        "completed",
        accepted=2,
        rejected=-4,
        written=MAX_DIAGNOSTIC_COUNT + 99,
        exception="provider secret",
        filename="private-memory.md",
        body="private memory body",
        prompt="private prompt",
    )

    assert event == {
        "operation": "extract",
        "status": "completed",
        "accepted": 2,
        "rejected": 0,
        "written": MAX_DIAGNOSTIC_COUNT,
    }
    assert "secret" not in str(event)
    assert "private" not in str(event)


def test_load_diagnostic_reports_only_bounded_recall_count() -> None:
    assert memory_diagnostic(
        "load",
        "recalled",
        count=MAX_DIAGNOSTIC_COUNT + 1,
        skipped=2,
        message="private provider message",
    ) == {
        "operation": "load",
        "status": "recalled",
        "count": MAX_DIAGNOSTIC_COUNT,
    }


def test_extraction_diagnostics_allow_only_safe_failure_details() -> None:
    assert memory_diagnostic(
        "extract",
        "truncated",
        limit=MAX_DIAGNOSTIC_COUNT + 1,
        message="private provider message",
    ) == {
        "operation": "extract",
        "status": "truncated",
        "limit": MAX_DIAGNOSTIC_COUNT,
    }
    assert memory_diagnostic(
        "extract",
        "provider_failed",
        code=ErrorCode.PROVIDER_RATE_LIMIT.value,
        message="private provider message",
    ) == {
        "operation": "extract",
        "status": "provider_failed",
        "code": ErrorCode.PROVIDER_RATE_LIMIT.value,
    }
    assert memory_diagnostic(
        "extract",
        "provider_failed",
        code="private-provider-code",
        message="private provider message",
    ) == {
        "operation": "extract",
        "status": "provider_failed",
    }


def test_selection_diagnostic_collapses_to_safe_failure() -> None:
    assert memory_diagnostic(
        "selection",
        "fallback",
        reason="timeout",
        provider_output="private output",
    ) == {
        "operation": "memory",
        "status": "failed",
    }


def test_unknown_diagnostic_shape_collapses_to_safe_failure() -> None:
    assert memory_diagnostic(
        "private-operation",
        "private-status",
        body="private memory body",
    ) == {
        "operation": "memory",
        "status": "failed",
    }
