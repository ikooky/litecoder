from __future__ import annotations

from litecoder.memory.diagnostics import memory_diagnostic


def test_phase_specific_timeouts_are_allowlisted_without_private_fields() -> None:
    assert memory_diagnostic(
        "extract",
        "timeout",
        prompt="private memory prompt",
    ) == {
        "operation": "extract",
        "status": "timeout",
    }
    assert memory_diagnostic(
        "dream",
        "timeout",
        provider_output="private provider output",
    ) == {
        "operation": "dream",
        "status": "timeout",
    }


def test_legacy_memory_timeout_collapses_to_safe_failure() -> None:
    assert memory_diagnostic(
        "memory",
        "timeout",
        prompt="private memory prompt",
    ) == {
        "operation": "memory",
        "status": "failed",
    }


def test_dream_skipped_collapses_to_safe_failure() -> None:
    assert memory_diagnostic("dream", "skipped") == {
        "operation": "memory",
        "status": "failed",
    }
