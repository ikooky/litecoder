from __future__ import annotations

import pytest

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.common.errors.classifier import ErrorClassifier
from litecoder.common.errors.recovery import RecoveryContext, RecoveryPolicy
from litecoder.common.errors.retry import RepairBudget, RetryBudget


def context_overflow_error() -> LiteCoderError:
    return LiteCoderError(
        ErrorCode.CONTEXT_OVERFLOW,
        "context too large",
        retryable=False,
    )


def test_rate_limit_retry_budget_is_bounded() -> None:
    budget = RetryBudget(max_attempts=3, base_delay=0.1)

    decisions = [budget.consume("rate_limit") for _ in range(4)]

    assert [decision.allowed for decision in decisions] == [
        True,
        True,
        True,
        False,
    ]
    assert [decision.delay_seconds for decision in decisions] == [
        0.1,
        0.2,
        0.4,
        0.0,
    ]


def test_default_provider_retry_budget_uses_five_step_exponential_backoff() -> None:
    budget = RetryBudget()

    decisions = [budget.consume("provider_transient") for _ in range(6)]

    assert [decision.allowed for decision in decisions] == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert [decision.delay_seconds for decision in decisions] == [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        0.0,
    ]


def test_provider_retry_categories_have_independent_budgets() -> None:
    budget = RetryBudget(max_attempts=1, base_delay=0.5)

    transient = budget.consume("provider_transient")
    rate_limit = budget.consume("provider_rate_limit")

    assert transient.allowed is True
    assert rate_limit.allowed is True


def test_response_repair_budget_limits_each_failure_and_total_repairs() -> None:
    budget = RepairBudget(max_attempts=2, max_attempts_per_category=1)

    first = budget.consume("malformed_tool_arguments")
    repeated = budget.consume("malformed_tool_arguments")
    second = budget.consume("empty_response")
    exhausted = budget.consume("unknown_stop_reason")

    assert first.allowed is True and first.attempt == 1
    assert repeated.allowed is False
    assert second.allowed is True and second.attempt == 2
    assert exhausted.allowed is False


def test_invalid_provider_response_selects_feedback_retry() -> None:
    policy = RecoveryPolicy()
    error = LiteCoderError(
        ErrorCode.PROVIDER_INVALID_RESPONSE,
        "invalid response",
        retryable=True,
        details={"provider_error_type": "invalid_tool_arguments"},
    )

    action = policy.choose(error, RecoveryContext())

    assert action.kind == "retry_with_feedback"
    assert action.failure_origin == "provider_response"
    assert action.failure_code == "invalid_tool_arguments"
    assert action.attempt == 1
    assert action.max_attempts == 2


def test_context_overflow_selects_compaction_before_retry() -> None:
    action = RecoveryPolicy().choose(
        context_overflow_error(), RecoveryContext(can_compact=True)
    )

    assert action.kind == "compact_then_retry"
    assert action.reason == "context_overflow"


def test_retryable_error_consumes_budget_then_stops_incomplete() -> None:
    policy = RecoveryPolicy(RetryBudget(max_attempts=1, base_delay=0.25))
    error = LiteCoderError(
        ErrorCode.PROVIDER_RATE_LIMIT,
        "slow down",
        retryable=True,
    )

    first = policy.choose(error, RecoveryContext())
    second = policy.choose(error, RecoveryContext())

    assert first.kind == "retry"
    assert first.delay_seconds == 0.25
    assert second.kind == "stop_incomplete"
    assert second.reason == "provider_rate_limit retry budget exhausted"


def test_classifier_preserves_litecoder_error_and_classifies_transient_io() -> None:
    original = LiteCoderError(
        ErrorCode.PROVIDER_TRANSIENT,
        "temporary",
        retryable=True,
    )

    assert ErrorClassifier().classify(original) is original
    classified = ErrorClassifier().classify(ConnectionError("network down"))

    assert classified.code is ErrorCode.PROVIDER_TRANSIENT
    assert classified.retryable is True
    assert classified.details["exception_type"] == "ConnectionError"

def test_context_overflow_compaction_is_limited_per_turn() -> None:
    policy = RecoveryPolicy(RetryBudget(max_attempts=2, base_delay=0.0))
    error = context_overflow_error()

    first = policy.choose(error, RecoveryContext(can_compact=True))
    repeated = policy.choose(
        error,
        RecoveryContext(
            can_compact=True,
            has_attempted_reactive_compaction=True,
        ),
    )
    next_turn = policy.choose(error, RecoveryContext(can_compact=True))

    assert first.kind == "compact_then_retry"
    assert repeated.kind == "stop_incomplete"
    assert repeated.reason == "context_overflow retry budget exhausted"
    assert next_turn.kind == "compact_then_retry"
