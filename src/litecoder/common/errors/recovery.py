"""Recovery actions and policies for failed operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from litecoder.common.errors.types import ErrorCode, LiteCoderError
from litecoder.common.errors.retry import RepairBudget, RetryBudget


RecoveryActionKind = Literal[
    "retry",
    "retry_with_feedback",
    "compact_then_retry",
    "stop_incomplete",
    "fail",
]


class FailureOrigin(StrEnum):
    """Enumeration of the failure origin values."""
    PROVIDER_TRANSPORT = "provider_transport"
    PROVIDER_RESPONSE = "provider_response"
    CONTEXT = "context"
    INTERNAL = "internal"


class RecoveryStrategy(StrEnum):
    """Enumeration of the recovery strategy values."""
    RETRY_SAME_REQUEST = "retry_same_request"
    RETRY_WITH_FEEDBACK = "retry_with_feedback"
    COMPACT_THEN_RETRY = "compact_then_retry"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    """Data model representing the recovery action."""
    kind: RecoveryActionKind
    reason: str
    delay_seconds: float = 0.0
    failure_origin: FailureOrigin = FailureOrigin.INTERNAL
    failure_code: str = ErrorCode.INTERNAL.value
    strategy: RecoveryStrategy = RecoveryStrategy.STOP
    attempt: int | None = None
    max_attempts: int | None = None
    feedback_code: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    """Data model representing the recovery context."""
    can_compact: bool = False
    has_attempted_reactive_compaction: bool = False


class RecoveryPolicy:
    """Component responsible for the recovery policy."""
    def __init__(
        self,
        retry_budget: RetryBudget | None = None,
        repair_budget: RepairBudget | None = None,
    ) -> None:
        self.retry_budget = retry_budget or RetryBudget()
        self.repair_budget = repair_budget or RepairBudget()

    def choose(
        self, error: LiteCoderError, context: RecoveryContext
    ) -> RecoveryAction:
        """Choose the appropriate recovery action."""
        if error.code is ErrorCode.CONTEXT_OVERFLOW:
            if not context.can_compact:
                return RecoveryAction(
                    "stop_incomplete",
                    "context_overflow cannot be compacted",
                    failure_origin=FailureOrigin.CONTEXT,
                    failure_code=ErrorCode.CONTEXT_OVERFLOW.value,
                )
            if context.has_attempted_reactive_compaction:
                return RecoveryAction(
                    "stop_incomplete",
                    "context_overflow retry budget exhausted",
                    failure_origin=FailureOrigin.CONTEXT,
                    failure_code=ErrorCode.CONTEXT_OVERFLOW.value,
                    attempt=1,
                    max_attempts=1,
                )
            decision = self.retry_budget.consume(error.code.value)
            if decision.allowed:
                return RecoveryAction(
                    "compact_then_retry",
                    ErrorCode.CONTEXT_OVERFLOW.value,
                    failure_origin=FailureOrigin.CONTEXT,
                    failure_code=ErrorCode.CONTEXT_OVERFLOW.value,
                    strategy=RecoveryStrategy.COMPACT_THEN_RETRY,
                    attempt=1,
                    max_attempts=1,
                )
            return RecoveryAction(
                "stop_incomplete",
                "context_overflow retry budget exhausted",
                failure_origin=FailureOrigin.CONTEXT,
                failure_code=ErrorCode.CONTEXT_OVERFLOW.value,
                attempt=1,
                max_attempts=1,
            )
        if error.code is ErrorCode.PROVIDER_INVALID_RESPONSE:
            failure_code = _response_failure_code(error)
            decision = self.repair_budget.consume(failure_code)
            if decision.allowed:
                return RecoveryAction(
                    "retry_with_feedback",
                    failure_code,
                    failure_origin=FailureOrigin.PROVIDER_RESPONSE,
                    failure_code=failure_code,
                    strategy=RecoveryStrategy.RETRY_WITH_FEEDBACK,
                    attempt=decision.attempt,
                    max_attempts=self.repair_budget.max_attempts,
                    feedback_code=failure_code,
                )
            return RecoveryAction(
                "stop_incomplete",
                "provider response repair budget exhausted",
                failure_origin=FailureOrigin.PROVIDER_RESPONSE,
                failure_code=failure_code,
                attempt=self.repair_budget.total_attempts,
                max_attempts=self.repair_budget.max_attempts,
            )
        if error.retryable:
            decision = self.retry_budget.consume(error.code.value)
            if decision.allowed:
                return RecoveryAction(
                    "retry",
                    error.code.value,
                    decision.delay_seconds,
                    failure_origin=FailureOrigin.PROVIDER_TRANSPORT,
                    failure_code=error.code.value,
                    strategy=RecoveryStrategy.RETRY_SAME_REQUEST,
                    attempt=decision.attempt,
                    max_attempts=self.retry_budget.max_attempts,
                )
            return RecoveryAction(
                "stop_incomplete",
                f"{error.code.value} retry budget exhausted",
                failure_origin=FailureOrigin.PROVIDER_TRANSPORT,
                failure_code=error.code.value,
                attempt=decision.attempt,
                max_attempts=self.retry_budget.max_attempts,
            )
        return RecoveryAction(
            "fail", error.code.value, failure_code=error.code.value
        )


def _response_failure_code(error: LiteCoderError) -> str:
    value = error.details.get("response_failure_code")
    if isinstance(value, str) and value.strip():
        return value
    kind = error.details.get("provider_error_type")
    if kind == "invalid_tool_arguments":
        reason = error.details.get("provider_data_reason")
        lowered = reason.lower() if isinstance(reason, str) else ""
        if "malformed" in lowered:
            return "malformed_tool_arguments"
        if "json object" in lowered:
            return "non_object_tool_arguments"
        if "missing" in lowered or "identity" in lowered:
            return "missing_tool_call_identity"
        if "unsupported" in lowered:
            return "unsupported_tool_arguments"
        return "invalid_tool_arguments"
    if kind in {"missing_finish_reason", "missing_response_completed"}:
        return "missing_response_completed"
    if kind == "invalid_provider_data":
        return "invalid_provider_stream"
    if isinstance(kind, str) and kind.strip():
        return kind
    return "invalid_provider_response"
