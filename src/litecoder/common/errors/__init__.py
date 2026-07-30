"""Public interfaces for the errors package."""

from litecoder.common.errors.classifier import ErrorClassifier
from litecoder.common.errors.recovery import (
    FailureOrigin,
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
    RecoveryStrategy,
)
from litecoder.common.errors.retry import RepairBudget, RetryBudget, RetryDecision
from litecoder.common.errors.types import ErrorCode, LiteCoderError

__all__ = [
    "ErrorClassifier",
    "ErrorCode",
    "FailureOrigin",
    "LiteCoderError",
    "RepairBudget",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryPolicy",
    "RecoveryStrategy",
    "RetryBudget",
    "RetryDecision",
]
