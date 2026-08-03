"""Public interfaces for the errors package."""

from litecoder.common.errors.classifier import ErrorClassifier
from litecoder.common.errors.recovery import (
    FailureOrigin,
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
    RecoveryStrategy,
)
from litecoder.common.errors.retry import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    RepairBudget,
    RetryBudget,
    RetryDecision,
    next_output_max_tokens,
)
from litecoder.common.errors.types import ErrorCode, LiteCoderError

__all__ = [
    "ErrorClassifier",
    "ErrorCode",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "FailureOrigin",
    "LiteCoderError",
    "RepairBudget",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryPolicy",
    "RecoveryStrategy",
    "RetryBudget",
    "RetryDecision",
    "next_output_max_tokens",
]
