"""Retry scheduling and bounded backoff helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


# RetryBudget.max_attempts counts retries after the initial request.
MODEL_RETRY_MAX_ATTEMPTS = 5
MODEL_RETRY_BASE_DELAY = 0.5
MODEL_RETRY_MAX_DELAY = 8.0
MODEL_CONTINUATION_MAX_ATTEMPTS = 3
DEFAULT_MAX_OUTPUT_TOKENS = 64_000


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Data model representing the retry decision."""
    allowed: bool
    delay_seconds: float
    attempt: int


@dataclass(slots=True)
class RetryBudget:
    """Data model representing the retry budget."""
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 8.0
    attempts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or self.max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays must be non-negative")
        if self.max_delay and self.base_delay > self.max_delay:
            raise ValueError("base_delay must not exceed max_delay")

    def consume(self, category: str) -> RetryDecision:
        """Consume and return the next queued item."""
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string")
        attempts = self.attempts.get(category, 0)
        if attempts >= self.max_attempts:
            return RetryDecision(False, 0.0, attempts)
        self.attempts[category] = attempts + 1
        delay = self.base_delay * (2 ** attempts)
        if self.max_delay:
            delay = min(delay, self.max_delay)
        return RetryDecision(True, delay, attempts + 1)


def next_output_max_tokens(
    current: int,
    *,
    cap: int = DEFAULT_MAX_OUTPUT_TOKENS,
    multiplier: int = 2,
) -> int:
    """Return the next bounded output-token limit for a retry."""
    if isinstance(current, bool) or not isinstance(current, int) or current <= 0:
        raise ValueError("current max_tokens must be a positive integer")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("max_tokens cap must be a positive integer")
    if current > cap:
        raise ValueError("current max_tokens must not exceed the cap")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 2:
        raise ValueError("multiplier must be an integer of at least 2")
    if current == cap:
        return current
    return min(cap, max(current + 1, current * multiplier))


@dataclass(slots=True)
class RepairBudget:
    """Data model representing the repair budget."""
    max_attempts: int = 2
    max_attempts_per_category: int = 1
    attempts: dict[str, int] = field(default_factory=dict)
    total_attempts: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_attempts", self.max_attempts),
            ("max_attempts_per_category", self.max_attempts_per_category),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def consume(self, category: str) -> RetryDecision:
        """Consume and return the next queued item."""
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string")
        category_attempts = self.attempts.get(category, 0)
        if (
            self.total_attempts >= self.max_attempts
            or category_attempts >= self.max_attempts_per_category
        ):
            return RetryDecision(False, 0.0, self.total_attempts)
        self.attempts[category] = category_attempts + 1
        self.total_attempts += 1
        return RetryDecision(True, 0.0, self.total_attempts)
