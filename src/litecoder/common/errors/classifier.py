"""Error classification for provider and tool failures."""

from __future__ import annotations

import asyncio

from litecoder.common.errors.types import ErrorCode, LiteCoderError


class ErrorClassifier:
    """Component responsible for the error classifier."""
    def classify(self, exception: BaseException) -> LiteCoderError:
        """Classify the requested operation."""
        if isinstance(exception, LiteCoderError):
            return exception
        if isinstance(exception, asyncio.CancelledError):
            return LiteCoderError(
                ErrorCode.CANCELLED,
                "Operation cancelled",
                retryable=False,
                details=_details(exception),
            )
        if isinstance(exception, (ConnectionError, TimeoutError, OSError)):
            return LiteCoderError(
                ErrorCode.PROVIDER_TRANSIENT,
                "Provider temporarily unavailable",
                retryable=True,
                details=_details(exception),
            )
        return LiteCoderError(
            ErrorCode.INTERNAL,
            "Internal error",
            retryable=False,
            details=_details(exception),
        )


def _details(exception: BaseException) -> dict[str, object]:
    return {"exception_type": type(exception).__name__}