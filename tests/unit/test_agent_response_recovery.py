from __future__ import annotations

import pytest

from litecoder.agent.loop import _invalid_completed_response_error
from litecoder.common.errors import ErrorCode
from litecoder.providers.models import StopReason


@pytest.mark.parametrize(
    ("blocks", "stop_reason", "raw_reason", "failure_code"),
    [
        (
            [],
            StopReason.UNKNOWN,
            "missing response.completed",
            "missing_response_completed",
        ),
        ([], StopReason.UNKNOWN, "future_stop", "unknown_stop_reason"),
        ([], StopReason.TOOL_USE, "tool_use", "tool_use_without_calls"),
        ([], StopReason.END_TURN, "end_turn", "empty_response"),
    ],
)
def test_invalid_completed_responses_receive_stable_repair_codes(
    blocks: list[dict[str, object]],
    stop_reason: StopReason,
    raw_reason: str,
    failure_code: str,
) -> None:
    error = _invalid_completed_response_error(blocks, stop_reason, raw_reason)

    assert error is not None
    assert error.code is ErrorCode.PROVIDER_INVALID_RESPONSE
    assert error.retryable is True
    assert error.details["provider_error_type"] == failure_code


def test_nonempty_completed_response_does_not_request_repair() -> None:
    error = _invalid_completed_response_error(
        [{"type": "text", "text": "done"}],
        StopReason.END_TURN,
        "end_turn",
    )

    assert error is None
