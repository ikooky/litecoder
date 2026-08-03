from __future__ import annotations

import pytest

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.providers.completion import complete_text_with_retry
from litecoder.providers.models import ModelRequest, ProviderEvent, StopReason
from tests.fakes.provider import FakeProvider


def request_factory(max_tokens: int) -> ModelRequest:
    return ModelRequest(
        model="model",
        system=[{"type": "text", "text": "system"}],
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "prompt"}],
        }],
        tools=[],
        max_tokens=max_tokens,
    )


async def no_sleep(delay: float) -> None:
    del delay


@pytest.mark.asyncio
async def test_max_tokens_retry_increases_the_request_limit() -> None:
    provider = FakeProvider([
        [
            ProviderEvent.text_delta(0, "part"),
            ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens"),
        ],
        [
            ProviderEvent.text_delta(0, "ial"),
            ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
        ],
    ])

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        sleep=no_sleep,
    )

    assert result.text == "partial"
    assert result.attempts == 2
    assert [request.max_tokens for request in provider.requests] == [2_000, 4_000]
    continuation_messages = provider.requests[1].messages[-2:]
    assert continuation_messages[0]["role"] == "assistant"
    assert continuation_messages[0]["content"] == [
        {"type": "text", "text": "part"}
    ]
    assert continuation_messages[1]["role"] == "user"
    assert "do not repeat text already emitted" in str(
        continuation_messages[1]["content"]
    )


@pytest.mark.asyncio
async def test_empty_response_is_retried_with_the_same_limit() -> None:
    provider = FakeProvider([
        [ProviderEvent.response_completed(StopReason.END_TURN, "end_turn")],
        [
            ProviderEvent.text_delta(0, "summary"),
            ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
        ],
    ])

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        sleep=no_sleep,
    )

    assert result.text == "summary"
    assert [request.max_tokens for request in provider.requests] == [2_000, 2_000]


@pytest.mark.asyncio
async def test_empty_continuation_completes_the_accumulated_response() -> None:
    provider = FakeProvider([
        [
            ProviderEvent.text_delta(0, "partial"),
            ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens"),
        ],
        [ProviderEvent.response_completed(StopReason.END_TURN, "end_turn")],
        [
            ProviderEvent.text_delta(0, "suffix"),
            ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
        ],
    ])

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        sleep=no_sleep,
    )

    assert result.text == "partialsuffix"
    assert result.stop_reason is StopReason.END_TURN
    assert len(provider.requests) == 3


@pytest.mark.asyncio
async def test_empty_continuation_does_not_return_a_partial_success() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.text_delta(0, "partial"),
                ProviderEvent.response_completed(
                    StopReason.MAX_TOKENS, "max_tokens"
                ),
            ],
        ]
        + [
            [ProviderEvent.response_completed(StopReason.END_TURN, "end_turn")]
            for _ in range(5)
        ]
    )

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        sleep=no_sleep,
    )

    assert result.text == ""
    assert result.stop_reason is StopReason.END_TURN
    assert len(provider.requests) == 6


@pytest.mark.asyncio
async def test_retryable_provider_error_is_retried() -> None:
    provider = FakeProvider([
        [
            ProviderEvent.provider_error(
                LiteCoderError(
                    ErrorCode.PROVIDER_TRANSIENT,
                    "temporary",
                    retryable=True,
                )
            )
        ],
        [
            ProviderEvent.text_delta(0, "summary"),
            ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
        ],
    ])

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        sleep=no_sleep,
    )

    assert result.text == "summary"
    assert result.provider_error is None
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_retryable_provider_error_uses_shared_exponential_backoff() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.provider_error(
                    LiteCoderError(
                        ErrorCode.PROVIDER_TRANSIENT,
                        "temporary",
                        retryable=True,
                    )
                )
            ]
            for _ in range(5)
        ]
        + [
            [
                ProviderEvent.text_delta(0, "summary"),
                ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
            ]
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        sleep=record_sleep,
    )

    assert result.text == "summary"
    assert len(provider.requests) == 6
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_max_tokens_holds_the_expanded_limit_for_followup_retries() -> None:
    provider = FakeProvider([
        [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
        [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
        [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
        [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
    ])

    result = await complete_text_with_retry(
        provider,
        request_factory,
        initial_max_tokens=2_000,
        max_output_tokens=8_000,
        sleep=no_sleep,
    )

    assert result.stop_reason is StopReason.MAX_TOKENS
    assert result.attempts == 4
    assert [request.max_tokens for request in provider.requests] == [
        2_000,
        4_000,
        4_000,
        4_000,
    ]
