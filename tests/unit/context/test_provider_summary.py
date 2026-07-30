from __future__ import annotations

import pytest

from litecoder.context.compaction import CompactionUnavailable, SummaryRequest
from litecoder.context.provider_summary import ProviderContextSummarizer
from litecoder.providers.models import ProviderEvent, StopReason
from tests.fakes.provider import FakeProvider


@pytest.mark.asyncio
async def test_provider_context_summarizer_returns_completed_text() -> None:
    provider = FakeProvider([[
        ProviderEvent.content_block_completed(
            0, {"type": "text", "text": "durable summary"}
        ),
        ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
    ]])
    summarize = ProviderContextSummarizer(provider, "model-a")

    result = await summarize(SummaryRequest(
        covered_through_sequence=2,
        messages=(
            {"role": "user", "content": [{"type": "text", "text": "old"}]},
        ),
    ))

    assert result == "durable summary"
    assert provider.requests[0].model == "model-a"
    assert provider.requests[0].tools == []
    assert provider.requests[0].max_tokens == 8_000
    prompt = str(provider.requests[0].system[0]["text"])
    assert "untrusted data" in prompt
    for heading in (
        "Objective and scope",
        "Constraints and decisions",
        "Files, code, and evidence",
        "Changes and validation",
        "Open work, blockers, and next action",
    ):
        assert heading in prompt


@pytest.mark.asyncio
async def test_provider_context_summarizer_uses_deltas_without_completed_block() -> None:
    provider = FakeProvider([[
        ProviderEvent.text_delta(0, "durable "),
        ProviderEvent.text_delta(0, "summary"),
        ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
    ]])

    result = await ProviderContextSummarizer(
        provider, "model-a", max_tokens=256
    )(SummaryRequest(1, ({"role": "user", "content": []},)))

    assert result == "durable summary"
    assert provider.requests[0].max_tokens == 256


@pytest.mark.asyncio
async def test_provider_context_summarizer_reports_output_limit() -> None:
    provider = FakeProvider([
        [
            ProviderEvent.text_delta(0, "partial"),
            ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens"),
        ],
        [
            ProviderEvent.text_delta(0, "still partial"),
            ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens"),
        ],
    ])

    with pytest.raises(
        CompactionUnavailable,
        match="summary generation exceeded output limit",
    ):
        await ProviderContextSummarizer(provider, "model-a")(
            SummaryRequest(1, ({"role": "user", "content": []},))
        )
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_provider_context_summarizer_retries_output_limit_once() -> None:
    provider = FakeProvider([
        [
            ProviderEvent.text_delta(0, "discarded partial"),
            ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens"),
        ],
        [
            ProviderEvent.content_block_completed(
                0, {"type": "text", "text": "durable summary"}
            ),
            ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
        ],
    ])

    result = await ProviderContextSummarizer(provider, "model-a")(
        SummaryRequest(1, ({"role": "user", "content": []},))
    )

    assert result == "durable summary"
    assert [request.max_tokens for request in provider.requests] == [8_000, 8_000]


@pytest.mark.asyncio
async def test_provider_context_summarizer_rejects_other_incomplete_responses() -> None:
    provider = FakeProvider([[
        ProviderEvent.text_delta(0, "partial"),
        ProviderEvent.response_completed(StopReason.REFUSAL, "refusal"),
    ]])

    with pytest.raises(RuntimeError, match="summary generation did not complete"):
        await ProviderContextSummarizer(provider, "model-a")(
            SummaryRequest(1, ({"role": "user", "content": []},))
        )


@pytest.mark.parametrize(
    ("model", "max_tokens", "message"),
    [("", 8_000, "model"), ("model-a", 0, "max_tokens")],
)
def test_provider_context_summarizer_validates_configuration(
    model: str,
    max_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProviderContextSummarizer(
            FakeProvider([]), model, max_tokens=max_tokens
        )
