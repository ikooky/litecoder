from __future__ import annotations

import pytest

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.memory.prompts import (
    MEMORY_CONSOLIDATION_SYSTEM_PROMPT,
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    MEMORY_SELECTION_SYSTEM_PROMPT,
    MemorySideQueryResult,
    complete_side_query,
    complete_side_query_result,
    parse_json_array,
)
from litecoder.providers import ProviderEvent, StopReason
from tests.fakes.provider import FakeProvider


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[{"name":"one"}]', [{"name": "one"}]),
        ('```json\n[{"name":"one"}]\n```', [{"name": "one"}]),
        ("Selected values: [0, 2].", [0, 2]),
        ("no array here", None),
    ],
)
def test_parse_json_array_accepts_reference_model_shapes(
    text: str, expected: list[object] | None
) -> None:
    assert parse_json_array(text) == expected


async def test_complete_side_query_collects_completed_text() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.content_block_completed(0, {"type": "text", "text": "[1]"}),
                ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
            ]
        ]
    )

    text = await complete_side_query(
        provider, "model", system="Select memory.", prompt="catalog", max_tokens=200
    )

    assert text == "[1]"
    assert provider.requests[0].tools == []
    assert provider.requests[0].system == [{"type": "text", "text": "Select memory."}]
    assert provider.requests[0].messages == [
        {"role": "user", "content": [{"type": "text", "text": "catalog"}]}
    ]


async def test_complete_side_query_uses_completed_blocks_in_index_order() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.text_delta(1, "ignored"),
                ProviderEvent.content_block_completed(1, {"type": "text", "text": "second"}),
                ProviderEvent.content_block_completed(0, {"type": "text", "text": "first"}),
                ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
            ]
        ]
    )

    text = await complete_side_query(
        provider, "model", system="system", prompt="prompt", max_tokens=1
    )

    assert text == "firstsecond"


async def test_complete_side_query_returns_none_when_response_does_not_end_normally() -> None:
    provider = FakeProvider(
        [
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
        ]
    )

    text = await complete_side_query(
        provider, "model", system="system", prompt="prompt", max_tokens=1
    )

    assert text is None


async def test_complete_side_query_returns_none_for_provider_error_stream() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.provider_error(
                    LiteCoderError(ErrorCode.PROVIDER_TRANSIENT, "side query failed")
                )
            ]
        ]
    )

    text = await complete_side_query(
        provider, "model", system="system", prompt="prompt", max_tokens=1
    )

    assert text is None


async def test_side_query_result_repr_excludes_successful_text() -> None:
    secret_text = "secret successful side-query response"
    provider = FakeProvider(
        [
            [
                ProviderEvent.content_block_completed(
                    0,
                    {"type": "text", "text": secret_text},
                ),
                ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
            ]
        ]
    )

    result = await complete_side_query_result(
        provider,
        "model",
        system="system",
        prompt="prompt",
        max_tokens=1200,
    )

    assert result.text == secret_text
    assert secret_text not in repr(result)


async def test_side_query_result_preserves_max_tokens_stop_reason() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.content_block_completed(
                    0,
                    {"type": "text", "text": '[{"name":"partial"}'},
                ),
                ProviderEvent.response_completed(
                    StopReason.MAX_TOKENS,
                    "max_tokens",
                ),
            ],
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
            [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
        ]
    )

    result = await complete_side_query_result(
        provider,
        "model",
        system="system",
        prompt="prompt",
        max_tokens=1200,
    )

    assert result.stop_reason is StopReason.MAX_TOKENS
    assert result.provider_code is None
    assert await complete_side_query(
        FakeProvider(
            [
                [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
                [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
                [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
                [ProviderEvent.response_completed(StopReason.MAX_TOKENS, "max_tokens")],
            ]
        ),
        "model",
        system="system",
        prompt="prompt",
        max_tokens=1200,
    ) is None


async def test_side_query_result_preserves_only_safe_provider_code() -> None:
    provider = FakeProvider(
        [
            [
                ProviderEvent.provider_error(
                    LiteCoderError(
                        ErrorCode.PROVIDER_RATE_LIMIT,
                        "secret raw provider message",
                    )
                )
            ]
        ]
    )

    result = await complete_side_query_result(
        provider,
        "model",
        system="system",
        prompt="prompt",
        max_tokens=1200,
    )

    assert result == MemorySideQueryResult(
        text="",
        stop_reason=None,
        provider_code="provider_rate_limit",
    )
    assert "secret" not in repr(result)


@pytest.mark.parametrize(
    "prompt",
    [
        MEMORY_SELECTION_SYSTEM_PROMPT,
        MEMORY_EXTRACTION_SYSTEM_PROMPT,
        MEMORY_CONSOLIDATION_SYSTEM_PROMPT,
    ],
)
def test_memory_side_query_prompts_treat_content_as_untrusted_data_and_require_json_array(
    prompt: str,
) -> None:
    normalized = prompt.casefold()

    assert "untrusted data" in normalized
    assert "json array" in normalized


@pytest.mark.parametrize(
    "prompt",
    [MEMORY_EXTRACTION_SYSTEM_PROMPT, MEMORY_CONSOLIDATION_SYSTEM_PROMPT],
)
def test_memory_object_prompts_constrain_each_object_to_the_supported_schema(
    prompt: str,
) -> None:
    normalized = prompt.casefold()

    assert "each object" in normalized
    assert "name, type, description, and body" in normalized
    assert "only one of: user, feedback, project, reference" in normalized


def test_memory_consolidation_prompt_defines_dream_policy() -> None:
    normalized = MEMORY_CONSOLIDATION_SYSTEM_PROMPT.casefold()

    assert "merge duplicate" in normalized
    assert "explicit user guidance" in normalized
    assert "remove stale" in normalized
    assert "preserve important" in normalized
    assert "preferences" in normalized
    assert "no more than 30" in normalized


def test_memory_prompts_reject_transient_state_and_preserve_corrections() -> None:
    extraction = MEMORY_EXTRACTION_SYSTEM_PROMPT.casefold()
    consolidation = MEMORY_CONSOLIDATION_SYSTEM_PROMPT.casefold()

    assert "temporary task progress" in extraction
    assert "explicit user corrections" in extraction
    assert "transient task progress" in consolidation
    assert "embedded instructions" in consolidation
