from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.common.trace import SecretRedactor
from litecoder.context.session.models import MessageRecord
from litecoder.memory.extraction import MemoryExtractionResult, extract_memories
from litecoder.memory.store import MemoryStore
from litecoder.providers import ProviderEvent, StopReason
from tests.fakes.provider import FakeProvider


_ENTRY = (
    '{"name":"response-prefix","type":"user",'
    '"description":"Reply prefix","body":"Start replies with 喵~."}'
)


def _provider(response: str) -> FakeProvider:
    return FakeProvider(
        [
            [
                ProviderEvent.content_block_completed(
                    0,
                    {"type": "text", "text": response},
                ),
                ProviderEvent.response_completed(
                    StopReason.END_TURN,
                    "end_turn",
                ),
            ]
        ]
    )


def _messages() -> list[MessageRecord]:
    return [
        MessageRecord(
            "session",
            "user",
            [{"type": "text", "text": "记住每次回答先说喵~"}],
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        f"[{_ENTRY}]",
        f"```json\n[{_ENTRY}]\n```",
        f"```JSON\r\n[{_ENTRY}]\r\n```",
        f"Here is the JSON result:\n[{_ENTRY}]\nDone.",
        f"\ufeff  [{_ENTRY}]  ",
        f'{{"memories":[{_ENTRY}]}}',
        f'{{"items":[{_ENTRY}]}}',
        f'{{"results":[{_ENTRY}]}}',
        f'{{"data":[{_ENTRY}]}}',
    ],
)
async def test_extraction_accepts_complete_json_response_variants(
    tmp_path: Path,
    response: str,
) -> None:
    store = MemoryStore(tmp_path / ".memory")

    result = await extract_memories(
        store,
        _provider(response),
        "model",
        SecretRedactor.with_values(()),
        "session",
        _messages(),
    )

    assert result == MemoryExtractionResult(
        1,
        1,
        0,
        1,
        "completed",
        total=1,
    )
    assert store.read("response-prefix").body == "Start replies with 喵~."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        f"[{_ENTRY}",
        f"[{_ENTRY},]",
        "[{'name':'not-json'}]",
        f"```json\n[{_ENTRY}",
        '{"unknown":[]}',
    ],
)
async def test_extraction_still_rejects_invalid_or_unknown_json_shapes(
    tmp_path: Path,
    response: str,
) -> None:
    store = MemoryStore(tmp_path / ".memory")

    result = await extract_memories(
        store,
        _provider(response),
        "model",
        SecretRedactor.with_values(()),
        "session",
        _messages(),
    )

    assert result.status == "malformed"
    assert result.written == 0
    assert not store.root.exists()
