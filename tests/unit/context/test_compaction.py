from __future__ import annotations

import copy
import json

import pytest

from litecoder.context.compaction import (
    CompactionPolicy,
    SummaryRequest,
    estimate_message_tokens,
)
from litecoder.context.session.models import MessageRecord
from litecoder.context.token_budget import estimate_tokens


def _message(
    sequence: int | None,
    role: str,
    *blocks: dict[str, object],
) -> MessageRecord:
    return MessageRecord("session", role, list(blocks), sequence=sequence)


def _text(sequence: int, role: str, text: str) -> MessageRecord:
    return _message(sequence, role, {"type": "text", "text": text})


def _tool_round(
    sequence: int,
    call_ids: tuple[str, ...],
    *,
    result_text: str,
) -> list[MessageRecord]:
    calls = [
        {
            "type": "tool_call",
            "call_id": call_id,
            "name": f"tool-{call_id}",
            "input": {"position": index},
        }
        for index, call_id in enumerate(call_ids)
    ]
    results = [
        {
            "type": "tool_result",
            "tool_call_id": call_id,
            "status": "success",
            "content": f"{result_text}:{call_id}",
            "metadata": {
                "artifact": {"path": f"outputs/{call_id}.txt", "bytes": 9999}
            },
        }
        for call_id in call_ids
    ]
    return [
        _message(sequence, "assistant", *calls),
        _message(sequence + 1, "user", *results),
    ]


def _call_ids(messages: list[MessageRecord]) -> list[str]:
    return [
        str(block["call_id"])
        for message in messages
        for block in message.content
        if block.get("type") == "tool_call"
    ]


def _result_ids(messages: list[MessageRecord]) -> list[str]:
    return [
        str(block["tool_call_id"])
        for message in messages
        for block in message.content
        if block.get("type") == "tool_result"
    ]


@pytest.mark.asyncio
async def test_exact_fit_does_not_call_summarizer() -> None:
    messages = [_text(1, "user", "hello"), _text(2, "assistant", "world")]
    calls: list[SummaryRequest] = []

    async def summarizer(request: SummaryRequest) -> str:
        calls.append(request)
        return "unused"

    budget = estimate_message_tokens(messages)
    result = await CompactionPolicy().compact(messages, budget, summarizer)

    assert result.summary is None
    assert result.messages == messages
    assert result.messages is not messages
    assert calls == []


@pytest.mark.asyncio
async def test_forced_compaction_summarizes_groups_snipped_to_budget() -> None:
    messages = [
        _text(1, "user", "old context " * 100),
        _text(2, "assistant", "recent answer"),
    ]
    requests: list[SummaryRequest] = []

    async def summarizer(request: SummaryRequest) -> str:
        requests.append(request)
        return "retained old context"

    result = await CompactionPolicy(keep_recent_tool_rounds=0).compact(
        messages,
        budget_tokens=estimate_message_tokens([messages[-1]]),
        summarizer=summarizer,
        summary_budget_tokens=10_000,
        force_summary=True,
    )

    assert result.summary == "retained old context"
    assert result.summary_request is not None
    assert result.summary_request.covered_through_sequence == 1
    assert [message.sequence for message in result.messages] == [2]
    assert requests == [result.summary_request]


@pytest.mark.asyncio
async def test_forced_compaction_summarizes_complete_tool_round_when_preview_fits(
) -> None:
    messages = _tool_round(
        1,
        ("large-result",),
        result_text="tool output " * 2_000,
    )
    original_tokens = estimate_message_tokens(messages)
    target = original_tokens * 2 // 3
    requests: list[SummaryRequest] = []

    async def summarizer(request: SummaryRequest) -> str:
        requests.append(request)
        return "tool round retained"

    result = await CompactionPolicy().compact(
        messages,
        budget_tokens=target,
        summarizer=summarizer,
        summary_budget_tokens=target,
        force_summary=True,
    )

    assert original_tokens > target
    assert result.summary == "tool round retained"
    assert result.messages == []
    assert requests == [result.summary_request]
    assert result.summary_request is not None
    request_messages = result.summary_request.messages
    assert [message["role"] for message in request_messages] == [
        "assistant",
        "user",
    ]
    request_call_ids = [
        block["call_id"]
        for message in request_messages
        for block in message["content"]
        if block.get("type") == "tool_call"
    ]
    request_result_ids = [
        block["tool_call_id"]
        for message in request_messages
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert request_call_ids == request_result_ids == ["large-result"]


@pytest.mark.asyncio
async def test_summary_budget_uses_original_persisted_tool_suffix() -> None:
    messages = [
        _text(1, "user", "short prefix"),
        *_tool_round(
            2,
            ("large-result",),
            result_text="tool output " * 2_000,
        ),
    ]
    before_tokens = estimate_message_tokens(messages)
    target = before_tokens * 2 // 3
    requests: list[SummaryRequest] = []

    async def summarizer(request: SummaryRequest) -> str:
        requests.append(request)
        return "retained context"

    result = await CompactionPolicy().compact(
        messages,
        budget_tokens=target,
        summarizer=summarizer,
        summary_budget_tokens=target,
        force_summary=True,
    )

    assert [request.covered_through_sequence for request in requests] == [1, 3]
    assert result.summary_request == requests[-1]
    assert result.messages == []
    final_request = requests[-1]
    request_call_ids = [
        block["call_id"]
        for message in final_request.messages
        for block in message["content"]
        if block.get("type") == "tool_call"
    ]
    request_result_ids = [
        block["tool_call_id"]
        for message in final_request.messages
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert request_call_ids == request_result_ids == ["large-result"]
    summary_record = _message(4, "system", {
        "type": "context_summary",
        "covered_through_sequence": 3,
        "text": "retained context",
    })
    persisted_suffix = [
        message
        for message in messages
        if message.sequence > final_request.covered_through_sequence
    ]
    assert estimate_message_tokens([summary_record, *persisted_suffix]) <= target


@pytest.mark.asyncio
async def test_summary_expands_prefix_until_summary_and_remaining_fit() -> None:
    messages = [
        _text(1, "user", "old context " * 100),
        _text(2, "assistant", "middle context " * 100),
        _text(3, "user", "recent request"),
    ]
    requests: list[SummaryRequest] = []

    async def summarizer(request: SummaryRequest) -> str:
        requests.append(request)
        return "retained facts"

    budget = estimate_message_tokens(messages[1:])
    result = await CompactionPolicy(keep_recent_tool_rounds=0).compact(
        messages,
        budget_tokens=budget,
        summarizer=summarizer,
        summary_budget_tokens=budget,
        force_summary=True,
    )

    assert [request.covered_through_sequence for request in requests] == [1, 2]
    assert result.summary_request == requests[-1]
    assert [message.sequence for message in result.messages] == [3]
    summary_record = _message(2, "system", {
        "type": "context_summary",
        "covered_through_sequence": 2,
        "text": "retained facts",
    })
    assert estimate_message_tokens([summary_record, *result.messages]) <= budget


@pytest.mark.asyncio
async def test_multi_tool_pair_is_preserved_in_original_call_order() -> None:
    messages = [
        _text(1, "user", "before"),
        *_tool_round(2, ("one", "two"), result_text="x" * 200),
        _text(4, "assistant", "after"),
    ]

    result = await CompactionPolicy(
        tool_result_budget_tokens=8,
        keep_recent_tool_rounds=0,
        compacted_tool_result_preview_tokens=2,
    ).compact(messages, budget_tokens=10_000)

    assert _call_ids(result.messages) == ["one", "two"]
    assert _result_ids(result.messages) == ["one", "two"]
    result_blocks = result.messages[2].content
    assert [block["tool_call_id"] for block in result_blocks] == ["one", "two"]
    assert all("artifact" in block["metadata"] for block in result_blocks)
    assert all(len(str(block["content"])) < 100 for block in result_blocks)


@pytest.mark.asyncio
async def test_snipping_never_separates_tool_call_and_result() -> None:
    messages = [
        _text(1, "user", "old" * 100),
        *_tool_round(2, ("old",), result_text="old-result" * 100),
        *_tool_round(4, ("recent",), result_text="recent-result" * 10),
        _text(6, "assistant", "latest useful answer"),
    ]

    async def summarize(request: SummaryRequest) -> str:
        return "older complete groups"

    result = await CompactionPolicy(
        keep_recent_tool_rounds=1,
        tool_result_budget_tokens=8,
    ).compact(messages, budget_tokens=90, summarizer=summarize)

    assert _call_ids(result.messages) == _result_ids(result.messages)
    assert result.summary_request is not None
    request_call_ids = [
        block["call_id"]
        for message in result.summary_request.messages
        for block in message["content"]
        if block.get("type") == "tool_call"
    ]
    request_result_ids = [
        block["tool_call_id"]
        for message in result.summary_request.messages
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert request_call_ids == request_result_ids == ["old", "recent"]
    assert estimate_message_tokens(result.messages) <= 90
    assert any(
        block.get("text") == "latest useful answer"
        for message in result.messages
        for block in message.content
    )


@pytest.mark.asyncio
async def test_malformed_orphan_history_does_not_manufacture_pairs() -> None:
    messages = [
        _message(
            1,
            "assistant",
            {"type": "text", "text": "keep assistant text"},
            {"type": "tool_call", "call_id": "orphan", "name": "read", "input": {}},
        ),
        _message(
            2,
            "user",
            {"type": "text", "text": "keep user text"},
            {"type": "tool_result", "tool_call_id": "different", "content": "bad"},
        ),
    ]

    result = await CompactionPolicy().compact(messages, budget_tokens=10_000)

    assert _call_ids(result.messages) == []
    assert _result_ids(result.messages) == []
    assert [message.content for message in result.messages] == [
        [{"type": "text", "text": "keep assistant text"}],
        [{"type": "text", "text": "keep user text"}],
    ]


@pytest.mark.asyncio
async def test_compaction_does_not_mutate_caller_owned_messages_or_content() -> None:
    messages = [
        *_tool_round(1, ("old",), result_text="large" * 100),
        _text(3, "assistant", "done"),
    ]
    original = copy.deepcopy(messages)

    result = await CompactionPolicy(
        keep_recent_tool_rounds=0,
        tool_result_budget_tokens=4,
        compacted_tool_result_preview_tokens=1,
    ).compact(messages, budget_tokens=10_000)

    assert messages == original
    assert result.messages is not messages
    assert result.messages[1].content is not messages[1].content
    assert result.messages[1].content[0] is not messages[1].content[0]


@pytest.mark.asyncio
async def test_recent_three_tool_rounds_remain_full_while_older_results_compact() -> None:
    messages: list[MessageRecord] = []
    for index in range(4):
        messages.extend(
            _tool_round(
                index * 2 + 1,
                (f"call-{index}",),
                result_text=f"round-{index}-" + "x" * 200,
            )
        )

    result = await CompactionPolicy(
        tool_result_budget_tokens=100_000,
        compacted_tool_result_preview_tokens=2,
    ).compact(messages, budget_tokens=100_000)

    contents = {
        block["tool_call_id"]: block["content"]
        for message in result.messages
        for block in message.content
        if block.get("type") == "tool_result"
    }
    assert "[compacted" in str(contents["call-0"])
    assert "[compacted" not in str(contents["call-1"])
    assert "[compacted" not in str(contents["call-2"])
    assert "[compacted" not in str(contents["call-3"])


@pytest.mark.asyncio
async def test_total_tool_result_budget_is_enforced_before_recent_round_retention() -> None:
    messages = _tool_round(1, ("recent",), result_text="x" * 1000)

    result = await CompactionPolicy(
        tool_result_budget_tokens=8,
        keep_recent_tool_rounds=3,
        compacted_tool_result_preview_tokens=1,
    ).compact(messages, budget_tokens=100_000)

    result_block = result.messages[1].content[0]
    assert len(str(result_block["content"])) < 100
    assert result_block["metadata"]["artifact"]["path"] == "outputs/recent.txt"


@pytest.mark.asyncio
async def test_total_tool_result_budget_counts_metadata_and_keeps_artifact() -> None:
    messages = _tool_round(1, ("recent",), result_text="ok")
    result_block = messages[1].content[0]
    result_block["metadata"]["detail"] = "x" * 4_000

    result = await CompactionPolicy(
        tool_result_budget_tokens=80,
        keep_recent_tool_rounds=3,
    ).compact(messages, budget_tokens=100_000)

    compacted = result.messages[1].content[0]
    serialized = json.dumps(
        compacted, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert estimate_tokens(serialized) <= 80
    assert compacted["metadata"]["artifact"]["path"] == "outputs/recent.txt"
    assert "detail" not in compacted["metadata"]

@pytest.mark.asyncio
async def test_over_budget_summary_request_has_exact_highest_original_cutoff() -> None:
    messages = [
        _text(1, "user", "old user " * 80),
        _text(2, "assistant", "old assistant " * 80),
        *_tool_round(3, ("recent",), result_text="result " * 20),
        _text(5, "assistant", "latest"),
    ]
    requests: list[SummaryRequest] = []

    async def summarizer(request: SummaryRequest) -> str:
        requests.append(request)
        return "stable summary"

    summary_message = _message(6, "system", {
        "type": "context_summary",
        "covered_through_sequence": 2,
        "text": "stable summary",
    })
    budget = estimate_message_tokens([messages[-1], summary_message])
    result = await CompactionPolicy(keep_recent_tool_rounds=1).compact(
        messages, budget_tokens=budget, summarizer=summarizer
    )

    assert result.summary == "stable summary"
    assert len(requests) == 1
    assert requests[0].covered_through_sequence == 4
    assert [message["role"] for message in requests[0].messages] == [
        "user",
        "assistant",
        "assistant",
        "user",
    ]
    assert all(isinstance(message["content"], list) for message in requests[0].messages)
    assert estimate_message_tokens(result.messages) <= budget
    assert [message.sequence for message in result.messages] == [5]
    request_call_ids = [
        block["call_id"]
        for message in requests[0].messages
        for block in message["content"]
        if block.get("type") == "tool_call"
    ]
    request_result_ids = [
        block["tool_call_id"]
        for message in requests[0].messages
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert request_call_ids == request_result_ids == ["recent"]


@pytest.mark.asyncio
async def test_summary_cutoff_only_covers_a_contiguous_removed_prefix() -> None:
    messages = [
        _text(1, "user", "old " * 100),
        *_tool_round(2, ("protected",), result_text="protected " * 100),
        _text(4, "assistant", "middle " * 100),
        _text(5, "user", "latest " * 100),
    ]

    async def summarize(request: SummaryRequest) -> str:
        return "prefix summary"

    summary_message = _message(6, "system", {
        "type": "context_summary",
        "covered_through_sequence": 2,
        "text": "stable summary",
    })
    budget = estimate_message_tokens([messages[-1], summary_message])
    result = await CompactionPolicy(keep_recent_tool_rounds=1).compact(
        messages, budget_tokens=budget, summarizer=summarize
    )

    assert result.summary_request is not None
    cutoff = result.summary_request.covered_through_sequence
    assert cutoff == 4
    assert all(message.sequence > cutoff for message in result.messages)
    assert estimate_message_tokens(result.messages) <= budget
    assert [message.sequence for message in result.messages] == [5]
    request_call_ids = [
        block["call_id"]
        for message in result.summary_request.messages
        for block in message["content"]
        if block.get("type") == "tool_call"
    ]
    request_result_ids = [
        block["tool_call_id"]
        for message in result.summary_request.messages
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert request_call_ids == request_result_ids == ["protected"]


@pytest.mark.asyncio
async def test_non_summary_system_message_is_included_when_cutoff_covers_it() -> None:
    messages = [
        _message(1, "system", {"type": "text", "text": "durable instruction"}),
        _text(2, "user", "old " * 100),
        _text(3, "assistant", "latest " * 100),
    ]
    requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        requests.append(request)
        return "summary"

    await CompactionPolicy(keep_recent_tool_rounds=0).compact(
        messages,
        budget_tokens=estimate_message_tokens([_message(4, "system", {
            "type": "context_summary",
            "covered_through_sequence": 3,
            "text": "summary",
        })]),
        summarizer=summarize
    )

    assert requests[0].messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "durable instruction"}],
    }


@pytest.mark.asyncio
async def test_over_budget_without_summarizer_fails_closed() -> None:
    messages = [
        _text(1, "user", "old " * 100),
        _text(2, "assistant", "latest " * 100),
    ]

    with pytest.raises(RuntimeError, match="summarizer"):
        await CompactionPolicy(keep_recent_tool_rounds=0).compact(
            messages, budget_tokens=1
        )

@pytest.mark.asyncio
async def test_single_complete_group_can_be_summarized_when_it_alone_is_over_budget() -> None:
    requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        requests.append(request)
        return "one-message summary"

    result = await CompactionPolicy(keep_recent_tool_rounds=0).compact(
        [_text(1, "user", "only message " * 100)],
        budget_tokens=estimate_message_tokens([_message(2, "system", {
            "type": "context_summary",
            "covered_through_sequence": 1,
            "text": "one-message summary",
        })]),
        summarizer=summarize,
    )

    assert requests[0].covered_through_sequence == 1
    assert result.messages == []
    assert result.summary == "one-message summary"

@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [True, 1.5, -1])
async def test_budget_validation_rejects_bool_non_integer_and_negative(
    budget: object,
) -> None:
    with pytest.raises(ValueError, match="budget_tokens"):
        await CompactionPolicy().compact([], budget)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_missing_sequence_is_rejected_without_calling_summarizer() -> None:
    called = False

    async def summarizer(request: SummaryRequest) -> str:
        nonlocal called
        called = True
        return "summary"

    with pytest.raises(ValueError, match="sequence"):
        await CompactionPolicy().compact(
            [_message(None, "user", {"type": "text", "text": "hello"})],
            10,
            summarizer,
        )

    assert called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", ["", "   ", 42, None])
async def test_invalid_summary_output_is_rejected(summary: object) -> None:
    messages = [_text(1, "user", "x" * 1000), _text(2, "assistant", "latest")]

    async def summarizer(request: SummaryRequest) -> object:
        return summary

    with pytest.raises(ValueError, match="summary"):
        await CompactionPolicy().compact(
            messages,
            budget_tokens=1,
            summarizer=summarizer,  # type: ignore[arg-type]
        )


def test_policy_rejects_invalid_integer_configuration() -> None:
    for kwargs in (
        {"tool_result_budget_tokens": True},
        {"tool_result_budget_tokens": -1},
        {"keep_recent_tool_rounds": 1.5},
        {"compacted_tool_result_preview_tokens": -1},
    ):
        with pytest.raises(ValueError):
            CompactionPolicy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("cutoff", [True, 0, -1, 1.5])
def test_summary_request_requires_positive_integer_cutoff(cutoff: object) -> None:
    with pytest.raises(ValueError, match="covered_through_sequence"):
        SummaryRequest(cutoff, ())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_message_sequences_must_be_strictly_increasing() -> None:
    messages = [_text(2, "user", "later"), _text(1, "assistant", "earlier")]

    with pytest.raises(ValueError, match="strictly increasing"):
        await CompactionPolicy().compact(messages, budget_tokens=100)

@pytest.mark.asyncio
async def test_oversized_summary_fails_closed() -> None:
    messages = [
        _text(1, "user", "old " * 200),
        _text(2, "assistant", "latest " * 100),
    ]

    async def summarize(request: SummaryRequest) -> str:
        return "x" * 10_000

    with pytest.raises(RuntimeError, match="summary.*budget"):
        await CompactionPolicy(keep_recent_tool_rounds=0).compact(
            messages, budget_tokens=20, summarizer=summarize
        )
