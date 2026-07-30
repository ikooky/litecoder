from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType

import pytest

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.providers.base import ModelProvider
from litecoder.providers.models import (
    AssistantContent,
    ModelRequest,
    ProviderEvent,
    StopReason,
    ToolCallBlock,
    Usage,
)
from tests.fakes.provider import FakeProvider


def request(*, model: str = "test-model") -> ModelRequest:
    return ModelRequest(
        model=model,
        system=[{"type": "text", "text": "system"}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        tools=[{"name": "read_file", "input_schema": {"type": "object"}}],
        max_tokens=128,
    )


def test_stop_reason_values_are_stable() -> None:
    assert [reason.value for reason in StopReason] == [
        "end_turn",
        "tool_use",
        "pause_turn",
        "max_tokens",
        "stop_sequence",
        "refusal",
        "context_exhausted",
        "unknown",
    ]


def test_usage_is_normalized_immutable_and_totals_input_and_output() -> None:
    provider_counts = {"reasoning_tokens": 7}
    usage = Usage(
        input_tokens=11,
        output_tokens=13,
        cache_read_tokens=5,
        cache_creation_tokens=3,
        extensions=provider_counts,
    )
    provider_counts["reasoning_tokens"] = 99

    assert usage.total_tokens == 24
    assert usage.extensions == {"reasoning_tokens": 7}
    with pytest.raises(AttributeError):
        usage.input_tokens = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", -1),
        ("output_tokens", True),
        ("cache_read_tokens", -1),
        ("cache_creation_tokens", 1.5),
    ],
)
def test_usage_rejects_invalid_counts(field: str, value: object) -> None:
    kwargs: dict[str, object] = {"input_tokens": 1, "output_tokens": 2, field: value}
    with pytest.raises(ValueError, match=f"^{field} must be a non-negative integer$"):
        Usage(**kwargs)  # type: ignore[arg-type]


def test_usage_rejects_invalid_or_unbounded_extensions() -> None:
    with pytest.raises(ValueError, match="usage extensions"):
        Usage(1, 2, extensions={"nested": 3.5})
    with pytest.raises(ValueError, match="at most 32 entries"):
        Usage(1, 2, extensions={f"metric_{index}": index for index in range(33)})


def test_usage_accepts_generic_mappings_and_can_be_reconstructed() -> None:
    source = {"reasoning_tokens": 3}
    usage = Usage(1, 2, extensions=MappingProxyType(source))
    source["reasoning_tokens"] = 99

    reconstructed = replace(usage)

    assert usage.extensions == {"reasoning_tokens": 3}
    assert reconstructed == usage
    with pytest.raises(TypeError):
        usage.extensions["reasoning_tokens"] = 4  # type: ignore[index]


def test_tool_call_block_snapshots_provider_neutral_values() -> None:
    parsed_input = {"path": "测试.txt", "ranges": [1, 2]}
    extensions = {"provider": {"opaque": ["a", "b"]}}
    block = ToolCallBlock("call-1", "read_file", parsed_input, extensions)

    parsed_input["ranges"].append(3)  # type: ignore[union-attr]
    extensions["provider"]["opaque"].append("c")  # type: ignore[index, union-attr]

    assert block.input == {"path": "测试.txt", "ranges": [1, 2]}
    assert block.extensions == {"provider": {"opaque": ["a", "b"]}}


@pytest.mark.parametrize(("call_id", "name"), [("", "tool"), ("call", "   ")])
def test_tool_call_block_requires_non_empty_identity(call_id: str, name: str) -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        ToolCallBlock(call_id, name, {})


def test_tool_call_event_snapshots_input_and_extensions_from_source_block() -> None:
    source = ToolCallBlock(
        "call-1",
        "read_file",
        {"paths": ["first.txt"]},
        {"provider": {"flags": ["one"]}},
    )

    event = ProviderEvent.tool_call_completed(0, source)
    source.input["paths"].append("second.txt")  # type: ignore[union-attr]
    source.extensions["provider"]["flags"].append("two")  # type: ignore[index, union-attr]

    assert event.tool_call is not source
    assert event.tool_call is not None
    assert event.tool_call.input == {"paths": ["first.txt"]}
    assert event.tool_call.extensions == {"provider": {"flags": ["one"]}}


def test_model_request_snapshots_inputs_and_validates_limits() -> None:
    system = [{"type": "text", "text": "system"}]
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    tools = [{"name": "read_file", "schema": {"type": "object"}}]
    model_request = ModelRequest("model", system, messages, tools, 64)

    system[0]["text"] = "changed"
    messages[0]["content"][0]["text"] = "changed"  # type: ignore[index]
    tools[0]["schema"]["type"] = "changed"  # type: ignore[index]

    assert model_request.system[0]["text"] == "system"
    assert model_request.messages[0]["content"][0]["text"] == "hello"  # type: ignore[index]
    assert model_request.tools[0]["schema"]["type"] == "object"  # type: ignore[index]
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        request(model=" ")
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        ModelRequest("model", [], [], [], 0)


@pytest.mark.parametrize(
    "invalid",
    [
        {"value": math.inf},
        {"value": math.nan},
        {1: "non-string key"},
        {"value": object()},
    ],
)
def test_contracts_reject_non_json_values_without_echoing_them(invalid: object) -> None:
    with pytest.raises(ValueError, match="^messages must contain only JSON-compatible values") as caught:
        ModelRequest("model", [], [invalid], [], 1)  # type: ignore[list-item]
    assert "object at" not in str(caught.value)


def test_contracts_reject_json_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="cycle"):
        ModelRequest("model", [], [{"content": cyclic}], [], 1)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ModelRequest("model", [], [{"content": ("not-json",)}], [], 1),
        lambda: AssistantContent(
            [{"type": "opaque", "payload": ("not-json",)}],
            StopReason.END_TURN,
            "end_turn",
            Usage(1, 1),
            "req-1",
        ),
        lambda: ToolCallBlock("call-1", "tool", {"value": ("not-json",)}),
        lambda: ProviderEvent(type="provider.future", block={"value": ("not-json",)}),
        lambda: ProviderEvent.provider_error(
            LiteCoderError(
                ErrorCode.PROVIDER_TRANSIENT,
                "temporary",
                details={"value": ("not-json",)},
            )
        ),
    ],
)
def test_provider_neutral_contracts_reject_nested_tuples(build: object) -> None:
    with pytest.raises(ValueError, match="JSON-compatible values"):
        build()  # type: ignore[operator]


def test_assistant_content_preserves_opaque_block_order_and_response_facts() -> None:
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "future.provider.block", "payload": {"unknown": True}},
        {"type": "text", "text": "world"},
    ]
    content = AssistantContent(
        blocks=blocks,
        stop_reason=StopReason.UNKNOWN,
        raw_stop_reason="future_reason",
        usage=Usage(10, 4),
        request_id="req-1",
    )
    blocks[1]["payload"]["unknown"] = False  # type: ignore[index]

    assert [block["type"] for block in content.blocks] == [
        "text",
        "future.provider.block",
        "text",
    ]
    assert content.blocks[1]["payload"] == {"unknown": True}
    assert content.stop_reason is StopReason.UNKNOWN
    assert content.raw_stop_reason == "future_reason"
    assert content.usage == Usage(10, 4)
    assert content.request_id == "req-1"


def test_provider_event_constructors_cover_normalized_stream_facts() -> None:
    tool_call = ToolCallBlock("call-1", "read_file", {"path": "README.md"})
    usage = Usage(8, 3)
    error = LiteCoderError(
        ErrorCode.PROVIDER_RATE_LIMIT,
        "provider unavailable",
        retryable=True,
        details={"status": 429},
    )
    events = [
        ProviderEvent.request_identified("req-1"),
        ProviderEvent.content_block_started(0, {"type": "text"}, request_id="req-1"),
        ProviderEvent.content_block_delta(0, {"text": "he"}, request_id="req-1"),
        ProviderEvent.text_delta(0, "hello", request_id="req-1"),
        ProviderEvent.tool_call_input_delta(1, "call-1", '{"path":', request_id="req-1"),
        ProviderEvent.tool_call_completed(1, tool_call, request_id="req-1"),
        ProviderEvent.content_block_completed(
            0, {"type": "text", "text": "hello"}, request_id="req-1"
        ),
        ProviderEvent.usage_updated(usage, request_id="req-1"),
        ProviderEvent.response_completed(
            StopReason.TOOL_USE,
            "tool_use",
            usage=usage,
            request_id="req-1",
        ),
        ProviderEvent.provider_error(error, request_id="req-1"),
    ]

    assert [event.type for event in events] == [
        "response.request_id",
        "content.started",
        "content.delta",
        "text.delta",
        "tool_call.input_delta",
        "tool_call.completed",
        "content.completed",
        "usage",
        "response.completed",
        "provider.error",
    ]
    assert events[4].tool_call_id == "call-1"
    assert events[4].delta == '{"path":'
    assert events[5].tool_call == tool_call
    assert events[8].stop_reason is StopReason.TOOL_USE
    assert events[8].raw_stop_reason == "tool_use"
    assert events[9].error is not error
    assert events[9].error is not None
    assert events[9].error.code is ErrorCode.PROVIDER_RATE_LIMIT
    assert events[9].error.retryable is True
    assert events[9].error.details == {"status": 429}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda error: setattr(error, "code", "provider_transient"),
            "code must be an ErrorCode",
        ),
        (lambda error: setattr(error, "args", (123,)), "message must be a string"),
        (lambda error: setattr(error, "retryable", 1), "retryable must be a bool"),
        (
            lambda error: setattr(error, "details", ["not", "a", "mapping"]),
            "details must be a JSON object",
        ),
    ],
)
def test_provider_error_rejects_malformed_common_error_facts(
    mutate: object, message: str
) -> None:
    error = LiteCoderError(ErrorCode.PROVIDER_TRANSIENT, "temporary")
    mutate(error)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        ProviderEvent.provider_error(error)


def test_unknown_stop_reason_retains_raw_value_and_fails_closed() -> None:
    event = ProviderEvent.response_completed(StopReason.UNKNOWN, "future_reason")

    assert event.stop_reason is StopReason.UNKNOWN
    assert event.raw_stop_reason == "future_reason"


@pytest.mark.parametrize(
    "event",
    [
        lambda: ProviderEvent(type="text.delta", delta={"text": "wrong"}, index=0),
        lambda: ProviderEvent(type="response.completed", stop_reason=StopReason.END_TURN),
        lambda: ProviderEvent(type="provider.error", error=None),
        lambda: ProviderEvent(type="usage", usage=Usage(1, 1), block={"type": "text"}),
    ],
)
def test_provider_event_rejects_incompatible_known_shapes(event: object) -> None:
    with pytest.raises(ValueError, match="incompatible fields|requires"):
        event()  # type: ignore[operator]


def test_provider_event_allows_forward_compatible_type_strings() -> None:
    event = ProviderEvent(type="provider.future_fact", block={"kind": "opaque"})

    assert event.type == "provider.future_fact"
    assert event.block == {"kind": "opaque"}


def test_forward_compatible_events_still_reject_invalid_typed_fields() -> None:
    with pytest.raises(ValueError, match="usage must be a Usage"):
        ProviderEvent(type="provider.future_fact", usage=object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fake_provider_conforms_and_streams_independent_rounds() -> None:
    first_event = ProviderEvent.content_block_completed(
        0, {"type": "text", "text": "first"}
    )
    first_round = [first_event]
    second_round = [ProviderEvent.response_completed(StopReason.END_TURN, "end_turn")]
    script = [first_round, second_round]
    provider = FakeProvider(script)
    assert first_event.block is not None
    first_event.block["text"] = "mutated"
    first_round.clear()
    script.clear()
    first_request = request(model="first-model")
    second_request = request(model="second-model")

    first_events = [event async for event in provider.stream(first_request)]
    second_events = [event async for event in provider.stream(second_request)]

    assert isinstance(provider, ModelProvider)
    assert [event.type for event in first_events] == ["content.completed"]
    assert first_events[0].block == {"type": "text", "text": "first"}
    assert [event.type for event in second_events] == ["response.completed"]
    assert provider.requests == [first_request, second_request]


@pytest.mark.asyncio
async def test_fake_provider_emits_scripted_errors_and_fails_clearly_when_exhausted() -> None:
    error_event = ProviderEvent.provider_error(
        LiteCoderError(ErrorCode.PROVIDER_TRANSIENT, "temporary", retryable=True)
    )
    provider = FakeProvider([[error_event]])

    emitted = [event async for event in provider.stream(request())]
    assert [event.type for event in emitted] == ["provider.error"]
    assert emitted[0].error is not None
    assert emitted[0].error.code is ErrorCode.PROVIDER_TRANSIENT
    assert emitted[0].error.retryable is True
    assert str(emitted[0].error) == "temporary"
    with pytest.raises(RuntimeError, match="fake provider script exhausted"):
        _ = [event async for event in provider.stream(request())]
