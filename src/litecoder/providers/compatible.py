"""Provider API-style normalization and streaming."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
from typing import Any, Literal
import uuid

from litecoder.providers._adapter import (
    InvalidProviderData,
    UsageAccumulator,
    classify_provider_error,
    invalid_stream_error,
    invalid_tool_arguments_error,
    managed_async_stream,
    parse_tool_input,
    plain_mapping,
    require_identity,
    require_index,
    require_text,
)
from litecoder.providers._json import JsonValue, snapshot_mapping
from litecoder.providers.models import (
    ModelRequest,
    ProviderEvent,
    StopReason,
    ToolCallBlock,
)


APIStyle = Literal[
    "anthropic-messages",
    "openai-chat-completions",
    "openai-responses",
]
ProviderCall = Callable[..., Awaitable[Any]]


_COMPLETION_STOP_MAP = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
    "context_length_exceeded": StopReason.CONTEXT_EXHAUSTED,
    "context_window_exceeded": StopReason.CONTEXT_EXHAUSTED,
}


def _normalize_completion_stop(value: object) -> tuple[StopReason, str]:
    raw = value if isinstance(value, str) and value else "unknown"
    return _COMPLETION_STOP_MAP.get(raw, StopReason.UNKNOWN), raw


@dataclass(slots=True)
class _CompletionBlock:
    """Data model representing the completion block."""
    index: int
    kind: str
    text: str = ""
    call_id: str | None = None
    name: str | None = None
    fragments: list[str] = field(default_factory=list)
    pending_fragments: list[str] = field(default_factory=list)
    started: bool = False
    native_thinking: dict[str, JsonValue] | None = None
    reasoning_items: list[dict[str, JsonValue]] = field(default_factory=list)


@dataclass(slots=True)
class _CompletionState:
    """Data model representing the completion state."""
    request_id: str | None = None
    active_choice: int | None = None
    blocks: dict[object, _CompletionBlock] = field(default_factory=dict)
    order: list[object] = field(default_factory=list)
    next_index: int = 0
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)
    stop_reason: object = None
    blocks_completed: bool = False
    terminal_seen: bool = False
    synthetic_call_seed: str = field(
        default_factory=lambda: uuid.uuid4().hex[:24]
    )

    def block(self, key: object, kind: str) -> tuple[_CompletionBlock, bool]:
        """Handle the block operation."""
        existing = self.blocks.get(key)
        if existing is not None:
            return existing, False
        block = _CompletionBlock(self.next_index, kind)
        self.next_index += 1
        self.blocks[key] = block
        self.order.append(key)
        return block, True


@dataclass(slots=True)
class _ResponsesBlock:
    """Data model representing the responses block."""
    index: int
    kind: str
    output_index: int
    text: str = ""
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)
    item: dict[str, JsonValue] | None = None
    started: bool = False


@dataclass(slots=True)
class _ResponsesState:
    """Data model representing the responses state."""
    request_id: str | None = None
    blocks: dict[object, _ResponsesBlock] = field(default_factory=dict)
    order: list[object] = field(default_factory=list)
    next_index: int = 0
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)
    status: str | None = None
    incomplete_reason: str | None = None
    terminal_seen: bool = False
    blocks_completed: bool = False

    def block(
        self, key: object, kind: str, output_index: int
    ) -> tuple[_ResponsesBlock, bool]:
        """Handle the block operation."""
        existing = self.blocks.get(key)
        if existing is not None:
            return existing, False
        block = _ResponsesBlock(self.next_index, kind, output_index)
        self.next_index += 1
        self.blocks[key] = block
        self.order.append(key)
        return block, True


class CompatibleProvider:
    """Component responsible for the compatible provider."""
    def __init__(
        self,
        *,
        completion: ProviderCall,
        responses: ProviderCall,
        model: str,
        api_style: APIStyle,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Provider model must be a non-empty string")
        if api_style not in {
            "anthropic-messages",
            "openai-chat-completions",
            "openai-responses",
        }:
            raise ValueError(f"Unsupported provider API style: {api_style}")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Provider API key must be a non-empty string")
        if base_url is not None and not base_url.strip():
            raise ValueError("Provider base URL must be non-empty when provided")
        self._completion = completion
        self._responses = responses
        self.model = model
        self.api_style = api_style
        self.api_key = api_key
        self.base_url = base_url

    async def stream(self, request: ModelRequest):
        """Handle the stream operation."""
        if self.api_style == "openai-responses":
            async for event in self._stream_responses(request):
                yield event
            return
        async for event in self._stream_completion(request):
            yield event

    async def _stream_completion(self, request: ModelRequest):
        state = _CompletionState()
        kwargs: dict[str, object] = {
            "model": self.model,
            "custom_llm_provider": (
                "anthropic"
                if self.api_style == "anthropic-messages"
                else "openai"
            ),
            "api_key": self.api_key,
            "messages": _to_chat_messages(request, self.api_style),
            "tools": _to_chat_tools(request.tools) or None,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": request.max_tokens,
            "max_retries": 0,
        }
        if self.base_url is not None:
            kwargs["api_base"] = _completion_api_base(
                self.base_url, self.api_style
            )
        try:
            stream = await self._completion(**kwargs)
            async with managed_async_stream(stream) as active_stream:
                async for raw_chunk in active_stream:
                    for event in self._normalize_completion_chunk(raw_chunk, state):
                        yield event
        except InvalidProviderData as error:
            reason = str(error)
            normalized = (
                invalid_tool_arguments_error(reason)
                if "tool argument" in reason
                else invalid_stream_error(reason=reason)
            )
            yield ProviderEvent.provider_error(
                normalized, request_id=state.request_id
            )
            return
        except Exception as error:
            yield ProviderEvent.provider_error(
                classify_provider_error(error), request_id=state.request_id
            )
            return

        if not state.terminal_seen:
            yield ProviderEvent.provider_error(
                invalid_stream_error("missing_finish_reason"),
                request_id=state.request_id,
            )
            return
        try:
            for event in self._complete_completion_blocks(state):
                yield event
        except InvalidProviderData as error:
            yield ProviderEvent.provider_error(
                invalid_tool_arguments_error(str(error)),
                request_id=state.request_id,
            )
            return
        normalized, raw = _normalize_completion_stop(state.stop_reason)
        yield ProviderEvent.response_completed(
            normalized,
            raw,
            usage=state.usage.current if state.usage.seen else None,
            request_id=state.request_id,
        )

    def _normalize_completion_chunk(
        self, raw_chunk: object, state: _CompletionState
    ) -> list[ProviderEvent]:
        """Normalize the completion chunk."""
        # Normalize chunks incrementally so malformed or late events cannot
        # overwrite already-emitted terminal state.
        chunk = plain_mapping(raw_chunk, "completion chunk")
        events: list[ProviderEvent] = []
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise InvalidProviderData("provider choices are invalid")
        if state.terminal_seen:
            usage = _completion_usage(chunk.get("usage"), state.usage)
            if usage is not None:
                events.append(
                    ProviderEvent.usage_updated(
                        usage, request_id=state.request_id
                    )
                )
            _validate_trailing_completion_choices(choices)
            return events

        request_id = chunk.get("id")
        if isinstance(request_id, str) and request_id:
            if state.request_id is None:
                state.request_id = request_id
                events.append(ProviderEvent.request_identified(request_id))
            elif state.request_id != request_id:
                raise InvalidProviderData("conflicting provider request identifiers")

        usage = _completion_usage(chunk.get("usage"), state.usage)
        if usage is not None:
            events.append(
                ProviderEvent.usage_updated(usage, request_id=state.request_id)
            )

        if len(choices) > 1:
            raise InvalidProviderData("multiple provider choices are unsupported")
        if not choices:
            return events

        choice = choices[0]
        if not isinstance(choice, dict):
            raise InvalidProviderData("provider choice is invalid")
        choice_index = require_index(choice.get("index", 0))
        if state.active_choice is None:
            state.active_choice = choice_index
        elif state.active_choice != choice_index:
            raise InvalidProviderData("multiple provider choices are unsupported")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise InvalidProviderData("provider choice delta is invalid")
        events.extend(self._completion_delta(delta, state))
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            state.stop_reason = require_identity(
                finish_reason, "provider finish reason"
            )
            state.terminal_seen = True
            events.extend(self._complete_completion_blocks(state))
        return events

    def _completion_delta(
        self, delta: dict[str, JsonValue], state: _CompletionState
    ) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        self._capture_completion_metadata(delta, state)

        content = delta.get("content")
        if content is not None:
            text = require_text(content, "provider text delta")
            if text:
                events.extend(
                    self._completion_text_delta("text", "text", text, state)
                )
        refusal = delta.get("refusal")
        if refusal is not None:
            text = require_text(refusal, "provider refusal delta")
            if text:
                events.extend(
                    self._completion_text_delta(
                        "refusal", "refusal", text, state
                    )
                )
        reasoning = delta.get("reasoning_content")
        if reasoning is not None:
            text = require_text(reasoning, "provider reasoning delta")
            if text:
                events.extend(
                    self._completion_text_delta(
                        "reasoning", "reasoning", text, state
                    )
                )

        tool_calls = delta.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise InvalidProviderData("provider tool calls are invalid")
            for raw_call in tool_calls:
                if not isinstance(raw_call, dict):
                    raise InvalidProviderData("provider tool call is invalid")
                provider_index = require_index(raw_call.get("index"))
                events.extend(
                    self._completion_tool_delta(
                        ("tool", provider_index), raw_call, state
                    )
                )
        function_call = delta.get("function_call")
        if function_call is not None:
            if not isinstance(function_call, dict):
                raise InvalidProviderData("provider function call is invalid")
            events.extend(
                self._completion_tool_delta(
                    ("function", 0),
                    {"index": 0, "function": function_call},
                    state,
                    legacy=True,
                )
            )
        return events

    def _capture_completion_metadata(
        self, delta: dict[str, JsonValue], state: _CompletionState
    ) -> None:
        # Providers may deliver reasoning metadata in a separate field or block.
        thinking_blocks = delta.get("thinking_blocks")
        provider_fields = delta.get("provider_specific_fields")
        if thinking_blocks is None and isinstance(provider_fields, dict):
            thinking_blocks = provider_fields.get("thinking_blocks")
        reasoning_items = delta.get("reasoning_items")
        if thinking_blocks is None and reasoning_items is None:
            return
        block, _ = state.block("reasoning", "reasoning")
        if thinking_blocks is not None:
            if not isinstance(thinking_blocks, list):
                raise InvalidProviderData("provider thinking blocks are invalid")
            for value in thinking_blocks:
                if not isinstance(value, dict):
                    raise InvalidProviderData("provider thinking block is invalid")
                candidate = snapshot_mapping(value, "provider thinking block")
                if candidate.get("type") in {"thinking", "redacted_thinking"}:
                    block.native_thinking = candidate
        if reasoning_items is not None:
            if not isinstance(reasoning_items, list):
                raise InvalidProviderData("provider reasoning items are invalid")
            block.reasoning_items = [
                snapshot_mapping(value, "provider reasoning item")
                for value in reasoning_items
                if isinstance(value, dict)
            ]

    def _completion_text_delta(
        self,
        key: object,
        kind: str,
        text: str,
        state: _CompletionState,
    ) -> list[ProviderEvent]:
        block, started = state.block(key, kind)
        events: list[ProviderEvent] = []
        if started:
            block.started = True
            start_type = "thinking" if kind == "reasoning" else kind
            events.append(
                ProviderEvent.content_block_started(
                    block.index,
                    {"type": start_type},
                    request_id=state.request_id,
                )
            )
        block.text += text
        if kind == "text":
            events.extend(
                [
                    ProviderEvent.content_block_delta(
                        block.index,
                        {"type": "text_delta", "text": text},
                        request_id=state.request_id,
                    ),
                    ProviderEvent.text_delta(
                        block.index, text, request_id=state.request_id
                    ),
                ]
            )
        elif kind == "reasoning":
            events.append(
                ProviderEvent.content_block_delta(
                    block.index,
                    {"type": "thinking_delta", "thinking": text},
                    request_id=state.request_id,
                )
            )
        else:
            events.append(
                ProviderEvent.content_block_delta(
                    block.index,
                    {"type": "refusal_delta", "text": text},
                    request_id=state.request_id,
                )
            )
        return events

    def _completion_tool_delta(
        self,
        key: object,
        raw_call: dict[str, JsonValue],
        state: _CompletionState,
        *,
        legacy: bool = False,
    ) -> list[ProviderEvent]:
        block, _ = state.block(key, "tool_call")
        if legacy and block.call_id is None:
            block.call_id = "legacy_function_call_0"
        call_id = raw_call.get("id")
        if isinstance(call_id, str) and not call_id.strip():
            call_id = None
        if call_id is not None:
            incoming_id = require_identity(call_id, "provider tool call id")
            if block.call_id is not None and block.call_id != incoming_id:
                raise InvalidProviderData("conflicting provider tool call id")
            block.call_id = incoming_id
        function = raw_call.get("function") or {}
        if not isinstance(function, dict):
            raise InvalidProviderData("provider tool function is invalid")
        name = function.get("name")
        if isinstance(name, str) and not name.strip():
            name = None
        if name is not None:
            incoming_name = require_identity(name, "provider tool name")
            if block.name is not None and block.name != incoming_name:
                raise InvalidProviderData("conflicting provider tool name")
            block.name = incoming_name
        arguments = function.get("arguments")
        if arguments is not None:
            fragment = require_text(arguments, "provider tool argument delta")
            block.fragments.append(fragment)
            block.pending_fragments.append(fragment)

        events: list[ProviderEvent] = []
        if block.call_id is None or block.name is None:
            return events
        if not block.started:
            block.started = True
            events.append(
                ProviderEvent.content_block_started(
                    block.index,
                    {
                        "type": "tool_call",
                        "call_id": block.call_id,
                        "name": block.name,
                    },
                    request_id=state.request_id,
                )
            )
        while block.pending_fragments:
            fragment = block.pending_fragments.pop(0)
            events.extend(
                [
                    ProviderEvent.content_block_delta(
                        block.index,
                        {"type": "tool_call_delta", "arguments": fragment},
                        request_id=state.request_id,
                    ),
                    ProviderEvent.tool_call_input_delta(
                        block.index,
                        block.call_id,
                        fragment,
                        request_id=state.request_id,
                    ),
                ]
            )
        return events

    def _complete_completion_blocks(
        self, state: _CompletionState
    ) -> list[ProviderEvent]:
        """Complete the completion blocks."""
        if state.blocks_completed:
            return []
        state.blocks_completed = True
        events: list[ProviderEvent] = []
        for key in state.order:
            block = state.blocks[key]
            if block.kind == "reasoning" and not (
                block.text or block.native_thinking or block.reasoning_items
            ):
                continue
            call: ToolCallBlock | None = None
            if block.kind == "text":
                completed: dict[str, JsonValue] = {
                    "type": "text",
                    "text": block.text,
                }
            elif block.kind == "refusal":
                completed = {"type": "refusal", "text": block.text}
            elif block.kind == "reasoning":
                completed = self._completed_reasoning_block(block)
            else:
                if block.name is None:
                    raise InvalidProviderData("tool arguments are missing name")
                if block.call_id is None:
                    block.call_id = (
                        f"call_litecoder_{state.synthetic_call_seed}_{block.index}"
                    )
                parsed = parse_tool_input("".join(block.fragments))
                call = ToolCallBlock(block.call_id, block.name, parsed)
                completed = {
                    "type": "tool_call",
                    "call_id": block.call_id,
                    "name": block.name,
                    "input": parsed,
                }
            if call is not None:
                events.append(
                    ProviderEvent.tool_call_completed(
                        block.index, call, request_id=state.request_id
                    )
                )
            events.append(
                ProviderEvent.content_block_completed(
                    block.index, completed, request_id=state.request_id
                )
            )
        return events

    def _completed_reasoning_block(
        self, block: _CompletionBlock
    ) -> dict[str, JsonValue]:
        if self.api_style == "anthropic-messages" and block.native_thinking:
            completed = dict(block.native_thinking)
            if completed.get("type") == "thinking" and block.text:
                completed["thinking"] = block.text
            return completed
        if block.reasoning_items:
            return {
                "type": "reasoning",
                "text": block.text,
                "items": block.reasoning_items,
            }
        return {"type": "thinking", "thinking": block.text}

    async def _stream_responses(self, request: ModelRequest):
        state = _ResponsesState()
        kwargs: dict[str, object] = {
            "model": self.model,
            "custom_llm_provider": "openai",
            "api_key": self.api_key,
            "input": _to_responses_input(request.messages),
            "instructions": _system_text(request.system) or None,
            "tools": _to_responses_tools(request.tools) or None,
            "stream": True,
            "max_output_tokens": request.max_tokens,
            "max_retries": 0,
        }
        if self.base_url is not None:
            kwargs["api_base"] = self.base_url
        try:
            stream = await self._responses(**kwargs)
            async with managed_async_stream(stream) as active_stream:
                async for raw_event in active_stream:
                    for event in self._normalize_responses_event(raw_event, state):
                        yield event
                    if state.terminal_seen:
                        break
        except InvalidProviderData as error:
            reason = str(error)
            normalized = (
                invalid_tool_arguments_error(reason)
                if "tool argument" in reason
                else invalid_stream_error(reason=reason)
            )
            yield ProviderEvent.provider_error(
                normalized, request_id=state.request_id
            )
            return
        except Exception as error:
            yield ProviderEvent.provider_error(
                classify_provider_error(error), request_id=state.request_id
            )
            return

        if not state.terminal_seen:
            yield ProviderEvent.provider_error(
                invalid_stream_error("missing_response_completed"),
                request_id=state.request_id,
            )
            return
        try:
            for event in self._complete_responses_blocks(state):
                yield event
        except InvalidProviderData as error:
            yield ProviderEvent.provider_error(
                invalid_tool_arguments_error(str(error)),
                request_id=state.request_id,
            )
            return
        stop_reason, raw_reason = _responses_stop(state)
        yield ProviderEvent.response_completed(
            stop_reason,
            raw_reason,
            usage=state.usage.current if state.usage.seen else None,
            request_id=state.request_id,
        )

    def _normalize_responses_event(
        self, raw_event: object, state: _ResponsesState
    ) -> list[ProviderEvent]:
        """Normalize the responses event."""
        event = plain_mapping(raw_event, "Responses event")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise InvalidProviderData("Responses event type is invalid")
        events: list[ProviderEvent] = []
        self._capture_responses_request_id(event, state, events)

        if event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict):
                events.extend(
                    self._responses_item_added(
                        item, require_index(event.get("output_index", 0)), state
                    )
                )
        elif event_type == "response.output_text.delta":
            text = require_text(event.get("delta"), "Responses text delta")
            output_index = require_index(event.get("output_index", 0))
            content_index = require_index(event.get("content_index", 0))
            events.extend(
                self._responses_text_delta(
                    ("text", output_index, content_index),
                    "text",
                    output_index,
                    text,
                    state,
                )
            )
        elif event_type == "response.reasoning_summary_text.delta":
            text = require_text(event.get("delta"), "Responses reasoning delta")
            output_index = require_index(event.get("output_index", 0))
            summary_index = require_index(event.get("summary_index", 0))
            events.extend(
                self._responses_text_delta(
                    ("reasoning", output_index, summary_index),
                    "reasoning",
                    output_index,
                    text,
                    state,
                )
            )
        elif event_type == "response.function_call_arguments.delta":
            fragment = require_text(
                event.get("delta"), "Responses tool argument delta"
            )
            output_index = require_index(event.get("output_index", 0))
            events.extend(
                self._responses_tool_delta(output_index, fragment, state)
            )
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                self._capture_responses_item(
                    item, require_index(event.get("output_index", 0)), state
                )
        elif event_type in {"response.completed", "response.incomplete"}:
            response = event.get("response")
            if not isinstance(response, dict):
                raise InvalidProviderData("Responses terminal payload is invalid")
            self._capture_responses_terminal(response, state)
            if state.status == "failed":
                raise InvalidProviderData("Responses request failed")
            state.terminal_seen = True
        elif event_type in {"response.failed", "error"}:
            raise InvalidProviderData("Responses request failed")
        return events

    def _capture_responses_request_id(
        self,
        event: dict[str, JsonValue],
        state: _ResponsesState,
        events: list[ProviderEvent],
    ) -> None:
        response = event.get("response")
        candidate = response.get("id") if isinstance(response, dict) else None
        if candidate is None:
            candidate = event.get("response_id")
        if not isinstance(candidate, str) or not candidate:
            return
        if state.request_id is None:
            state.request_id = candidate
            events.append(ProviderEvent.request_identified(candidate))
        elif state.request_id != candidate:
            raise InvalidProviderData("conflicting Responses request identifiers")

    def _responses_item_added(
        self,
        item: dict[str, JsonValue],
        output_index: int,
        state: _ResponsesState,
    ) -> list[ProviderEvent]:
        item_type = item.get("type")
        if item_type != "function_call":
            return []
        block, _ = state.block(("tool", output_index), "tool_call", output_index)
        block.item = snapshot_mapping(item, "Responses function call")
        call_id = item.get("call_id", item.get("id"))
        name = item.get("name")
        if call_id is not None:
            block.call_id = require_identity(call_id, "Responses tool call id")
        if name is not None:
            block.name = require_identity(name, "Responses tool name")
        if block.started or block.call_id is None or block.name is None:
            return []
        block.started = True
        return [
            ProviderEvent.content_block_started(
                block.index,
                {
                    "type": "tool_call",
                    "call_id": block.call_id,
                    "name": block.name,
                },
                request_id=state.request_id,
            )
        ]

    def _responses_text_delta(
        self,
        key: object,
        kind: str,
        output_index: int,
        text: str,
        state: _ResponsesState,
    ) -> list[ProviderEvent]:
        if not text:
            return []
        block, started = state.block(key, kind, output_index)
        events: list[ProviderEvent] = []
        if started:
            block.started = True
            events.append(
                ProviderEvent.content_block_started(
                    block.index,
                    {"type": "thinking" if kind == "reasoning" else "text"},
                    request_id=state.request_id,
                )
            )
        block.text += text
        if kind == "text":
            events.extend(
                [
                    ProviderEvent.content_block_delta(
                        block.index,
                        {"type": "text_delta", "text": text},
                        request_id=state.request_id,
                    ),
                    ProviderEvent.text_delta(
                        block.index, text, request_id=state.request_id
                    ),
                ]
            )
        else:
            events.append(
                ProviderEvent.content_block_delta(
                    block.index,
                    {"type": "thinking_delta", "thinking": text},
                    request_id=state.request_id,
                )
            )
        return events

    def _responses_tool_delta(
        self,
        output_index: int,
        fragment: str,
        state: _ResponsesState,
    ) -> list[ProviderEvent]:
        block, _ = state.block(("tool", output_index), "tool_call", output_index)
        block.arguments.append(fragment)
        if block.call_id is None:
            return []
        return [
            ProviderEvent.tool_call_input_delta(
                block.index,
                block.call_id,
                fragment,
                request_id=state.request_id,
            )
        ]

    def _capture_responses_item(
        self,
        item: dict[str, JsonValue],
        output_index: int,
        state: _ResponsesState,
    ) -> None:
        item_type = item.get("type")
        if item_type == "function_call":
            block, _ = state.block(
                ("tool", output_index), "tool_call", output_index
            )
            block.item = snapshot_mapping(item, "Responses function call")
            call_id = item.get("call_id", item.get("id"))
            name = item.get("name")
            if call_id is not None:
                block.call_id = require_identity(call_id, "Responses tool call id")
            if name is not None:
                block.name = require_identity(name, "Responses tool name")
            arguments = item.get("arguments")
            if isinstance(arguments, str) and not block.arguments:
                block.arguments.append(arguments)
        elif item_type == "reasoning":
            block, _ = state.block(
                ("reasoning", output_index, 0), "reasoning", output_index
            )
            block.item = snapshot_mapping(item, "Responses reasoning item")
            if not block.text:
                block.text = _reasoning_item_text(item)

    def _capture_responses_terminal(
        self, response: dict[str, JsonValue], state: _ResponsesState
    ) -> None:
        request_id = response.get("id")
        if isinstance(request_id, str) and request_id:
            if state.request_id is None:
                state.request_id = request_id
            elif state.request_id != request_id:
                raise InvalidProviderData("conflicting Responses request identifiers")
        status = response.get("status")
        state.status = status if isinstance(status, str) else "completed"
        incomplete = response.get("incomplete_details")
        if isinstance(incomplete, dict):
            reason = incomplete.get("reason")
            if isinstance(reason, str):
                state.incomplete_reason = reason
        output = response.get("output") or []
        if not isinstance(output, list):
            raise InvalidProviderData("Responses output is invalid")
        for output_index, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            self._capture_responses_item(item, output_index, state)
            if item.get("type") == "message":
                content = item.get("content") or []
                if not isinstance(content, list):
                    continue
                for content_index, part in enumerate(content):
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if part.get("type") in {"output_text", "text"} and isinstance(
                        text, str
                    ):
                        block, _ = state.block(
                            ("text", output_index, content_index),
                            "text",
                            output_index,
                        )
                        if not block.text:
                            block.text = text
        _responses_usage(response.get("usage"), state.usage)

    def _complete_responses_blocks(
        self, state: _ResponsesState
    ) -> list[ProviderEvent]:
        """Complete the responses blocks."""
        if state.blocks_completed:
            return []
        state.blocks_completed = True
        events: list[ProviderEvent] = []
        for key in state.order:
            block = state.blocks[key]
            call: ToolCallBlock | None = None
            if block.kind == "text":
                completed: dict[str, JsonValue] = {
                    "type": "text",
                    "text": block.text,
                }
            elif block.kind == "reasoning":
                completed = {
                    "type": "reasoning",
                    "text": block.text,
                }
                if block.item is not None:
                    completed["item"] = _reasoning_input_item(block.item)
            else:
                if block.call_id is None or block.name is None:
                    raise InvalidProviderData("tool arguments are missing identity")
                parsed = parse_tool_input("".join(block.arguments))
                call = ToolCallBlock(block.call_id, block.name, parsed)
                completed = {
                    "type": "tool_call",
                    "call_id": block.call_id,
                    "name": block.name,
                    "input": parsed,
                }
            if call is not None:
                events.append(
                    ProviderEvent.tool_call_completed(
                        block.index, call, request_id=state.request_id
                    )
                )
            events.append(
                ProviderEvent.content_block_completed(
                    block.index, completed, request_id=state.request_id
                )
            )
        return events


def _completion_api_base(base_url: str, api_style: APIStyle) -> str:
    if api_style != "anthropic-messages":
        return base_url
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def _validate_trailing_completion_choices(
    choices: list[JsonValue],
) -> None:
    """Validate the trailing completion choices."""
    if not choices:
        return
    if len(choices) != 1 or not isinstance(choices[0], dict):
        raise InvalidProviderData("provider choice arrived after finish_reason")
    choice = choices[0]
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        raise InvalidProviderData("provider choice delta is invalid")
    if choice.get("finish_reason") is not None or any(
        value not in (None, "", [], {}) for value in delta.values()
    ):
        raise InvalidProviderData("provider choice arrived after finish_reason")


def _to_chat_messages(
    request: ModelRequest, api_style: APIStyle
) -> list[dict[str, JsonValue]]:
    messages: list[dict[str, JsonValue]] = []
    if request.system:
        messages.append(
            {
                "role": "system",
                "content": [
                    snapshot_mapping(block, "system block")
                    for block in request.system
                ],
            }
        )
    for message in request.messages:
        copied = snapshot_mapping(message, "chat message")
        role = copied.get("role")
        content = copied.get("content")
        if not isinstance(content, list):
            messages.append(copied)
            continue
        if role == "assistant":
            assistant_content: list[JsonValue] = []
            tool_calls: list[JsonValue] = []
            reasoning_items: list[JsonValue] = []
            reasoning_text = ""
            for value in content:
                if not isinstance(value, dict):
                    raise InvalidProviderData("assistant message content is invalid")
                block = snapshot_mapping(value, "assistant message block")
                block_type = block.get("type")
                if block_type == "text":
                    assistant_content.append(
                        {
                            "type": "text",
                            "text": require_text(
                                block.get("text"), "assistant replay text"
                            ),
                        }
                    )
                elif block_type == "refusal":
                    assistant_content.append(
                        {
                            "type": "text",
                            "text": require_text(
                                block.get("text"), "assistant replay refusal"
                            ),
                        }
                    )
                elif block_type in {"thinking", "redacted_thinking"}:
                    if api_style == "anthropic-messages":
                        assistant_content.append(block)
                    elif block_type == "thinking":
                        reasoning_text += require_text(
                            block.get("thinking"), "assistant replay thinking"
                        )
                elif block_type == "reasoning":
                    text = block.get("text")
                    if isinstance(text, str):
                        reasoning_text += text
                    item = block.get("item")
                    if isinstance(item, dict):
                        reasoning_items.append(item)
                    items = block.get("items")
                    if isinstance(items, list):
                        reasoning_items.extend(
                            item for item in items if isinstance(item, dict)
                        )
                elif block_type == "tool_call":
                    tool_calls.append(
                        {
                            "id": require_identity(
                                block.get("call_id"), "tool call id"
                            ),
                            "type": "function",
                            "function": {
                                "name": require_identity(
                                    block.get("name"), "tool name"
                                ),
                                "arguments": json.dumps(
                                    snapshot_mapping(
                                        block.get("input"), "tool input"
                                    ),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    )
            chat_message: dict[str, JsonValue] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if tool_calls:
                chat_message["tool_calls"] = tool_calls
            if reasoning_items:
                chat_message["reasoning_items"] = reasoning_items
            if reasoning_text:
                chat_message["reasoning_content"] = reasoning_text
            messages.append(chat_message)
            continue
        if role == "user":
            regular: list[JsonValue] = []
            tool_messages: list[dict[str, JsonValue]] = []
            for value in content:
                if not isinstance(value, dict):
                    raise InvalidProviderData("user message content is invalid")
                if value.get("type") != "tool_result":
                    regular.append(snapshot_mapping(value, "user message block"))
                    continue
                call_id = value.get("tool_call_id", value.get("tool_use_id"))
                result_content = value.get("content", "")
                if not isinstance(result_content, str):
                    result_content = json.dumps(
                        result_content,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": require_identity(
                            call_id, "tool result call id"
                        ),
                        "content": result_content,
                    }
                )
            if regular:
                messages.append({"role": "user", "content": regular})
            messages.extend(tool_messages)
            continue
        messages.append(copied)
    return messages


def _to_chat_tools(
    tools: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    converted: list[dict[str, JsonValue]] = []
    for tool in tools:
        name = require_identity(tool.get("name"), "tool name")
        schema = tool.get("input_schema", tool.get("parameters", {}))
        parameters = snapshot_mapping(schema, "tool input schema")
        function: dict[str, JsonValue] = {
            "name": name,
            "parameters": parameters,
        }
        description = tool.get("description")
        if description is not None:
            function["description"] = require_text(
                description, "tool description"
            )
        converted.append({"type": "function", "function": function})
    return converted


def _system_text(system: list[dict[str, JsonValue]]) -> str:
    return "\n\n".join(
        str(block["text"])
        for block in system
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _to_responses_input(
    messages: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    converted: list[dict[str, JsonValue]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if role == "assistant":
            text_parts: list[JsonValue] = []
            for value in content:
                if not isinstance(value, dict):
                    raise InvalidProviderData("assistant message content is invalid")
                block = snapshot_mapping(value, "Responses assistant block")
                block_type = block.get("type")
                if block_type in {"text", "refusal"}:
                    text_parts.append(
                        {
                            "type": "output_text",
                            "text": require_text(
                                block.get("text"), "Responses assistant text"
                            ),
                        }
                    )
                elif block_type == "reasoning":
                    if text_parts:
                        converted.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": text_parts,
                            }
                        )
                        text_parts = []
                    item = block.get("item")
                    if isinstance(item, dict):
                        converted.append(_reasoning_input_item(item))
                elif block_type == "tool_call":
                    if text_parts:
                        converted.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": text_parts,
                            }
                        )
                        text_parts = []
                    converted.append(
                        {
                            "type": "function_call",
                            "call_id": require_identity(
                                block.get("call_id"), "Responses tool call id"
                            ),
                            "name": require_identity(
                                block.get("name"), "Responses tool name"
                            ),
                            "arguments": json.dumps(
                                snapshot_mapping(
                                    block.get("input"), "Responses tool input"
                                ),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
            if text_parts:
                converted.append(
                    {"type": "message", "role": "assistant", "content": text_parts}
                )
            continue
        if role == "user":
            text_parts: list[JsonValue] = []
            for value in content:
                if not isinstance(value, dict):
                    raise InvalidProviderData("user message content is invalid")
                if value.get("type") == "tool_result":
                    if text_parts:
                        converted.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": text_parts,
                            }
                        )
                        text_parts = []
                    call_id = value.get(
                        "tool_call_id", value.get("tool_use_id")
                    )
                    output = value.get("content", "")
                    if not isinstance(output, str):
                        output = json.dumps(
                            output,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    converted.append(
                        {
                            "type": "function_call_output",
                            "call_id": require_identity(
                                call_id, "Responses tool result call id"
                            ),
                            "output": output,
                        }
                    )
                elif value.get("type") == "text":
                    text_parts.append(
                        {
                            "type": "input_text",
                            "text": require_text(
                                value.get("text"), "Responses user text"
                            ),
                        }
                    )
            if text_parts:
                converted.append(
                    {"type": "message", "role": "user", "content": text_parts}
                )
    return converted


def _to_responses_tools(
    tools: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    converted: list[dict[str, JsonValue]] = []
    for tool in tools:
        converted_tool: dict[str, JsonValue] = {
            "type": "function",
            "name": require_identity(tool.get("name"), "tool name"),
            "parameters": snapshot_mapping(
                tool.get("input_schema", tool.get("parameters", {})),
                "tool input schema",
            ),
        }
        description = tool.get("description")
        if description is not None:
            converted_tool["description"] = require_text(
                description, "tool description"
            )
        converted.append(converted_tool)
    return converted


def _reasoning_input_item(
    item: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    allowed = {"type", "id", "summary", "encrypted_content"}
    return {
        key: value
        for key, value in snapshot_mapping(item, "Responses reasoning item").items()
        if key in allowed and value is not None
    }


def _reasoning_item_text(item: dict[str, JsonValue]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return ""
    return " ".join(
        str(value["text"])
        for value in summary
        if isinstance(value, dict) and isinstance(value.get("text"), str)
    )


def _completion_usage(value: object, accumulator: UsageAccumulator):
    if value is None:
        return None
    usage = plain_mapping(value, "provider usage")
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(prompt_details, dict) or not isinstance(
        completion_details, dict
    ):
        raise InvalidProviderData("provider usage details are invalid")
    extensions: dict[str, object] = {}
    for key, item in completion_details.items():
        extensions[key] = item
    for key, item in prompt_details.items():
        if key != "cached_tokens":
            extensions[f"prompt_{key}"] = item
    return accumulator.update(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cache_read_tokens=prompt_details.get("cached_tokens"),
        extensions=extensions,
    )


def _responses_usage(value: object, accumulator: UsageAccumulator):
    if value is None:
        return None
    usage = plain_mapping(value, "Responses usage")
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    if not isinstance(input_details, dict) or not isinstance(output_details, dict):
        raise InvalidProviderData("Responses usage details are invalid")
    extensions: dict[str, object] = {}
    for key, item in output_details.items():
        extensions[key] = item
    for key, item in input_details.items():
        if key != "cached_tokens":
            extensions[f"input_{key}"] = item
    return accumulator.update(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=input_details.get("cached_tokens"),
        extensions=extensions,
    )


def _responses_stop(state: _ResponsesState) -> tuple[StopReason, str]:
    if any(state.blocks[key].kind == "tool_call" for key in state.order):
        return StopReason.TOOL_USE, "tool_calls"
    if state.status == "incomplete":
        raw = state.incomplete_reason or "incomplete"
        if raw in {"max_output_tokens", "max_tokens"}:
            return StopReason.MAX_TOKENS, raw
        if raw in {"content_filter", "content_filtered"}:
            return StopReason.REFUSAL, raw
        return StopReason.UNKNOWN, raw
    if state.status == "failed":
        return StopReason.UNKNOWN, "failed"
    return StopReason.END_TURN, state.status or "completed"
