"""Data models for the surrounding subsystem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.providers._json import JsonValue, snapshot_mapping, snapshot_object_list


class StopReason(StrEnum):
    """Enumeration of the stop reason values."""
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    CONTEXT_EXHAUSTED = "context_exhausted"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Usage:
    """Data model representing the usage."""
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    extensions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer")

        if not isinstance(self.extensions, Mapping):
            raise ValueError("usage extensions must be a JSON object")
        if len(self.extensions) > 32:
            raise ValueError("usage extensions must contain at most 32 entries")
        extensions: dict[str, int] = {}
        for key, value in self.extensions.items():
            if not isinstance(key, str) or not key:
                raise ValueError("usage extensions must use non-empty string keys")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("usage extensions must contain non-negative integers")
            extensions[key] = value
        object.__setattr__(self, "extensions", MappingProxyType(extensions))

    @property
    def total_tokens(self) -> int:
        """Handle the total tokens operation."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ToolCallBlock:
    """Data model representing the tool call block."""
    call_id: str
    name: str
    input: dict[str, JsonValue]
    extensions: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.call_id, "call_id")
        _require_non_empty(self.name, "name")
        object.__setattr__(self, "input", snapshot_mapping(self.input, "tool call input"))
        object.__setattr__(
            self,
            "extensions",
            snapshot_mapping(self.extensions, "tool call extensions"),
        )


@dataclass(frozen=True, slots=True)
class AssistantContent:
    """Data model representing the assistant content."""
    blocks: list[dict[str, JsonValue]]
    stop_reason: StopReason
    raw_stop_reason: str
    usage: Usage
    request_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", snapshot_object_list(self.blocks, "assistant blocks"))
        if not isinstance(self.stop_reason, StopReason):
            raise ValueError("stop_reason must be a StopReason")
        _require_non_empty(self.raw_stop_reason, "raw_stop_reason")
        if not isinstance(self.usage, Usage):
            raise ValueError("usage must be a Usage")
        _require_non_empty(self.request_id, "request_id")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Data model representing the model request."""
    model: str
    system: list[dict[str, JsonValue]]
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    max_tokens: int

    def __post_init__(self) -> None:
        _require_non_empty(self.model, "model")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        object.__setattr__(self, "system", snapshot_object_list(self.system, "system"))
        object.__setattr__(self, "messages", snapshot_object_list(self.messages, "messages"))
        object.__setattr__(self, "tools", snapshot_object_list(self.tools, "tools"))


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    """Data model representing the provider event."""
    type: str
    index: int | None = None
    delta: str | dict[str, JsonValue] | None = None
    block: dict[str, JsonValue] | None = None
    tool_call_id: str | None = None
    tool_call: ToolCallBlock | None = None
    usage: Usage | None = None
    stop_reason: StopReason | None = None
    raw_stop_reason: str | None = None
    request_id: str | None = None
    error: LiteCoderError | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.type, "type")
        if self.index is not None and (
            isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0
        ):
            raise ValueError("index must be a non-negative integer")
        if self.request_id is not None:
            _require_non_empty(self.request_id, "request_id")
        if self.tool_call_id is not None:
            _require_non_empty(self.tool_call_id, "tool_call_id")
        if self.raw_stop_reason is not None:
            _require_non_empty(self.raw_stop_reason, "raw_stop_reason")
        if self.block is not None:
            object.__setattr__(self, "block", snapshot_mapping(self.block, "event block"))
        if isinstance(self.delta, dict):
            object.__setattr__(self, "delta", snapshot_mapping(self.delta, "event delta"))
        elif self.delta is not None and not isinstance(self.delta, str):
            raise ValueError("event delta must be text or a JSON object")
        if self.error is not None:
            if not isinstance(self.error, LiteCoderError):
                raise ValueError("error must be a LiteCoderError")
            if not isinstance(self.error.code, ErrorCode):
                raise ValueError("provider error code must be an ErrorCode")
            if len(self.error.args) != 1 or not isinstance(self.error.args[0], str):
                raise ValueError("provider error message must be a string")
            if not isinstance(self.error.retryable, bool):
                raise ValueError("provider error retryable must be a bool")
            details = snapshot_mapping(self.error.details, "provider error details")
            object.__setattr__(
                self,
                "error",
                LiteCoderError(
                    self.error.code,
                    self.error.args[0],
                    retryable=self.error.retryable,
                    details=details,
                ),
            )
        self._validate_known_shape()
        if self.tool_call is not None:
            object.__setattr__(
                self,
                "tool_call",
                ToolCallBlock(
                    self.tool_call.call_id,
                    self.tool_call.name,
                    self.tool_call.input,
                    self.tool_call.extensions,
                ),
            )

    def _validate_known_shape(self) -> None:
        """Validate the known shape."""
        if self.stop_reason is not None and not isinstance(self.stop_reason, StopReason):
            raise ValueError("stop_reason must be a StopReason")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise ValueError("usage must be a Usage")
        if self.tool_call is not None and not isinstance(self.tool_call, ToolCallBlock):
            raise ValueError("tool_call must be a ToolCallBlock")
        fields = {
            "index": self.index,
            "delta": self.delta,
            "block": self.block,
            "tool_call_id": self.tool_call_id,
            "tool_call": self.tool_call,
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "raw_stop_reason": self.raw_stop_reason,
            "request_id": self.request_id,
            "error": self.error,
        }
        rules: dict[str, tuple[set[str], set[str]]] = {
            "response.request_id": ({"request_id"}, {"request_id"}),
            "content.started": ({"index", "block"}, {"index", "block", "request_id"}),
            "content.delta": ({"index", "delta"}, {"index", "delta", "request_id"}),
            "text.delta": ({"index", "delta"}, {"index", "delta", "request_id"}),
            "tool_call.input_delta": (
                {"index", "delta", "tool_call_id"},
                {"index", "delta", "tool_call_id", "request_id"},
            ),
            "tool_call.completed": (
                {"index", "tool_call"},
                {"index", "tool_call", "request_id"},
            ),
            "content.completed": ({"index", "block"}, {"index", "block", "request_id"}),
            "usage": ({"usage"}, {"usage", "request_id"}),
            "response.completed": (
                {"stop_reason", "raw_stop_reason"},
                {"stop_reason", "raw_stop_reason", "usage", "request_id"},
            ),
            "provider.error": ({"error"}, {"error", "request_id"}),
        }
        rule = rules.get(self.type)
        if rule is None:
            return
        required, allowed = rule
        present = {name for name, value in fields.items() if value is not None}
        missing = required - present
        if missing:
            raise ValueError(f"{self.type} requires fields: {', '.join(sorted(missing))}")
        if present - allowed:
            raise ValueError(f"{self.type} has incompatible fields")
        if self.type == "text.delta" and not isinstance(self.delta, str):
            raise ValueError("text.delta has incompatible fields")
        if self.type == "content.delta" and not isinstance(self.delta, dict):
            raise ValueError("content.delta has incompatible fields")
        if self.type == "tool_call.input_delta" and not isinstance(self.delta, str):
            raise ValueError("tool_call.input_delta has incompatible fields")

    @classmethod
    def request_identified(cls, request_id: str) -> ProviderEvent:
        """Handle the request identified operation."""
        return cls("response.request_id", request_id=request_id)

    @classmethod
    def content_block_started(
        cls, index: int, block: dict[str, JsonValue], *, request_id: str | None = None
    ) -> ProviderEvent:
        """Handle the content block started operation."""
        return cls("content.started", index=index, block=block, request_id=request_id)

    @classmethod
    def content_block_delta(
        cls, index: int, delta: dict[str, JsonValue], *, request_id: str | None = None
    ) -> ProviderEvent:
        """Handle the content block delta operation."""
        return cls("content.delta", index=index, delta=delta, request_id=request_id)

    @classmethod
    def text_delta(
        cls, index: int, text: str, *, request_id: str | None = None
    ) -> ProviderEvent:
        """Handle the text delta operation."""
        return cls("text.delta", index=index, delta=text, request_id=request_id)

    @classmethod
    def tool_call_input_delta(
        cls,
        index: int,
        call_id: str,
        delta: str,
        *,
        request_id: str | None = None,
    ) -> ProviderEvent:
        """Handle the tool call input delta operation."""
        return cls(
            "tool_call.input_delta",
            index=index,
            delta=delta,
            tool_call_id=call_id,
            request_id=request_id,
        )

    @classmethod
    def tool_call_completed(
        cls,
        index: int,
        tool_call: ToolCallBlock,
        *,
        request_id: str | None = None,
    ) -> ProviderEvent:
        """Handle the tool call completed operation."""
        return cls(
            "tool_call.completed",
            index=index,
            tool_call=tool_call,
            request_id=request_id,
        )

    @classmethod
    def content_block_completed(
        cls, index: int, block: dict[str, JsonValue], *, request_id: str | None = None
    ) -> ProviderEvent:
        """Handle the content block completed operation."""
        return cls("content.completed", index=index, block=block, request_id=request_id)

    @classmethod
    def usage_updated(cls, usage: Usage, *, request_id: str | None = None) -> ProviderEvent:
        """Handle the usage updated operation."""
        return cls("usage", usage=usage, request_id=request_id)

    @classmethod
    def response_completed(
        cls,
        stop_reason: StopReason,
        raw_stop_reason: str,
        *,
        usage: Usage | None = None,
        request_id: str | None = None,
    ) -> ProviderEvent:
        """Handle the response completed operation."""
        return cls(
            "response.completed",
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
            usage=usage,
            request_id=request_id,
        )

    @classmethod
    def provider_error(
        cls, error: LiteCoderError, *, request_id: str | None = None
    ) -> ProviderEvent:
        """Handle the provider error operation."""
        return cls("provider.error", error=error, request_id=request_id)


def _require_non_empty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
