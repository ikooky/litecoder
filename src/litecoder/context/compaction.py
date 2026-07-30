"""Context compaction policies and summarization."""

from __future__ import annotations

import copy
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from litecoder.context.session.models import MessageRecord
from litecoder.context.token_budget import _non_negative_integer, estimate_tokens


DEFAULT_TOOL_RESULT_BUDGET_DIVISOR = 2
Summarizer = Callable[["SummaryRequest"], Awaitable[str]]


class CompactionUnavailable(RuntimeError):
    """The current context cannot be reduced safely."""


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    """Data model representing the summary request."""
    covered_through_sequence: int
    messages: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        _positive_integer(
            self.covered_through_sequence, "covered_through_sequence"
        )


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Data model representing the compaction result."""
    messages: list[MessageRecord]
    summary: str | None = None
    summary_request: SummaryRequest | None = None


@dataclass(slots=True)
class _MessageGroup:
    """Data model representing the message group."""
    messages: list[MessageRecord]
    tool_round: bool = False

    @property
    def highest_sequence(self) -> int:
        """Handle the highest sequence operation."""
        return max(_sequence(message) for message in self.messages)


class CompactionPolicy:
    """Component responsible for the compaction policy."""
    def __init__(
        self,
        *,
        tool_result_budget_tokens: int | None = None,
        keep_recent_tool_rounds: int = 3,
        compacted_tool_result_preview_tokens: int = 64,
    ) -> None:
        if tool_result_budget_tokens is not None:
            _non_negative_integer(
                tool_result_budget_tokens, "tool_result_budget_tokens"
            )
        _non_negative_integer(keep_recent_tool_rounds, "keep_recent_tool_rounds")
        _non_negative_integer(
            compacted_tool_result_preview_tokens,
            "compacted_tool_result_preview_tokens",
        )
        self.tool_result_budget_tokens = tool_result_budget_tokens
        self.keep_recent_tool_rounds = keep_recent_tool_rounds
        self.compacted_tool_result_preview_tokens = (
            compacted_tool_result_preview_tokens
        )

    async def compact(
        self,
        messages: Sequence[MessageRecord],
        budget_tokens: int,
        summarizer: Summarizer | None = None,
        *,
        summary_budget_tokens: int | None = None,
        force_summary: bool = False,
    ) -> CompactionResult:
        """Compact the selected context or session."""
        budget_tokens = _non_negative_integer(budget_tokens, "budget_tokens")
        if not isinstance(force_summary, bool):
            raise ValueError("force_summary must be a bool")
        if summary_budget_tokens is None:
            summary_budget_tokens = budget_tokens
        summary_budget_tokens = _non_negative_integer(
            summary_budget_tokens, "summary_budget_tokens"
        )
        groups = _valid_groups(messages)
        persisted_messages = copy.deepcopy(list(messages))
        original_over_budget = (
            estimate_message_tokens(persisted_messages) > budget_tokens
        )
        tool_budget = self.tool_result_budget_tokens
        if tool_budget is None:
            tool_budget = budget_tokens // DEFAULT_TOOL_RESULT_BUDGET_DIVISOR
        _enforce_total_tool_result_budget(
            groups,
            tool_budget=tool_budget,
            preview_tokens=self.compacted_tool_result_preview_tokens,
        )
        _compact_older_tool_results(
            groups,
            keep_recent=self.keep_recent_tool_rounds,
            preview_tokens=self.compacted_tool_result_preview_tokens,
        )
        working = _flatten(groups)
        if (
            estimate_message_tokens(working) <= budget_tokens
            and not (force_summary and original_over_budget)
        ):
            return CompactionResult(working)

        remaining, snipped = _snip_complete_groups(
            groups,
            budget_tokens=budget_tokens,
            keep_recent_tool_rounds=self.keep_recent_tool_rounds,
        )
        working = _flatten(remaining)
        if force_summary and original_over_budget and not snipped and remaining:
            snipped.append(remaining.pop(0))
            working = _flatten(remaining)
        if (
            estimate_message_tokens(working) <= budget_tokens
            and not (force_summary and snipped)
        ):
            return CompactionResult(working)

        remaining, snipped = _expand_summary_prefix(
            groups,
            already_snipped=snipped,
            budget_tokens=budget_tokens,
        )
        working = _flatten(remaining)
        if not snipped:
            return CompactionResult(working)
        if summarizer is None:
            raise RuntimeError(
                "context remains over budget and requires a summarizer"
            )

        while True:
            request = SummaryRequest(
                covered_through_sequence=max(
                    group.highest_sequence for group in snipped
                ),
                messages=tuple(
                    _provider_message(message)
                    for group in snipped
                    for message in group.messages
                ),
            )
            summary = await summarizer(request)
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("summary output must be a non-empty string")
            summary = summary.strip()
            summary_record = MessageRecord(
                session_id=snipped[0].messages[0].session_id,
                role="system",
                content=[{
                    "type": "context_summary",
                    "covered_through_sequence": request.covered_through_sequence,
                    "text": summary,
                }],
                sequence=request.covered_through_sequence,
            )
            persisted_suffix = [
                message
                for message in persisted_messages
                if _sequence(message) > request.covered_through_sequence
            ]
            if (
                estimate_message_tokens([summary_record, *persisted_suffix])
                <= summary_budget_tokens
            ):
                return CompactionResult(working, summary, request)
            if not remaining:
                raise CompactionUnavailable(
                    "summary output exceeds the context budget"
                )
            snipped.append(remaining.pop(0))
            working = _flatten(remaining)


def estimate_message_tokens(messages: Sequence[MessageRecord]) -> int:
    """Handle the estimate message tokens operation."""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages must be a sequence")
    total = 0
    for message in messages:
        if not isinstance(message, MessageRecord):
            raise ValueError("messages must contain MessageRecord values")
        payload = {"role": message.role, "content": message.content}
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        total += estimate_tokens(rendered)
    return total


def _valid_groups(messages: Sequence[MessageRecord]) -> list[_MessageGroup]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages must be a sequence")
    snapshots: list[MessageRecord] = []
    previous_sequence = 0
    for message in messages:
        if not isinstance(message, MessageRecord):
            raise ValueError("messages must contain MessageRecord values")
        sequence = _sequence(message)
        if sequence <= previous_sequence:
            raise ValueError(
                "message sequence values must be strictly increasing"
            )
        previous_sequence = sequence
        snapshots.append(copy.deepcopy(message))

    groups: list[_MessageGroup] = []
    index = 0
    while index < len(snapshots):
        current = snapshots[index]
        call_ids = _tool_call_ids(current)
        if call_ids and index + 1 < len(snapshots):
            following = snapshots[index + 1]
            if (
                current.role == "assistant"
                and following.role == "user"
                and call_ids == _tool_result_ids(following)
                and len(call_ids) == len(set(call_ids))
            ):
                groups.append(_MessageGroup([current, following], tool_round=True))
                index += 2
                continue
        sanitized = _remove_orphan_tool_blocks(current)
        if sanitized.content:
            groups.append(_MessageGroup([sanitized]))
        index += 1
    return groups


def _tool_call_ids(message: MessageRecord) -> list[str]:
    ids: list[str] = []
    for block in message.content:
        if block.get("type") != "tool_call":
            continue
        call_id = block.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return []
        ids.append(call_id)
    return ids


def _tool_result_ids(message: MessageRecord) -> list[str]:
    ids: list[str] = []
    for block in message.content:
        if block.get("type") != "tool_result":
            continue
        call_id = block.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            return []
        ids.append(call_id)
    return ids


def _remove_orphan_tool_blocks(message: MessageRecord) -> MessageRecord:
    message.content = [
        block
        for block in message.content
        if block.get("type") not in {"tool_call", "tool_result"}
    ]
    return message


def _enforce_total_tool_result_budget(
    groups: list[_MessageGroup],
    *,
    tool_budget: int,
    preview_tokens: int,
) -> None:
    blocks = _tool_result_blocks(groups)
    for block in blocks:
        if _tool_result_tokens(groups) <= tool_budget:
            return
        _compact_tool_result(block, preview_tokens)
    for replacement in ("[compacted]", ""):
        for block in blocks:
            if _tool_result_tokens(groups) <= tool_budget:
                return
            block["content"] = replacement
    for block in blocks:
        if _tool_result_tokens(groups) <= tool_budget:
            return
        _compact_tool_result_metadata(block)


def _compact_tool_result_metadata(block: dict[str, object]) -> None:
    metadata = block.get("metadata")
    if not isinstance(metadata, dict):
        block.pop("metadata", None)
        return
    priority = (
        "artifact",
        "artifact_error",
        "automatic_retry",
        "changed_workspace",
        "workspace_version",
    )
    block["metadata"] = {
        key: copy.deepcopy(metadata[key])
        for key in priority
        if key in metadata
    }


def _compact_older_tool_results(
    groups: list[_MessageGroup],
    *,
    keep_recent: int,
    preview_tokens: int,
) -> None:
    tool_groups = [group for group in groups if group.tool_round]
    protected_ids = (
        {id(group) for group in tool_groups[-keep_recent:]}
        if keep_recent
        else set()
    )
    for group in tool_groups:
        if id(group) in protected_ids:
            continue
        for message in group.messages:
            for block in message.content:
                if block.get("type") == "tool_result":
                    _compact_tool_result(block, preview_tokens)


def _compact_tool_result(block: dict[str, object], preview_tokens: int) -> None:
    content = block.get("content")
    if not isinstance(content, str):
        content = json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    original_bytes = len(content.encode("utf-8"))
    preview = _utf8_prefix(content, preview_tokens * 4)
    compacted = f"{preview}\n[compacted:{original_bytes}B]"
    if estimate_tokens(compacted) < estimate_tokens(content):
        block["content"] = compacted


def _tool_result_tokens(groups: Sequence[_MessageGroup]) -> int:
    total = 0
    for block in _tool_result_blocks(groups):
        rendered = json.dumps(
            block,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        total += estimate_tokens(rendered)
    return total


def _tool_result_blocks(
    groups: Sequence[_MessageGroup],
) -> list[dict[str, object]]:
    return [
        block
        for group in groups
        for message in group.messages
        for block in message.content
        if block.get("type") == "tool_result"
    ]


def _snip_complete_groups(
    groups: list[_MessageGroup],
    *,
    budget_tokens: int,
    keep_recent_tool_rounds: int,
) -> tuple[list[_MessageGroup], list[_MessageGroup]]:
    remaining = list(groups)
    snipped: list[_MessageGroup] = []
    protected_tool_ids = (
        {
            id(group)
            for group in [item for item in groups if item.tool_round][
                -keep_recent_tool_rounds:
            ]
        }
        if keep_recent_tool_rounds
        else set()
    )
    while estimate_message_tokens(_flatten(remaining)) > budget_tokens:
        if len(remaining) <= 1:
            break
        removable = remaining[0]
        if id(removable) in protected_tool_ids:
            break
        remaining.remove(removable)
        snipped.append(removable)
    return remaining, snipped


def _expand_summary_prefix(
    groups: list[_MessageGroup],
    *,
    already_snipped: list[_MessageGroup],
    budget_tokens: int,
) -> tuple[list[_MessageGroup], list[_MessageGroup]]:
    snipped = list(already_snipped)
    remaining = list(groups[len(snipped):])
    while (
        estimate_message_tokens(_flatten(remaining)) > budget_tokens
        and remaining
    ):
        snipped.append(remaining.pop(0))
    return remaining, snipped


def _flatten(groups: Sequence[_MessageGroup]) -> list[MessageRecord]:
    return [message for group in groups for message in group.messages]


def _provider_message(message: MessageRecord) -> dict[str, object]:
    return {"role": message.role, "content": copy.deepcopy(message.content)}


def _sequence(message: MessageRecord) -> int:
    sequence = message.sequence
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise ValueError("message sequence must be a positive integer")
    return sequence


def _positive_integer(value: object, field_name: str) -> int:
    value = _non_negative_integer(value, field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _utf8_prefix(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")
