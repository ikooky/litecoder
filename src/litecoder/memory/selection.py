"""Relevant-memory selection helpers."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from litecoder.context.session.models import MessageRecord
from litecoder.memory.models import MemoryMetadata
from litecoder.memory.prompts import MEMORY_SELECTION_SYSTEM_PROMPT, complete_side_query, parse_json_array
from litecoder.memory.store import MemoryStore
from litecoder.providers import ModelProvider


SELECTION_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class MemorySelectionResult:
    """Data model representing the memory selection result."""
    filenames: tuple[str, ...]
    source: Literal["model", "fallback", "skipped"]
    reason: Literal["timeout", "provider_failed", "malformed"] | None = None

async def select_relevant_memories(
    store: MemoryStore,
    provider: ModelProvider,
    model: str,
    messages: Sequence[MessageRecord],
    *,
    max_items: int = 5,
) -> list[str]:
    """Select managed memory filenames relevant to recent user messages."""
    result = await _select_relevant_memories_result(
        store,
        provider,
        model,
        messages,
        max_items=max_items,
    )
    return list(result.filenames)


async def _select_relevant_memories_result(
    store: MemoryStore,
    provider: ModelProvider,
    model: str,
    messages: Sequence[MessageRecord],
    *,
    max_items: int = 5,
) -> MemorySelectionResult:
    """Select the relevant memories result."""
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 0:
        raise ValueError("max_items must be a non-negative integer")

    catalog = store.scan()
    recent = _recent_user_text(messages, limit=3, max_chars=2_000)
    if not catalog or not recent or max_items == 0:
        return MemorySelectionResult((), "skipped")

    task = asyncio.create_task(
        complete_side_query(
            provider,
            model,
            system=MEMORY_SELECTION_SYSTEM_PROMPT,
            prompt=_selection_prompt(recent, catalog),
            max_tokens=200,
        )
    )
    try:
        done, _ = await asyncio.wait(
            {task},
            timeout=SELECTION_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        _cancel_and_consume(task)
        raise

    if not done:
        _cancel_and_consume(task)
        return MemorySelectionResult(
            tuple(_unicode_fallback(recent, catalog, max_items)),
            "fallback",
            "timeout",
        )

    try:
        text = task.result()
    except asyncio.CancelledError:
        text = None
    except Exception:
        text = None
    if text is None:
        return MemorySelectionResult(
            tuple(_unicode_fallback(recent, catalog, max_items)),
            "fallback",
            "provider_failed",
        )

    parsed = parse_json_array(text)
    if parsed is None:
        return MemorySelectionResult(
            tuple(_unicode_fallback(recent, catalog, max_items)),
            "fallback",
            "malformed",
        )
    return MemorySelectionResult(
        tuple(_validated_indices(parsed, catalog, max_items)),
        "model",
    )


def _cancel_and_consume(task: asyncio.Task[str | None]) -> None:
    task.cancel()
    task.add_done_callback(_consume_task)


def _consume_task(task: asyncio.Task[str | None]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass

def _recent_user_text(
    messages: Sequence[MessageRecord], *, limit: int, max_chars: int
) -> str:
    recent: list[str] = []
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = _message_text(message)
        if text:
            recent.append(text)
            if len(recent) == limit:
                break
    return "\n".join(reversed(recent))[-max_chars:]


def _message_text(message: MessageRecord) -> str:
    return "\n".join(
        block["text"]
        for block in message.content
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _selection_prompt(recent: str, catalog: Sequence[MemoryMetadata]) -> str:
    entries = "\n".join(
        f"{index}: {entry.filename} | {entry.type} | {entry.description}"
        for index, entry in enumerate(catalog)
    )
    return f"Recent user messages:\n{recent}\n\nMemory catalog:\n{entries}"


def _validated_indices(
    values: Sequence[object], catalog: Sequence[MemoryMetadata], max_items: int
) -> list[str]:
    selected: list[str] = []
    seen: set[int] = set()
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= len(catalog)
            or value in seen
        ):
            continue
        seen.add(value)
        selected.append(catalog[value].filename)
        if len(selected) == max_items:
            break
    return selected


def _unicode_fallback(
    recent: str, catalog: Sequence[MemoryMetadata], max_items: int
) -> list[str]:
    query_terms = _unicode_terms(recent)
    if not query_terms:
        return []
    ranked: list[tuple[int, int, str]] = []
    for index, entry in enumerate(catalog):
        entry_terms = _unicode_terms(
            f"{entry.filename} {entry.name} {entry.description} {entry.type}"
        )
        score = len(query_terms & entry_terms)
        if score:
            ranked.append((-score, index, entry.filename))
    ranked.sort()
    return [filename for _, _, filename in ranked[:max_items]]


def _unicode_terms(text: str) -> set[str]:
    normalized = text.casefold()
    terms = set(re.findall(r"\w+", normalized, flags=re.UNICODE))
    for segment in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", normalized):
        terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms
