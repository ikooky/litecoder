"""Memory consolidation workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from litecoder.common.trace import SecretRedactor
from litecoder.memory.extraction import memory_entry_from_candidate
from litecoder.memory.models import MemoryEntry
from litecoder.memory.prompts import (
    MEMORY_CONSOLIDATION_SYSTEM_PROMPT,
    complete_side_query,
    parse_json_array,
)
from litecoder.memory.store import MemoryConflictError, MemoryStore
from litecoder.providers import ModelProvider


DREAM_THRESHOLD = 10
DREAM_MAX_ENTRIES = 30
DREAM_MAX_TOKENS = 3_000


@dataclass(frozen=True, slots=True)
class MemoryConsolidationResult:
    """Data model representing the memory consolidation result."""
    status: str
    before: int
    after: int


async def consolidate_memories(
    store: MemoryStore,
    provider: ModelProvider,
    model: str,
    redactor: SecretRedactor,
) -> MemoryConsolidationResult:
    """Handle the consolidate memories operation."""
    snapshot = store.snapshot()
    count = len(snapshot.entries)
    if count < DREAM_THRESHOLD:
        return MemoryConsolidationResult("skipped", count, count)

    rejected = MemoryConsolidationResult("rejected", count, count)
    failed = MemoryConsolidationResult("failed", count, count)
    try:
        text = await complete_side_query(
            provider,
            model,
            system=MEMORY_CONSOLIDATION_SYSTEM_PROMPT,
            prompt=_consolidation_prompt(snapshot.entries),
            max_tokens=DREAM_MAX_TOKENS,
        )
    except Exception:
        return failed

    if text is None:
        return failed
    payload = parse_json_array(text)
    if payload is None or len(payload) > DREAM_MAX_ENTRIES:
        return rejected

    try:
        replacement = _validated_replacement(payload, redactor)
    except ValueError:
        return rejected

    try:
        store.replace_all(replacement, expected=snapshot)
    except MemoryConflictError:
        return MemoryConsolidationResult(
            "conflict",
            count,
            len(store.snapshot().entries),
        )
    except (OSError, ValueError):
        return failed
    return MemoryConsolidationResult("completed", count, len(replacement))


def _consolidation_prompt(entries: Sequence[MemoryEntry]) -> str:
    payload = {
        "untrusted_entries": [
            {
                "name": entry.name,
                "type": entry.type,
                "description": entry.description,
                "body": entry.body,
            }
            for entry in entries
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validated_replacement(
    payload: Sequence[object],
    redactor: SecretRedactor,
) -> tuple[MemoryEntry, ...]:
    entries: list[MemoryEntry] = []
    names: set[str] = set()
    for item in payload:
        entry = memory_entry_from_candidate(item, redactor)
        key = entry.name.casefold()
        if key in names:
            raise ValueError("memory rejected")
        names.add(key)
        entries.append(entry)
    return tuple(entries)
