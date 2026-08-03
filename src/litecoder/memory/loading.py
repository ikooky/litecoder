"""Memory loading and prompt rendering."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from litecoder.context.session.models import MessageRecord
from litecoder.context.token_budget import estimate_tokens
from litecoder.memory.models import MemoryEntry
from litecoder.memory.selection import _select_relevant_memories_result
from litecoder.memory.store import MemoryStore
from litecoder.providers import ModelProvider


MAX_LOADED_FILES = 5
MAX_ENTRY_BODY_BYTES = 4_096
MAX_RENDERED_BYTES = 20_480


@dataclass(frozen=True, slots=True)
class LoadedMemories:
    """Data model representing the loaded memories."""
    entries: tuple[MemoryEntry, ...]
    rendered: str
    all_memory_tokens: int = 0
    memory_index_tokens: int = 0
    selected_names: tuple[str, ...] = ()


async def load_memories(
    store: MemoryStore,
    provider: ModelProvider,
    model: str,
    messages: Sequence[MessageRecord],
) -> LoadedMemories:
    """Read selected memory files and render a bounded request-only payload."""
    if not store.index_exists():
        return LoadedMemories((), "")

    all_memory_tokens, memory_index_tokens = _catalog_token_counts(store)

    try:
        selection = await _select_relevant_memories_result(
            store,
            provider,
            model,
            messages,
        )
    except Exception:
        return LoadedMemories(
            (),
            "",
            all_memory_tokens=all_memory_tokens,
            memory_index_tokens=memory_index_tokens,
        )

    entries: list[MemoryEntry] = []
    for filename in selection.filenames[:MAX_LOADED_FILES]:
        try:
            entry = store.read(Path(filename).stem)
            entry.render()
            entries.append(entry)
        except Exception:
            continue
    loaded = _fit_payload(entries)
    return replace(
        loaded,
        all_memory_tokens=all_memory_tokens,
        memory_index_tokens=memory_index_tokens,
    )


def _fit_payload(entries: list[MemoryEntry]) -> LoadedMemories:
    if not entries:
        return LoadedMemories((), "")

    bounded = [
        replace(
            entry,
            body=_escaped_utf8_prefix(entry.body, MAX_ENTRY_BODY_BYTES),
        )
        for entry in entries
    ]
    while bounded:
        rendered = _render_payload(bounded)
        if len(rendered.encode("utf-8")) <= MAX_RENDERED_BYTES:
            return LoadedMemories(
                tuple(bounded),
                rendered,
                selected_names=tuple(entry.name for entry in bounded),
            )

        excess = len(rendered.encode("utf-8")) - MAX_RENDERED_BYTES
        reduced = False
        for index in range(len(bounded) - 1, -1, -1):
            body_bytes = _escaped_length(bounded[index].body)
            if body_bytes == 0:
                continue
            target = max(0, body_bytes - excess)
            body = _escaped_utf8_prefix(bounded[index].body, target)
            if _escaped_length(body) < body_bytes:
                bounded[index] = replace(bounded[index], body=body)
                reduced = True
                break
        if not reduced:
            bounded.pop()

    return LoadedMemories((), "")


def _catalog_token_counts(store: MemoryStore) -> tuple[int, int]:
    """Return token estimates for the complete catalog and its index."""
    try:
        snapshot = store.snapshot()
    except (OSError, ValueError):
        return 0, 0
    catalog = _render_payload(snapshot.entries) if snapshot.entries else ""
    return (
        estimate_tokens(catalog) if catalog else 0,
        estimate_tokens(snapshot.index) if snapshot.index else 0,
    )


def _render_payload(entries: Sequence[MemoryEntry]) -> str:
    parts = ["<relevant_memories>"]
    parts.extend(_render_untrusted_entry(entry) for entry in entries)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def _render_untrusted_entry(entry: MemoryEntry) -> str:
    return (
        "---\n"
        f"name: {html.escape(entry.name, quote=False)}\n"
        f"description: {html.escape(entry.description, quote=False)}\n"
        f"type: {html.escape(entry.type, quote=False)}\n"
        "---\n\n"
        f"{html.escape(entry.body.rstrip(), quote=False)}\n"
    )


def _escaped_utf8_prefix(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    parts: list[str] = []
    used = 0
    try:
        for character in value:
            escaped = html.escape(character, quote=False)
            size = len(escaped.encode("utf-8"))
            if used + size > max_bytes:
                break
            parts.append(character)
            used += size
    except UnicodeEncodeError:
        return ""
    return "".join(parts)


def _escaped_length(value: str) -> int:
    try:
        return len(html.escape(value, quote=False).encode("utf-8"))
    except UnicodeEncodeError:
        return 0
