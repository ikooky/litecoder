"""Memory candidate extraction and validation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from litecoder.common.trace import SecretRedactor
from litecoder.context.session.models import MessageRecord
from litecoder.memory.models import MemoryEntry, MemoryMetadata
from litecoder.memory.prompts import (
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    complete_side_query_result,
    parse_json_array,
)
from litecoder.memory.store import (
    MEMORY_FILE_MAX_BYTES,
    MemoryStore,
    write_memory_files,
)
from litecoder.providers import ModelProvider, StopReason


EXTRACTION_MAX_TOKENS = 8_000

_MEMORY_EXTRACTION_RETRY_SYSTEM_PROMPT = (
    MEMORY_EXTRACTION_SYSTEM_PROMPT
    + "\nThe user explicitly asked to persist information. Return a valid, "
    "non-empty JSON array when the request contains a safe durable memory."
)
_EXPLICIT_MEMORY_PATTERN = re.compile(
    r"\b(?:remember|save|store)\b|"
    r"\u8bb0\u4f4f|\u8bb0\u4e0b|\u8bb0\u5f55\u4e0b|\u4fdd\u5b58\u5230\u8bb0\u5fc6|"
    r"\u5b58\u5230\u8bb0\u5fc6",
    re.IGNORECASE,
)


_UNSAFE_MEMORY_PATTERNS = (
    re.compile(
        r"\b(?:pid|process[\s_-]+id|session[\s_-]+id|request[\s_-]+id|"
        r"trace[\s_-]+id)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcurrent[\s_-]+(?:branch|commit|process|session|request|trace|time|"
        r"date|working[\s_-]+directory)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btemporary[\s_-]+(?:file|path|directory|state|value)\b",
        re.IGNORECASE,
    ),
)
_UNSAFE_DIRECTIVE_PATTERN = re.compile(
    r"(?:^|[\n:;.!?]\s*|[-*]\s+)"
    r"(?:(?:please|now)\s+|you\s+(?:must|should|need\s+to)\s+)?"
    r"(?:ignore|disregard|override|bypass|replace|forget)\b.{0,120}"
    r"\b(?:(?:system|developer|previous|prior)\s+)*"
    r"(?:instruction|prompt|polic(?:y|ies)|rule)s?\b",
    re.IGNORECASE | re.DOTALL,
)
_UNSAFE_NONCOMPLIANCE_PATTERN = re.compile(
    r"(?:^|[\n:;.!?]\s*|[-*]\s+)(?:do\s+not|don't|never)\s+"
    r"(?:follow|obey|respect)\b.{0,120}"
    r"\b(?:(?:system|developer|previous|prior)\s+)*"
    r"(?:instruction|prompt|polic(?:y|ies)|rule)s?\b",
    re.IGNORECASE | re.DOTALL,
)
_WRAPPER_TAGS = ("<relevant_memories>", "</relevant_memories>")
_OVERRIDE_VERBS = (
    "\u5ffd\u7565",
    "\u65e0\u89c6",
    "\u7ed5\u8fc7",
    "\u8986\u76d6",
    "\u66ff\u4ee3",
    "\u4e0d\u8981\u9075\u5b88",
)
_PROTECTED_TARGETS = (
    "\u7cfb\u7edf",
    "\u5f00\u53d1\u8005",
    "\u4e4b\u524d",
    "\u5148\u524d",
    "\u4e0a\u8ff0",
    "\u4ee5\u4e0a",
)
_INSTRUCTION_NOUNS = (
    "\u6307\u4ee4",
    "\u63d0\u793a",
    "\u89c4\u5219",
    "\u653f\u7b56",
)

MemoryExtractionStatus = Literal[
    "completed",
    "empty",
    "provider_failed",
    "truncated",
    "malformed",
    "partial_rejected",
    "failed",
]


@dataclass(frozen=True, slots=True)
class MemoryExtractionResult:
    """Data model representing the memory extraction result."""
    proposed: int
    accepted: int
    rejected: int
    written: int
    status: MemoryExtractionStatus
    total: int = 0
    provider_code: str | None = None
    limit: int | None = None


async def extract_memories(
    store: MemoryStore,
    provider: ModelProvider,
    model: str,
    redactor: SecretRedactor,
    session_id: str,
    messages: Sequence[MessageRecord],
) -> MemoryExtractionResult:
    """Extract the memories."""
    catalog = store.scan() if store.index_exists() else ()
    try:
        outcome = await complete_side_query_result(
            provider,
            model,
            system=MEMORY_EXTRACTION_SYSTEM_PROMPT,
            prompt=_extraction_prompt(session_id, messages, catalog),
            max_tokens=EXTRACTION_MAX_TOKENS,
        )
    except Exception:
        return MemoryExtractionResult(0, 0, 0, 0, "failed")

    if outcome.provider_code is not None:
        return MemoryExtractionResult(
            0,
            0,
            0,
            0,
            "provider_failed",
            provider_code=outcome.provider_code,
        )
    if outcome.stop_reason is StopReason.MAX_TOKENS:
        return MemoryExtractionResult(
            0,
            0,
            0,
            0,
            "truncated",
            limit=EXTRACTION_MAX_TOKENS,
        )
    if outcome.stop_reason is not StopReason.END_TURN:
        return MemoryExtractionResult(0, 0, 0, 0, "failed")

    payload = parse_json_array(outcome.text)
    if payload is None or not payload:
        initial_status: MemoryExtractionStatus = (
            "malformed" if payload is None else "empty"
        )
        if not is_explicit_memory_request(messages):
            return MemoryExtractionResult(0, 0, 0, 0, initial_status)
        try:
            retry = await complete_side_query_result(
                provider,
                model,
                system=_MEMORY_EXTRACTION_RETRY_SYSTEM_PROMPT,
                prompt=_extraction_prompt(
                    session_id,
                    messages,
                    catalog,
                    retry=True,
                ),
                max_tokens=EXTRACTION_MAX_TOKENS,
            )
        except Exception:
            return MemoryExtractionResult(0, 0, 0, 0, initial_status)
        if (
            retry.provider_code is not None
            or retry.stop_reason is not StopReason.END_TURN
        ):
            return MemoryExtractionResult(0, 0, 0, 0, initial_status)
        payload = parse_json_array(retry.text)
        if payload is None or not payload:
            return MemoryExtractionResult(0, 0, 0, 0, initial_status)

    accepted: list[MemoryEntry] = []
    rejected = 0
    for item in payload:
        try:
            accepted.append(memory_entry_from_candidate(item, redactor))
        except ValueError:
            rejected += 1

    status: MemoryExtractionStatus = (
        "partial_rejected" if rejected else "completed"
    )
    if not accepted:
        return MemoryExtractionResult(
            proposed=len(payload),
            accepted=0,
            rejected=rejected,
            written=0,
            status=status,
        )

    accepted_names = {entry.name.casefold() for entry in accepted}
    if len(accepted_names) != len(accepted):
        return MemoryExtractionResult(
            proposed=len(payload),
            accepted=0,
            rejected=rejected + len(accepted),
            written=0,
            status="partial_rejected",
        )

    try:
        write_result = write_memory_files(store, redactor, accepted)
    except (OSError, ValueError):
        return MemoryExtractionResult(
            proposed=len(payload),
            accepted=len(accepted),
            rejected=rejected,
            written=0,
            status="failed",
        )

    return MemoryExtractionResult(
        proposed=len(payload),
        accepted=len(accepted),
        rejected=rejected,
        written=len(write_result.paths),
        status=status,
        total=write_result.total,
    )

def _extraction_prompt(
    session_id: str,
    messages: Sequence[MessageRecord],
    catalog: Sequence[MemoryMetadata],
    retry: bool = False,
) -> str:
    payload = {
        "session_id": session_id,
        "messages": [
            {
                "role": message.role,
                "content": [dict(block) for block in message.content],
            }
            for message in messages
        ],
        "catalog": [f"{entry.name}: {entry.description}" for entry in catalog],
    }
    if retry:
        payload["retry"] = (
            "The first extraction response was empty or malformed. Extract the "
            "explicit durable-memory request now."
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def is_explicit_memory_request(messages: Sequence[MessageRecord]) -> bool:
    """Return whether the explicit memory request condition holds."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = "\n".join(
            str(block.get("text", ""))
            for block in message.content
            if block.get("type") == "text"
        )
        return _EXPLICIT_MEMORY_PATTERN.search(text) is not None
    return False


def memory_entry_from_candidate(
    item: object,
    redactor: SecretRedactor,
) -> MemoryEntry:
    """Validate a model-proposed memory using the shared safety rules."""
    if not isinstance(item, dict) or not isinstance(redactor, SecretRedactor):
        raise ValueError("memory rejected")
    fields = (
        item.get("name"),
        item.get("description"),
        item.get("type"),
        item.get("body"),
    )
    if not all(isinstance(value, str) for value in fields):
        raise ValueError("memory rejected")
    if not fields[3].strip():
        raise ValueError("memory rejected")
    try:
        entry = MemoryEntry(
            name=fields[0],
            description=fields[1],
            type=fields[2],
            body=fields[3],
        )
    except ValueError as error:
        raise ValueError("memory rejected") from error

    content = _normalize_for_safety(f"{entry.description}\n{entry.body}")
    if any(tag in content for tag in _WRAPPER_TAGS):
        raise ValueError("memory rejected")
    if any(pattern.search(content) is not None for pattern in _UNSAFE_MEMORY_PATTERNS):
        raise ValueError("memory rejected")
    if _UNSAFE_DIRECTIVE_PATTERN.search(content) is not None:
        raise ValueError("memory rejected")
    if _UNSAFE_NONCOMPLIANCE_PATTERN.search(content) is not None:
        raise ValueError("memory rejected")
    if _contains_multilingual_override(content):
        raise ValueError("memory rejected")

    rendered = entry.render()
    try:
        rendered_size = len(rendered.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("memory rejected") from error
    if rendered_size > MEMORY_FILE_MAX_BYTES:
        raise ValueError("memory rejected")
    if redactor.redact_text(rendered) != rendered:
        raise ValueError("memory rejected")
    return entry


def _normalize_for_safety(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not _is_default_ignorable(character)
    )


def _is_default_ignorable(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character) == "Cf"
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
        or codepoint
        in {
            0x034F,
            0x115F,
            0x1160,
            0x17B4,
            0x17B5,
            0x180B,
            0x180C,
            0x180D,
            0x180F,
            0x3164,
            0xFFA0,
        }
    )


def _find_all_positions(value: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = value.find(needle, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


_OVERRIDE_TERMS = tuple(
    (term, "verb") for term in _OVERRIDE_VERBS
) + tuple(
    (term, "target") for term in _PROTECTED_TARGETS
) + tuple(
    (term, "noun") for term in _INSTRUCTION_NOUNS
)
_OVERRIDE_PATTERN = re.compile(
    "|".join(re.escape(term) for term, _ in _OVERRIDE_TERMS)
)


def _find_all_positions(value: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = value.find(needle, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


def _contains_multilingual_override(content: str) -> bool:
    compact = re.sub(r"\s+", "", content)
    # Single pass over the compacted text instead of one str.find scan per term
    # (14 terms previously meant 14 linear traversals).
    label_by_term = {term: label for term, label in _OVERRIDE_TERMS}
    occurrences: list[tuple[int, str]] = [
        (match.start(), label_by_term[match.group()])
        for match in _OVERRIDE_PATTERN.finditer(compact)
    ]
    occurrences.sort(key=lambda occurrence: occurrence[0])
    category_counts = {"verb": 0, "target": 0, "noun": 0}
    left = 0
    for position, category in occurrences:
        category_counts[category] += 1
        while position - occurrences[left][0] > 120:
            category_counts[occurrences[left][1]] -= 1
            left += 1
        if all(category_counts.values()):
            return True
    return False