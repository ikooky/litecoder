"""Prompt assembly from detached runtime state."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litecoder.providers._json import JsonValue


PROMPT_SECTION_MAX_BYTES = 65_536
PROJECT_INSTRUCTIONS_MAX_BYTES = 32_768
_SECTION_ORDER = (
    "identity",
    "runtime",
    "project_instructions",
    "skills",
    "memories",
    "tasks",
    "team",
)


@dataclass(frozen=True, slots=True)
class PromptInputs:
    """Data model representing the prompt inputs."""
    identity: object
    runtime: object
    project_instructions: str | None
    skill_catalog: list[dict[str, object]]
    memories: object
    tasks: list[dict[str, object]]
    team: list[dict[str, object]]
    todos: list[dict[str, object]] | None = None


class PromptAssembler:
    """Build deterministic provider-neutral JSON text sections.

    Callers that provide session TODO state receive a ``todos`` section before
    ``tasks``. Tool schemas are intentionally excluded: providers receive them
    through their native tool-definition channel, which is the single source
    of truth for callable tools.
    """

    def __init__(self, *, section_max_bytes: int = PROMPT_SECTION_MAX_BYTES) -> None:
        if (
            isinstance(section_max_bytes, bool)
            or not isinstance(section_max_bytes, int)
            or section_max_bytes < 128
        ):
            raise ValueError("section_max_bytes must be an integer of at least 128")
        self.section_max_bytes = section_max_bytes

    def build(self, inputs: PromptInputs, *, total_max_bytes: int | None = None) -> list[dict[str, JsonValue]]:
        """Build the requested object."""
        values: dict[str, Any] = {
            "identity": inputs.identity,
            "runtime": inputs.runtime,
            "project_instructions": inputs.project_instructions,
            "skills": inputs.skill_catalog,
            "memories": inputs.memories,
            "tasks": inputs.tasks,
            "team": inputs.team,
        }
        section_order = _SECTION_ORDER
        if inputs.todos is not None:
            values["todos"] = inputs.todos
            section_order = (*_SECTION_ORDER[:-2], "todos", *_SECTION_ORDER[-2:])
        if total_max_bytes is not None and (
            isinstance(total_max_bytes, bool)
            or not isinstance(total_max_bytes, int)
            or total_max_bytes < 128 * len(section_order)
        ):
            raise ValueError("total_max_bytes is too small for prompt sections")

        def render(section_max_bytes: int) -> list[dict[str, JsonValue]]:
            blocks: list[dict[str, JsonValue]] = []
            for name in section_order:
                section = {"name": name, "content": values[name]}
                blocks.append(
                    {
                        "type": "text",
                        "text": _bounded_json(section, section_max_bytes),
                    }
                )
            return blocks

        blocks = render(self.section_max_bytes)
        if total_max_bytes is None:
            return blocks
        total_bytes = sum(
            len(str(block["text"]).encode("utf-8")) for block in blocks
        )
        if total_bytes <= total_max_bytes:
            return blocks
        return render(max(128, total_max_bytes // len(section_order)))


def load_project_instructions(
    workspace_root: Path, *, max_bytes: int = PROJECT_INSTRUCTIONS_MAX_BYTES
) -> str | None:
    """Load the project instructions."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    try:
        root = workspace_root.resolve(strict=True)
        candidate = root / "LITECODER.md"
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if _is_reparse_or_symlink(candidate) or not resolved.is_file():
            return None
        with resolved.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes or b"\x00" in raw:
            return None
        return raw.decode("utf-8")
    except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
        return None


def _bounded_json(section: dict[str, object], max_bytes: int) -> str:
    text = json.dumps(
        section,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    source = json.dumps(
        section["content"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    low, high, best = 0, len(source), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {
                "name": section["name"],
                "content": {"preview": source[:middle], "truncated": True},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(candidate.encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise ValueError("section_max_bytes is too small for a prompt section")
    return best


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        status = os.lstat(path)
    except OSError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def provider_neutral_summary(
    messages: list[dict[str, JsonValue]], *, limit: int = 20
) -> dict[str, JsonValue]:
    """Handle the provider neutral summary operation."""
    lines: list[str] = []
    for message in messages[-limit:]:
        role = str(message.get("role", "unknown"))
        content = message.get("content", [])
        text_parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ] if isinstance(content, list) else []
        lines.append(
            f"{role}: {' '.join(text_parts)}" if text_parts
            else f"{role}: [{len(content) if isinstance(content, list) else 1} content block(s)]"
        )
    return {
        "type": "context_summary",
        "text": "Provider-neutral parent session summary:\n" + "\n".join(lines),
    }