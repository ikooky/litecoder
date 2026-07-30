"""Skill discovery and loading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)


SKILL_MAX_BYTES = 65_536
DESCRIPTION_MAX_CHARS = 240
SKILL_CATALOG_CONTEXT_PERCENT = 0.01
SKILL_CATALOG_CHARS_PER_TOKEN = 4
DEFAULT_SKILL_CATALOG_CHARS = 8_000
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Data model representing the skill metadata."""
    name: str
    source: str
    description: str
    path_identity: str

    def prompt_metadata(self) -> dict[str, str]:
        """Handle the prompt metadata operation."""
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class _SkillEntry:
    """Data model representing the skill entry."""
    metadata: SkillMetadata
    path: Path
    source_root: Path
    content_sha256: str = field(repr=False)


class SkillCatalog:
    """Component responsible for the skill catalog."""
    def __init__(self, entries: dict[str, _SkillEntry] | None = None) -> None:
        self._entries = dict(entries or {})

    @classmethod
    def discover(
        cls, project_root: Path, user_dir: Path, bundled_root: Path
    ) -> SkillCatalog:
        """Discover the requested operation."""
        roots = (
            ("project", project_root / ".litecoder" / "skills"),
            ("user", user_dir / "skills"),
            ("bundled", bundled_root),
        )
        selected: dict[str, _SkillEntry] = {}
        for source, configured_root in roots:
            for entry in _discover_root(source, configured_root):
                key = entry.metadata.name.casefold()
                if key in selected:
                    continue
                selected[key] = entry
        return cls(selected)

    def list(self) -> tuple[SkillMetadata, ...]:
        """Return the available entries."""
        return tuple(
            entry.metadata
            for entry in sorted(
                self._entries.values(),
                key=lambda item: (item.metadata.name.casefold(), item.metadata.name),
            )
        )

    def resolve(self, name: str) -> SkillMetadata:
        """Resolve the requested operation."""
        return self._resolve_entry(name).metadata

    def _resolve_entry(self, name: str) -> _SkillEntry:
        _validate_name(name)
        try:
            return self._entries[name.casefold()]
        except KeyError:
            raise KeyError("Unknown skill") from None

    def prompt_metadata(self, *, max_chars: int | None = None) -> list[dict[str, str]]:
        """Handle the prompt metadata operation."""
        full = [metadata.prompt_metadata() for metadata in self.list()]
        if max_chars is None:
            return full
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        if _catalog_chars(full) <= max_chars:
            return full

        names_only = [
            {"name": item["name"], "source": item["source"], "description": ""}
            for item in full
        ]
        if _catalog_chars(names_only) > max_chars:
            return _names_with_omission_marker(names_only, max_chars)

        rendered = [dict(item) for item in names_only]
        for index, source in enumerate(full):
            description = source["description"]
            if not description:
                continue
            remaining_slots = len(rendered) - index
            remaining = max_chars - _catalog_chars(rendered)
            if remaining <= 0:
                break
            target = min(len(description), max(1, remaining // remaining_slots))
            rendered[index]["description"] = _largest_fitting_description(
                rendered, index, description, target, max_chars
            )
        return rendered


def _names_with_omission_marker(
    names_only: list[dict[str, str]], max_chars: int
) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for entry in names_only:
        candidate = [*rendered, entry]
        if _catalog_chars(candidate) > max_chars:
            break
        rendered.append(entry)
    omitted = len(names_only) - len(rendered)
    if omitted:
        marker = {"truncated": f"{omitted} additional skills omitted"}
        if _catalog_chars([*rendered, marker]) <= max_chars:
            rendered.append(marker)
    return rendered


def _largest_fitting_description(
    entries: list[dict[str, str]],
    index: int,
    description: str,
    limit: int,
    max_chars: int,
) -> str:

    low, high, best = 0, limit, ""
    while low <= high:
        middle = (low + high) // 2
        entries[index]["description"] = _truncate_description(description, middle)
        if _catalog_chars(entries) <= max_chars:
            best = entries[index]["description"]
            low = middle + 1
        else:
            high = middle - 1
    entries[index]["description"] = best
    return best


def _truncate_description(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 0:
        return ""
    if limit == 1:
        return "…"
    return value[: limit - 1] + "…"


def _catalog_chars(entries: list[dict[str, str]]) -> int:
    return len(json.dumps(entries, ensure_ascii=False, separators=(",", ":")))



SkillCatalogResolver = Callable[[Path], SkillCatalog]


class LoadSkillTool:
    """Component responsible for the load skill tool."""
    spec = ToolSpec(
        name="load_skill",
        description="Load one discovered Skill only when its description is relevant to the current task. Apply it within runtime and user constraints; do not load unrelated skills speculatively.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        concurrency="shared",
        permission_risk="safe",
    )

    def __init__(
        self,
        catalog: SkillCatalog | None = None,
        *,
        catalog_resolver: SkillCatalogResolver | None = None,
        max_bytes: int = SKILL_MAX_BYTES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if catalog is None and catalog_resolver is None:
            raise ValueError("catalog or catalog_resolver is required")
        if catalog is not None and catalog_resolver is not None:
            raise ValueError("catalog and catalog_resolver are mutually exclusive")
        self.catalog = catalog
        self.catalog_resolver = catalog_resolver
        self.max_bytes = max_bytes

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        name = call.arguments.get("name")
        if not isinstance(name, str):
            raise ToolFailure("Skill name must be text")
        try:
            entry = self._catalog_for(context)._resolve_entry(name)
        except (KeyError, ValueError):
            raise ToolFailure("Skill is unavailable") from None
        try:
            path = _revalidate(entry)
            with path.open("rb") as handle:
                raw = handle.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes or b"\x00" in raw:
                raise ValueError
            content = raw.decode("utf-8")
            _revalidate(entry)
            if hashlib.sha256(raw).hexdigest() != entry.content_sha256:
                raise ValueError
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
            raise ToolFailure("Skill is unavailable") from None
        return ToolExecution.success(context.redactor.redact_text(content))

    def _catalog_for(self, context: ToolContext) -> SkillCatalog:
        if self.catalog_resolver is None:
            assert self.catalog is not None
            return self.catalog
        return self.catalog_resolver(context.workspace_root)


def _discover_root(source: str, configured_root: Path) -> tuple[_SkillEntry, ...]:
    """Discover the root."""
    if not configured_root.exists():
        return ()
    if _is_reparse_or_symlink(configured_root):
        raise ValueError("Skill source root is unsafe")
    try:
        root = configured_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return ()
    if not root.is_dir():
        raise ValueError("Skill source root must be a directory")
    found: dict[str, _SkillEntry] = {}
    try:
        children = sorted(
            root.iterdir(),
            key=lambda item: (item.name.casefold(), item.name),
        )
    except OSError as error:
        raise ValueError("Skill source root is unavailable") from error
    for child in children:
        name = child.name
        candidate = child / "SKILL.md"
        if not candidate.exists():
            if _is_reparse_or_symlink(candidate):
                raise ValueError("Skill path is unsafe")
            continue
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("Unsafe skill name")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            raise ValueError("Skill path escapes its source root") from None
        if (
            _is_reparse_or_symlink(child)
            or _is_reparse_or_symlink(candidate)
            or not resolved.is_file()
        ):
            raise ValueError("Skill path is unsafe")
        key = name.casefold()
        if key in found:
            raise ValueError("Skill name case collision")
        content = _read_skill(resolved)
        metadata = SkillMetadata(
            name=name,
            source=source,
            description=_description(content),
            path_identity=_path_identity(resolved),
        )
        found[key] = _SkillEntry(
            metadata=metadata,
            path=resolved,
            source_root=root,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
    return tuple(found.values())


def _read_skill(path: Path) -> str:
    """Read the skill."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(SKILL_MAX_BYTES + 1)
    except OSError as error:
        raise ValueError("Skill file is unavailable") from error
    if len(raw) > SKILL_MAX_BYTES or b"\x00" in raw:
        raise ValueError("Skill file is invalid")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Skill file is invalid") from None


def _description(content: str) -> str:
    lines = content.splitlines()
    if lines[:1] == ["---"]:
        for line in lines[1:]:
            if line == "---":
                break
            key, separator, value = line.partition(":")
            if separator and key.strip() == "description":
                return value.strip()[:DESCRIPTION_MAX_CHARS]
    return ""


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError("Unsafe skill name")


def _revalidate(entry: _SkillEntry) -> Path:
    _validate_name(entry.metadata.name)
    expected = entry.source_root / entry.metadata.name / "SKILL.md"
    if (
        expected != entry.path
        or _is_reparse_or_symlink(expected)
        or _is_reparse_or_symlink(expected.parent)
    ):
        raise ValueError
    resolved = expected.resolve(strict=True)
    resolved.relative_to(entry.source_root)
    if resolved != entry.path or not resolved.is_file():
        raise ValueError
    return resolved


def _path_identity(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve(strict=True)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        status = os.lstat(path)
    except OSError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)
