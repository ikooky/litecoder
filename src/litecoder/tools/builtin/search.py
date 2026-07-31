"""Built-in search tools."""

from __future__ import annotations

import asyncio
import codecs
import re
from dataclasses import dataclass
from pathlib import Path
from re import _constants as sre_constants
from re import _parser as sre_parse

from litecoder.tools.builtin._common import (
    MAX_FILE_BYTES,
    MAX_REGEX_PATTERN_CHARS,
    MAX_SEARCH_LINE_BYTES,
    PROCESS_READ_CHUNK_BYTES,
    matches_glob,
    optional_bool,
    result_limit,
    truncate_utf8,
    validate_glob_pattern,
)
from litecoder.tools.builtin.secure_path import (
    TraversalState,
    secure_iter_files,
    secure_read_chunks,
)
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)


_ROUND_GLOB_SNAPSHOT = "glob_files.snapshot"


@dataclass(frozen=True, slots=True)
class _GlobSnapshot:
    paths: tuple[str, ...]
    traversal: TraversalState


class GlobFilesTool:
    """Component responsible for the glob files tool."""
    spec = ToolSpec(
        "glob_files",
        "Discover workspace files by relative glob when the exact path is unknown; narrow the pattern before reading or editing files.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10_000},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        False,
        concurrency="traversal",
        permission_risk="safe",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return _glob_guard(call.arguments.get("pattern"))

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        pattern = validate_glob_pattern(call.arguments.get("pattern"))
        limit = result_limit(call.arguments)
        snapshot = await _round_glob_snapshot(context)
        if snapshot is not None:
            matches, result_truncated = _glob_matches(
                snapshot.paths, pattern, limit
            )
            return _glob_result(
                context,
                pattern,
                limit,
                matches,
                result_truncated,
                snapshot.traversal,
            )

        traversal = TraversalState()
        matches: list[str] = []
        result_truncated = False
        async for relative in secure_iter_files(context.workspace_root, traversal):
            if not matches_glob(relative, pattern):
                continue
            if len(matches) == limit:
                result_truncated = True
                break
            matches.append(relative)
        return _glob_result(
            context, pattern, limit, matches, result_truncated, traversal
        )


async def _round_glob_snapshot(context: ToolContext) -> _GlobSnapshot | None:
    if context.metadata.get("glob_batch_size") is None:
        return None
    existing = context.round_state.get(_ROUND_GLOB_SNAPSHOT)
    if existing is None:
        existing = asyncio.create_task(_collect_glob_snapshot(context.workspace_root))
        context.round_state[_ROUND_GLOB_SNAPSHOT] = existing
    if not isinstance(existing, asyncio.Task):
        raise RuntimeError("glob round state is invalid")
    return await existing


async def _collect_glob_snapshot(root: Path) -> _GlobSnapshot:
    traversal = TraversalState()
    paths: list[str] = []
    async for relative in secure_iter_files(root, traversal):
        paths.append(relative)
    return _GlobSnapshot(tuple(paths), traversal)


def _glob_matches(
    paths: tuple[str, ...], pattern: str, limit: int
) -> tuple[list[str], bool]:
    matches: list[str] = []
    for relative in paths:
        if not matches_glob(relative, pattern):
            continue
        if len(matches) == limit:
            return matches, True
        matches.append(relative)
    return matches, False


def _glob_result(
    context: ToolContext,
    pattern: str,
    limit: int,
    matches: list[str],
    result_truncated: bool,
    traversal: TraversalState,
) -> ToolExecution:
    rendered = context.redactor.redact_text("\n".join(matches))
    rendered, output_truncated = truncate_utf8(rendered)
    metadata = {
        "pattern": context.redactor.redact_text(pattern),
        "count": len(matches),
        "limit": limit,
        "truncated": result_truncated or traversal.truncated or output_truncated,
        "traversal_truncated": traversal.truncated,
        "directory_entries_truncated": traversal.directory_entries_truncated,
        "total_entries_truncated": traversal.total_entries_truncated,
        "traversed_entries": traversal.traversed_entries,
        "changed_workspace": False,
    }
    return ToolExecution.success(rendered, metadata=metadata, preview=matches)


class SearchTextTool:
    """Component responsible for the search text tool."""
    spec = ToolSpec(
        "search_text",
        "Locate definitions, usages, or relevant text in workspace files. Prefer literal search; enable regex only when its semantics are required and scope it with glob when possible.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "regex": {"type": "boolean", "default": False},
                "case_sensitive": {"type": "boolean", "default": True},
                "glob": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10_000},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        False,
        concurrency="traversal",
        permission_risk="safe",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        value = call.arguments.get("glob")
        return None if value is None else _glob_guard(value)

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query:
            raise ToolFailure("Invalid tool arguments", metadata={"field": "query"})
        regex = optional_bool(call.arguments, "regex", False)
        if not regex and len(query) > MAX_REGEX_PATTERN_CHARS:
            raise ToolFailure("Invalid search pattern")
        case_sensitive = optional_bool(call.arguments, "case_sensitive", True)
        pattern = validate_glob_pattern(call.arguments.get("glob", "**/*"))
        limit = result_limit(call.arguments)
        compiled: re.Pattern[str] | None = None
        if regex:
            _validate_safe_regex(query)
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(query, flags)
            except re.error:
                raise ToolFailure("Invalid search pattern") from None

        matches: list[dict[str, object]] = []
        skipped_files = 0
        result_truncated = False
        traversal = TraversalState()
        search_incomplete = False
        async for relative in secure_iter_files(context.workspace_root, traversal):
            if not matches_glob(relative, pattern):
                continue
            try:
                _, chunks = secure_read_chunks(
                    context.workspace_root,
                    relative,
                    chunk_bytes=PROCESS_READ_CHUNK_BYTES,
                    max_bytes=MAX_FILE_BYTES,
                )
                scan_state: dict[str, object] = {}
                file_matches, oversized, has_more = _search_chunks(
                    chunks,
                    relative=relative,
                    compiled=compiled,
                    context=context,
                    remaining_limit=limit - len(matches),
                    literal_query=None if regex else query,
                    case_sensitive=case_sensitive,
                    scan_state=scan_state,
                )
                search_incomplete = search_incomplete or bool(
                    scan_state.get("incomplete", False)
                )
            except (ToolDenied, ToolFailure, UnicodeError):
                skipped_files += 1
                continue
            if oversized:
                skipped_files += 1
                continue
            matches.extend(file_matches)
            if has_more:
                result_truncated = True
                break
        rendered = "\n".join(
            f"{item['path']}:{item['line']}:{item['column']}:{item['text']}"
            for item in matches
        )
        rendered = context.redactor.redact_text(rendered)
        rendered, output_truncated = truncate_utf8(rendered)
        metadata = {
            "count": len(matches),
            "limit": limit,
            "truncated": (
                result_truncated
                or traversal.truncated
                or output_truncated
                or search_incomplete
            ),
            "skipped_files": skipped_files,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "glob": context.redactor.redact_text(pattern),
            "matches": matches,
            "search_incomplete": search_incomplete,
            "traversal_truncated": traversal.truncated,
            "directory_entries_truncated": traversal.directory_entries_truncated,
            "total_entries_truncated": traversal.total_entries_truncated,
            "traversed_entries": traversal.traversed_entries,
            "changed_workspace": False,
        }
        return ToolExecution.success(rendered, metadata=metadata, preview=matches)


def _search_chunks(
    chunks: object,
    *,
    relative: str,
    compiled: re.Pattern[str] | None,
    context: ToolContext,
    remaining_limit: int,
    literal_query: str | None = None,
    case_sensitive: bool = True,
    scan_state: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], bool, bool]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    line_prefix = ""
    line_prefix_bytes = 0
    line_truncated = False
    line_char_count = 0
    literal_columns: list[int] = []
    literal_state = 0
    pending_cr = False
    total = 0
    line_number = 0
    matches: list[dict[str, object]] = []
    more = False
    incomplete = False
    literal_units = (
        tuple(_literal_unit(character, case_sensitive) for character in literal_query)
        if literal_query is not None
        else ()
    )
    literal_failure = _kmp_failure(literal_units)

    def append_line(fragment: str) -> None:
        """Handle the append line operation."""
        nonlocal line_prefix, line_prefix_bytes, line_truncated
        nonlocal line_char_count, literal_state
        if not fragment:
            return
        if literal_query is not None:
            capacity = remaining_limit - len(matches) + 1
            for offset, character in enumerate(fragment):
                unit = _literal_unit(character, case_sensitive)
                while literal_state and unit != literal_units[literal_state]:
                    literal_state = literal_failure[literal_state - 1]
                if unit == literal_units[literal_state]:
                    literal_state += 1
                if literal_state == len(literal_units):
                    if len(literal_columns) < capacity:
                        end = line_char_count + offset
                        literal_columns.append(end - len(literal_units) + 2)
                    literal_state = 0
        line_char_count += len(fragment)
        remaining = MAX_SEARCH_LINE_BYTES - line_prefix_bytes
        if remaining <= 0:
            line_truncated = True
            return
        bounded, truncated = truncate_utf8(fragment, remaining)
        line_prefix += bounded
        line_prefix_bytes += len(bounded.encode("utf-8"))
        line_truncated = line_truncated or truncated

    def append_match(column: int, preview: str, preview_truncated: bool) -> None:
        matches.append(
            {
                "path": relative,
                "line": line_number,
                "column": column,
                "text": preview,
                "line_truncated": line_truncated or preview_truncated,
            }
        )

    def inspect_line() -> None:
        """Handle the inspect line operation."""
        nonlocal line_number, more, incomplete
        line_number += 1
        if literal_query is None and line_truncated:
            incomplete = True
        preview = context.redactor.redact_text(line_prefix)
        preview, preview_truncated = truncate_utf8(preview, MAX_SEARCH_LINE_BYTES)
        if literal_query is not None:
            columns = literal_columns
        else:
            assert compiled is not None
            columns = [found.start() + 1 for found in compiled.finditer(line_prefix)]
        for column in columns:
            if len(matches) == remaining_limit:
                more = True
                return
            append_match(column, preview, preview_truncated)

    def finish_line() -> None:
        nonlocal line_prefix, line_prefix_bytes, line_truncated
        nonlocal line_char_count, literal_columns, literal_state
        inspect_line()
        line_prefix = ""
        line_prefix_bytes = 0
        line_truncated = False
        line_char_count = 0
        literal_columns = []
        literal_state = 0

    def process_text(text: str, *, final: bool) -> None:
        """Process the text."""
        nonlocal pending_cr
        index = 0
        if pending_cr:
            if text.startswith("\n"):
                index = 1
            finish_line()
            pending_cr = False
            if more:
                return
        segment_start = index
        while index < len(text):
            character = text[index]
            if character not in {"\r", "\n"}:
                index += 1
                continue
            append_line(text[segment_start:index])
            if character == "\r" and index + 1 == len(text) and not final:
                pending_cr = True
                return
            if character == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            finish_line()
            if more:
                return
            index += 1
            segment_start = index
        append_line(text[segment_start:])

    for chunk in chunks:  # type: ignore[union-attr]
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            return [], True, False
        if b"\x00" in chunk:
            raise ToolFailure("Search skipped non-text file")
        process_text(decoder.decode(chunk, final=False), final=False)
        if more:
            break
    if not more:
        process_text(decoder.decode(b"", final=True), final=True)
        if pending_cr:
            finish_line()
            pending_cr = False
        elif line_char_count or line_truncated:
            finish_line()
    if scan_state is not None:
        scan_state["incomplete"] = incomplete
    return matches, False, more


_IGNORECASE_LITERAL_SPECIALS = {
    "İ": "i",
    "ı": "i",
    "ſ": "s",
    "K": "k",
}


def _literal_unit(character: str, case_sensitive: bool) -> str:
    if case_sensitive:
        return character
    special = _IGNORECASE_LITERAL_SPECIALS.get(character)
    if special is not None:
        return special
    folded = character.casefold()
    if len(folded) == 1:
        return folded
    lowered = character.lower()
    return lowered if len(lowered) == 1 else character


def _kmp_failure(units: tuple[str, ...]) -> tuple[int, ...]:
    failure = [0] * len(units)
    matched = 0
    for index in range(1, len(units)):
        while matched and units[index] != units[matched]:
            matched = failure[matched - 1]
        if units[index] == units[matched]:
            matched += 1
            failure[index] = matched
    return tuple(failure)


def _validate_safe_regex(pattern: str) -> None:
    """Validate the safe regex."""
    if len(pattern) > MAX_REGEX_PATTERN_CHARS:
        raise ToolFailure("Invalid search pattern")
    try:
        parsed = sre_parse.parse(pattern, 0)
    except re.error:
        raise ToolFailure("Invalid search pattern") from None
    if not _safe_subpattern(parsed, inside_repeat=False):
        raise ToolFailure("Unsafe search pattern")


def _safe_subpattern(subpattern: object, *, inside_repeat: bool) -> bool:
    for operation, argument in subpattern:  # type: ignore[union-attr]
        if operation in {
            sre_constants.ASSERT,
            sre_constants.ASSERT_NOT,
            sre_constants.GROUPREF,
            sre_constants.GROUPREF_EXISTS,
        }:
            return False
        if operation is sre_constants.SUBPATTERN:
            if not _safe_subpattern(argument[-1], inside_repeat=inside_repeat):
                return False
        elif operation is sre_constants.BRANCH:
            if inside_repeat:
                return False
            if any(
                not _safe_subpattern(branch, inside_repeat=False)
                for branch in argument[1]
            ):
                return False
        elif operation in {sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT}:
            if inside_repeat:
                return False
            repeated = argument[2]
            if _contains_branch_or_repeat(repeated):
                return False
            if not _safe_subpattern(repeated, inside_repeat=True):
                return False
    return True


def _contains_branch_or_repeat(subpattern: object) -> bool:
    for operation, argument in subpattern:  # type: ignore[union-attr]
        if operation in {
            sre_constants.BRANCH,
            sre_constants.MAX_REPEAT,
            sre_constants.MIN_REPEAT,
        }:
            return True
        if operation is sre_constants.SUBPATTERN and _contains_branch_or_repeat(
            argument[-1]
        ):
            return True
    return False


def _glob_guard(value: object) -> str | None:
    try:
        validate_glob_pattern(value)
    except ToolDenied as error:
        return error.safe_message
    return None
