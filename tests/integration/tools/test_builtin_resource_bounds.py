from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from litecoder.tools import ToolCall, ToolContext, ToolFailure
from litecoder.tools.builtin import ReadFileTool, RunShellTool, SearchTextTool
from litecoder.tools.builtin._common import (
    MAX_DIRECTORY_ENTRIES,
    MAX_FILE_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_REGEX_PATTERN_CHARS,
    MAX_SEARCH_LINE_BYTES,
    MAX_TRAVERSAL_ENTRIES,
    PROCESS_READ_CHUNK_BYTES,
)


def context(root: Path, *, secrets: tuple[str, ...] = ()) -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": "ask"},
        secret_values=secrets,
    )


def test_binding_resource_constants() -> None:
    assert MAX_FILE_BYTES == 1_048_576
    assert MAX_OUTPUT_BYTES == 65_536
    assert PROCESS_READ_CHUNK_BYTES == 16_384
    assert MAX_SEARCH_LINE_BYTES == 4_096
    assert MAX_DIRECTORY_ENTRIES == 10_000
    assert MAX_TRAVERSAL_ENTRIES == 100_000
    assert MAX_REGEX_PATTERN_CHARS == 512


@pytest.mark.asyncio
async def test_shell_streams_and_bounds_both_pipes_with_boundary_secret(
    tmp_path: Path,
) -> None:
    secret = "split-secret-value"
    script = (
        "import sys;"
        f"sys.stdout.write('a'*{MAX_OUTPUT_BYTES - 5} + {secret!r} + 'z'*{MAX_OUTPUT_BYTES});"
        f"sys.stderr.write('e'*{MAX_OUTPUT_BYTES * 3})"
    )

    result = await RunShellTool().execute(
        ToolCall(
            "bounded-shell",
            "run_shell",
            {"argv": [sys.executable, "-c", script]},
        ),
        context(tmp_path, secrets=(secret,)),
    )

    assert result.status == "success"
    assert secret not in result.content
    assert secret not in repr(result.metadata)
    assert len(result.metadata["stdout"].encode("utf-8")) <= MAX_OUTPUT_BYTES
    assert len(result.metadata["stderr"].encode("utf-8")) <= MAX_OUTPUT_BYTES
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True


@pytest.mark.asyncio
async def test_read_file_reads_only_limit_plus_one(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_bytes(b"x" * (MAX_FILE_BYTES * 3))

    with pytest.raises(ToolFailure) as raised:
        await ReadFileTool().execute(
            ToolCall("large-read", "read_file", {"path": "large.txt"}),
            context(tmp_path),
        )

    assert raised.value.metadata["size"] == MAX_FILE_BYTES + 1
    assert raised.value.metadata["max_size"] == MAX_FILE_BYTES


@pytest.mark.asyncio
async def test_search_truncates_long_redacted_line_preview(tmp_path: Path) -> None:
    secret = "line-secret"
    (tmp_path / "long.txt").write_text(
        "needle " + secret + " " + ("q" * (MAX_SEARCH_LINE_BYTES * 3)),
        encoding="utf-8",
    )

    result = await SearchTextTool().execute(
        ToolCall("long-line", "search_text", {"query": "needle"}),
        context(tmp_path, secrets=(secret,)),
    )

    match = result.metadata["matches"][0]
    assert secret not in result.content
    assert secret not in repr(result.metadata)
    assert len(match["text"].encode("utf-8")) <= MAX_SEARCH_LINE_BYTES
    assert match["line_truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [r"(a+)+$", r"(a|aa)+$"])
async def test_adversarial_regex_is_rejected_promptly(
    tmp_path: Path, pattern: str
) -> None:
    (tmp_path / "input.txt").write_text("a" * 100_000 + "!", encoding="utf-8")

    with pytest.raises(ToolFailure, match="Unsafe search pattern"):
        await asyncio.wait_for(
            SearchTextTool().execute(
                ToolCall(
                    "unsafe-regex",
                    "search_text",
                    {"query": pattern, "regex": True},
                ),
                context(tmp_path),
            ),
            timeout=0.5,
        )


@pytest.mark.asyncio
async def test_regex_length_limit_and_simple_expression(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("foo middle bar", encoding="utf-8")
    with pytest.raises(ToolFailure, match="Invalid search pattern"):
        await SearchTextTool().execute(
            ToolCall(
                "long-regex",
                "search_text",
                {"query": "a" * (MAX_REGEX_PATTERN_CHARS + 1), "regex": True},
            ),
            context(tmp_path),
        )

    result = await SearchTextTool().execute(
        ToolCall(
            "simple-regex",
            "search_text",
            {"query": r"^foo.*bar$", "regex": True},
        ),
        context(tmp_path),
    )
    assert result.status == "success"
    assert result.metadata["count"] == 1
