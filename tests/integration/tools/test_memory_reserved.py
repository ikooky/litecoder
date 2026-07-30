from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.tools.builtin import (
    EditFileTool,
    GlobFilesTool,
    ReadFileTool,
    RunShellTool,
    SearchTextTool,
    WriteFileTool,
)
from litecoder.tools.models import ToolCall, ToolContext


def _context(root: Path) -> ToolContext:
    return ToolContext("root", "workspace", root)


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        (
            ReadFileTool(),
            ToolCall("read", "read_file", {"path": ".memory/MEMORY.md"}),
        ),
        (
            WriteFileTool(),
            ToolCall(
                "write",
                "write_file",
                {"path": ".memory/item.md", "content": "blocked"},
            ),
        ),
        (
            EditFileTool(),
            ToolCall(
                "edit",
                "edit_file",
                {
                    "path": ".memory/item.md",
                    "old_text": "a",
                    "new_text": "b",
                },
            ),
        ),
        (
            GlobFilesTool(),
            ToolCall("glob", "glob_files", {"pattern": ".memory/**/*"}),
        ),
        (
            SearchTextTool(),
            ToolCall(
                "search",
                "search_text",
                {"query": "secret", "glob": ".memory/**/*"},
            ),
        ),
        (
            RunShellTool(),
            ToolCall(
                "shell",
                "run_shell",
                {"argv": ["cmd.exe", "/c", "mkdir", ".memory"]},
            ),
        ),
    ],
)
def test_general_tools_reject_explicit_memory_store_access(
    tmp_path: Path, tool: object, call: ToolCall
) -> None:
    assert getattr(tool, "hard_guard")(call, _context(tmp_path))


@pytest.mark.asyncio
async def test_broad_glob_hides_memory_store(tmp_path: Path) -> None:
    (tmp_path / ".memory").mkdir()
    (tmp_path / ".memory" / "secret.md").write_text("hidden", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("shown", encoding="utf-8")

    result = await GlobFilesTool().execute(
        ToolCall("glob", "glob_files", {"pattern": "**/*"}),
        _context(tmp_path),
    )

    assert result.preview == ["visible.txt"]
    assert ".memory" not in result.content
