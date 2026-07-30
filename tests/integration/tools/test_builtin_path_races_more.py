from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path

import pytest

from litecoder.tools import ToolCall, ToolContext
from litecoder.tools.builtin import (
    EditFileTool,
    GlobFilesTool,
    SearchTextTool,
    WriteFileTool,
)


def _directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    result = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        pytest.skip("account cannot create a directory symlink or junction")


def _install_barrier(monkeypatch, entered, release, *, occurrence: int = 1) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    seen = 0
    if os.name == "nt":
        original = secure_path._win_open_existing

        def open_with_barrier(path: Path, *, directory: bool):
            nonlocal seen
            handle = original(path, directory=directory)
            if directory and path.name == "victim":
                seen += 1
                if seen == occurrence:
                    entered.set()
                    assert release.wait(timeout=5)
            return handle

        monkeypatch.setattr(secure_path, "_win_open_existing", open_with_barrier)
    else:
        original = secure_path.os.open

        def open_with_barrier(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal seen
            descriptor = original(path, flags, mode, dir_fd=dir_fd)
            if path == "victim" and dir_fd is not None:
                seen += 1
                if seen == occurrence:
                    entered.set()
                    assert release.wait(timeout=5)
            return descriptor

        monkeypatch.setattr(secure_path.os, "open", open_with_barrier)


def _swap(victim: Path, workspace: Path, outside: Path) -> bool:
    try:
        victim.rename(workspace / "parked")
    except PermissionError:
        assert os.name == "nt"
        return False
    _directory_link(victim, outside)
    return True


async def _execute(tool, call: ToolCall, workspace: Path):
    return await asyncio.to_thread(
        lambda: asyncio.run(
            tool.execute(call, ToolContext("agent", "workspace", workspace))
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "tool_call", "expected"),
    [
        (
            WriteFileTool(),
            ToolCall(
                "race-write",
                "write_file",
                {"path": "victim/target.txt", "content": "new"},
            ),
            "new",
        ),
        (
            EditFileTool(),
            ToolCall(
                "race-edit",
                "edit_file",
                {
                    "path": "victim/target.txt",
                    "old_text": "inside",
                    "new_text": "edited",
                },
            ),
            "edited",
        ),
    ],
)
async def test_write_and_edit_pin_parent_during_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool,
    tool_call: ToolCall,
    expected: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    (victim / "target.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = "outside-race-sentinel"
    (outside / "target.txt").write_text(sentinel, encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    # EditFileTool reads then writes, opening the victim parent twice; the
    # barrier must pin the second (write-phase) open so the swap happens
    # before the write commits, not between read and write.
    occurrence = 2 if isinstance(tool, EditFileTool) else 1
    _install_barrier(monkeypatch, entered, release, occurrence=occurrence)

    operation = asyncio.create_task(_execute(tool, tool_call, workspace))
    assert await asyncio.to_thread(entered.wait, 5)
    swapped = _swap(victim, workspace, outside)
    release.set()
    result = await operation

    assert result.status == "success"
    changed = (workspace / "parked" if swapped else victim) / "target.txt"
    assert changed.read_text(encoding="utf-8") == expected
    assert (outside / "target.txt").read_text(encoding="utf-8") == sentinel
    assert str(outside) not in repr(result.metadata)


@pytest.mark.asyncio
async def test_glob_pins_directory_during_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    (victim / "inside.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside-only.txt").write_text("outside", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    _install_barrier(monkeypatch, entered, release)
    operation = asyncio.create_task(
        _execute(
            GlobFilesTool(),
            ToolCall("race-glob", "glob_files", {"pattern": "**/*.txt"}),
            workspace,
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    _swap(victim, workspace, outside)
    release.set()
    result = await operation

    assert result.status == "success"
    assert "victim/inside.txt" in result.content
    assert "outside-only.txt" not in result.content


@pytest.mark.asyncio
async def test_search_pins_parent_during_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    (victim / "inside.txt").write_text("inside-needle", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = "outside-race-sentinel"
    (outside / "inside.txt").write_text(sentinel, encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    _install_barrier(monkeypatch, entered, release, occurrence=2)
    operation = asyncio.create_task(
        _execute(
            SearchTextTool(),
            ToolCall("race-search", "search_text", {"query": "needle"}),
            workspace,
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    _swap(victim, workspace, outside)
    release.set()
    result = await operation

    assert result.status == "success"
    assert "inside-needle" in result.content
    assert sentinel not in result.content
    assert sentinel not in repr(result.metadata)
