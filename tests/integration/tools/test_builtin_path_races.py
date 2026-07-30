from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from litecoder.hooks import HookManager
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    WorkspaceStateRegistry,
)
from litecoder.tools.builtin import ReadFileTool
from litecoder.tools import ToolDenied


class PermissionBarrierTrace:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def record(self, fact: Mapping[str, object]) -> None:
        if fact.get("event") == "tool.runtime" and fact.get("stage") == "permission":
            self.entered.set()
            await self.release.wait()


def _executor(trace: PermissionBarrierTrace) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    return ToolExecutor(
        registry,
        HookManager(trace_hook=trace),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(),
        WorkspaceStateRegistry(),
    )


def _directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    completed = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        pytest.skip("account cannot create a directory symlink or junction")


@pytest.mark.asyncio
async def test_read_denies_ancestor_swapped_after_hard_guard(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    (victim / "target.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = "outside-race-sentinel"
    (outside / "target.txt").write_text(sentinel, encoding="utf-8")

    trace = PermissionBarrierTrace()
    task = asyncio.create_task(
        _executor(trace).execute(
            ToolCall("race-read", "read_file", {"path": "victim/target.txt"}),
            ToolContext("agent", "workspace", workspace),
        )
    )
    await trace.entered.wait()
    victim.rename(workspace / "parked")
    _directory_link(victim, outside)
    trace.release.set()
    result = await task

    assert result.status == "denied"
    assert sentinel not in result.content
    assert sentinel not in repr(result.metadata)
    assert str(outside) not in result.content
    assert str(outside) not in repr(result.metadata)


@pytest.mark.asyncio
async def test_read_uses_pinned_ancestor_during_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    (victim / "target.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = "outside-race-sentinel"
    (outside / "target.txt").write_text(sentinel, encoding="utf-8")

    entered = threading.Event()
    release = threading.Event()
    swapped = False

    if os.name == "nt":
        original_open = secure_path._win_open_existing

        def barrier_open(path: Path, *, directory: bool):
            handle = original_open(path, directory=directory)
            if directory and path.name == "victim":
                entered.set()
                assert release.wait(timeout=5)
            return handle

        monkeypatch.setattr(secure_path, "_win_open_existing", barrier_open)
    else:
        original_open = secure_path.os.open

        def barrier_open(path, flags, mode=0o777, *, dir_fd=None):
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if path == "victim" and dir_fd is not None:
                entered.set()
                assert release.wait(timeout=5)
            return descriptor

        monkeypatch.setattr(secure_path.os, "open", barrier_open)

    operation = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(
                ReadFileTool().execute(
                    ToolCall(
                        "race-read-open",
                        "read_file",
                        {"path": "victim/target.txt"},
                    ),
                    ToolContext("agent", "workspace", workspace),
                )
            )
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        victim.rename(workspace / "parked")
    except PermissionError:
        assert os.name == "nt"
    else:
        swapped = True
        _directory_link(victim, outside)
    release.set()
    result = await operation

    assert result.status == "success"
    assert result.content == "inside"
    assert sentinel not in result.content
    assert (outside / "target.txt").read_text(encoding="utf-8") == sentinel
    if os.name != "nt":
        assert swapped is True