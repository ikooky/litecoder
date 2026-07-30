from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from litecoder.hooks import HookManager
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
    WorkspaceStateRegistry,
)


class _Trace:
    async def record(self, _payload: object) -> None:
        return None


class _ReadSharedTool:
    spec = ToolSpec("read_shared", "Read shared file", {}, False)

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.execution_count = 0

    async def execute(self, call: ToolCall, _context: ToolContext) -> ToolExecution:
        self.execution_count += 1
        path = self.workspace / str(call.arguments["path"])
        content = path.read_text(encoding="utf-8")
        return ToolExecution.success(content, preview={"content": content})


class _WriteSharedTool:
    spec = ToolSpec("write_shared", "Write shared file", {}, True)

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, call: ToolCall, _context: ToolContext) -> ToolExecution:
        path = self.workspace / str(call.arguments["path"])
        path.write_text(str(call.arguments["content"]), encoding="utf-8")
        return ToolExecution.success("written", changed_workspace=True)


def _executor(workspace: Path) -> tuple[ToolExecutor, _ReadSharedTool]:
    registry = ToolRegistry()
    read = _ReadSharedTool(workspace)
    registry.register(read)
    registry.register(_WriteSharedTool(workspace))
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=_Trace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"),
        WorkspaceStateRegistry(),
    )
    return executor, read


def _context(tmp_path: Path, agent: str, *, round_number: int) -> ToolContext:
    return ToolContext(
        agent,
        "workspace-1",
        tmp_path,
        metadata={"round_number": round_number, "permission_mode": "ask"},
    )


@pytest.mark.asyncio
async def test_same_read_executes_once_per_agent(tmp_path: Path) -> None:
    (tmp_path / "shared.txt").write_text("old", encoding="utf-8")
    executor, read = _executor(tmp_path)
    first, second = await asyncio.gather(
        executor.execute(
            ToolCall("read-a", "read_shared", {"path": "shared.txt"}),
            _context(tmp_path, "agent-a", round_number=1),
        ),
        executor.execute(
            ToolCall("read-b", "read_shared", {"path": "shared.txt"}),
            _context(tmp_path, "agent-b", round_number=1),
        ),
    )

    assert first.status == second.status == "success"
    assert first.content == second.content == "old"
    assert read.execution_count == 2


@pytest.mark.asyncio
async def test_agent_write_invalidates_other_agent_old_version_cache(
    tmp_path: Path,
) -> None:
    (tmp_path / "shared.txt").write_text("old", encoding="utf-8")
    executor, read = _executor(tmp_path)

    first = await executor.execute(
        ToolCall("read-1", "read_shared", {"path": "shared.txt"}),
        _context(tmp_path, "agent-b", round_number=1),
    )
    written = await executor.execute(
        ToolCall(
            "write-1",
            "write_shared",
            {"path": "shared.txt", "content": "new"},
        ),
        _context(tmp_path, "agent-a", round_number=1),
    )
    second = await executor.execute(
        ToolCall("read-2", "read_shared", {"path": "shared.txt"}),
        _context(tmp_path, "agent-b", round_number=2),
    )

    assert first.content == "old"
    assert written.status == "success"
    assert second.status == "success"
    assert second.content == "new"
    assert read.execution_count == 2