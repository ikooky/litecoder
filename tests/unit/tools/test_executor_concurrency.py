from __future__ import annotations

import asyncio

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
from tests.unit.tools.test_executor_pipeline import RecordingTrace


@pytest.mark.asyncio
async def test_executor_allows_shared_readers_and_excludes_writer(tmp_path) -> None:
    readers_entered = 0
    both_readers = asyncio.Event()
    release_readers = asyncio.Event()
    writer_entered = asyncio.Event()

    class ReadTool:
        spec = ToolSpec("read", "read", {}, False)

        async def execute(self, call, context):
            nonlocal readers_entered
            readers_entered += 1
            if readers_entered == 2:
                both_readers.set()
            await release_readers.wait()
            return ToolExecution.success("read")

    class WriteTool:
        spec = ToolSpec("write", "write", {}, True)

        async def execute(self, call, context):
            writer_entered.set()
            return ToolExecution.success("write", changed_workspace=True)

    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"),
        WorkspaceStateRegistry(),
    )
    context = ToolContext("agent", "workspace", tmp_path, metadata={"round_number": 1})
    reads = [
        asyncio.create_task(executor.execute(ToolCall(f"r-{index}", "read", {"index": index}), context))
        for index in range(2)
    ]
    await both_readers.wait()
    writer = asyncio.create_task(executor.execute(ToolCall("w", "write", {}), context))
    assert writer_entered.is_set() is False
    release_readers.set()
    await asyncio.gather(*reads, writer)
    assert writer_entered.is_set()

@pytest.mark.asyncio
async def test_orchestrator_can_wait_for_child_writer_without_self_deadlock(
    tmp_path,
) -> None:
    registry = ToolRegistry()

    class WriteTool:
        spec = ToolSpec("write", "write", {}, True)

        async def execute(self, call, context):
            return ToolExecution.success("written", changed_workspace=True)

    class OrchestratorTool:
        spec = ToolSpec(
            "orchestrate", "wait for child work", {}, False, workspace_lock=False
        )

        async def execute(self, call, context):
            result = await executor.execute(
                ToolCall("child-write", "write", {}), context
            )
            assert result.status == "success"
            return ToolExecution.success("child completed")

    registry.register(WriteTool())
    registry.register(OrchestratorTool())
    executor = ToolExecutor(
        registry, HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"), WorkspaceStateRegistry(),
    )
