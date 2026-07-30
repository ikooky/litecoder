from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

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
from tests.unit.tools.test_task4_review_executor import Trace


def context(tmp_path, round_number: int) -> ToolContext:
    return ToolContext(
        "agent", "workspace", tmp_path, metadata={"round_number": round_number}
    )


@pytest.mark.asyncio
async def test_dedupe_none_stale_generation_cannot_advance_fresh_round_cursor(
    tmp_path,
) -> None:
    prepared = asyncio.Event()
    release = asyncio.Event()

    class PausingGuard(DuplicateGuard):
        @asynccontextmanager
        async def execution_lease(self, *args, **kwargs):
            prepared.set()
            await release.wait()
            async with super().execution_lease(*args, **kwargs) as value:
                yield value

    class Tool:
        spec = ToolSpec("status", "status", {}, False, dedupe_policy="none")

        async def execute(self, call, context):
            return ToolExecution.success("ok")

    guard = PausingGuard(annotation=lambda **_: None)
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=Trace()),
        guard,
        PermissionService(),
        WorkspaceStateRegistry(),
    )

    old = asyncio.create_task(
        executor.execute(ToolCall("old", "status", {}), context(tmp_path, 3))
    )
    await prepared.wait()
    await guard.start_user_message("agent")
    release.set()
    assert (await old).status == "success"

    fresh = await executor.execute(
        ToolCall("fresh", "status", {}), context(tmp_path, 0)
    )
    assert fresh.status == "success"
    current = await executor.execute(
        ToolCall("current", "status", {}), context(tmp_path, 1)
    )
    assert current.status == "success"
    with pytest.raises(ValueError, match="round_number must be monotonic"):
        await executor.execute(
            ToolCall("decreasing", "status", {}), context(tmp_path, 0)
        )
    assert guard.record_count == 0
    assert guard.lease_count == 0
