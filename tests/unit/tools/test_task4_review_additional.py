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
from tests.unit.tools.test_task4_review_executor import Trace


def context(tmp_path, round_number: int = 1) -> ToolContext:
    return ToolContext(
        "agent", "workspace", tmp_path, metadata={
            "round_number": round_number,
            "permission_mode": "bypass",
            "bypass_authorized": True,
        }
    )


@pytest.mark.asyncio
async def test_cancelled_success_bookkeeping_finishes_before_lock_release(tmp_path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingGuard(DuplicateGuard):
        async def record_prepared_success(self, *args, **kwargs):
            entered.set()
            await release.wait()
            await super().record_prepared_success(*args, **kwargs)

    class Tool:
        spec = ToolSpec("write", "write", {}, True)

        async def execute(self, call, context):
            return ToolExecution.success("ok", changed_workspace=True)

    guard = BlockingGuard(annotation=lambda **_: None)
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=Trace()),
        guard,
        PermissionService(),
        WorkspaceStateRegistry(),
    )
    task = asyncio.create_task(
        executor.execute(ToolCall("one", "write", {}), context(tmp_path))
    )
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    repeated = await executor.execute(
        ToolCall("two", "write", {}), context(tmp_path)
    )
    assert repeated.status == "duplicate_blocked"
    assert executor.workspaces.get("workspace").version == 1


@pytest.mark.asyncio
async def test_dedupe_none_still_enforces_monotonic_rounds() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    spec = ToolSpec("status", "status", {}, False, dedupe_policy="none")
    call = ToolCall("one", "status", {})
    assert await guard.check(
        "agent", "workspace", 0, round_number=4, call=call, spec=spec
    ) is None
    with pytest.raises(ValueError, match="round_number must be monotonic"):
        await guard.check(
            "agent", "workspace", 0,
            round_number=3, call=call, spec=spec,
        )
