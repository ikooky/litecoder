from __future__ import annotations

import asyncio

import pytest

from litecoder.hooks import HookManager, HookOutcome, HookPoint
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    PromptChoice,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
    WorkspaceStateRegistry,
)
from tests.unit.tools.test_task4_review_executor import Trace


def context(tmp_path) -> ToolContext:
    return ToolContext(
        "agent", "workspace", tmp_path,
        metadata={"round_number": 1, "permission_mode": "ask"},
    )


class Tool:
    def __init__(self, *, mutates: bool, entered=None, release=None) -> None:
        self.spec = ToolSpec(
            "work", "work", {}, mutates,
            permission_risk="external" if not mutates else "workspace",
        )
        self.entered = entered
        self.release = release
        self.calls = 0

    async def execute(self, call, context):
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return ToolExecution.success("ok", changed_workspace=self.spec.mutates_workspace)


def build(tool, *, trace=None, guard=None, permission=None, workspaces=None):
    registry = ToolRegistry()
    registry.register(tool)
    hooks = HookManager(trace_hook=trace or Trace())
    executor = ToolExecutor(
        registry,
        hooks,
        guard or DuplicateGuard(annotation=lambda **_: None),
        permission or PermissionService(prompt=lambda _: PromptChoice.ALLOW_ONCE),
        workspaces or WorkspaceStateRegistry(),
    )
    return executor, hooks


async def assert_cancelled(task: asyncio.Task[object]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancellation_during_pre_stops_before_execute(tmp_path) -> None:
    entered = asyncio.Event()
    tool = Tool(mutates=True)
    executor, hooks = build(tool)

    async def blocking(envelope):
        entered.set()
        await asyncio.Event().wait()
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.PRE_TOOL_USE, blocking, name="blocking-pre")
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await entered.wait()
    await assert_cancelled(task)
    assert tool.calls == 0
    assert executor.workspaces.get("workspace").version == 0


@pytest.mark.asyncio
async def test_cancellation_during_duplicate_annotation_releases_lease(tmp_path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def annotation(**_):
        entered.set()
        await release.wait()

    guard = DuplicateGuard(annotation=annotation)
    original = ToolCall("old", "work", {})
    await guard.record_success("agent", "workspace", 0, round_number=1, call=original)
    tool = Tool(mutates=False)
    executor, _ = build(tool, guard=guard)
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await entered.wait()
    await assert_cancelled(task)
    assert tool.calls == 0
    assert guard.lease_count == 0


@pytest.mark.asyncio
async def test_cancellation_during_permission_stops_before_execute(tmp_path) -> None:
    entered = asyncio.Event()

    async def prompt(_):
        entered.set()
        await asyncio.Event().wait()

    tool = Tool(mutates=False)
    executor, _ = build(tool, permission=PermissionService(prompt=prompt))
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await entered.wait()
    await assert_cancelled(task)
    assert tool.calls == 0
    assert executor.workspaces.get("workspace").version == 0


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_workspace_lock_has_no_mutation(tmp_path) -> None:
    workspaces = WorkspaceStateRegistry()
    state = workspaces.get("workspace")
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold() -> None:
        async with state.lock.write():
            holder_entered.set()
            await release_holder.wait()

    holder = asyncio.create_task(hold())
    await holder_entered.wait()
    tool = Tool(mutates=True)
    executor, _ = build(tool, workspaces=workspaces)
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await executor.hooks.record_runtime_fact("test.sync", {"ready": True})
    await assert_cancelled(task)
    release_holder.set()
    await holder
    assert tool.calls == 0
    assert state.version == 0


@pytest.mark.asyncio
async def test_cancellation_during_mutating_execute_versions_once_without_cache(tmp_path) -> None:
    entered = asyncio.Event()
    tool = Tool(mutates=True, entered=entered, release=asyncio.Event())
    guard = DuplicateGuard(annotation=lambda **_: None)
    executor, _ = build(tool, guard=guard)
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await entered.wait()
    await assert_cancelled(task)
    assert executor.workspaces.get("workspace").version == 1
    assert guard.record_count == 0


@pytest.mark.asyncio
async def test_cancellation_at_version_fact_keeps_committed_success(tmp_path) -> None:
    trace = Trace(block_stage="version")
    guard = DuplicateGuard(annotation=lambda **_: None)
    tool = Tool(mutates=True)
    executor, _ = build(tool, trace=trace, guard=guard)
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await trace.entered.wait()
    await assert_cancelled(task)
    trace.release.set()
    repeated = await executor.execute(ToolCall("two", "work", {}), context(tmp_path))
    assert repeated.status == "duplicate_blocked"
    assert tool.calls == 1
    assert executor.workspaces.get("workspace").version == 1


@pytest.mark.asyncio
async def test_cancellation_during_post_keeps_committed_success(tmp_path) -> None:
    entered = asyncio.Event()
    guard = DuplicateGuard(annotation=lambda **_: None)
    tool = Tool(mutates=True)
    executor, hooks = build(tool, guard=guard)

    async def blocking(envelope):
        entered.set()
        await asyncio.Event().wait()
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.POST_TOOL_USE, blocking, name="blocking-post")
    task = asyncio.create_task(executor.execute(ToolCall("one", "work", {}), context(tmp_path)))
    await entered.wait()
    await assert_cancelled(task)
    hooks.clear(HookPoint.POST_TOOL_USE)
    repeated = await executor.execute(ToolCall("two", "work", {}), context(tmp_path))
    assert repeated.status == "duplicate_blocked"
    assert tool.calls == 1
    assert executor.workspaces.get("workspace").version == 1
