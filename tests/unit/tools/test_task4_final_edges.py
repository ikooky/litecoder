from __future__ import annotations

import asyncio

import pytest

from litecoder.hooks import HookManager, HookOutcome, HookPoint
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolExecutor,
    ToolPartialFailure,
    ToolRegistry,
    ToolSpec,
    WorkspaceStateRegistry,
)
from tests.unit.tools.test_task4_review_executor import Trace


def context(tmp_path, round_number: int = 0) -> ToolContext:
    return ToolContext(
        "agent", "workspace", tmp_path, metadata={"round_number": round_number}
    )


def build(tool, *, trace=None, guard=None):
    registry = ToolRegistry()
    registry.register(tool)
    hooks = HookManager(trace_hook=trace or Trace())
    executor = ToolExecutor(
        registry,
        hooks,
        guard or DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"),
        WorkspaceStateRegistry(),
    )
    return executor, hooks


@pytest.mark.asyncio
async def test_read_partial_mutation_commits_version_before_partial_fact(tmp_path) -> None:
    class Tool:
        spec = ToolSpec("read", "read", {}, False)

        async def execute(self, call, context):
            raise ToolPartialFailure("partly changed", changed_workspace=True)

    trace = Trace(fail_stage="partial")
    executor, _ = build(Tool(), trace=trace)
    with pytest.raises(RuntimeError, match="mandatory version fact failed"):
        await executor.execute(ToolCall("one", "read", {}), context(tmp_path))
    assert executor.workspaces.get("workspace").version == 1
    assert executor.duplicates.record_count == 0


@pytest.mark.asyncio
async def test_reset_discards_inflight_success_from_previous_generation(tmp_path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Tool:
        spec = ToolSpec("read", "read", {}, False)

        async def execute(self, call, context):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return ToolExecution.success("ok")

    guard = DuplicateGuard(annotation=lambda **_: None)
    executor, _ = build(Tool(), guard=guard)
    old = asyncio.create_task(
        executor.execute(ToolCall("old", "read", {}), context(tmp_path, 3))
    )
    await entered.wait()
    await guard.start_user_message("agent")
    release.set()
    assert (await old).status == "success"
    fresh = await executor.execute(
        ToolCall("fresh", "read", {}), context(tmp_path, 0)
    )
    assert fresh.status == "success"
    repeated = await executor.execute(
        ToolCall("repeat", "read", {}), context(tmp_path, 0)
    )
    assert repeated.status == "duplicate_blocked"
    assert calls == 2


def test_tool_execution_is_success_only() -> None:
    with pytest.raises(ValueError, match="ToolExecution status must be success"):
        ToolExecution("failure", "bad")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutates", "changed", "expected_status"),
    [(False, True, "contract_violation"), (True, False, "tool_error")],
)
async def test_mutated_non_success_execution_routes_tool_error_without_cache_or_post(
    tmp_path, mutates: bool, changed: bool, expected_status: str
) -> None:
    class Tool:
        spec = ToolSpec("work", "work", {}, mutates)

        async def execute(self, call, context):
            execution = ToolExecution.success("not-success", changed_workspace=changed)
            execution.status = "failure"
            return execution

    executor, hooks = build(Tool())
    post_called = False
    error_called = False

    async def post(envelope):
        nonlocal post_called
        post_called = True
        return HookOutcome(envelope.payload)

    async def error(envelope):
        nonlocal error_called
        error_called = True
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.POST_TOOL_USE, post, name="post")
    hooks.register(HookPoint.TOOL_ERROR, error, name="error")
    result = await executor.execute(
        ToolCall("one", "work", {}), context(tmp_path)
    )
    assert result.status == expected_status
    assert result.metadata["automatic_retry"] is False
    assert executor.workspaces.get("workspace").version == 1
    assert executor.duplicates.record_count == 0
    assert error_called is True
    assert post_called is False
