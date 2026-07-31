from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from litecoder.hooks import HookManager
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


class Trace:
    def __init__(self, *, block_stage: str | None = None, fail_stage: str | None = None) -> None:
        self.facts: list[dict[str, object]] = []
        self.block_stage = block_stage
        self.fail_stage = fail_stage
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.pre_count = 0
        self.two_pre = asyncio.Event()

    async def record(self, fact: Mapping[str, object]) -> None:
        saved = dict(fact)
        self.facts.append(saved)
        if saved.get("stage") == "pre":
            self.pre_count += 1
            if self.pre_count == 2:
                self.two_pre.set()
        if self.fail_stage is not None and saved.get("stage") == self.fail_stage:
            raise RuntimeError("mandatory version fact failed")
        if self.block_stage is not None and saved.get("stage") == self.block_stage:
            self.entered.set()
            await self.release.wait()


def context(tmp_path, *, round_number: int = 1) -> ToolContext:
    return ToolContext(
        "agent", "workspace", tmp_path,
        metadata={"round_number": round_number, "permission_mode": "ask"},
    )


def executor_for(tool, *, trace=None, guard=None, permission=None, workspaces=None):
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(
        registry,
        HookManager(trace_hook=trace or Trace()),
        guard or DuplicateGuard(annotation=lambda **_: None),
        permission or PermissionService(prompt=lambda _: PromptChoice.ALLOW_ONCE),
        workspaces or WorkspaceStateRegistry(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutates", [False, True])
async def test_identical_concurrent_calls_lease_before_prompt_and_execute(tmp_path, mutates: bool) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    executions = 0
    prompts = 0

    class Tool:
        spec = ToolSpec("work", "work", {}, mutates, permission_risk="external")

        async def execute(self, call, context):
            nonlocal executions
            executions += 1
            entered.set()
            await release.wait()
            return ToolExecution.success("ok", changed_workspace=mutates)

    async def prompt(_):
        nonlocal prompts
        prompts += 1
        return PromptChoice.ALLOW_ONCE

    guard = DuplicateGuard(annotation=lambda **_: None)
    trace = Trace()
    executor = executor_for(Tool(), trace=trace, guard=guard, permission=PermissionService(prompt=prompt))
    first = asyncio.create_task(executor.execute(ToolCall("one", "work", {"same": True}), context(tmp_path)))
    await entered.wait()
    second = asyncio.create_task(executor.execute(ToolCall("two", "work", {"same": True}), context(tmp_path)))
    await trace.two_pre.wait()
    assert executions == prompts == 1
    release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert first_result.status == "success"
    assert second_result.status == "duplicate_blocked"
    assert executions == prompts == 1
    assert guard.lease_count == 0


@pytest.mark.asyncio
async def test_denied_lease_releases_and_next_identical_call_can_prompt(tmp_path) -> None:
    choices = iter((PromptChoice.DENY, PromptChoice.ALLOW_ONCE))
    prompts = 0

    class Tool:
        spec = ToolSpec("work", "work", {}, False, permission_risk="external")

        async def execute(self, call, context):
            return ToolExecution.success("ok")

    async def prompt(_):
        nonlocal prompts
        prompts += 1
        return next(choices)

    executor = executor_for(Tool(), permission=PermissionService(prompt=prompt))
    first = await executor.execute(ToolCall("one", "work", {}), context(tmp_path))
    second = await executor.execute(ToolCall("two", "work", {}), context(tmp_path))
    assert (first.status, second.status, prompts) == ("denied", "success", 2)


@pytest.mark.asyncio
async def test_version_fact_failure_leaves_success_cache_committed(tmp_path) -> None:
    executions = 0

    class Tool:
        spec = ToolSpec("write", "write", {}, True)

        async def execute(self, call, context):
            nonlocal executions
            executions += 1
            return ToolExecution.success("ok", changed_workspace=True)

    trace = Trace(fail_stage="version")
    guard = DuplicateGuard(annotation=lambda **_: None)
    executor = executor_for(Tool(), trace=trace, guard=guard)
    call = ToolCall("one", "write", {"path": "a"})
    with pytest.raises(RuntimeError, match="mandatory version fact failed"):
        await executor.execute(call, context(tmp_path))
    trace.fail_stage = None
    repeated = await executor.execute(ToolCall("two", "write", call.arguments), context(tmp_path))
    assert repeated.status == "duplicate_blocked"
    assert executions == 1


@pytest.mark.asyncio
async def test_tool_cannot_mutate_decision_identity_or_fingerprint(tmp_path) -> None:
    trace = Trace()

    class Tool:
        spec = ToolSpec("read", "read", {}, False)

        async def execute(self, call, context):
            call.id = "evil-id"
            call.name = "evil-name"
            call.arguments["path"] = "evil"
            return ToolExecution.success("ok")

    executor = executor_for(Tool(), trace=trace)
    first = await executor.execute(ToolCall("one", "read", {"path": "safe"}), context(tmp_path))
    repeated = await executor.execute(ToolCall("two", "read", {"path": "safe"}), context(tmp_path))
    assert first.tool_call_id == "one"
    assert repeated.status == "duplicate_blocked"
    runtime = [fact for fact in trace.facts if fact.get("event") == "tool.runtime"]
    assert {fact["tool_name"] for fact in runtime} == {"read"}
    assert all(fact["tool_call_id"] in {"one", "two"} for fact in runtime)


@pytest.mark.asyncio
async def test_executor_forces_exclusive_for_corrupted_mutating_spec(tmp_path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    second_entered = asyncio.Event()

    class Tool:
        spec = ToolSpec("write", "write", {}, True)

        async def execute(self, call, context):
            if call.arguments["index"] == 1:
                entered.set()
                await release.wait()
            else:
                second_entered.set()
            return ToolExecution.success("ok", changed_workspace=True)

    object.__setattr__(Tool.spec, "concurrency", "shared")
    executor = executor_for(Tool())
    first = asyncio.create_task(executor.execute(ToolCall("one", "write", {"index": 1}), context(tmp_path)))
    await entered.wait()
    second = asyncio.create_task(executor.execute(ToolCall("two", "write", {"index": 2}), context(tmp_path)))
    assert second_entered.is_set() is False
    release.set()
    await asyncio.gather(first, second)
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_executor_serializes_traversals_per_workspace(tmp_path) -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    class Tool:
        spec = ToolSpec("glob_files", "Glob", {}, False, concurrency="traversal")

        async def execute(self, call, context):
            del context
            if call.id == "first":
                first_entered.set()
                await release_first.wait()
            else:
                second_entered.set()
            return ToolExecution.success("ok")

    executor = executor_for(Tool())
    first = asyncio.create_task(
        executor.execute(
            ToolCall("first", "glob_files", {"pattern": "*.py"}),
            context(tmp_path),
        )
    )
    await first_entered.wait()
    second = asyncio.create_task(
        executor.execute(
            ToolCall("second", "glob_files", {"pattern": "*.txt"}),
            context(tmp_path),
        )
    )

    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first, second)
    assert second_entered.is_set()
