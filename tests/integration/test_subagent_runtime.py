from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from litecoder.agent.result import AgentResult
from litecoder.agent.factory import DefaultAgentRuntimeFactory, _ChildExecutor
from litecoder.agent.loop import RuntimeBudgets
from litecoder.context.manager import ContextManager
from litecoder.providers.models import Usage
from litecoder.tasks.subagents import (
    AgentCaller,
    AgentCreationDenied,
    ChildAgentRequest,
    ChildAuthority,
    SubagentManager,
    profile_tools,
    profile_instructions,
)
from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskCreate
from litecoder.tasks.store import TaskStore
from litecoder.tools.builtin.agents import SpawnSubagentTool
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


class RuntimeDouble:
    def __init__(
        self,
        session_id: str,
        *,
        status: str = "completed",
        reason: str = "done",
    ) -> None:
        self.session_id = session_id
        self.status = status
        self.reason = reason
        self.duplicates = object()
        self.objectives: list[str] = []

    async def run(self, objective: str) -> AgentResult:
        self.objectives.append(objective)
        return AgentResult(
            self.session_id,
            self.status,
            self.reason,
            Usage(3, 5),
        )


class FactoryDouble:
    def __init__(self, runtime: RuntimeDouble) -> None:
        self.runtime = runtime
        self.requests: list[ChildAgentRequest] = []

    async def create_child(self, request: ChildAgentRequest) -> RuntimeDouble:
        self.requests.append(request)
        return self.runtime


@pytest.mark.asyncio
async def test_child_workspace_mutation_requires_active_task_assignment(
    tmp_path: Path,
) -> None:
    class WriteToolDouble:
        spec = ToolSpec("write_file", "write", {"type": "object"}, True)

    class ExecutorDouble:
        async def execute(self, call, context):
            return ToolResult(call.id, "success", "written")

    registry = ToolRegistry()
    registry.register(WriteToolDouble())
    tasks = TaskManager(TaskStore(tmp_path / "tasks"))
    await tasks.create(TaskCreate("task-1", "Implement", "Change solution"))
    executor = _ChildExecutor(
        ExecutorDouble(),
        registry=registry,
        allowed_tools=frozenset({"write_file"}),
        max_calls=2,
        task_manager=tasks,
        task_id="task-1",
        agent_id="worker-1",
    )
    call = ToolCall("write-1", "write_file", {})
    tool_context = ToolContext("worker-session", "workspace-1", tmp_path)

    denied = await executor.execute(call, tool_context)
    pending = await tasks.get("task-1")

    assert denied.status == "denied"
    assert denied.metadata == {
        "stage": "workspace_mutation",
        "code": "task_not_assigned",
        "task_id": "task-1",
        "agent_id": "worker-1",
    }
    assert pending.owner_agent_id is None

    await tasks.assign_and_start("task-1", "worker-1")
    allowed = await executor.execute(
        ToolCall("write-2", "write_file", {}), tool_context
    )

    assert allowed.status == "success"

    unbound = _ChildExecutor(
        ExecutorDouble(),
        registry=registry,
        allowed_tools=frozenset({"write_file"}),
        max_calls=1,
        task_manager=tasks,
        task_id=None,
        agent_id="worker-2",
    )
    missing_task = await unbound.execute(
        ToolCall("write-3", "write_file", {}), tool_context
    )

    assert missing_task.status == "denied"
    assert missing_task.metadata["code"] == "missing_delegated_task"


def full_authority() -> ChildAuthority:
    return ChildAuthority(
        tools=frozenset({"read_file", "search_text"}),
        workspace_id="workspace-1",
        permission_mode="ask",
        task_ids=frozenset({"task-1"}),
        max_rounds=10,
        max_tool_calls=20,
    )


@pytest.mark.asyncio
async def test_child_receives_independent_session_and_cache() -> None:
    child_runtime = RuntimeDouble("child-session")
    lead_runtime = RuntimeDouble("lead-session")
    manager = SubagentManager(FactoryDouble(child_runtime))
    lead_caller = AgentCaller(
        "lead", "lead-session", full_authority(), runtime=lead_runtime
    )
    child_request = ChildAgentRequest(
        "summarize",
        ChildAuthority(
            tools=frozenset({"read_file"}),
            workspace_id="workspace-1",
            permission_mode="ask",
            task_ids=frozenset({"task-1"}),
            max_rounds=3,
            max_tool_calls=5,
        ),
        "call-1",
    )

    handle = await manager.spawn(child_request, caller=lead_caller)

    assert handle.session_id != lead_caller.session_id
    assert handle.runtime.duplicates is not lead_caller.runtime.duplicates


@pytest.mark.asyncio
async def test_spawn_subagent_tool_builds_restricted_request(tmp_path) -> None:
    child_runtime = RuntimeDouble("child-session")
    factory = FactoryDouble(child_runtime)
    manager = SubagentManager(factory)
    lead = AgentCaller("lead", "lead-session", full_authority())
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: lead)
    context = ToolContext("lead-session", "workspace-1", tmp_path)

    result = await tool.execute(
        ToolCall(
            "call-7",
            "spawn_subagent",
            {
                "objective": "read only",
                "tools": ["read_file"],
                "budget": {"max_rounds": 2, "max_tool_calls": 4},
            },
        ),
        context,
    )

    assert result.status == "success"
    assert result.metadata["session_id"] == "child-session"
    assert factory.requests[0].authority.tools == frozenset({"read_file"})
    assert factory.requests[0].authority.max_rounds == 2
    assert factory.requests[0].authority.max_tool_calls == 4
    assert tool.spec.workspace_lock is False


@pytest.mark.asyncio
async def test_spawn_subagent_tool_inherits_parent_budget_when_omitted(
    tmp_path,
) -> None:
    child_runtime = RuntimeDouble("child-session")
    factory = FactoryDouble(child_runtime)
    manager = SubagentManager(factory)
    lead = AgentCaller("lead", "lead-session", full_authority())
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: lead)

    result = await tool.execute(
        ToolCall(
            "call-8",
            "spawn_subagent",
            {"objective": "read only", "tools": ["read_file"]},
        ),
        ToolContext("lead-session", "workspace-1", tmp_path),
    )

    assert result.status == "success"
    assert factory.requests[0].authority.max_rounds == 10
    assert factory.requests[0].authority.max_tool_calls == 20


@pytest.mark.asyncio
async def test_subagent_result_contains_the_child_final_text() -> None:
    class StoreDouble:
        async def load_context(self, session_id: str) -> object:
            assert session_id == "child-session"
            return SimpleNamespace(
                messages=(
                    SimpleNamespace(
                        role="assistant",
                        content=(
                            {"type": "text", "text": "CHILD FINAL REPORT"},
                        ),
                    ),
                )
            )

    child_runtime = RuntimeDouble("child-session")
    child_runtime.store = StoreDouble()  # type: ignore[attr-defined]
    manager = SubagentManager(FactoryDouble(child_runtime))
    handle = await manager.spawn(
        ChildAgentRequest(
            "read only",
            full_authority(),
            "call-output",
        ),
        caller=AgentCaller("lead", "lead-session", full_authority()),
    )

    assert handle.result.content == "CHILD FINAL REPORT"
    assert handle.result.metadata["session_id"] == "child-session"


@pytest.mark.asyncio
async def test_spawn_subagent_tool_reports_child_failure() -> None:
    child_runtime = RuntimeDouble(
        "child-session",
        status="failed",
        reason="provider unavailable",
    )
    manager = SubagentManager(FactoryDouble(child_runtime))
    lead = AgentCaller("lead", "lead-session", full_authority())
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: lead)
    context = ToolContext("lead-session", "workspace-1", Path.cwd())

    with pytest.raises(ToolFailure, match="finished with failed"):
        await tool.execute(
            ToolCall(
                "call-failed",
                "spawn_subagent",
                {
                    "objective": "read only",
                    "tools": ["read_file"],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                },
            ),
            context,
        )


@pytest.mark.asyncio
async def test_spawn_subagent_tool_translates_creation_denial() -> None:
    manager = SubagentManager(FactoryDouble(RuntimeDouble("child-session")))
    child = AgentCaller("child", "child-session", full_authority())
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: child)
    context = ToolContext("child-session", "workspace-1", Path.cwd())

    with pytest.raises(ToolDenied, match="only user or lead"):
        await tool.execute(
            ToolCall(
                "call-denied",
                "spawn_subagent",
                {
                    "objective": "read only",
                    "tools": ["read_file"],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                },
            ),
            context,
        )


@pytest.mark.asyncio
async def test_spawn_subagent_tool_reports_unexpected_runtime_failure(
    tmp_path: Path,
) -> None:
    class FailingManager:
        factory = object()

        async def spawn(self, request, *, caller):
            raise RuntimeError("child workspace unavailable")

    lead = AgentCaller("lead", "lead-session", full_authority())
    tool = SpawnSubagentTool(  # type: ignore[arg-type]
        FailingManager(), caller_resolver=lambda context: lead
    )

    with pytest.raises(ToolFailure) as captured:
        await tool.execute(
            ToolCall(
                "call-runtime-failure",
                "spawn_subagent",
                {
                    "objective": "Inspect files",
                    "tools": ["read_file"],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                },
            ),
            ToolContext("lead-session", "workspace-1", tmp_path),
        )

    assert str(captured.value) == (
        "Subagent execution failed (RuntimeError): child workspace unavailable"
    )
    assert captured.value.metadata == {
        "stage": "spawn_subagent",
        "code": "runtime_error",
        "failure_type": "RuntimeError",
        "reason": "child workspace unavailable",
    }


@pytest.mark.asyncio
async def test_mutating_subagent_requires_task_and_worktree(tmp_path: Path) -> None:
    class WriteToolDouble:
        spec = ToolSpec("write_file", "write", {"type": "object"}, True)

    registry = ToolRegistry()
    registry.register(WriteToolDouble())
    manager = SubagentManager(FactoryDouble(RuntimeDouble("child-session")))
    lead = AgentCaller(
        "lead",
        "lead-session",
        ChildAuthority(
            tools=frozenset(
                {"write_file", "task_complete", "task_fail"}
            ),
            workspace_id="workspace-1",
            permission_mode="ask",
            task_ids=frozenset({"task-1"}),
            max_rounds=10,
            max_tool_calls=20,
        ),
    )
    tool = SpawnSubagentTool(
        manager,
        caller_resolver=lambda context: lead,
        tool_registry=registry,
    )

    with pytest.raises(ToolFailure) as captured:
        await tool.execute(
            ToolCall(
                "call-write",
                "spawn_subagent",
                {
                    "objective": "Implement the task",
                    "tools": [
                        "write_file",
                        "task_complete",
                        "task_fail",
                    ],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                    "task_id": "task-1",
                },
            ),
            ToolContext("lead-session", "workspace-1", tmp_path),
        )

    assert captured.value.metadata == {
        "stage": "spawn_subagent",
        "code": "missing_task_worktree",
    }


@pytest.mark.asyncio
async def test_task_subagent_requires_lifecycle_tools(tmp_path: Path) -> None:
    manager = SubagentManager(FactoryDouble(RuntimeDouble("child-session")))
    lead = AgentCaller("lead", "lead-session", full_authority())
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: lead)

    with pytest.raises(ToolFailure) as captured:
        await tool.execute(
            ToolCall(
                "call-task",
                "spawn_subagent",
                {
                    "objective": "Inspect the task",
                    "tools": ["read_file"],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                    "task_id": "task-1",
                },
            ),
            ToolContext("lead-session", "workspace-1", tmp_path),
        )

    assert captured.value.metadata == {
        "stage": "spawn_subagent",
        "code": "missing_task_tools",
        "missing_tools": ["task_complete", "task_fail"],
    }


@pytest.mark.asyncio
async def test_spawn_subagent_delegates_verified_worktree_workspace(
    tmp_path: Path,
) -> None:
    binding = SimpleNamespace(
        id="binding-1",
        task_id="task-1",
        workspace_id="workspace-2",
    )
    worktrees = SimpleNamespace(list=lambda: None)

    async def list_worktrees():
        return (binding,)

    worktrees.list = list_worktrees
    child_runtime = RuntimeDouble("child-session")
    factory = FactoryDouble(child_runtime)
    tasks = TaskManager(TaskStore(tmp_path / "tasks"))
    await tasks.create(
        TaskCreate(
            "task-1",
            "implement",
            "Implement the task",
            worktree_id="binding-1",
        )
    )
    manager = SubagentManager(factory, task_manager=tasks)
    delegated_tools = frozenset(
        {"read_file", "task_complete", "task_fail"}
    )
    lead = AgentCaller(
        "lead",
        "lead-session",
        ChildAuthority(
            tools=delegated_tools,
            workspace_id="workspace-1",
            permission_mode="ask",
            task_ids=frozenset({"task-1"}),
            max_rounds=10,
            max_tool_calls=20,
            task_workspaces=frozenset({"workspace-2"}),
        ),
    )
    tool = SpawnSubagentTool(
        manager,
        caller_resolver=lambda context: lead,
        worktrees=worktrees,
        task_manager=tasks,
    )

    result = await tool.execute(
        ToolCall(
            "call-worktree",
            "spawn_subagent",
            {
                "objective": "Implement the task",
                "tools": sorted(delegated_tools),
                "budget": {"max_rounds": 2, "max_tool_calls": 4},
                "task_id": "task-1",
                "worktree_id": "binding-1",
            },
        ),
        ToolContext("lead-session", "workspace-1", tmp_path),
    )

    assert result.status == "success"
    assert factory.requests[0].authority.workspace_id == "workspace-2"
    assert factory.requests[0].worktree_id == "binding-1"


def _profile_authority(profile: str) -> ChildAuthority:
    tools = profile_tools(profile)  # type: ignore[arg-type]
    assert tools is not None
    return ChildAuthority(
        tools=tools,
        workspace_id="workspace-1",
        permission_mode="ask",
        task_ids=frozenset(),
        max_rounds=10,
        max_tool_calls=20,
    )


@pytest.mark.asyncio
async def test_spawn_subagent_explore_profile_uses_fixed_read_only_tools(tmp_path) -> None:
    child_runtime = RuntimeDouble("child-session")
    factory = FactoryDouble(child_runtime)
    manager = SubagentManager(factory)
    lead = AgentCaller("lead", "lead-session", _profile_authority("explore"))
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: lead)

    result = await tool.execute(
        ToolCall(
            "call-profile",
            "spawn_subagent",
            {
                "objective": "Locate the runtime entry point and report evidence.",
                "profile": "explore",
                "budget": {"max_rounds": 2, "max_tool_calls": 4},
            },
        ),
        ToolContext("lead-session", "workspace-1", tmp_path),
    )

    assert result.status == "success"
    request = factory.requests[0]
    assert request.profile == "explore"
    assert request.authority.tools == profile_tools("explore")
    assert "run_shell" not in request.authority.tools
    assert "write_file" not in request.authority.tools


@pytest.mark.asyncio
async def test_spawn_subagent_profile_rejects_caller_selected_tools(tmp_path) -> None:
    manager = SubagentManager(FactoryDouble(RuntimeDouble("child-session")))
    lead = AgentCaller("lead", "lead-session", _profile_authority("explore"))
    tool = SpawnSubagentTool(manager, caller_resolver=lambda context: lead)

    with pytest.raises(ToolFailure, match="Profile tools are fixed"):
        await tool.execute(
            ToolCall(
                "call-profile-tools",
                "spawn_subagent",
                {
                    "objective": "Inspect files",
                    "profile": "explore",
                    "tools": ["write_file"],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                },
            ),
            ToolContext("lead-session", "workspace-1", tmp_path),
        )


@pytest.mark.asyncio
async def test_subagent_manager_rejects_forged_profile_authority() -> None:
    parent = _profile_authority("explore")
    forged = ChildAgentRequest(
        "Inspect files",
        ChildAuthority(
            tools=frozenset({"read_file"}),
            workspace_id="workspace-1",
            permission_mode="ask",
            task_ids=frozenset(),
            max_rounds=2,
            max_tool_calls=4,
        ),
        "call-forged",
        profile="explore",
    )

    with pytest.raises(AgentCreationDenied, match="profile tools are fixed"):
        await SubagentManager(FactoryDouble(RuntimeDouble("child-session"))).spawn(
            forged, caller=AgentCaller("lead", "lead-session", parent)
        )


def test_child_factory_injects_read_only_profile_instructions() -> None:
    class ToolDouble:
        def __init__(self, name: str) -> None:
            self.spec = ToolSpec(name, name, {"type": "object"}, False)

    class LoopDouble:
        def __init__(self) -> None:
            self.context = ContextManager(object(), model="model")  # type: ignore[arg-type]
            self.tools = ToolRegistry()
            for name in profile_tools("explore") or ():
                self.tools.register(ToolDouble(name))
            self.executor = object()
            self.duplicates = object()
            self.budgets = RuntimeBudgets()

    loop = LoopDouble()
    parent = type("ParentRuntime", (), {
        "loop_factory": staticmethod(lambda _provider, _model, _turn: loop),
    })()
    factory = DefaultAgentRuntimeFactory(parent)  # type: ignore[arg-type]
    request = ChildAgentRequest(
        "Inspect files",
        _profile_authority("explore"),
        "call-profile-context",
        profile="explore",
    )

    child_loop = factory._child_loop("fake", "model", object(), request)

    assert child_loop.context.agent_instructions == profile_instructions("explore")
    assert {tool.spec.name for tool in child_loop.tools.list()} == profile_tools("explore")
    assert "strictly read-only" in str(child_loop.context.agent_instructions)


@pytest.mark.asyncio
async def test_child_factory_forks_executor_with_the_child_duplicate_guard(
    tmp_path: Path,
) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import (
        DuplicateGuard,
        PermissionService,
        ToolExecutor,
        WorkspaceStateRegistry,
    )

    class TraceDouble:
        async def record(self, payload) -> None:
            return None

    class ToolDouble:
        spec = ToolSpec("read_file", "read", {"type": "object"}, False)

        async def execute(self, call, context):
            return ToolExecution.success(str(call.arguments["round"]))

    class LoopDouble:
        def __init__(self) -> None:
            self.context = ContextManager(object(), model="model")  # type: ignore[arg-type]
            self.tools = ToolRegistry()
            self.tools.register(ToolDouble())
            self.executor = ToolExecutor(
                self.tools,
                HookManager(trace_hook=TraceDouble()),
                DuplicateGuard(annotation=lambda **_: None),
                PermissionService(),
                WorkspaceStateRegistry(),
            )
            self.duplicates = self.executor.duplicates
            self.budgets = RuntimeBudgets()

    loop = LoopDouble()
    parent_executor = loop.executor
    parent = type(
        "ParentRuntime",
        (),
        {"loop_factory": staticmethod(lambda _provider, _model, _turn: loop)},
    )()
    request = ChildAgentRequest(
        "Inspect files",
        ChildAuthority(
            tools=frozenset({"read_file"}),
            workspace_id="workspace-1",
            permission_mode="ask",
            task_ids=frozenset(),
            max_rounds=2,
            max_tool_calls=4,
        ),
        "call-child-executor",
    )

    child_loop = DefaultAgentRuntimeFactory(parent)._child_loop(  # type: ignore[arg-type]
        "fake", "model", object(), request
    )
    child_executor = child_loop.executor.executor

    assert isinstance(child_executor, ToolExecutor)
    assert child_executor is not parent_executor
    assert child_executor.duplicates is child_loop.duplicates
    assert child_executor.registry is child_loop.tools
    assert child_executor.permission is parent_executor.permission
    assert child_executor.workspaces is parent_executor.workspaces

    await child_loop.duplicates.start_user_message("child-session")
    first = await child_executor.execute(
        ToolCall("round-8", "read_file", {"round": 8}),
        ToolContext(
            "child-session",
            "workspace-1",
            tmp_path,
            metadata={
                "round_number": 8,
                "permission_mode": "bypass",
                "bypass_authorized": True,
            },
        ),
    )
    await child_loop.duplicates.start_user_message("child-session")
    resumed = await child_executor.execute(
        ToolCall("round-1", "read_file", {"round": 1}),
        ToolContext(
            "child-session",
            "workspace-1",
            tmp_path,
            metadata={
                "round_number": 1,
                "permission_mode": "bypass",
                "bypass_authorized": True,
            },
        ),
    )

    assert first.status == "success"
    assert resumed.status == "success"
