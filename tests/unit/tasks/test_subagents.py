from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.agent.result import AgentResult
from litecoder.providers.models import Usage
from litecoder.tasks.subagents import (
    AgentCaller,
    AgentCreationDenied,
    ChildAgentRequest,
    ChildAuthority,
    SubagentManager,
)
from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskCreate, TaskStatus
from litecoder.tasks.store import TaskStore


class FakeRuntime:
    def __init__(self, session_id: str = "child-session") -> None:
        self.session_id = session_id
        self.duplicates = object()
        self.objectives: list[str] = []

    async def run(self, objective: str) -> AgentResult:
        self.objectives.append(objective)
        return AgentResult(self.session_id, "completed", "done", Usage(1, 2))


class FakeFactory:
    def __init__(self, runtime: FakeRuntime | None = None) -> None:
        self.runtime = runtime or FakeRuntime()
        self.requests: list[ChildAgentRequest] = []

    async def create_child(self, request: ChildAgentRequest) -> FakeRuntime:
        self.requests.append(request)
        return self.runtime


def authority(**overrides: object) -> ChildAuthority:
    values = {
        "tools": frozenset({"read_file", "search_text"}),
        "workspace_id": "workspace-1",
        "permission_mode": "ask",
        "task_ids": frozenset({"task-1", "task-2"}),
        "max_rounds": 8,
        "max_tool_calls": 16,
        "task_workspaces": frozenset({"workspace-2"}),
    }
    values.update(overrides)
    return ChildAuthority(**values)  # type: ignore[arg-type]


def request(**overrides: object) -> ChildAgentRequest:
    values = {
        "objective": "inspect the code",
        "authority": authority(tools=frozenset({"read_file"}), task_ids=frozenset({"task-1"}), max_rounds=2, max_tool_calls=4),
        "tool_call_id": "call-1",
        "task_id": "task-1",
        "worktree_id": None,
    }
    values.update(overrides)
    return ChildAgentRequest(**values)  # type: ignore[arg-type]


def caller(kind: str = "lead") -> AgentCaller:
    return AgentCaller(
        kind=kind,
        session_id="lead-session",
        authority=authority(),
        runtime=object(),
    )


def test_authority_restricts_to_parent_tools_tasks_workspace_and_budgets() -> None:
    parent = authority()

    assert ChildAuthority.restrict(parent, request().authority) == request().authority
    assert ChildAuthority.restrict(
        parent,
        authority(
            tools=frozenset({"read_file"}),
            workspace_id="workspace-2",
            task_ids=frozenset({"task-1"}),
            max_rounds=2,
            max_tool_calls=4,
        ),
    ).workspace_id == "workspace-2"
    with pytest.raises(AgentCreationDenied, match="authority"):
        ChildAuthority.restrict(parent, authority(tools=frozenset({"write_file"})))
    with pytest.raises(AgentCreationDenied, match="authority"):
        ChildAuthority.restrict(parent, authority(task_ids=frozenset({"task-9"})))
    with pytest.raises(AgentCreationDenied, match="workspace"):
        ChildAuthority.restrict(parent, authority(workspace_id="workspace-9"))
    with pytest.raises(AgentCreationDenied, match="budget"):
        ChildAuthority.restrict(parent, authority(max_rounds=99))
    with pytest.raises(AgentCreationDenied, match="workspace"):
        ChildAuthority.restrict(
            parent,
            authority(task_workspaces=frozenset({"workspace-9"})),
        )


@pytest.mark.asyncio
async def test_request_task_must_belong_to_restricted_authority() -> None:
    manager = SubagentManager(FakeFactory())

    with pytest.raises(AgentCreationDenied, match="task"):
        await manager.spawn(request(task_id="task-2"), caller=caller("lead"))


@pytest.mark.asyncio
async def test_non_lead_agent_cannot_spawn_subagent() -> None:
    manager = SubagentManager(FakeFactory())

    with pytest.raises(AgentCreationDenied, match="only user or lead"):
        await manager.spawn(request(), caller=caller("child"))


@pytest.mark.asyncio
async def test_lead_spawn_creates_child_and_returns_tool_result() -> None:
    runtime = FakeRuntime("child-session")
    factory = FakeFactory(runtime)
    manager = SubagentManager(factory)

    handle = await manager.spawn(request(), caller=caller("lead"))

    assert factory.requests == [request()]
    assert runtime.objectives == ["inspect the code"]
    assert handle.session_id == "child-session"
    assert handle.result.tool_call_id == "call-1"
    assert handle.result.status == "success"
    assert handle.result.metadata["session_id"] == "child-session"
    assert handle.result.metadata["agent_status"] == "completed"
    assert manager.spawn_history == [
        {
            "agent_id": "",
            "task_id": "task-1",
            "worktree_id": "",
            "status": "completed",
            "result_returned": 1,
            "failure": "",
            "session_id": "child-session",
            "input_tokens": 1,
            "output_tokens": 2,
            "reason": "done",
        }
    ]


@pytest.mark.asyncio
async def test_subagent_marks_unfinished_claimed_task_failed(tmp_path: Path) -> None:
    tasks = TaskManager(TaskStore(tmp_path / "tasks"))
    await tasks.create(TaskCreate("task-1", "inspect", "inspect the code"))

    class ClaimingRuntime(FakeRuntime):
        agent_id = "child-1"
        task_manager = tasks

        async def run(self, objective: str) -> AgentResult:
            await tasks.claim("task-1", self.agent_id)
            return await super().run(objective)

    manager = SubagentManager(FakeFactory(ClaimingRuntime()))

    await manager.spawn(request(), caller=caller("lead"))

    task = await tasks.get("task-1")
    assert task.status is TaskStatus.FAILED
    assert task.owner_agent_id == "child-1"
