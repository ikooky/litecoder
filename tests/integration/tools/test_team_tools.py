from __future__ import annotations

import asyncio

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from litecoder.agent.result import AgentResult
from litecoder.hooks import HookManager
from litecoder.providers.models import Usage
from litecoder.tasks.message_bus import MessageBus, TeamMessage
from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskCreate, TaskStatus
from litecoder.tasks.store import TaskStore
from litecoder.tasks.subagents import AgentCaller, AgentCreationDenied, ChildAgentRequest, ChildAuthority
from litecoder.tasks.teams import TeamManager
from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolRegistry, WorkspaceStateRegistry
from litecoder.tools.builtin.team import TeamCreateTool, TeamListTool, TeamReceiveTool, TeamSendTool, _worktree_binding, register_team_tools
from litecoder.tools.models import ToolDenied, ToolFailure, ToolSpec


class Trace:
    async def record(self, fact):
        return None


class RuntimeDouble:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.duplicates = object()

    async def run(self, objective: str) -> AgentResult:
        return AgentResult(self.session_id, "completed", objective, Usage(0, 0))


class FactoryDouble:
    def __init__(self) -> None:
        self.calls: list[ChildAgentRequest] = []
        self.runtimes: list[RuntimeDouble] = []

    async def create_child(self, request: ChildAgentRequest) -> RuntimeDouble:
        self.calls.append(request)
        runtime = RuntimeDouble(f"teammate-session-{len(self.runtimes) + 1}")
        self.runtimes.append(runtime)
        return runtime


class AssignmentFailureRuntime(RuntimeDouble):
    def __init__(
        self, session_id: str, tasks: TaskManager, agent_id: str, task_id: str
    ) -> None:
        super().__init__(session_id)
        self.tasks = tasks
        self.agent_id = agent_id
        self.task_id = task_id

    async def run(self, objective: str) -> AgentResult:
        del objective
        task = await self.tasks.get(self.task_id)
        assert task.status is TaskStatus.IN_PROGRESS
        assert task.owner_agent_id == self.agent_id
        return AgentResult(
            self.session_id,
            "incomplete",
            "round budget exhausted",
            Usage(0, 0),
        )

    async def resume(self, session_id: str, prompt: str) -> AgentResult:
        del session_id, prompt
        raise AssertionError("failed worker must not resume")


class AssignmentFailureFactory(FactoryDouble):
    def __init__(self, tasks: TaskManager) -> None:
        super().__init__()
        self.tasks = tasks

    async def create_child(self, request: ChildAgentRequest) -> RuntimeDouble:
        assert request.agent_id is not None
        assert request.task_id is not None
        self.calls.append(request)
        runtime = AssignmentFailureRuntime(
            "failing-worker-session",
            self.tasks,
            request.agent_id,
            request.task_id,
        )
        self.runtimes.append(runtime)
        return runtime


def authority() -> ChildAuthority:
    return ChildAuthority(
        tools=frozenset(
            {
                "read_file",
                "search_text",
                "task_complete",
                "task_fail",
                "team_send",
            }
        ),
        workspace_id="workspace-1",
        permission_mode="ask",
        task_ids=frozenset({"task-1"}),
        max_rounds=8,
        max_tool_calls=16,
    )


def caller(kind: str = "lead") -> AgentCaller:
    return AgentCaller(kind, f"{kind}-session", authority())


def context(tmp_path: Path, session_id: str = "lead-session", **metadata: object) -> ToolContext:
    values = {"round_number": 1}
    values.update(metadata)
    return ToolContext(session_id, "workspace-1", tmp_path, metadata=values)


def build_executor(*tools):
    registry = ToolRegistry()
    registry.register_many(tools)
    return ToolExecutor(
        registry,
        HookManager(trace_hook=Trace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"),
        WorkspaceStateRegistry(),
    )


def create_call(
    call_id: str,
    display_name: str = "reviewer",
    *,
    task_id: str | None = None,
) -> ToolCall:
    arguments: dict[str, object] = {
        "display_name": display_name,
        "objective": "inspect",
        "tools": ["read_file", "task_complete", "task_fail", "team_send"],
        "budget": {"max_rounds": 2, "max_tool_calls": 4},
    }
    if task_id is not None:
        arguments["task_id"] = task_id
    return ToolCall(call_id, "team_create", arguments)


async def create_member(manager: TeamManager, display_name: str):
    return await manager.create_teammate(
        ChildAgentRequest("inspect", authority(), f"create-{display_name}"),
        caller=caller(),
        display_name=display_name,
    )


def test_team_send_schema_documents_lead_recipient() -> None:
    agent_id = TeamSendTool.spec.input_schema["properties"]["agent_id"]

    assert "lead" in TeamSendTool.spec.description
    assert "Use 'lead'" in agent_id["description"]


@pytest.mark.asyncio
async def test_failed_team_worker_updates_assigned_task(tmp_path: Path) -> None:
    tasks = TaskManager(TaskStore(tmp_path / "tasks"))
    await tasks.create(TaskCreate("task-1", "work", "perform delegated work"))
    manager = TeamManager(
        AssignmentFailureFactory(tasks),
        task_manager=tasks,
        id_factory=lambda: "worker-1",
    )

    member = await manager.create_teammate(
        ChildAgentRequest(
            "perform work",
            authority(),
            "create-worker",
            task_id="task-1",
        ),
        caller=caller(),
    )
    worker = manager._workers[member.agent_id]
    await asyncio.gather(worker, return_exceptions=True)
    await asyncio.sleep(0)

    task = await tasks.get("task-1")
    assert task.status is TaskStatus.FAILED
    assert task.owner_agent_id == "worker-1"
    assert member.state == "failed"
    assert member.failure == "round budget exhausted"


@pytest.mark.asyncio
async def test_team_executor_does_not_dedupe_identical_sends(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    send = TeamSendTool(manager)
    executor = build_executor(send)
    member = await create.execute(create_call("create-1"), context(tmp_path))
    agent_id = member.metadata["agent_id"]
    arguments = {"agent_id": agent_id, "body": "same"}

    first = await executor.execute(
        ToolCall("send-1", "team_send", arguments),
        context(tmp_path, agent_id="lead"),
    )
    second = await executor.execute(
        ToolCall("send-2", "team_send", arguments),
        context(tmp_path, agent_id="lead"),
    )

    assert first.status == second.status == "success"
    assert [message.body for message in await bus.receive(agent_id)] == ["same", "same"]


@pytest.mark.asyncio
async def test_team_executor_receive_can_run_again_after_new_message(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    send = TeamSendTool(manager)
    receive = TeamReceiveTool(manager)
    executor = build_executor(send, receive)
    member = await create.execute(create_call("create-1"), context(tmp_path))
    agent_id = member.metadata["agent_id"]

    await executor.execute(
        ToolCall("send-1", "team_send", {"agent_id": "lead", "body": "one"}),
        context(tmp_path, agent_id=agent_id),
    )
    first = await executor.execute(
        ToolCall("receive-1", "team_receive", {}),
        context(tmp_path, agent_id="lead"),
    )
    await executor.execute(
        ToolCall("send-2", "team_send", {"agent_id": "lead", "body": "two"}),
        context(tmp_path, agent_id=agent_id),
    )
    second = await executor.execute(
        ToolCall("receive-2", "team_receive", {}),
        context(tmp_path, agent_id="lead"),
    )

    assert json.loads(first.content)[0]["body"] == "one"
    assert json.loads(second.content)[0]["body"] == "two"


@pytest.mark.asyncio
async def test_team_executor_list_reflects_later_roster_changes(tmp_path: Path) -> None:
    factory = FactoryDouble()
    manager = TeamManager(factory)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    listing = TeamListTool(manager)
    executor = build_executor(create, listing)

    before = await executor.execute(ToolCall("list-1", "team_list", {}), context(tmp_path))
    await executor.execute(create_call("create-1"), context(tmp_path))
    after = await executor.execute(ToolCall("list-2", "team_list", {}), context(tmp_path))

    assert json.loads(before.content) == []
    assert len(json.loads(after.content)) == 1
    assert json.loads(after.content)[0]["display_name"] == "reviewer"


@pytest.mark.asyncio
async def test_two_identical_explicit_creates_both_run(tmp_path: Path) -> None:
    factory = FactoryDouble()
    manager = TeamManager(factory)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    executor = build_executor(create)

    first = await executor.execute(create_call("create-1"), context(tmp_path))
    second = await executor.execute(create_call("create-2"), context(tmp_path))

    assert first.status == second.status == "success"
    assert len(factory.calls) == 2
    assert len(manager.list()) == 2


@pytest.mark.asyncio
async def test_create_send_receive_binds_session_to_generated_agent_id(tmp_path: Path) -> None:
    factory = FactoryDouble()
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(factory, message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    send = TeamSendTool(manager)
    receive = TeamReceiveTool(manager)
    create_executor = build_executor(create)

    created = await create_executor.execute(create_call("create-1"), context(tmp_path))
    agent_id = created.metadata["agent_id"]
    member = manager.list()[0]
    sent = await send.execute(
        ToolCall("send-1", "team_send", {"agent_id": "reviewer", "body": "hello"}),
        context(tmp_path, agent_id="lead"),
    )

    received = await receive.execute(ToolCall("receive-1", "team_receive", {}), context(tmp_path, session_id=member.session_id))

    assert agent_id in created.content
    assert factory.calls[0].agent_id == agent_id == member.agent_id
    assert sent.metadata == {
        "agent_id": agent_id,
        "requested_agent_id": "reviewer",
    }
    assert json.loads(received.content)[0]["body"] == "hello"
    assert not (tmp_path / "mailboxes" / f"{agent_id}.jsonl").exists()


@pytest.mark.asyncio
async def test_team_create_assigns_the_delegated_task_before_worker_start(
    tmp_path: Path,
) -> None:
    tasks = TaskManager(TaskStore(tmp_path / "tasks"))
    await tasks.create(TaskCreate("task-1", "Inspect", "Inspect the solution"))
    manager = TeamManager(FactoryDouble(), id_factory=lambda: "worker-1")
    create = TeamCreateTool(
        manager,
        caller_resolver=lambda _: caller(),
        task_manager=tasks,
    )

    created = await create.execute(
        create_call("create-1", task_id="task-1"), context(tmp_path)
    )
    task = await tasks.get("task-1")

    assert created.metadata["agent_id"] == "worker-1"
    assert task.status.value == "in_progress"
    assert task.owner_agent_id == "worker-1"


@pytest.mark.asyncio
async def test_mutating_teammate_requires_task_and_worktree(tmp_path: Path) -> None:
    class WriteToolDouble:
        spec = ToolSpec("write_file", "write", {"type": "object"}, True)

    tool_registry = ToolRegistry()
    tool_registry.register(WriteToolDouble())
    create = TeamCreateTool(
        TeamManager(FactoryDouble()),
        caller_resolver=lambda _: caller(),
        tool_registry=tool_registry,
    )

    with pytest.raises(
        ToolFailure, match="require both task_id and worktree_id"
    ):
        await create.execute(
            ToolCall(
                "create-writer",
                "team_create",
                {
                    "display_name": "writer",
                    "objective": "edit solution",
                    "tools": ["write_file"],
                    "budget": {"max_rounds": 2, "max_tool_calls": 4},
                },
            ),
            context(tmp_path),
        )


@pytest.mark.asyncio
async def test_worktree_binding_returns_binding_and_rejects_lead_owned_task(
    tmp_path: Path,
) -> None:
    class Worktrees:
        async def list(self):
            return [binding]

    tasks = TaskManager(TaskStore(tmp_path / "tasks"))
    await tasks.create(TaskCreate("task-1", "Implement", "Change solution"))
    await tasks.bind_worktree("task-1", "worktree-1")
    binding = SimpleNamespace(
        id="worktree-1",
        task_id="task-1",
        workspace_id="workspace-child",
    )

    resolved = await _worktree_binding(
        Worktrees(), "worktree-1", "task-1", tasks
    )

    assert resolved is binding

    await tasks.assign_and_start("task-1", "lead")
    with pytest.raises(ToolFailure, match="pending and unassigned"):
        await _worktree_binding(Worktrees(), "worktree-1", "task-1", tasks)


@pytest.mark.asyncio
async def test_team_send_rejects_ambiguous_display_name(tmp_path: Path) -> None:
    factory = FactoryDouble()
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(factory, message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    send = TeamSendTool(manager)

    await create.execute(create_call("create-1"), context(tmp_path))
    await create.execute(create_call("create-2"), context(tmp_path))

    with pytest.raises(ToolFailure, match="Ambiguous team recipient"):
        await send.execute(
            ToolCall(
                "send-1",
                "team_send",
                {"agent_id": "reviewer", "body": "hello"},
            ),
            context(tmp_path),
        )


@pytest.mark.asyncio
async def test_team_mailbox_rejects_non_members(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    send = TeamSendTool(manager)
    receive = TeamReceiveTool(manager)
    created = await create.execute(create_call("create-1"), context(tmp_path))
    member_id = created.metadata["agent_id"]

    with pytest.raises(ToolFailure, match="mailbox access"):
        await send.execute(
            ToolCall("unknown-recipient", "team_send", {"agent_id": "outside", "body": "hello"}),
            context(tmp_path, agent_id="lead"),
        )
    with pytest.raises(ToolFailure, match="mailbox access"):
        await send.execute(
            ToolCall("unknown-sender", "team_send", {"agent_id": member_id, "body": "hello"}),
            context(tmp_path, agent_id="outside"),
        )
    with pytest.raises(ToolFailure, match="mailbox access"):
        await receive.execute(
            ToolCall("unknown-receive", "team_receive", {}),
            context(tmp_path, agent_id="outside"),
        )


@pytest.mark.asyncio
async def test_team_tool_contracts_and_registration(tmp_path: Path) -> None:
    factory = FactoryDouble()
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(factory, message_bus=bus)
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())

    assert {tool.spec.name for tool in registry.list()} == {
        "team_create",
        "team_send",
        "team_receive",
        "team_list",
        "team_request_plan_approval",
        "team_respond_plan_approval",
        "team_request_shutdown",
        "team_respond_shutdown",
    }
    assert all(tool.spec.dedupe_policy == "none" for tool in registry.list())
    create = registry.require("team_create")
    assert create.spec.input_schema["required"] == ["display_name", "objective", "tools"]
    assert create.spec.input_schema["properties"]["budget"] == {"type": "object", "properties": {"max_rounds": {"type": "integer", "minimum": 1}, "max_tool_calls": {"type": "integer", "minimum": 1}}, "required": ["max_rounds", "max_tool_calls"], "additionalProperties": False}

    with pytest.raises(ToolFailure):
        await create.execute(ToolCall("bad", "team_create", {"display_name": "x", "objective": "x", "tools": [], "budget": {"max_rounds": 0, "max_tool_calls": 1}}), context(tmp_path))
    denied = TeamCreateTool(manager, caller_resolver=lambda _: caller("child"))
    with pytest.raises(ToolDenied, match="only user or lead"):
        await denied.execute(create_call("denied"), context(tmp_path, session_id="child-session"))
    created = await create.execute(create_call("create-member"), context(tmp_path))
    member_id = created.metadata["agent_id"]
    sent = await registry.require("team_send").execute(
        ToolCall("send", "team_send", {"agent_id": member_id, "body": "hello"}),
        context(tmp_path, agent_id="lead"),
    )
    assert sent.status == "success"
    received = await registry.require("team_receive").execute(
        ToolCall("receive", "team_receive", {}),
        context(tmp_path, agent_id=member_id),
    )
    assert json.loads(received.content)[0]["body"] == "hello"
    assert json.loads(received.content)[0]["sender"] == "lead"
    listed = await registry.require("team_list").execute(ToolCall("list", "team_list", {}), context(tmp_path))
    assert isinstance(json.loads(listed.content), list)


@pytest.mark.asyncio
async def test_team_create_inherits_parent_budget_when_omitted(tmp_path: Path) -> None:
    manager = TeamManager(FactoryDouble())
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())

    result = await create.execute(
        ToolCall(
            "inherit-budget",
            "team_create",
            {
                "display_name": "reviewer",
                "objective": "inspect",
                "tools": ["read_file"],
            },
        ),
        context(tmp_path),
    )

    assert result.status == "success"
    assert manager.factory.calls[0].authority.max_rounds == 8
    assert manager.factory.calls[0].authority.max_tool_calls == 16


@pytest.mark.asyncio
async def test_registration_binds_explicit_bus_to_protocol_notifications(
    tmp_path: Path,
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble())
    registry = ToolRegistry()
    register_team_tools(
        registry, manager, bus=bus, caller_resolver=lambda _: caller()
    )
    worker = await create_member(manager, "worker")

    requested = await registry.require("team_request_shutdown").execute(
        ToolCall(
            "request",
            "team_request_shutdown",
            {"agent_id": worker.agent_id},
        ),
        context(tmp_path, agent_id="lead"),
    )
    messages = await bus.receive(worker.agent_id)
    request = manager.protocols.pending_requests[requested.metadata["request_id"]]

    assert manager.message_bus is bus
    assert manager.protocols.message_bus is bus
    assert len(messages) == 1
    assert messages[0].sender == "lead"
    assert json.loads(messages[0].body)["phase"] == "request"
    request.future.cancel()
    await asyncio.sleep(0)


def test_registration_rejects_conflicting_bus_without_partial_rebind(
    tmp_path: Path,
) -> None:
    manager_bus = MessageBus(tmp_path / "manager-mailboxes")
    registration_bus = MessageBus(tmp_path / "registration-mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=manager_bus)
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="conflicting MessageBus"):
        register_team_tools(
            registry,
            manager,
            bus=registration_bus,
            caller_resolver=lambda _: caller(),
        )

    assert manager.message_bus is manager_bus
    assert manager.protocols.message_bus is manager_bus
    assert registry.list() == ()


@pytest.mark.parametrize("tool_type", [TeamSendTool, TeamReceiveTool])
def test_direct_message_tool_binds_the_authoritative_bus(
    tmp_path: Path, tool_type
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble())

    tool = tool_type(manager, bus)

    assert tool.bus is bus
    assert manager.message_bus is bus
    assert manager.protocols.message_bus is bus


@pytest.mark.parametrize("tool_type", [TeamSendTool, TeamReceiveTool])
def test_direct_message_tool_rejects_a_conflicting_bus_atomically(
    tmp_path: Path, tool_type
) -> None:
    manager_bus = MessageBus(tmp_path / "manager-mailboxes")
    tool_bus = MessageBus(tmp_path / "tool-mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=manager_bus)

    with pytest.raises(ValueError, match="conflicting MessageBus"):
        tool_type(manager, tool_bus)

    assert manager.message_bus is manager_bus
    assert manager.protocols.message_bus is manager_bus

@pytest.mark.asyncio
async def test_two_teammate_round_trip_uses_generated_agent_ids(tmp_path: Path) -> None:
    factory = FactoryDouble()
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(factory, message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())
    send = TeamSendTool(manager)
    receive = TeamReceiveTool(manager)
    executor = build_executor(create, send, receive)

    await executor.execute(create_call("create-a", "A"), context(tmp_path))
    await executor.execute(create_call("create-b", "B"), context(tmp_path))
    member_a, member_b = manager.list()

    await executor.execute(
        ToolCall("send-a", "team_send", {"agent_id": member_b.agent_id, "body": "from A"}),
        context(tmp_path, session_id=member_a.session_id),
    )
    received_by_b = await executor.execute(
        ToolCall("receive-b", "team_receive", {}),
        context(tmp_path, session_id=member_b.session_id),
    )
    b_messages = json.loads(received_by_b.content)
    assert b_messages[0]["sender"] == member_a.agent_id

    await executor.execute(
        ToolCall("reply-b", "team_send", {"agent_id": b_messages[0]["sender"], "body": "reply from B"}),
        context(tmp_path, session_id=member_b.session_id),
    )
    received_by_a = await executor.execute(
        ToolCall("receive-a", "team_receive", {}),
        context(tmp_path, session_id=member_a.session_id),
    )
    assert json.loads(received_by_a.content)[0]["body"] == "reply from B"
    assert not (tmp_path / "mailboxes" / f"{member_a.session_id}.jsonl").exists()
    assert not (tmp_path / "mailboxes" / f"{member_b.session_id}.jsonl").exists()
    await manager.end_turn()
    assert manager.last_turn_peer_messages_sent == 2

@pytest.mark.asyncio
async def test_protocol_tools_correlate_requester_and_exact_responder(
    tmp_path: Path,
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())
    reviewer = await create_member(manager, "reviewer")
    other = await create_member(manager, "other")

    requested = await registry.require("team_request_plan_approval").execute(
        ToolCall(
            "request-plan",
            "team_request_plan_approval",
            {"agent_id": reviewer.agent_id, "plan": {"tasks": ["t1"]}},
        ),
        context(tmp_path, agent_id="lead"),
    )
    request_id = requested.metadata["request_id"]
    pending = manager.protocols.pending_requests[request_id]

    with pytest.raises(ToolFailure, match="unexpected responder"):
        await registry.require("team_respond_plan_approval").execute(
            ToolCall(
                "wrong-response",
                "team_respond_plan_approval",
                {"request_id": request_id, "approved": True},
            ),
            context(tmp_path, agent_id=other.agent_id),
        )

    responded = await registry.require("team_respond_plan_approval").execute(
        ToolCall(
            "response",
            "team_respond_plan_approval",
            {"request_id": request_id, "approved": False, "reason": "revise"},
        ),
        context(tmp_path, agent_id=reviewer.agent_id),
    )

    assert responded.status == "success"
    assert (await pending).approved is False
    assert request_id not in manager.protocols.pending_requests


@pytest.mark.asyncio
async def test_shutdown_tools_use_the_shutdown_protocol_kind(
    tmp_path: Path,
) -> None:
    manager = TeamManager(
        FactoryDouble(), message_bus=MessageBus(tmp_path / "mailboxes")
    )
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())
    worker = await create_member(manager, "worker")

    requested = await registry.require("team_request_shutdown").execute(
        ToolCall(
            "request-shutdown",
            "team_request_shutdown",
            {"agent_id": worker.agent_id, "reason": "maintenance"},
        ),
        context(tmp_path, agent_id="lead"),
    )
    request_id = requested.metadata["request_id"]
    pending = manager.protocols.pending_requests[request_id]
    await registry.require("team_respond_shutdown").execute(
        ToolCall(
            "respond-shutdown",
            "team_respond_shutdown",
            {"request_id": request_id, "approved": True},
        ),
        context(tmp_path, agent_id=worker.agent_id),
    )

    assert pending.kind == "shutdown"
    assert (await pending).approved is True

def test_protocol_tools_are_exported_from_builtin_package() -> None:
    from litecoder.tools.builtin import (
        TeamRequestPlanApprovalTool,
        TeamRequestShutdownTool,
        TeamRespondPlanApprovalTool,
        TeamRespondShutdownTool,
    )

    assert TeamRequestPlanApprovalTool.spec.name == "team_request_plan_approval"
    assert TeamRespondPlanApprovalTool.spec.name == "team_respond_plan_approval"
    assert TeamRequestShutdownTool.spec.name == "team_request_shutdown"
    assert TeamRespondShutdownTool.spec.name == "team_respond_shutdown"

class ProtocolToolBus:
    def __init__(self) -> None:
        self.messages = []
        self.response_failures = 0
        self.response_started = asyncio.Event()
        self.release_response = asyncio.Event()
        self.release_response.set()

    async def send(self, agent_id, message) -> None:
        phase = json.loads(message.body)["phase"]
        if phase == "response":
            self.response_started.set()
            await self.release_response.wait()
            if self.response_failures:
                self.response_failures -= 1
                raise OSError("mailbox write failed")
        self.messages.append(message)


@pytest.mark.asyncio
async def test_response_tool_reports_notification_failure_and_allows_retry(
    tmp_path: Path,
) -> None:
    bus = ProtocolToolBus()
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())
    worker = await create_member(manager, "worker")
    requested = await registry.require("team_request_shutdown").execute(
        ToolCall(
            "request",
            "team_request_shutdown",
            {"agent_id": worker.agent_id},
        ),
        context(tmp_path, agent_id="lead"),
    )
    request_id = requested.metadata["request_id"]
    request = manager.protocols.pending_requests[request_id]
    bus.response_failures = 1

    with pytest.raises(ToolFailure, match="protocol response notification failed"):
        await registry.require("team_respond_shutdown").execute(
            ToolCall(
                "respond-fail",
                "team_respond_shutdown",
                {"request_id": request_id, "approved": True},
            ),
            context(tmp_path, agent_id=worker.agent_id),
        )

    assert manager.protocols.pending_requests == {request_id: request}
    assert not request.future.done()
    result = await registry.require("team_respond_shutdown").execute(
        ToolCall(
            "respond-retry",
            "team_respond_shutdown",
            {"request_id": request_id, "approved": True},
        ),
        context(tmp_path, agent_id=worker.agent_id),
    )
    assert result.status == "success"
    assert (await request).approved is True


@pytest.mark.asyncio
async def test_response_tool_cancellation_leaves_request_retryable(
    tmp_path: Path,
) -> None:
    bus = ProtocolToolBus()
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())
    worker = await create_member(manager, "worker")
    requested = await registry.require("team_request_shutdown").execute(
        ToolCall(
            "request",
            "team_request_shutdown",
            {"agent_id": worker.agent_id},
        ),
        context(tmp_path, agent_id="lead"),
    )
    request_id = requested.metadata["request_id"]
    request = manager.protocols.pending_requests[request_id]
    bus.release_response.clear()
    responding = asyncio.create_task(
        registry.require("team_respond_shutdown").execute(
            ToolCall(
                "respond-cancel",
                "team_respond_shutdown",
                {"request_id": request_id, "approved": True},
            ),
            context(tmp_path, agent_id=worker.agent_id),
        )
    )
    await bus.response_started.wait()

    responding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await responding

    assert manager.protocols.pending_requests == {request_id: request}
    assert not request.future.done()
    bus.release_response.set()
    await registry.require("team_respond_shutdown").execute(
        ToolCall(
            "respond-retry",
            "team_respond_shutdown",
            {"request_id": request_id, "approved": True},
        ),
        context(tmp_path, agent_id=worker.agent_id),
    )
    assert (await request).approved is True


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
async def test_protocol_tools_reject_non_finite_timeout(
    tmp_path: Path, timeout: float
) -> None:
    bus = ProtocolToolBus()
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())
    worker = await create_member(manager, "worker")

    try:
        with pytest.raises(ToolFailure, match="timeout"):
            raw_call = type(
                "RawToolCall",
                (),
                {
                    "id": "request",
                    "arguments": {"agent_id": worker.agent_id, "timeout": timeout},
                },
            )()
            await registry.require("team_request_shutdown").execute(
                raw_call,
                context(tmp_path, agent_id="lead"),
            )
    finally:
        for request in tuple(manager.protocols.pending_requests.values()):
            request.future.cancel()
        await asyncio.sleep(0)

@pytest.mark.asyncio
async def test_tool_created_timeout_exception_is_observed_and_awaitable(
    tmp_path: Path,
) -> None:
    bus = ProtocolToolBus()
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    registry = ToolRegistry()
    register_team_tools(registry, manager, caller_resolver=lambda _: caller())
    worker = await create_member(manager, "worker")
    requested = await registry.require("team_request_shutdown").execute(
        ToolCall(
            "request-timeout",
            "team_request_shutdown",
            {"agent_id": worker.agent_id, "timeout": 0.01},
        ),
        context(tmp_path, agent_id="lead"),
    )
    request = manager.protocols.pending_requests[requested.metadata["request_id"]]

    await asyncio.sleep(0.03)

    assert request.future._log_traceback is False
    with pytest.raises(TimeoutError, match="protocol request timed out"):
        await request

@pytest.mark.asyncio
async def test_team_end_turn_resets_roster_and_preserves_detached_evidence(
    tmp_path: Path,
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    manager = TeamManager(FactoryDouble(), message_bus=bus)
    create = TeamCreateTool(manager, caller_resolver=lambda _: caller())

    first = await create.execute(create_call("create-1"), context(tmp_path))
    agent_id = first.metadata["agent_id"]
    await bus.send(agent_id, TeamMessage("lead", agent_id, "work"))
    assert len(await bus.receive(agent_id)) == 1

    await manager.end_turn()

    assert manager.list() == ()
    assert len(manager.last_turn_members) == 1
    assert manager.last_turn_messages_sent == 1
    assert manager.last_turn_messages_received == 1

    await create.execute(create_call("create-2"), context(tmp_path))
    assert len(manager.list()) == 1
    await manager.close()
