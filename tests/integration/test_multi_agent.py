from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.agent.loop import AgentLoop
from litecoder.agent.result import AgentResult
from litecoder.agent.runtime import AgentRuntime, RuntimeContext
from litecoder.common.locks import NamedFileLock
from litecoder.context.manager import ContextManager
from litecoder.context.session.models import SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.paths import AppPaths
from litecoder.providers.models import ProviderEvent, StopReason, Usage
from litecoder.tasks.subagents import (
    AgentCaller,
    AgentCreationDenied,
    ChildAgentRequest,
    ChildAuthority,
    SubagentManager,
)
from litecoder.tasks.teams import TeamManager
from litecoder.tools.models import ToolCall, ToolContext, ToolResult, ToolSpec
from litecoder.tools.permission import PermissionDecision, PermissionService
from litecoder.tools.registry import ToolRegistry
from tests.fakes.provider import FakeProvider


class _Executor:
    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return ToolResult(call.id, "success", "ok")


class _Duplicates:
    async def start_user_message(self, _agent_session_id: str) -> None:
        return None


class _RuntimeDouble:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.duplicates = object()

    async def run(self, _objective: str) -> AgentResult:
        return AgentResult(self.session_id, "completed", "done", Usage(1, 1))


class _Factory:
    async def create_child(self, _request: ChildAgentRequest) -> _RuntimeDouble:
        return _RuntimeDouble("child-session")


def _answer_round() -> list[ProviderEvent]:
    return [
        ProviderEvent.request_identified("request"),
        ProviderEvent.content_block_completed(
            0, {"type": "text", "text": "done"}, request_id="request"
        ),
        ProviderEvent.response_completed(
            StopReason.END_TURN, "end_turn", usage=Usage(1, 1), request_id="request"
        ),
    ]


def _paths(tmp_path: Path) -> AppPaths:
    return AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-1",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


def _authority(kind: str = "lead") -> AgentCaller:
    authority = ChildAuthority(
        tools=frozenset({"read_shared"}),
        workspace_id="workspace-1",
        permission_mode="ask",
        task_ids=frozenset({"task-1"}),
        max_rounds=4,
        max_tool_calls=8,
    )
    return AgentCaller(kind, f"{kind}-session", authority)


@pytest.mark.asyncio
async def test_child_spans_share_root_trace_but_keep_session_identity(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    turns: list[RuntimeContext] = []

    def loop_factory(_provider: str, model: str, turn: RuntimeContext) -> AgentLoop:
        turns.append(turn)
        return AgentLoop(
            store=store,
            provider=FakeProvider([_answer_round()]),
            context=ContextManager(store, model=model),
            tools=ToolRegistry(),
            executor=_Executor(),
            duplicates=_Duplicates(),
            trace_recorder=turn.trace_recorder,
            trace_id=turn.trace_id,
            root_session_id=turn.root_session_id,
            span_id=turn.span_id,
            parent_span_id=turn.parent_span_id,
            agent_id=turn.agent_id,
            redactor=turn.redactor,
        )

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="fake-model",
        loop_factory=loop_factory,
        id_factory=lambda: "lead-session",
    )
    lead = await runtime.run("lead")
    await store.create_session(
        SessionRecord.new(
            "child-session",
            paths.project_id,
            paths.workspace_id,
            "fake",
            "fake-model",
            workspace_path=str(paths.workspace_root),
            session_type="child",
            parent_session_id=lead.session_id,
        )
    )
    child = await runtime.resume("child-session", "child")
    await runtime.close()

    assert lead.session_id != child.session_id
    assert turns[0].root_session_id == turns[1].root_session_id == "lead-session"
    assert turns[0].trace_id == turns[1].trace_id
    assert turns[0].span_id == "root"
    assert turns[1].span_id != turns[0].span_id
    assert turns[1].parent_span_id == turns[0].span_id


@pytest.mark.asyncio
async def test_child_permission_prompts_bubble_to_root_broker(tmp_path: Path) -> None:
    class Broker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def request_from_child(self, request: object) -> PermissionDecision:
            self.requests.append(request)
            return PermissionDecision(True, "allow", "root approved")

    async def local_prompt(_request: object) -> str:
        raise AssertionError("child service must not prompt directly")

    broker = Broker()
    service = PermissionService(prompt=local_prompt)
    decision = await service.decide(
        ToolSpec("write_shared", "write", {}, True),
        ToolCall("call-1", "write_shared", {"path": "shared.txt"}),
        ToolContext(
            "child-session",
            "workspace-1",
            tmp_path,
            metadata={"permission_mode": "ask", "root_session_id": "lead-session"},
            parent_permission_broker=broker,
        ),
    )

    assert decision.allowed is True
    assert len(broker.requests) == 1
    request = broker.requests[0]
    assert request.agent_session_id == "child-session"
    assert request.tool_name == "write_shared"
    assert request.root_session_id == "lead-session"


@pytest.mark.asyncio
async def test_child_cannot_create_more_agents_or_autonomously_assign_work() -> None:
    child = _authority("child")
    request = ChildAgentRequest(
        "nested",
        child.authority,
        "call-1",
        task_id="task-1",
    )

    with pytest.raises(AgentCreationDenied, match="only user or lead"):
        await SubagentManager(_Factory()).spawn(request, caller=child)
    with pytest.raises(AgentCreationDenied, match="only user or lead"):
        await TeamManager(_Factory()).create_teammate(request, caller=child)

@pytest.mark.asyncio
async def test_root_turn_lease_allows_child_same_tree_and_is_revoked_after_turn(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    turns: list[RuntimeContext] = []
    children: list[AgentRuntime] = []
    parent: AgentRuntime

    class InlineLoop:
        def __init__(self, turn: RuntimeContext) -> None:
            self.turn = turn

        async def run_turn(self, session_id: str, _prompt: str) -> AgentResult:
            turns.append(self.turn)
            if self.turn.agent_id == "lead":
                lease = parent.active_root_turn_lease
                assert lease is not None
                child = AgentRuntime(
                    store=store,
                    paths=paths,
                    provider_name="fake",
                    model="fake-model",
                    loop_factory=loop_factory,  # type: ignore[arg-type]
                    id_factory=lambda: "leased-child",
                    session_lock_factory=lambda root_id: NamedFileLock.session_tree(
                        root_id, paths.user_dir
                    ),
                    session_type="child",
                    parent_session_id=session_id,
                    agent_id="leased-child",
                    root_turn_lease=lease,
                    owns_store=False,
                    declared_session_id="leased-child",
                )
                children.append(child)
                await child.run("child")
            return AgentResult(session_id, "completed", "done", Usage(1, 1))

    def loop_factory(
        _provider: str, _model: str, turn: RuntimeContext
    ) -> InlineLoop:
        return InlineLoop(turn)

    parent = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="fake-model",
        loop_factory=loop_factory,  # type: ignore[arg-type]
        id_factory=lambda: "leased-lead",
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )

    result = await parent.run("lead")
    child = children[0]

    assert result.status == "completed"
    assert [turn.agent_id for turn in turns] == ["lead", "leased-child"]
    assert turns[0].root_session_id == turns[1].root_session_id == "leased-lead"
    assert turns[0].trace_id == turns[1].trace_id
    assert turns[0].trace_recorder is turns[1].trace_recorder
    assert parent.active_root_turn_lease is None
    with pytest.raises(RuntimeError, match="lease"):
        await child.resume("leased-child", "too late")

    await child.close()
    await parent.close()
