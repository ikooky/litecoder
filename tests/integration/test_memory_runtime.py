from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from litecoder.agent.loop import AgentLoop
from litecoder.agent.runtime import AgentRuntime, RuntimeContext
from litecoder.common.trace import SecretRedactor
from litecoder.context.manager import ContextManager
from litecoder.context.session.models import MessageRecord, SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.memory import (
    MemoryConsolidationResult,
    MemoryCoordinator,
    MemoryEntry,
    MemoryExtractionResult,
    MemoryService,
    MemoryStore,
)
from litecoder.paths import AppPaths
from litecoder.providers.models import ProviderEvent, StopReason, Usage
from litecoder.tools.models import ToolCall, ToolResult
from litecoder.tools.registry import ToolRegistry
from tests.fakes.provider import FakeProvider


class RecordingDuplicates:
    async def start_user_message(self, agent_session_id: str) -> None:
        del agent_session_id


class RecordingTrace:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def record(self, payload: object) -> None:
        assert isinstance(payload, dict)
        self.records.append(dict(payload))


class NoOpExecutor:
    async def execute(self, call: ToolCall, context: object) -> ToolResult:
        del call, context
        raise AssertionError("memory runtime tests do not execute tools")


class StubMemoryService:
    pass


class DiagnosticMemoryService:
    def __init__(
        self,
        *,
        hang_extract: bool = False,
        recalled: bool = False,
    ) -> None:
        self.hang_extract = hang_extract
        self.recalled = recalled
        self.extract_started = asyncio.Event()

    def system_payload(self) -> dict[str, object]:
        return {"directory": ".memory", "index": "", "instructions": []}

    async def load_memories(self, messages: Sequence[MessageRecord]):
        del messages
        from litecoder.memory.loading import LoadedMemories

        if not self.recalled:
            return LoadedMemories((), "")
        entry = MemoryEntry(
            "reply-style",
            "Stable user reply preference",
            "user",
            "Start replies with meow.",
        )
        return LoadedMemories((entry,), entry.render())

    async def extract_memories(
        self,
        session_id: str,
        messages: Sequence[MessageRecord],
    ) -> MemoryExtractionResult:
        del session_id, messages
        self.extract_started.set()
        if self.hang_extract:
            await asyncio.Event().wait()
        return MemoryExtractionResult(
            2,
            1,
            1,
            1,
            "partial_rejected",
            total=10,
        )

    async def consolidate_memories(self) -> MemoryConsolidationResult:
        return MemoryConsolidationResult("completed", 10, 5)


async def _build_diagnostic_loop(
    root: Path,
    service: DiagnosticMemoryService,
    coordinator: MemoryCoordinator,
    *,
    trace_recorder: RecordingTrace | None = None,
) -> tuple[AgentLoop, SQLiteSessionStore, object]:
    from litecoder.ui.sink import RecordingUISink

    store = SQLiteSessionStore(root / "sessions.db")
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        "project-1",
        "workspace-1",
        "fake",
        "fake-model",
        workspace_path=str(root),
    ))
    sink = RecordingUISink()
    loop = AgentLoop(
        store=store,
        provider=FakeProvider([_text_round("done")]),
        context=ContextManager(
            store,
            model="fake-model",
            memory_service=service,  # type: ignore[arg-type]
        ),
        tools=ToolRegistry(),
        executor=NoOpExecutor(),
        duplicates=RecordingDuplicates(),
        memory_service=service,  # type: ignore[arg-type]
        memory_coordinator=coordinator,
        ui_sink=sink,
        trace_recorder=trace_recorder,
        trace_id="trace-1",
        root_session_id="session-1",
    )
    return loop, store, sink


@dataclass(frozen=True, slots=True)
class Submission:
    service: object
    session_id: str
    messages: tuple[MessageRecord, ...]


class RecordingCoordinator:
    def __init__(self) -> None:
        self.submissions: list[Submission] = []

    def submit(
        self,
        service: object,
        session_id: str,
        messages: Sequence[MessageRecord],
        diagnostic: object,
    ) -> None:
        del diagnostic
        self.submissions.append(Submission(service, session_id, tuple(messages)))


def _text_round(
    text: str,
    *,
    stop_reason: StopReason = StopReason.END_TURN,
) -> list[ProviderEvent]:
    return [
        ProviderEvent.content_block_completed(
            0,
            {"type": "text", "text": text},
            request_id="request",
        ),
        ProviderEvent.response_completed(
            stop_reason,
            stop_reason.value,
            usage=Usage(1, 1),
            request_id="request",
        ),
    ]


async def _build_loop(
    root: Path,
    *,
    coordinator: RecordingCoordinator,
    memory_eligible: bool = True,
    stop_reason: StopReason = StopReason.END_TURN,
) -> tuple[AgentLoop, SQLiteSessionStore, StubMemoryService]:
    store = SQLiteSessionStore(root / "sessions.db")
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        "project-1",
        "workspace-1",
        "fake",
        "fake-model",
        workspace_path=str(root),
    ))
    service = StubMemoryService()
    provider_rounds = [
        _text_round("done", stop_reason=stop_reason)
        for _ in range(5 if stop_reason is StopReason.MAX_TOKENS else 1)
    ]
    return (
        AgentLoop(
            store=store,
            provider=FakeProvider(provider_rounds),
            context=ContextManager(store, model="fake-model"),
            tools=ToolRegistry(),
            executor=NoOpExecutor(),
            duplicates=RecordingDuplicates(),
            memory_service=service,
            memory_coordinator=coordinator,
            memory_eligible=memory_eligible,
        ),
        store,
        service,
    )


@pytest.mark.asyncio
async def test_completed_turn_submits_memory_without_waiting_for_extraction(
    tmp_path: Path,
) -> None:
    coordinator = RecordingCoordinator()
    loop, store, service = await _build_loop(tmp_path, coordinator=coordinator)
    try:
        result = await loop.run_turn("session-1", "以后回答先说喵")
    finally:
        await store.close()

    assert result.status == "completed"
    assert len(coordinator.submissions) == 1
    assert coordinator.submissions[0].service is service
    assert coordinator.submissions[0].session_id == "session-1"
    assert [item.role for item in coordinator.submissions[0].messages] == [
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_eligible", "stop_reason"),
    [
        (True, StopReason.MAX_TOKENS),
        (False, StopReason.END_TURN),
    ],
)
async def test_non_completed_or_non_lead_turn_does_not_submit_memory(
    tmp_path: Path,
    memory_eligible: bool,
    stop_reason: StopReason,
) -> None:
    coordinator = RecordingCoordinator()
    loop, store, _ = await _build_loop(
        tmp_path,
        coordinator=coordinator,
        memory_eligible=memory_eligible,
        stop_reason=stop_reason,
    )
    try:
        await loop.run_turn("session-1", "remember this")
    finally:
        await store.close()

    assert coordinator.submissions == []


@pytest.mark.asyncio
async def test_lead_runtime_only_root_session_submits_memory(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "root-session",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "fake-model",
        workspace_path=str(tmp_path),
    ))
    await store.create_session(SessionRecord.new(
        "child-session",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "fake-model",
        session_type="derived",
        parent_session_id="root-session",
        workspace_path=str(tmp_path),
    ))
    coordinator = RecordingCoordinator()
    service = StubMemoryService()
    provider = FakeProvider([
        _text_round("child done"),
        _text_round("root done"),
    ])

    def loop_factory(
        selected_provider: str,
        selected_model: str,
        turn: RuntimeContext,
    ) -> AgentLoop:
        assert selected_provider == "fake"
        return AgentLoop(
            store=store,
            provider=provider,
            context=ContextManager(store, model=selected_model),
            tools=ToolRegistry(),
            executor=NoOpExecutor(),
            duplicates=RecordingDuplicates(),
            memory_service=service,
            memory_coordinator=coordinator,
            memory_eligible=turn.memory_eligible,
        )

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="fake-model",
        loop_factory=loop_factory,
        trace_redactor=SecretRedactor.with_values(()),
    )
    try:
        child_result = await runtime.resume("child-session", "child turn")
        root_result = await runtime.resume("root-session", "root turn")
    finally:
        await runtime.close()
        await store.close()

    assert child_result.status == "completed"
    assert root_result.status == "completed"
    assert [item.session_id for item in coordinator.submissions] == [
        "root-session",
    ]


def _paths(tmp_path: Path) -> AppPaths:
    user_dir = tmp_path / ".litecoder"
    return AppPaths(
        user_dir=user_dir,
        sessions_db=user_dir / "sessions.db",
        project_id="project-1",
        project_dir=user_dir / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


async def _build_runtime(
    tmp_path: Path,
    provider: FakeProvider,
    memory_store: MemoryStore,
) -> tuple[AgentRuntime, SQLiteSessionStore]:
    paths = _paths(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    coordinator = MemoryCoordinator(timeout=0.5, close_timeout=0.5)
    redactor = SecretRedactor.with_values(())
    service = MemoryService(memory_store, provider, "fake-model", redactor)

    def loop_factory(
        selected_provider: str,
        selected_model: str,
        turn: RuntimeContext,
    ) -> AgentLoop:
        assert selected_provider == "fake"
        return AgentLoop(
            store=store,
            provider=provider,
            context=ContextManager(
                store,
                model=selected_model,
                memory_service=service,
            ),
            tools=ToolRegistry(),
            executor=NoOpExecutor(),
            duplicates=RecordingDuplicates(),
            memory_service=service,
            memory_coordinator=coordinator,
            memory_eligible=turn.memory_eligible,
            redactor=turn.redactor,
        )

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="fake-model",
        loop_factory=loop_factory,
        trace_redactor=redactor,
        id_factory=lambda: "session-1",
        closeables=(coordinator,),
    )
    return runtime, store


@pytest.mark.asyncio
async def test_runtime_close_drains_real_memory_job(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / ".memory")
    provider = FakeProvider([
        _text_round("noted"),
        _text_round(json.dumps([{
            "name": "reply-style",
            "type": "user",
            "description": "Stable user reply preference",
            "body": "Start replies with meow.",
        }])),
    ])
    runtime, _ = await _build_runtime(tmp_path, provider, memory_store)

    await runtime.run("以后回答先说喵")
    await runtime.close()

    assert memory_store.read("reply-style").body == "Start replies with meow."


@pytest.mark.asyncio
async def test_new_session_recalls_chinese_preference_request_only(
    tmp_path: Path,
) -> None:
    memory_store = MemoryStore(tmp_path / ".memory")
    memory_store.replace_all((
        MemoryEntry(
            "reply-style",
            "Stable user reply preference",
            "user",
            "Start replies with meow.",
        ),
    ))
    provider = FakeProvider([
        _text_round("[0]"),
        _text_round("I remember."),
        _text_round("[]"),
    ])
    runtime, store = await _build_runtime(tmp_path, provider, memory_store)

    result = await runtime.run("你还记得我的回复偏好吗？")
    persisted = await store.load_context(result.session_id)
    await runtime.close()

    selection_request = json.dumps(provider.requests[0].messages, ensure_ascii=False)
    agent_request = json.dumps(provider.requests[1].messages, ensure_ascii=False)
    persisted_messages = json.dumps(
        [message.content for message in persisted.messages],
        ensure_ascii=False,
    )
    assert "你还记得我的回复偏好吗？" in selection_request
    assert "Start replies with meow." in agent_request
    assert "Start replies with meow." not in persisted_messages


@pytest.mark.asyncio
async def test_agent_loop_shows_only_recalled_memory_and_traces_background_lifecycle(
    tmp_path: Path,
) -> None:
    service = DiagnosticMemoryService(recalled=True)
    trace = RecordingTrace()
    coordinator = MemoryCoordinator(timeout=0.2, close_timeout=0.2)
    loop, store, sink = await _build_diagnostic_loop(
        tmp_path, service, coordinator, trace_recorder=trace
    )
    try:
        result = await loop.run_turn("session-1", "remember this")
        await service.extract_started.wait()
        await coordinator.close()
    finally:
        await store.close()

    assert result.status == "completed"
    diagnostics = [
        event.payload["memory"]
        for event in sink.events
        if event.type.value == "diagnostic"
    ]
    assert diagnostics == [
        {"operation": "load", "status": "recalled", "count": 1}
    ]
    assert trace.records == [
        {
            "event": "memory.lifecycle",
            "trace_id": "trace-1",
            "span_id": "root",
            "parent_span_id": None,
            "root_session_id": "session-1",
            "session_id": "session-1",
            "agent_id": "lead",
            "attributes": {
                "operation": "extract",
                "status": "partial_rejected",
                "accepted": 1,
                "rejected": 1,
                "written": 1,
            },
        },
        {
            "event": "memory.lifecycle",
            "trace_id": "trace-1",
            "span_id": "root",
            "parent_span_id": None,
            "root_session_id": "session-1",
            "session_id": "session-1",
            "agent_id": "lead",
            "attributes": {
                "operation": "dream",
                "status": "completed",
                "before": 10,
                "after": 5,
            },
        },
    ]


@pytest.mark.asyncio
async def test_agent_loop_shows_explicit_empty_extraction_and_traces_it(
    tmp_path: Path,
) -> None:
    class EmptyMemoryService(DiagnosticMemoryService):
        async def extract_memories(
            self,
            session_id: str,
            messages: Sequence[MessageRecord],
        ) -> MemoryExtractionResult:
            del session_id, messages
            self.extract_started.set()
            return MemoryExtractionResult(0, 0, 0, 0, "empty")

    service = EmptyMemoryService()
    trace = RecordingTrace()
    coordinator = MemoryCoordinator(timeout=0.2, close_timeout=0.2)
    loop, store, sink = await _build_diagnostic_loop(
        tmp_path, service, coordinator, trace_recorder=trace
    )
    try:
        result = await loop.run_turn("session-1", "remember this")
        await service.extract_started.wait()
        await coordinator.close()
    finally:
        await store.close()

    assert result.status == "completed"
    diagnostics = [
        event.payload["memory"]
        for event in sink.events
        if event.type.value == "diagnostic"
    ]
    assert diagnostics == [
        {
            "operation": "extract",
            "status": "empty",
            "accepted": 0,
            "rejected": 0,
            "written": 0,
            "visible": True,
        }
    ]
    assert trace.records == [
        {
            "event": "memory.lifecycle",
            "trace_id": "trace-1",
            "span_id": "root",
            "parent_span_id": None,
            "root_session_id": "session-1",
            "session_id": "session-1",
            "agent_id": "lead",
            "attributes": {
                "operation": "extract",
                "status": "empty",
                "accepted": 0,
                "rejected": 0,
                "written": 0,
            },
        }
    ]


@pytest.mark.asyncio
async def test_agent_loop_shows_explicit_extract_timeout_and_traces_it(
    tmp_path: Path,
) -> None:
    service = DiagnosticMemoryService(hang_extract=True)
    trace = RecordingTrace()
    coordinator = MemoryCoordinator(timeout=0.01, close_timeout=0.1)
    loop, store, sink = await _build_diagnostic_loop(
        tmp_path, service, coordinator, trace_recorder=trace
    )
    try:
        result = await loop.run_turn("session-1", "remember this")
        await service.extract_started.wait()
        await coordinator.close()
    finally:
        await store.close()

    assert result.status == "completed"
    diagnostics = [
        event.payload["memory"]
        for event in sink.events
        if event.type.value == "diagnostic"
    ]
    assert diagnostics == [
        {
            "operation": "extract",
            "status": "timeout",
            "visible": True,
        }
    ]
    assert trace.records == [
        {
            "event": "memory.lifecycle",
            "trace_id": "trace-1",
            "span_id": "root",
            "parent_span_id": None,
            "root_session_id": "session-1",
            "session_id": "session-1",
            "agent_id": "lead",
            "attributes": {"operation": "extract", "status": "timeout"},
        }
    ]
