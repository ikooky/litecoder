from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from litecoder.agent.loop import AgentLoop
from litecoder.context.manager import ContextManager
from litecoder.context.session.models import SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.providers.models import ProviderEvent, StopReason, Usage
from litecoder.tools.background import BackgroundManager
from litecoder.tools.models import ToolCall, ToolResult
from litecoder.tools.registry import ToolRegistry
from tests.fakes.provider import FakeProvider


class Duplicates:
    async def start_user_message(self, agent_session_id: str) -> None:
        pass


class Executor:
    async def execute(self, call: ToolCall, context: object) -> ToolResult:
        raise AssertionError("no tool call expected")


def answer_round() -> list[ProviderEvent]:
    return [
        ProviderEvent.content_block_completed(
            0, {"type": "text", "text": "ack"}
        ),
        ProviderEvent.response_completed(
            StopReason.END_TURN,
            "end_turn",
            usage=Usage(3, 1),
        ),
    ]


@pytest.mark.asyncio
async def test_background_notifications_are_injected_before_model_boundary(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        "project-1",
        "workspace-1",
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    background = BackgroundManager(id_factory=lambda: "bg-1")
    await background.start(
        asyncio.sleep(0, result="done"),
        {"tool": "run_shell", "agent_session_id": "session-1"},
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    provider = FakeProvider([answer_round()])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=ContextManager(store, model="model"),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        background=background,
    )

    result = await loop.run_turn("session-1", "check background")
    restored = await store.load_context("session-1")
    request_text = json.dumps(provider.requests[0].messages)

    assert result.status == "completed"
    assert "bg-1" in request_text
    assert "done" in request_text
    assert [message.role for message in restored.messages] == [
        "user",
        "user",
        "assistant",
    ]
    await background.close()
    await store.close()


@pytest.mark.asyncio
async def test_background_notifications_are_scoped_to_agent_session(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    for session_id in ("session-a", "session-b"):
        await store.create_session(SessionRecord.new(
            session_id,
            "project-1",
            "workspace-1",
            "fake",
            "model",
            workspace_path=str(tmp_path),
        ))
    background = BackgroundManager(id_factory=lambda: "bg-1")
    await background.start(
        asyncio.sleep(0, result="owned-by-a"),
        {"tool": "run_shell", "agent_session_id": "session-a"},
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    provider = FakeProvider([answer_round(), answer_round()])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=ContextManager(store, model="model"),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        background=background,
    )

    await loop.run_turn("session-b", "check b")
    b_request_text = json.dumps(provider.requests[0].messages)
    b_context = await store.load_context("session-b")
    await loop.run_turn("session-a", "check a")
    a_request_text = json.dumps(provider.requests[1].messages)
    a_context = await store.load_context("session-a")

    assert "owned-by-a" not in b_request_text
    assert [message.role for message in b_context.messages] == [
        "user",
        "assistant",
    ]
    assert "owned-by-a" in a_request_text
    assert [message.role for message in a_context.messages] == [
        "user",
        "user",
        "assistant",
    ]
    await background.close()
    await store.close()