from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.agent.loop import AgentLoop
from litecoder.context.manager import ContextManager
from litecoder.context.session.models import SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.providers import ProviderEvent, StopReason, ToolCallBlock
from litecoder.tools.models import ToolCall, ToolResult
from litecoder.tools.registry import ToolRegistry
from tests.fakes.provider import FakeProvider


class _Duplicates:
    async def start_user_message(self, agent_session_id: str) -> None:
        del agent_session_id


class _Executor:
    async def execute(self, call: ToolCall, context: object) -> ToolResult:
        del context
        return ToolResult(
            call.id,
            "success",
            "done",
            {"changed_workspace": True},
        )


class _Coordinator:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, ...]] = []

    def submit(self, *arguments: object) -> None:
        self.submissions.append(arguments)


def _tool_round(name: str) -> list[ProviderEvent]:
    call = ToolCallBlock("call-1", name, {})
    return [
        ProviderEvent.tool_call_completed(0, call),
        ProviderEvent.content_block_completed(
            0,
            {
                "type": "tool_call",
                "call_id": call.call_id,
                "name": call.name,
                "input": call.input,
            },
        ),
        ProviderEvent.response_completed(StopReason.TOOL_USE, "tool_use"),
    ]


def _answer_round() -> list[ProviderEvent]:
    return [
        ProviderEvent.content_block_completed(
            0,
            {"type": "text", "text": "done"},
        ),
        ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "expected_submissions"),
    [
        ("memory_update", 0),
        ("memory_delete", 0),
        ("read_file", 1),
    ],
)
async def test_successful_memory_mutation_skips_background_extraction(
    tmp_path: Path,
    tool_name: str,
    expected_submissions: int,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(
        SessionRecord.new(
            "session",
            "project",
            "workspace",
            "fake",
            "model",
            workspace_path=str(tmp_path),
        )
    )
    coordinator = _Coordinator()
    loop = AgentLoop(
        store=store,
        provider=FakeProvider([_tool_round(tool_name), _answer_round()]),
        context=ContextManager(store, model="model"),
        tools=ToolRegistry(),
        executor=_Executor(),
        duplicates=_Duplicates(),
        memory_service=object(),  # type: ignore[arg-type]
        memory_coordinator=coordinator,  # type: ignore[arg-type]
    )

    try:
        result = await loop.run_turn("session", "remember this")
    finally:
        await store.close()

    assert result.status == "completed"
    assert len(coordinator.submissions) == expected_submissions
