from __future__ import annotations

import copy
from pathlib import Path

import pytest

from litecoder.context.todos import (
    TODO_WRITE_SUCCESS_TEXT,
    TodoItem,
    TodoService,
    TodoStatus,
    TodoWriteTool,
)
from litecoder.tasks.store import TaskStore
from litecoder.tools.models import ToolCall, ToolContext, ToolDenied


class MemoryTodoStore:
    def __init__(self) -> None:
        self.values: dict[str, list[dict[str, object]]] = {}
        self.writes: list[tuple[str, list[dict[str, str]]]] = []

    async def list_todos(self, session_id: str) -> list[dict[str, object]]:
        return copy.deepcopy(self.values.get(session_id, []))

    async def replace_todos(
        self, session_id: str, todos: list[dict[str, str]]
    ) -> None:
        replacement = copy.deepcopy(todos)
        self.values[session_id] = replacement
        self.writes.append((session_id, replacement))


def _context(tmp_path: Path, agent_id: str) -> ToolContext:
    return ToolContext(
        "session-1",
        "workspace-1",
        tmp_path,
        metadata={"agent_id": agent_id},
    )


@pytest.mark.asyncio
async def test_todo_service_replaces_the_full_list_without_creating_tasks(
    tmp_path: Path,
) -> None:
    store = MemoryTodoStore()
    service = TodoService(store)
    task_store = TaskStore(tmp_path / "tasks")
    first = TodoItem("Plan", "Planning", TodoStatus.IN_PROGRESS)
    replacement = TodoItem("Implement", "Implementing", TodoStatus.PENDING)

    assert await service.replace("session-1", [first]) == (first,)
    assert await service.replace("session-1", [replacement]) == (replacement,)
    assert await service.list("session-1") == (replacement,)
    assert store.writes == [
        ("session-1", [first.to_json()]),
        ("session-1", [replacement.to_json()]),
    ]
    assert task_store.read_all() == []


@pytest.mark.asyncio
async def test_todo_write_is_lead_only_and_replaces_session_todos(
    tmp_path: Path,
) -> None:
    store = MemoryTodoStore()
    service = TodoService(store)
    tool = TodoWriteTool(service)
    call = ToolCall(
        "todo-1",
        "todo_write",
        {"todos": [{"content": "Plan", "active_form": "Planning", "status": "pending"}]},
    )

    worker = _context(tmp_path, "worker-1")
    assert tool.hard_guard(call, worker) == "Todo updates are restricted to the lead agent"
    with pytest.raises(ToolDenied, match="lead agent"):
        await tool.execute(call, worker)
    assert await service.list("session-1") == ()

    result = await tool.execute(call, _context(tmp_path, "lead"))
    assert result.metadata["todos"] == call.arguments["todos"]
    await tool.execute(ToolCall("todo-2", "todo_write", {"todos": []}), _context(tmp_path, "lead"))
    assert await service.list("session-1") == ()


@pytest.mark.asyncio
async def test_todo_write_returns_previous_state_and_clears_completed_list(
    tmp_path: Path,
) -> None:
    store = MemoryTodoStore()
    service = TodoService(store)
    tool = TodoWriteTool(service)
    first = ToolCall(
        "todo-1",
        "todo_write",
        {"todos": [{
            "content": "Plan",
            "active_form": "Planning",
            "status": "in_progress",
        }]},
    )
    completed = ToolCall(
        "todo-2",
        "todo_write",
        {"todos": [{
            "content": "Plan",
            "active_form": "Planning",
            "status": "completed",
        }]},
    )

    await tool.execute(first, _context(tmp_path, "lead"))
    result = await tool.execute(completed, _context(tmp_path, "lead"))

    assert result.content == TODO_WRITE_SUCCESS_TEXT
    assert result.metadata["old_todos"] == first.arguments["todos"]
    assert result.metadata["todos"] == completed.arguments["todos"]
    assert result.metadata["new_todos"] == completed.arguments["todos"]
    assert await service.list("session-1") == ()
    assert store.writes[-1] == ("session-1", [])


def test_todo_tool_description_requires_live_sequential_updates() -> None:
    description = TodoWriteTool.spec.description

    assert "three or more meaningful steps" in description
    assert "Inspect only enough" in description
    assert "at most one item in_progress" in description
    assert "update it when work changes state" in description
    assert "outcome and relevant validation" in description
