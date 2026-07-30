from __future__ import annotations

import json

import pytest

from litecoder.context.prompt import PromptAssembler, PromptInputs
from litecoder.context.prompt_state import PromptStateProvider
from litecoder.context.todos import TodoItem, TodoStatus
from litecoder.tasks.models import TaskRecord


class TodoServiceDouble:
    def __init__(self, values: tuple[TodoItem, ...]) -> None:
        self.values = values

    async def list(self, _session_id: str) -> tuple[TodoItem, ...]:
        return self.values


class TaskManagerDouble:
    def __init__(self, values: list[TaskRecord]) -> None:
        self.values = values

    async def list(self) -> list[TaskRecord]:
        return self.values


class TeamMemberDouble:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return self.payload


class TeamManagerDouble:
    def __init__(self, values: list[TeamMemberDouble]) -> None:
        self.values = values

    def list(self) -> list[TeamMemberDouble]:
        return self.values


def _sections(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [json.loads(str(block["text"])) for block in blocks]


@pytest.mark.asyncio
async def test_prompt_state_is_sorted_and_detached_from_live_records() -> None:
    todos = (
        TodoItem("Second", "Working on second", TodoStatus.PENDING),
        TodoItem("First", "Working on first", TodoStatus.IN_PROGRESS),
    )
    tasks = [
        TaskRecord("z-task", "Z", "last"),
        TaskRecord("a-task", "A", "first"),
    ]
    team = [
        TeamMemberDouble({"agent_id": "worker-z", "nested": {"value": "z"}}),
        TeamMemberDouble({"agent_id": "worker-a", "nested": {"value": "a"}}),
    ]
    provider = PromptStateProvider(
        todo_service=TodoServiceDouble(todos),
        task_manager=TaskManagerDouble(tasks),
        team_manager=TeamManagerDouble(team),  # type: ignore[arg-type]
    )

    state = await provider.snapshot("session-1")
    assert [item["content"] for item in state.todos] == ["Second", "First"]
    assert [item["id"] for item in state.tasks] == ["a-task", "z-task"]
    assert [item["agent_id"] for item in state.team] == ["worker-a", "worker-z"]

    tasks[0].subject = "changed live task"
    team[0].payload["nested"] = {"value": "changed live team"}
    rendered = state.to_json()
    assert rendered["tasks"][-1]["subject"] == "Z"
    assert rendered["team"][-1]["nested"] == {"value": "z"}

    rendered["tasks"][0]["subject"] = "mutated render"
    rendered["team"][0]["nested"] = {"value": "mutated render"}
    assert state.to_json()["tasks"][0]["subject"] == "A"
    assert state.to_json()["team"][0]["nested"] == {"value": "a"}


def test_prompt_assembler_places_a_detached_todos_section_before_tasks() -> None:
    todos = [{"content": "Plan", "active_form": "Planning", "status": "pending"}]
    inputs = PromptInputs(
        identity="LiteCoder",
        runtime={},
        project_instructions=None,
        skill_catalog=[],
        memories=[],
        todos=todos,
        tasks=[{"id": "task-1"}],
        team=[],
    )

    blocks = PromptAssembler().build(inputs)
    todos[0]["content"] = "changed after assembly"
    sections = _sections(blocks)

    assert [section["name"] for section in sections] == [
        "identity", "runtime", "project_instructions", "skills", "memories",
        "todos", "tasks", "team",
    ]
    assert sections[5]["content"] == [{
        "content": "Plan", "active_form": "Planning", "status": "pending",
    }]
