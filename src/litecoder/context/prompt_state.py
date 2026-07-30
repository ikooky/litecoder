"""Safe prompt-state snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litecoder.context.todos import TodoService, TodoStateError
from litecoder.providers._json import snapshot_json
from litecoder.tasks.manager import TaskManager, TaskManagerError
from litecoder.tasks.teams import TeamManager


@dataclass(frozen=True, slots=True)
class PromptState:
    """Provider-neutral, detached state intended for prompt construction only."""

    todos: list[dict[str, object]]
    tasks: list[dict[str, object]]
    team: list[dict[str, object]]

    def to_json(self) -> dict[str, list[dict[str, object]]]:
        """Convert this object to a JSON-compatible value."""
        return {
            "todos": _safe_record_list(self.todos),
            "tasks": _safe_record_list(self.tasks),
            "team": _safe_record_list(self.team),
        }


class PromptStateProvider:
    """Collect only safe snapshots; prompt building never receives live records."""

    def __init__(
        self,
        *,
        todo_service: TodoService | None = None,
        task_manager: TaskManager | None = None,
        team_manager: TeamManager | None = None,
    ) -> None:
        self.todo_service = todo_service
        self.task_manager = task_manager
        self.team_manager = team_manager

    async def snapshot(
        self,
        session_id: str,
        *,
        task_ids: frozenset[str] | None = None,
    ) -> PromptState:
        """Return an immutable snapshot of the current state."""
        todos = await self._todos(session_id)
        tasks = await self._tasks(task_ids)
        team = self._team()
        return PromptState(todos=todos, tasks=tasks, team=team)

    async def _todos(self, session_id: str) -> list[dict[str, object]]:
        if self.todo_service is None:
            return []
        try:
            values = await self.todo_service.list(session_id)
        except TodoStateError:
            return []
        return _safe_record_list([value.to_json() for value in values])

    async def _tasks(
        self, task_ids: frozenset[str] | None
    ) -> list[dict[str, object]]:
        if self.task_manager is None:
            return []
        try:
            values = await self.task_manager.list()
        except (TaskManagerError, ValueError):
            return []
        rendered = _safe_record_list([value.to_json() for value in values])
        if task_ids is not None:
            rendered = [
                value for value in rendered if value.get("id") in task_ids
            ]
        return sorted(rendered, key=lambda value: str(value.get("id", "")))

    def _team(self) -> list[dict[str, object]]:
        if self.team_manager is None:
            return []
        try:
            values = [member.to_dict() for member in self.team_manager.list()]
        except (ValueError, KeyError, TypeError, AttributeError):
            return []
        rendered = _safe_record_list(values)
        return sorted(rendered, key=lambda value: str(value.get("agent_id", "")))


def _safe_record_list(value: object) -> list[dict[str, object]]:
    """Deep-copy JSON-compatible state and discard malformed or live objects."""
    try:
        snapshot = snapshot_json(value, "prompt state")
    except ValueError:
        return []
    if not isinstance(snapshot, list) or any(
        not isinstance(item, dict) for item in snapshot
    ):
        return []
    return [dict(item) for item in snapshot if isinstance(item, dict)]