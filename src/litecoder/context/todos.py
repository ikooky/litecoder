"""Session-scoped TODO progress management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from litecoder.providers._json import snapshot_json
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


TODO_WRITE_DESCRIPTION = (
    "Lead-only: manage the current session's TODO list for work with three or "
    "more meaningful steps, changed scope, or visible progress needs. Do not use "
    "it for a trivial one-step or purely informational request, and do not "
    "duplicate durable cross-agent tasks. Inspect only enough to make the list "
    "accurate before extended work. Keep at most one item in_progress, update it "
    "when work changes state, and reconcile it before the final response. Complete "
    "an item only when its outcome and relevant validation are finished; keep "
    "blocked or failed work incomplete and track the blocker. Remove irrelevant "
    "items. content must be an imperative action and active_form its "
    "present-continuous form."
)
TODO_WRITE_SUCCESS_TEXT = (
    "TODO list updated successfully. Continue with the current in-progress item "
    "if work remains, and update the list immediately when a tracked task changes state."
)


class TodoStatus(StrEnum):
    """Enumeration of the todo status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class TodoItem:
    """One session-scoped progress item, separate from durable TaskRecord."""

    content: str
    active_form: str
    status: TodoStatus = TodoStatus.PENDING

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _required_text(self.content, "content"))
        object.__setattr__(
            self, "active_form", _required_text(self.active_form, "active_form")
        )
        object.__setattr__(self, "status", TodoStatus(self.status))

    def to_json(self) -> dict[str, str]:
        """Convert this object to a JSON-compatible value."""
        return {
            "active_form": self.active_form,
            "content": self.content,
            "status": self.status.value,
        }

    @classmethod
    def from_json(cls, value: object) -> TodoItem:
        """Construct a value from json data."""
        if not isinstance(value, dict):
            raise ValueError("todo item is invalid")
        return cls(
            content=_required_text(value.get("content"), "content"),
            active_form=_required_text(value.get("active_form"), "active_form"),
            status=TodoStatus(value.get("status")),
        )


class TodoStore(Protocol):
    """The session-store contract for atomically replacing a TODO list."""

    async def list_todos(self, session_id: str) -> list[dict[str, object]]: ...

    async def replace_todos(
        self, session_id: str, todos: list[dict[str, str]]
    ) -> None: ...


class TodoStateError(RuntimeError):
    """Raised when the todo state error conditions occur."""
    pass


class TodoService:
    """Service providing the todo service operations."""
    def __init__(self, store: TodoStore) -> None:
        self.store = store

    async def list(self, session_id: str) -> tuple[TodoItem, ...]:
        """Return the available entries."""
        _required_text(session_id, "session_id")
        try:
            values = await self.store.list_todos(session_id)
            if not isinstance(values, list):
                raise ValueError("todo state is invalid")
            return tuple(TodoItem.from_json(value) for value in values)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise TodoStateError("session todos are unavailable") from error

    async def replace(
        self, session_id: str, todos: tuple[TodoItem, ...] | list[TodoItem]
    ) -> tuple[TodoItem, ...]:
        """Handle the replace operation."""
        _required_text(session_id, "session_id")
        if not isinstance(todos, (tuple, list)) or any(
            not isinstance(todo, TodoItem) for todo in todos
        ):
            raise ValueError("todo list is invalid")
        snapshot = tuple(todos)
        active_snapshot = () if all(
            todo.status is TodoStatus.COMPLETED for todo in snapshot
        ) else snapshot
        try:
            await self.store.replace_todos(
                session_id, [todo.to_json() for todo in active_snapshot]
            )
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise TodoStateError("session todos are unavailable") from error
        return snapshot


class TodoWriteTool:
    """Component responsible for the todo write tool."""
    spec = ToolSpec(
        "todo_write",
        TODO_WRITE_DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "active_form": {"type": "string", "minLength": 1},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "active_form", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        concurrency="exclusive",
        permission_risk="safe",
        dedupe_policy="none",
    )

    def __init__(self, service: TodoService) -> None:
        self.service = service

    def hard_guard(self, _call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return None if _is_lead(context) else "Todo updates are restricted to the lead agent"

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        if not _is_lead(context):
            raise ToolDenied("Todo updates are restricted to the lead agent")
        values = call.arguments.get("todos")
        if not isinstance(values, list):
            raise ToolFailure("Invalid tool arguments", metadata={"field": "todos"})
        try:
            previous = await self.service.list(context.agent_session_id)
            todos = tuple(TodoItem.from_json(value) for value in values)
            written = await self.service.replace(context.agent_session_id, todos)
        except ValueError:
            raise ToolFailure(
                "Invalid tool arguments", metadata={"field": "todos"}
            ) from None
        except TodoStateError:
            raise ToolFailure("Session todos are unavailable") from None
        payload = [todo.to_json() for todo in written]
        old_payload = [todo.to_json() for todo in previous]
        return ToolExecution.success(
            TODO_WRITE_SUCCESS_TEXT,
            metadata={
                "old_todos": old_payload,
                "new_todos": payload,
                "todos": payload,
                "changed_workspace": False,
            },
            preview=snapshot_json(
                {"old_todos": old_payload, "new_todos": payload}, "todos"
            ),
        )


def register_todo_tools(registry: ToolRegistry, service: TodoService) -> None:
    """Register the todo tools."""
    if not isinstance(registry, ToolRegistry):
        raise ValueError("registry is invalid")
    registry.register(TodoWriteTool(service))


def _is_lead(context: ToolContext) -> bool:
    return context.metadata.get("agent_id") == "lead"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"todo {field_name} is invalid")
    return value
