"""Task-management tool adapters."""

from __future__ import annotations

import json

from litecoder.tasks.manager import (
    InvalidTaskTransition,
    TaskAlreadyExists,
    TaskBlocked,
    TaskManager,
    TaskManagerError,
    TaskNotFound,
    TaskOwnershipError,
)
from litecoder.tasks.models import TaskCreate, TaskRecord
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


_TASK_ID_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 128}


class TaskCreateTool:
    """Component responsible for the task create tool."""
    spec = ToolSpec(
        "task_create",
        "Lead-only: create one durable task only for cross-agent coordination, dependencies, worktree binding, or recovery across turns. Use a specific outcome, implementation context, and real dependencies; do not duplicate an ordinary TodoWrite item.",
        {
            "type": "object",
            "properties": {
                "id": _TASK_ID_SCHEMA,
                "subject": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "dependencies": {"type": "array", "items": _TASK_ID_SCHEMA},
            },
            "required": ["id", "subject", "description"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        concurrency="exclusive",
        permission_risk="safe",
        dedupe_policy="none",
    )

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    def hard_guard(self, _call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return None if _is_lead(context) else "Task creation is not delegated to child agents"

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        if not _is_lead(context):
            raise ToolDenied("Task creation is not delegated to child agents")
        dependencies = call.arguments.get("dependencies", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(value, str) for value in dependencies
        ):
            raise _invalid_arguments("dependencies")
        try:
            request = TaskCreate(
                _text_argument(call, "id"),
                _text_argument(call, "subject"),
                _text_argument(call, "description"),
                dependencies=tuple(dependencies),
            )
        except ValueError:
            raise _invalid_arguments() from None
        try:
            created = await self.manager.create(request)
        except TaskAlreadyExists:
            raise ToolFailure("Task already exists") from None
        except TaskManagerError:
            raise ToolFailure("Task could not be created") from None
        return _task_success("Created task.", created)


class TaskListTool:
    """Component responsible for the task list tool."""
    spec = ToolSpec(
        "task_list",
        "List durable project tasks to coordinate dependencies, choose delegated work, or reconcile task state. Use TodoWrite instead for ordinary single-agent progress.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        mutates_workspace=False,
        concurrency="shared",
        permission_risk="safe",
        dedupe_policy="none",
    )

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    async def execute(self, _call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            tasks = await self.manager.list()
            if not _is_lead(context):
                delegated = _delegated_task_ids(context)
                tasks = tuple(task for task in tasks if task.id in delegated)
        except (ValueError, TaskManagerError):
            raise ToolFailure("Tasks are unavailable") from None
        payload = [_task_payload(task) for task in tasks]
        return ToolExecution.success(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            metadata={"tasks": payload, "count": len(payload)},
            preview=payload,
        )


class TaskGetTool:
    """Component responsible for the task get tool."""
    spec = ToolSpec(
        "task_get",
        "Read the latest state of one durable project task before transitioning, delegating, or relying on its ownership or dependencies.",
        {
            "type": "object",
            "properties": {"id": _TASK_ID_SCHEMA},
            "required": ["id"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        concurrency="shared",
        permission_risk="safe",
        dedupe_policy="none",
    )

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            task_id = _text_argument(call, "id")
            _require_task_access(context, task_id)
            task = await self.manager.get(task_id)
        except TaskNotFound:
            raise ToolFailure("Task is unavailable") from None
        except TaskManagerError:
            raise ToolFailure("Tasks are unavailable") from None
        except ValueError:
            raise _invalid_arguments("id") from None
        return _task_success("Read task.", task)


class _TaskTransitionTool:
    """Internal helper for the task transition tool."""
    action = ""
    description = ""

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            task_id = _text_argument(call, "id")
            _require_task_access(context, task_id)
            operation = getattr(self.manager, self.action)
            updated = await operation(task_id, _agent_id(context))
        except TaskNotFound:
            raise ToolFailure("Task is unavailable") from None
        except TaskBlocked as error:
            raise ToolFailure(
                "Task is blocked", metadata={"blocking_task_ids": error.blocking_ids}
            ) from None
        except TaskOwnershipError:
            raise ToolFailure("Task is not owned by this agent") from None
        except InvalidTaskTransition:
            raise ToolFailure("Task transition is invalid") from None
        except TaskManagerError:
            raise ToolFailure("Task could not be updated") from None
        except ValueError:
            raise _invalid_arguments("id") from None
        return _task_success(self.description, updated)


class TaskCompleteTool(_TaskTransitionTool):
    """Component responsible for the task complete tool."""
    action = "complete"
    description = "Completed task."
    spec = ToolSpec(
        "task_complete", "Mark the current agent's durable task completed only after its requested outcome and relevant validation are finished; send the evidence to the lead when delegated and report unresolved work instead.",
        {"type": "object", "properties": {"id": _TASK_ID_SCHEMA}, "required": ["id"], "additionalProperties": False},
        False, concurrency="exclusive", permission_risk="safe", dedupe_policy="none",
    )


class TaskFailTool(_TaskTransitionTool):
    """Component responsible for the task fail tool."""
    action = "fail"
    description = "Marked task failed."
    spec = ToolSpec(
        "task_fail", "Mark the current agent's durable task failed when a blocker, failed validation, or missing authority prevents completion. Include the confirmed blocker or validation evidence and preserve the failure for recovery instead of marking success.",
        {"type": "object", "properties": {"id": _TASK_ID_SCHEMA}, "required": ["id"], "additionalProperties": False},
        False, concurrency="exclusive", permission_risk="safe", dedupe_policy="none",
    )


class TaskCancelTool(_TaskTransitionTool):
    """Component responsible for the task cancel tool."""
    action = "cancel"
    description = "Cancelled task."
    spec = ToolSpec(
        "task_cancel", "Cancel only an unowned durable task or the current agent's task when it is superseded or no longer needed; do not erase active teammate work.",
        {"type": "object", "properties": {"id": _TASK_ID_SCHEMA}, "required": ["id"], "additionalProperties": False},
        False, concurrency="exclusive", permission_risk="safe", dedupe_policy="none",
    )


def register_task_tools(registry: ToolRegistry, manager: TaskManager) -> None:
    """Register the task tools."""
    if not isinstance(registry, ToolRegistry):
        raise ValueError("registry is invalid")
    registry.register_many((
        TaskCreateTool(manager), TaskListTool(manager), TaskGetTool(manager),
        TaskCompleteTool(manager), TaskFailTool(manager),
        TaskCancelTool(manager),
    ))


def _task_success(content: str, task: TaskRecord) -> ToolExecution:
    payload = task.to_json()
    return ToolExecution.success(
        content,
        metadata={"task": payload, "changed_workspace": False},
        preview=payload,
    )


def _task_payload(task: TaskRecord) -> dict[str, object]:
    return task.to_json()


def _text_argument(call: ToolCall, name: str) -> str:
    value = call.arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name)
    return value


def _is_lead(context: ToolContext) -> bool:
    return context.metadata.get("agent_id") == "lead"


def _delegated_task_ids(context: ToolContext) -> frozenset[str]:
    raw = context.metadata.get("task_ids", ())
    if not isinstance(raw, (list, tuple, set, frozenset)) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        return frozenset()
    return frozenset(raw)


def _require_task_access(context: ToolContext, task_id: str) -> None:
    if _is_lead(context):
        return
    if task_id not in _delegated_task_ids(context):
        raise ToolDenied("Task is outside delegated authority")

def _agent_id(context: ToolContext) -> str:
    configured = context.metadata.get("agent_id")
    if isinstance(configured, str) and configured.strip():
        return configured
    return context.agent_session_id


def _invalid_arguments(field: str | None = None) -> ToolFailure:
    metadata = {} if field is None else {"field": field}
    return ToolFailure("Invalid tool arguments", metadata=metadata)
