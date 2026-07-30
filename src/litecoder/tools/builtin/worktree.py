"""Built-in worktree management tool."""

from __future__ import annotations

import json

from litecoder.tasks.manager import TaskManager, TaskManagerError
from litecoder.tasks.worktrees import (
    WorktreeBinding,
    WorktreeError,
    WorktreeManager,
    validate_binding_id,
    validate_branch,
    validate_task_id,
)
from litecoder.tools.builtin._common import require_string
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


class WorktreeCreateTool:
    """Component responsible for the worktree create tool."""
    spec = ToolSpec(
        "worktree_create",
        "Create an isolated Git worktree only for a durable task whose changes need separation from the current workspace or another agent. Verify the task and branch intent first, then bind the result to that task.",
        {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "branch": {"type": "string", "minLength": 1},
            },
            "required": ["task_id", "branch"],
            "additionalProperties": False,
        },
        True,
        concurrency="exclusive",
        permission_risk="workspace",
        dedupe_policy="none",
    )

    def __init__(
        self,
        manager: WorktreeManager,
        task_manager: TaskManager | None = None,
    ) -> None:
        self.manager = manager
        self.task_manager = task_manager

    def hard_guard(self, call: ToolCall, _context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return _validation_guard(call, branch=True)

    async def execute(self, call: ToolCall, _context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            task_id = require_string(call.arguments, "task_id")
            if self.task_manager is not None:
                await _reconcile_task_bindings(self.manager, self.task_manager)
                await self.task_manager.get(task_id)
            binding = await self.manager.create(
                task_id,
                require_string(call.arguments, "branch"),
            )
        except (ValueError, TaskManagerError, WorktreeError):
            raise ToolFailure("Worktree could not be created") from None
        if self.task_manager is not None:
            try:
                await self.task_manager.bind_worktree(task_id, binding.id)
            except (ValueError, TaskManagerError):
                try:
                    await self.manager.remove(binding.id, discard=True)
                except WorktreeError:
                    raise ToolFailure(
                        "Worktree binding failed; cleanup is required",
                        metadata={
                            "binding": _binding_payload(binding),
                            "reconciliation_required": True,
                        },
                    ) from None
                raise ToolFailure("Worktree could not be bound to task") from None
        payload = _binding_payload(binding)
        rendered = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return ToolExecution.success(
            rendered, metadata=payload, changed_workspace=True, preview=payload
        )


class WorktreeListTool:
    """Component responsible for the worktree list tool."""
    spec = ToolSpec(
        "worktree_list",
        "List and reconcile task-bound Git worktrees before assigning, resuming, or cleaning isolated work.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        False,
        concurrency="shared",
        permission_risk="safe",
        dedupe_policy="none",
    )

    def __init__(
        self, manager: WorktreeManager, task_manager: TaskManager | None = None
    ) -> None:
        self.manager = manager
        self.task_manager = task_manager

    async def execute(self, _call: ToolCall, _context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            bindings = await self.manager.list()
            if self.task_manager is not None:
                await _reconcile_task_bindings(self.manager, self.task_manager)
        except (TaskManagerError, WorktreeError):
            raise ToolFailure("Worktrees could not be listed") from None
        payload = [_binding_payload(binding) for binding in bindings]
        return ToolExecution.success(
            json.dumps(payload, sort_keys=True, ensure_ascii=False), preview=payload
        )


class WorktreeRemoveTool:
    """Component responsible for the worktree remove tool."""
    spec = ToolSpec(
        "worktree_remove",
        "Remove a completed or abandoned task Git worktree only after checking its task binding and changes. This is destructive; use discard only when loss of unmerged work is intended.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "discard": {"type": "boolean"},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        True,
        concurrency="exclusive",
        permission_risk="high",
        dedupe_policy="none",
        requires_confirmation=True,
    )

    def __init__(
        self,
        manager: WorktreeManager,
        task_manager: TaskManager | None = None,
    ) -> None:
        self.manager = manager
        self.task_manager = task_manager

    def hard_guard(self, call: ToolCall, _context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return _validation_guard(call)

    async def execute(self, call: ToolCall, _context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            discard = _discard_argument(call.arguments)
            binding_id = require_string(call.arguments, "id")
            if self.task_manager is not None:
                bindings = await self.manager.list()
                await _reconcile_task_bindings(self.manager, self.task_manager)
                expected = next(
                    (item for item in bindings if item.id == binding_id),
                    None,
                )
                if expected is None:
                    raise WorktreeError("worktree binding was not found")
                task = await self.task_manager.get(expected.task_id)
                if task.worktree_id != expected.id:
                    raise TaskManagerError("task/worktree binding is inconsistent")
            binding = await self.manager.remove(binding_id, discard=discard)
            if self.task_manager is not None:
                await self.task_manager.unbind_worktree(
                    binding.task_id, binding.id
                )
        except (ValueError, TaskManagerError, WorktreeError):
            raise ToolFailure("Worktree could not be removed") from None
        return ToolExecution.success(
            "Removed task worktree",
            metadata=_binding_payload(binding),
            changed_workspace=True,
            preview=_binding_payload(binding),
        )


def register_worktree_tools(
    registry: ToolRegistry,
    manager: WorktreeManager,
    *,
    task_manager: TaskManager | None = None,
) -> None:
    """Register the worktree tools."""
    registry.register_many(
        (
            WorktreeCreateTool(manager, task_manager),
            WorktreeListTool(manager, task_manager),
            WorktreeRemoveTool(manager, task_manager),
        )
    )


def _validation_guard(call: ToolCall, *, branch: bool = False) -> str | None:
    try:
        validator = validate_task_id if branch else validate_binding_id
        validator(call.arguments.get("task_id" if branch else "id"))
        if branch:
            validate_branch(call.arguments.get("branch"))
        else:
            _discard_argument(call.arguments)
    except ValueError:
        return ToolDenied().safe_message
    return None


def _discard_argument(arguments: dict[str, object]) -> bool:
    value = arguments.get("discard", False)
    if not isinstance(value, bool):
        raise ValueError("discard must be a boolean")
    return value


def _binding_payload(binding: WorktreeBinding) -> dict[str, object]:
    return {
        "id": binding.id,
        "task_id": binding.task_id,
        "branch": binding.branch,
        "project_id": binding.project_id,
        "workspace_id": binding.workspace_id,
        "path": str(binding.path),
    }


async def _reconcile_task_bindings(
    manager: WorktreeManager, task_manager: TaskManager
) -> None:
    """Repair incomplete task/worktree compensation from a prior operation."""
    bindings = {binding.id: binding for binding in await manager.list()}
    tasks = await task_manager.list()
    for task in tasks:
        worktree_id = task.worktree_id
        if worktree_id is not None and worktree_id not in bindings:
            await task_manager.unbind_worktree(task.id, worktree_id)
    for binding in bindings.values():
        task = await task_manager.get(binding.task_id)
        if task.worktree_id is None:
            await task_manager.bind_worktree(task.id, binding.id)
