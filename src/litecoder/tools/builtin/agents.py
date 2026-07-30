"""Built-in subagent tools."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from litecoder.tasks.subagents import (
    AgentCaller,
    AgentCreationDenied,
    ChildProfile,
    ChildAgentRequest,
    ChildAuthority,
    SubagentManager,
    profile_tools,
    resolve_lead_delegation,
)
from litecoder.tasks.manager import TaskManager, TaskManagerError
from litecoder.tasks.worktrees import WorktreeManager
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


CallerResolver = Callable[[ToolContext], AgentCaller | Awaitable[AgentCaller]]


class SpawnSubagentTool:
    """Component responsible for the spawn subagent tool."""
    spec = ToolSpec(
        "spawn_subagent",
        "Delegate only a bounded investigation or genuinely independent task. Give a normal child a precise objective, relevant context, scope, explicit tools, write authority, and expected deliverable. Use explore or plan for fixed read-only work and omit tools; do not delegate simple local inspection.",
        {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "tools": {"type": "array", "items": {"type": "string"}},
                "profile": {"type": "string", "enum": ["explore", "plan"]},
                "budget": {
                    "type": "object",
                    "properties": {
                        "max_rounds": {"type": "integer", "minimum": 1},
                        "max_tool_calls": {"type": "integer", "minimum": 1},
                    },
                },
                "task_id": {"type": "string"},
                "worktree_id": {
                    "type": "string",
                    "description": "Opaque ID returned by worktree_create for task_id.",
                },
            },
            "required": ["objective", "budget"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        permission_risk="high",
        workspace_lock=False,
    )

    def __init__(
        self,
        manager: SubagentManager,
        *,
        caller_resolver: CallerResolver,
        worktrees: WorktreeManager | None = None,
        task_manager: TaskManager | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.manager = manager
        self.caller_resolver = caller_resolver
        self.worktrees = worktrees or getattr(manager.factory, "worktrees", None)
        self.task_manager = task_manager
        self.tool_registry = tool_registry

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext,
    ) -> ToolExecution:
        """Execute the requested tool call."""
        try:
            caller = await _resolve_caller(self.caller_resolver, context)
            requested = await _request_from_call(
                call,
                context,
                caller,
                self.worktrees,
                self.task_manager,
                self.tool_registry,
            )
            caller = await resolve_lead_delegation(
                caller,
                task_id=requested.task_id,
                workspace_id=requested.authority.workspace_id,
                task_manager=self.task_manager,
            )
            handle = await self.manager.spawn(requested, caller=caller)
        except AgentCreationDenied as error:
            raise ToolDenied(str(error)) from error
        except ToolFailure:
            raise
        except Exception as error:
            reason = str(error).strip()
            detail = f": {reason}" if reason else ""
            raise ToolFailure(
                f"Subagent execution failed ({type(error).__name__}){detail}",
                metadata={
                    "stage": "spawn_subagent",
                    "code": "runtime_error",
                    "failure_type": type(error).__name__,
                    "reason": reason,
                },
            ) from error
        if handle.result.status != "success":
            raise ToolFailure(
                handle.result.content,
                metadata=dict(handle.result.metadata),
            )
        return ToolExecution.success(
            handle.result.content,
            metadata=dict(handle.result.metadata),
            preview={"session_id": handle.session_id},
        )


def register_agent_tools(
    registry: ToolRegistry,
    manager: SubagentManager,
    *,
    caller_resolver: CallerResolver,
    worktrees: WorktreeManager | None = None,
    task_manager: TaskManager | None = None,
) -> None:
    """Register the agent tools."""
    registry.register(
        SpawnSubagentTool(
            manager,
            caller_resolver=caller_resolver,
            worktrees=worktrees,
            task_manager=task_manager,
            tool_registry=registry,
        )
    )


async def _request_from_call(
    call: ToolCall,
    context: ToolContext,
    caller: AgentCaller,
    worktrees: WorktreeManager | None,
    task_manager: TaskManager | None,
    tool_registry: ToolRegistry | None,
) -> ChildAgentRequest:
    objective = _required_str(call.arguments.get("objective"), "objective")
    profile = _profile(call.arguments.get("profile"))
    if profile is None:
        tools = _string_collection(call.arguments.get("tools"), "tools")
    elif "tools" in call.arguments:
        raise ToolFailure("Profile tools are fixed", metadata={"field": "tools"})
    else:
        tools = tuple(sorted(profile_tools(profile) or ()))
    budget = call.arguments.get("budget")
    if not isinstance(budget, dict):
        raise ToolFailure("Invalid subagent arguments", metadata={"field": "budget"})
    max_rounds = _positive_int(
        budget.get("max_rounds"), "budget.max_rounds"
    )
    max_tool_calls = _positive_int(
        budget.get("max_tool_calls"), "budget.max_tool_calls"
    )
    task_id = call.arguments.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        raise ToolFailure("Invalid subagent arguments", metadata={"field": "task_id"})
    worktree_id = call.arguments.get("worktree_id")
    if worktree_id is not None and not isinstance(worktree_id, str):
        raise ToolFailure("Invalid subagent arguments", metadata={"field": "worktree_id"})
    if _requests_workspace_mutation(tools, tool_registry) and (
        not task_id or not worktree_id
    ):
        raise ToolFailure(
            "Workspace-mutating subagents require both task_id and worktree_id",
            metadata={
                "stage": "spawn_subagent",
                "code": "missing_task_worktree",
            },
        )
    if task_id is not None:
        required = {"task_claim", "task_complete", "task_fail"}
        missing = sorted(required - set(tools))
        if missing:
            raise ToolFailure(
                "Task subagents require delegated lifecycle tools",
                metadata={
                    "stage": "spawn_subagent",
                    "code": "missing_task_tools",
                    "missing_tools": missing,
                },
            )
    workspace_id = context.workspace_id
    if worktree_id is not None:
        if worktrees is None:
            raise ToolFailure("Worktree delegation is not configured")
        binding = await _worktree_binding(
            worktrees, worktree_id, task_id, task_manager
        )
        workspace_id = binding.workspace_id
    task_ids = frozenset({task_id}) if isinstance(task_id, str) else frozenset()
    authority = ChildAuthority(
        tools=frozenset(tools),
        workspace_id=workspace_id,
        permission_mode=caller.authority.permission_mode,
        task_ids=task_ids,
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
    )
    return ChildAgentRequest(
        objective,
        authority,
        call.id,
        task_id=task_id,
        worktree_id=worktree_id,
        profile=profile,
    )


def _requests_workspace_mutation(
    tools: tuple[str, ...], registry: ToolRegistry | None
) -> bool:
    if registry is None:
        return False
    for name in tools:
        try:
            tool = registry.require(name)
        except KeyError:
            continue
        if tool.spec.mutates_workspace:
            return True
    return False


def _required_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolFailure("Invalid subagent arguments", metadata={"field": field_name})
    return value


def _string_collection(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ToolFailure("Invalid subagent arguments", metadata={"field": field_name})
    return tuple(dict.fromkeys(value))


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolFailure("Invalid subagent arguments", metadata={"field": field_name})
    return value


def _profile(value: object) -> ChildProfile | None:
    if value is None:
        return None
    if value == "explore":
        return "explore"
    if value == "plan":
        return "plan"
    raise ToolFailure("Invalid subagent arguments", metadata={"field": "profile"})


async def _resolve_caller(
    resolver: CallerResolver, context: ToolContext
) -> AgentCaller:
    caller = resolver(context)
    if inspect.isawaitable(caller):
        caller = await caller
    if not isinstance(caller, AgentCaller):
        raise ToolDenied("Caller authority is unavailable")
    return caller


async def _worktree_binding(
    manager: WorktreeManager,
    worktree_id: str,
    task_id: object,
    task_manager: TaskManager | None,
):
    if not isinstance(task_id, str) or not task_id:
        raise ToolFailure("A task_id is required for worktree delegation")
    try:
        bindings = await manager.list()
    except Exception as error:
        raise ToolFailure("Worktree delegation could not be verified") from error
    binding = next((item for item in bindings if item.id == worktree_id), None)
    if binding is None or binding.task_id != task_id:
        raise ToolFailure("Worktree does not belong to delegated task")
    if task_manager is not None:
        try:
            task = await task_manager.get(task_id)
        except (TaskManagerError, ValueError) as error:
            raise ToolFailure(
                "Worktree delegation could not be verified"
            ) from error
        if task.worktree_id != binding.id:
            raise ToolFailure("Task is not bound to delegated worktree")
    return binding
