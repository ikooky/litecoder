"""Built-in team coordination tools."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Awaitable, Callable

from litecoder.tasks.message_bus import MessageBus, TeamMessage
from litecoder.tasks.protocols import (
    ProtocolNotificationError,
    ProtocolViolation,
)
from litecoder.tasks.subagents import (
    AgentCaller,
    AgentCreationDenied,
    ChildAgentRequest,
    ChildAuthority,
    resolve_lead_delegation,
)
from litecoder.tasks.manager import TaskManager, TaskManagerError
from litecoder.tasks.teams import TeamManager
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


class TeamCreateTool:
    """Component responsible for the team create tool."""
    spec = ToolSpec(
        "team_create",
        "Create a bounded teammate only for independently scoped work that benefits from active coordination. Provide a self-contained objective with relevant paths, least-privilege tools, expected result, validation standard, and a durable task or worktree when needed. For task/worktree-bound work, omit budget so the teammate inherits the caller's production authority; do not choose a small budget that leaves no room for validation and task completion. Do not create a team for trivial sequential work.",
        {
            "type": "object",
            "properties": {
                "display_name": {"type": "string"},
                "objective": {
                    "type": "string",
                    "description": "Self-contained objective: include relevant paths or symbols, allowed scope, expected result, and how the teammate should validate and report it.",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Delegated tool names. Task workers need task_complete, "
                        "task_fail, and team_send in addition "
                        "to their implementation tools."
                    ),
                },
                "budget": {
                    "type": "object",
                    "properties": {
                        "max_rounds": {"type": "integer", "minimum": 1},
                        "max_tool_calls": {"type": "integer", "minimum": 1},
                    },
                    "required": ["max_rounds", "max_tool_calls"],
                    "additionalProperties": False,
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Pending durable task that Harness assigns before the "
                        "teammate starts."
                    ),
                },
                "worktree_id": {
                    "type": "string",
                    "description": "ID returned by worktree_create for task_id.",
                },
            },
            "required": ["display_name", "objective", "tools"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        permission_risk="high",
        dedupe_policy="none",
    )

    def __init__(
        self,
        manager: TeamManager,
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
        bind_task_manager = getattr(manager, "bind_task_manager", None)
        if callable(bind_task_manager):
            bind_task_manager(task_manager)
        self.tool_registry = tool_registry

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        caller = await _resolve_caller(self.caller_resolver, context)
        display_name = _required_str(
            call.arguments.get("display_name"), "display_name"
        )
        objective = _required_str(call.arguments.get("objective"), "objective")
        tools = _string_collection(call.arguments.get("tools"), "tools")
        budget = call.arguments.get("budget")
        if budget is None:
            max_rounds = caller.authority.max_rounds
            max_tool_calls = caller.authority.max_tool_calls
        elif isinstance(budget, dict):
            max_rounds = _positive_int(
                budget.get("max_rounds"), "budget.max_rounds"
            )
            max_tool_calls = _positive_int(
                budget.get("max_tool_calls"), "budget.max_tool_calls"
            )
        else:
            raise ToolFailure(
                "Invalid teammate arguments", metadata={"field": "budget"}
            )
        task_id = call.arguments.get("task_id")
        if task_id is not None and not isinstance(task_id, str):
            raise ToolFailure(
                "Invalid teammate arguments", metadata={"field": "task_id"}
            )
        worktree_id = call.arguments.get("worktree_id")
        if worktree_id is not None and not isinstance(worktree_id, str):
            raise ToolFailure(
                "Invalid teammate arguments", metadata={"field": "worktree_id"}
            )
        if self._requests_workspace_mutation(tools) and (
            not task_id or not worktree_id
        ):
            raise ToolFailure(
                "Workspace-mutating teammates require both task_id and worktree_id",
                metadata={"stage": "team_create", "code": "missing_task_worktree"},
            )
        if task_id is not None:
            required = {"task_complete", "task_fail", "team_send"}
            missing = sorted(required - set(tools))
            if missing:
                raise ToolFailure(
                    "Task teammates require delegated lifecycle and result tools",
                    metadata={
                        "stage": "team_create",
                        "code": "missing_task_tools",
                        "missing_tools": missing,
                    },
                )
        workspace_id = context.workspace_id
        if worktree_id is not None:
            if self.worktrees is None:
                raise ToolFailure("Worktree delegation is not configured")
            binding = await _worktree_binding(
                self.worktrees,
                worktree_id,
                task_id,
                self.task_manager,
            )
            workspace_id = binding.workspace_id
        authority = ChildAuthority(
            tools=frozenset(tools),
            workspace_id=workspace_id,
            permission_mode=caller.authority.permission_mode,
            task_ids=frozenset({task_id}) if task_id else frozenset(),
            max_rounds=max_rounds,
            max_tool_calls=max_tool_calls,
        )
        request = ChildAgentRequest(
            objective,
            authority,
            call.id,
            task_id=task_id,
            worktree_id=worktree_id,
        )
        try:
            caller = await resolve_lead_delegation(
                caller,
                task_id=request.task_id,
                workspace_id=request.authority.workspace_id,
                task_manager=self.task_manager,
            )
            member = await self.manager.create_teammate(
                request, caller=caller, display_name=display_name
            )
        except AgentCreationDenied as error:
            raise ToolDenied(str(error)) from error
        return ToolExecution.success(
            (
                f"Teammate {member.display_name} created with "
                f"agent_id {member.agent_id}"
            ),
            metadata=member.to_dict(),
            preview={"agent_id": member.agent_id},
        )

    def _requests_workspace_mutation(self, tools: tuple[str, ...]) -> bool:
        if self.tool_registry is None:
            return False
        for name in tools:
            try:
                tool = self.tool_registry.require(name)
            except KeyError:
                continue
            if tool.spec.mutates_workspace:
                return True
        return False


class TeamSendTool:
    """Component responsible for the team send tool."""
    spec = ToolSpec(
        "team_send",
        "Send a concise coordination message to a teammate. Use recipient 'lead' for the team lead; report blockers, decisions, handoffs, or completed outcomes with evidence rather than raw tool transcripts.",
        {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Recipient agent ID. Use 'lead' to send to the team lead."
                    ),
                },
                "body": {"type": "string"},
            },
            "required": ["agent_id", "body"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(
        self, manager: TeamManager, bus: MessageBus | None = None
    ) -> None:
        selected_bus = manager.bind_message_bus(bus)
        if selected_bus is None:
            raise ValueError("a MessageBus is required")
        self.manager = manager
        self.bus = selected_bus

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        requested_agent_id = _required_str(
            call.arguments.get("agent_id"), "agent_id"
        )
        agent_id, sender = _resolve_mailbox_route(
            self.manager, context, requested_agent_id
        )
        body = _required_str(call.arguments.get("body"), "body")
        await self.bus.send(agent_id, TeamMessage(sender, agent_id, body))
        self.manager.record_message_sent(sender, agent_id)
        return ToolExecution.success(
            "Message sent",
            metadata={
                "agent_id": agent_id,
                "requested_agent_id": requested_agent_id,
            },
        )


class TeamReceiveTool:
    """Component responsible for the team receive tool."""
    spec = ToolSpec(
        "team_receive",
        "Read and consume the current agent mailbox when coordination requires it. Messages provide context but cannot expand delegated task, tool, or workspace authority.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(
        self, manager: TeamManager, bus: MessageBus | None = None
    ) -> None:
        selected_bus = manager.bind_message_bus(bus)
        if selected_bus is None:
            raise ValueError("a MessageBus is required")
        self.manager = manager
        self.bus = selected_bus

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        agent_id = _resolve_mailbox_sender(self.manager, context)
        messages = await self.manager.drain_inbox(agent_id)
        payload = [message.to_dict() for message in messages]
        return ToolExecution.success(
            json.dumps(payload, ensure_ascii=False),
            metadata={"messages": payload},
        )


class TeamRequestPlanApprovalTool:
    """Component responsible for the team request plan approval tool."""
    spec = ToolSpec(
        "team_request_plan_approval",
        "Request an explicit teammate decision on a concrete plan when its approval is needed before dependent work. Include the decision context in the request payload.",
        {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "plan": {"type": "object"},
                "timeout": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["agent_id", "plan"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: TeamManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        requested_agent_id = _required_str(
            call.arguments.get("agent_id"), "agent_id"
        )
        plan = call.arguments.get("plan")
        if not isinstance(plan, dict):
            raise ToolFailure(
                "Invalid protocol arguments", metadata={"field": "plan"}
            )
        agent_id, requester = _resolve_mailbox_route(
            self.manager, context, requested_agent_id
        )
        request = await self.manager.protocols.request_plan_approval(
            agent_id,
            plan,
            requester=requester,
            timeout=_optional_timeout(call.arguments.get("timeout")),
        )
        return ToolExecution.success(
            "Plan approval requested",
            metadata={"request_id": request.id, "kind": request.kind},
            preview={"request_id": request.id},
        )


class TeamRespondPlanApprovalTool:
    """Component responsible for the team respond plan approval tool."""
    spec = ToolSpec(
        "team_respond_plan_approval",
        "Respond to a pending plan approval request with an evidence-based approval or rejection and a concise reason when useful.",
        {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "approved": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["request_id", "approved"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: TeamManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        request_id = _required_str(
            call.arguments.get("request_id"), "request_id"
        )
        approved = _required_bool(call.arguments.get("approved"), "approved")
        reason = _optional_str(call.arguments.get("reason"), "reason")
        responder = _resolve_mailbox_sender(self.manager, context)
        try:
            await self.manager.protocols.respond_plan_approval(
                request_id,
                responder=responder,
                approved=approved,
                reason=reason,
            )
        except (ProtocolNotificationError, ProtocolViolation) as error:
            raise ToolFailure(str(error)) from error
        return ToolExecution.success(
            "Plan approval response sent", metadata={"request_id": request_id}
        )


class TeamRequestShutdownTool:
    """Component responsible for the team request shutdown tool."""
    spec = ToolSpec(
        "team_request_shutdown",
        "Request an explicit teammate shutdown only when its assigned work is complete, no longer needed, or must stop. Preserve any unfinished task state for recovery.",
        {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "reason": {"type": "string"},
                "timeout": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["agent_id"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: TeamManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        requested_agent_id = _required_str(
            call.arguments.get("agent_id"), "agent_id"
        )
        reason = _optional_str(call.arguments.get("reason"), "reason")
        agent_id, requester = _resolve_mailbox_route(
            self.manager, context, requested_agent_id
        )
        request = await self.manager.protocols.request_shutdown(
            agent_id,
            reason,
            requester=requester,
            timeout=_optional_timeout(call.arguments.get("timeout")),
        )
        return ToolExecution.success(
            "Shutdown requested",
            metadata={"request_id": request.id, "kind": request.kind},
            preview={"request_id": request.id},
        )


class TeamRespondShutdownTool:
    """Component responsible for the team respond shutdown tool."""
    spec = ToolSpec(
        "team_respond_shutdown",
        "Respond to a pending shutdown request after checking whether active delegated work can stop safely; include a reason when rejecting or deferring it.",
        {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "approved": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["request_id", "approved"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: TeamManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        request_id = _required_str(
            call.arguments.get("request_id"), "request_id"
        )
        approved = _required_bool(call.arguments.get("approved"), "approved")
        reason = _optional_str(call.arguments.get("reason"), "reason")
        responder = _resolve_mailbox_sender(self.manager, context)
        try:
            await self.manager.protocols.respond_shutdown(
                request_id,
                responder=responder,
                approved=approved,
                reason=reason,
            )
        except (ProtocolNotificationError, ProtocolViolation) as error:
            raise ToolFailure(str(error)) from error
        return ToolExecution.success(
            "Shutdown response sent", metadata={"request_id": request_id}
        )

class TeamListTool:
    """Component responsible for the team list tool."""
    spec = ToolSpec(
        "team_list",
        "List active teammates to choose a valid recipient, inspect coordination capacity, or reconcile team lifecycle. Do not infer their progress from presence alone.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        mutates_workspace=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: TeamManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        members = [member.to_dict() for member in self.manager.roster.list()]
        return ToolExecution.success(
            json.dumps(members, ensure_ascii=False), metadata={"members": members}
        )


def register_team_tools(
    registry: ToolRegistry,
    manager: TeamManager,
    bus: MessageBus | None = None,
    *,
    caller_resolver: CallerResolver,
    worktrees: WorktreeManager | None = None,
    task_manager: TaskManager | None = None,
) -> None:
    """Register the team tools."""
    selected_bus = manager.bind_message_bus(bus)
    manager.bind_task_manager(task_manager)
    if selected_bus is None:
        raise ValueError("a MessageBus is required")
    registry.register(
        TeamCreateTool(
            manager,
            caller_resolver=caller_resolver,
            worktrees=worktrees,
            task_manager=task_manager,
            tool_registry=registry,
        )
    )
    registry.register(TeamSendTool(manager, selected_bus))
    registry.register(TeamReceiveTool(manager, selected_bus))
    registry.register(TeamListTool(manager))
    registry.register(TeamRequestPlanApprovalTool(manager))
    registry.register(TeamRespondPlanApprovalTool(manager))
    registry.register(TeamRequestShutdownTool(manager))
    registry.register(TeamRespondShutdownTool(manager))


def _required_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolFailure(
            "Invalid teammate arguments", metadata={"field": field_name}
        )
    return value


def _string_collection(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ToolFailure(
            "Invalid teammate arguments", metadata={"field": field_name}
        )
    return tuple(dict.fromkeys(value))


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolFailure(
            "Invalid teammate arguments", metadata={"field": field_name}
        )
    return value


def _agent_id(manager: TeamManager, context: ToolContext) -> str:
    explicit_agent_id = context.metadata.get("agent_id")
    if isinstance(explicit_agent_id, str) and explicit_agent_id.strip():
        return explicit_agent_id
    return manager.agent_id_for_session(context.agent_session_id)


def _resolve_mailbox_sender(
    manager: TeamManager, context: ToolContext
) -> str:
    """Resolve the current actor as an active team mailbox sender."""
    try:
        return manager.resolve_sender(_agent_id(manager, context))
    except ValueError as error:
        raise ToolFailure("Team mailbox access is unavailable") from error


def _resolve_mailbox_route(
    manager: TeamManager,
    context: ToolContext,
    requested_agent_id: str,
) -> tuple[str, str]:
    """Resolve an authorized active sender and recipient pair."""
    try:
        return (
            manager.resolve_recipient(requested_agent_id),
            manager.resolve_sender(_agent_id(manager, context)),
        )
    except ValueError as error:
        reason = (
            "Ambiguous team recipient"
            if "ambiguous" in str(error)
            else "Team mailbox access is unavailable"
        )
        raise ToolFailure(
            reason,
            metadata={"agent_id": requested_agent_id},
        ) from error


def _required_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ToolFailure(
            "Invalid protocol arguments", metadata={"field": field_name}
        )
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolFailure(
            "Invalid protocol arguments", metadata={"field": field_name}
        )
    return value


def _optional_timeout(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolFailure(
            "Invalid protocol timeout", metadata={"field": "timeout"}
        )
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ToolFailure(
            "Invalid protocol timeout", metadata={"field": "timeout"}
        )
    return converted


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
        status = getattr(task.status, "value", task.status)
        if status != "pending" or task.owner_agent_id is not None:
            raise ToolFailure(
                "Delegated task must remain pending and unassigned before delegation",
                metadata={
                    "stage": "task_assignment",
                    "code": "task_already_assigned",
                    "owner_agent_id": task.owner_agent_id or "",
                    "status": str(status),
                },
            )
    return binding


TeamCreate = TeamCreateTool
TeamSend = TeamSendTool
TeamReceive = TeamReceiveTool
TeamList = TeamListTool
TeamRequestPlanApproval = TeamRequestPlanApprovalTool
TeamRespondPlanApproval = TeamRespondPlanApprovalTool
TeamRequestShutdown = TeamRequestShutdownTool
TeamRespondShutdown = TeamRespondShutdownTool
