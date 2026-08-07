"""Subagent task coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from litecoder.agent.factory import AgentRuntimeFactory
from litecoder.agent.prompt_policy import (
    EXPLORE_SUBAGENT_INSTRUCTIONS,
    PLAN_SUBAGENT_INSTRUCTIONS,
)
from litecoder.common.trace.redaction import current_secret_redactor
from litecoder.tasks.manager import TaskManager, TaskManagerError
from litecoder.tasks.models import TaskStatus
from litecoder.hooks import HookManager, HookPoint
from litecoder.agent.result import AgentResult
from litecoder.tools.models import ToolResult


_SUBAGENT_OUTPUT_MAX_CHARS = 32_000


CallerKind = Literal["user", "lead", "child", "teammate"]
ChildProfile = Literal["explore", "plan"]

_PROFILE_TOOLS: dict[ChildProfile, frozenset[str]] = {
    "explore": frozenset({
        "read_file", "glob_files", "search_text", "git_status", "git_diff",
    }),
    "plan": frozenset({
        "read_file", "glob_files", "search_text", "git_status", "git_diff",
        "task_list", "task_get",
    }),
}

_PROFILE_INSTRUCTIONS: dict[ChildProfile, str] = {
    "explore": EXPLORE_SUBAGENT_INSTRUCTIONS,
    "plan": PLAN_SUBAGENT_INSTRUCTIONS,
}


def profile_tools(profile: ChildProfile | None) -> frozenset[str] | None:
    """Handle the profile tools operation."""
    return None if profile is None else _PROFILE_TOOLS[profile]


def profile_instructions(profile: ChildProfile | None) -> str | None:
    """Handle the profile instructions operation."""
    return None if profile is None else _PROFILE_INSTRUCTIONS[profile]



class AgentCreationDenied(PermissionError):
    """Component responsible for the agent creation denied."""
    pass


@dataclass(frozen=True, slots=True)
class ChildAuthority:
    """Data model representing the child authority."""
    tools: frozenset[str]
    workspace_id: str
    permission_mode: str
    task_ids: frozenset[str]
    max_rounds: int
    max_tool_calls: int
    task_workspaces: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", _string_set(self.tools, "tools"))
        object.__setattr__(
            self, "task_ids", _string_set(self.task_ids, "task_ids")
        )
        object.__setattr__(
            self,
            "task_workspaces",
            _string_set(self.task_workspaces, "task_workspaces"),
        )
        _non_empty(self.workspace_id, "workspace_id")
        _non_empty(self.permission_mode, "permission_mode")
        _positive(self.max_rounds, "max_rounds")
        _positive(self.max_tool_calls, "max_tool_calls")

    @classmethod
    def restrict(
        cls,
        parent: ChildAuthority,
        requested: ChildAuthority,
    ) -> ChildAuthority:
        """Restrict the child context to the allowed capabilities."""
        if (
            not requested.tools <= parent.tools
            or not requested.task_ids <= parent.task_ids
        ):
            raise AgentCreationDenied("requested authority exceeds parent")
        if (
            requested.workspace_id != parent.workspace_id
            and requested.workspace_id not in parent.task_workspaces
        ):
            raise AgentCreationDenied("requested workspace is not delegated")
        if not requested.task_workspaces <= parent.task_workspaces:
            raise AgentCreationDenied(
                "requested workspace authority exceeds parent"
            )
        if (
            requested.max_rounds > parent.max_rounds
            or requested.max_tool_calls > parent.max_tool_calls
        ):
            raise AgentCreationDenied("requested budget exceeds parent")
        if requested.permission_mode != parent.permission_mode:
            raise AgentCreationDenied("requested authority exceeds parent")
        return requested


@dataclass(frozen=True, slots=True)
class ChildAgentRequest:
    """Data model representing the child agent request."""
    objective: str
    authority: ChildAuthority
    tool_call_id: str
    task_id: str | None = None
    worktree_id: str | None = None
    profile: ChildProfile | None = None
    agent_id: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.objective, "objective")
        _non_empty(self.tool_call_id, "tool_call_id")
        if not isinstance(self.authority, ChildAuthority):
            raise ValueError("authority must be a ChildAuthority")
        if self.profile not in {None, "explore", "plan"}:
            raise ValueError("child agent profile is invalid")
        if self.task_id is not None:
            _non_empty(self.task_id, "task_id")
        if self.worktree_id is not None:
            _non_empty(self.worktree_id, "worktree_id")
        if self.agent_id is not None:
            _non_empty(self.agent_id, "agent_id")

    def with_authority(self, authority: ChildAuthority) -> ChildAgentRequest:
        """Handle the with authority operation."""
        return ChildAgentRequest(
            self.objective,
            authority,
            self.tool_call_id,
            task_id=self.task_id,
            worktree_id=self.worktree_id,
            profile=self.profile,
            agent_id=self.agent_id,
        )

    def with_agent_id(self, agent_id: str) -> ChildAgentRequest:
        """Handle the with agent id operation."""
        return ChildAgentRequest(
            self.objective,
            self.authority,
            self.tool_call_id,
            task_id=self.task_id,
            worktree_id=self.worktree_id,
            profile=self.profile,
            agent_id=agent_id,
        )


@dataclass(frozen=True, slots=True)
class AgentCaller:
    """Data model representing the agent caller."""
    kind: CallerKind
    session_id: str
    authority: ChildAuthority
    runtime: object | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"user", "lead", "child", "teammate"}:
            raise ValueError("caller kind is invalid")
        _non_empty(self.session_id, "session_id")
        if not isinstance(self.authority, ChildAuthority):
            raise ValueError("authority must be a ChildAuthority")


@dataclass(frozen=True, slots=True)
class ChildAgentHandle:
    """Data model representing the child agent handle."""
    session_id: str
    result: ToolResult
    runtime: object
    authority: ChildAuthority


class SubagentManager:
    """Manager coordinating the subagent manager."""
    def __init__(
        self,
        factory: AgentRuntimeFactory,
        *,
        hooks: HookManager | None = None,
        task_manager: TaskManager | None = None,
    ) -> None:
        self.factory = factory
        self.hooks = hooks
        self.task_manager = task_manager
        self.spawn_history: list[dict[str, object]] = []

    def bind_task_manager(
        self, task_manager: TaskManager | None = None
    ) -> TaskManager | None:
        """Bind the durable task manager used for delegated execution."""
        if (
            self.task_manager is not None
            and task_manager is not None
            and self.task_manager is not task_manager
        ):
            raise ValueError("conflicting TaskManager instances")
        if self.task_manager is None:
            self.task_manager = task_manager
        return self.task_manager

    async def spawn(
        self,
        request: ChildAgentRequest,
        *,
        caller: AgentCaller,
    ) -> ChildAgentHandle:
        """Spawn a child agent task."""
        if caller.kind not in {"user", "lead"}:
            raise AgentCreationDenied("only user or lead may create agents")
        authority = ChildAuthority.restrict(caller.authority, request.authority)
        restricted = request.with_authority(authority)
        expected_profile_tools = profile_tools(restricted.profile)
        if (
            expected_profile_tools is not None
            and restricted.authority.tools != expected_profile_tools
        ):
            raise AgentCreationDenied("child profile tools are fixed")
        if (
            restricted.task_id is not None
            and restricted.task_id not in authority.task_ids
        ):
            raise AgentCreationDenied(
                "requested task is not delegated by child authority"
            )
        if self.hooks is not None:
            outcome = await self.hooks.dispatch_pre(
                HookPoint.SUBAGENT_START, _hook_payload(restricted, caller)
            )
            if outcome.blocked:
                raise AgentCreationDenied("subagent creation was blocked by a hook")
        runtime = await self.factory.create_child(restricted)
        agent_id = _runtime_agent_id(runtime, restricted)
        task_manager = self.task_manager or getattr(runtime, "task_manager", None)
        if restricted.task_id is not None:
            if task_manager is None:
                close = getattr(runtime, "close", None)
                if callable(close):
                    await close()
                raise AgentCreationDenied("delegated task manager is unavailable")
            try:
                await task_manager.assign_and_start(restricted.task_id, agent_id)
            except (TaskManagerError, ValueError) as error:
                close = getattr(runtime, "close", None)
                if callable(close):
                    await close()
                raise AgentCreationDenied(
                    "delegated task could not be assigned"
                ) from error
        child_evidence: dict[str, object] = {
            "agent_id": agent_id,
            "task_id": restricted.task_id or "",
            "worktree_id": restricted.worktree_id or "",
            "status": "running",
            "result_returned": 0,
            "failure": "",
        }
        self.spawn_history.append(child_evidence)
        result: AgentResult | None = None
        failure: BaseException | None = None
        try:
            result = await runtime.run(restricted.objective)
            tool_result = await _agent_result_to_tool_result(
                result, restricted.tool_call_id, runtime
            )
            child_evidence.update(
                session_id=result.session_id,
                status=result.status,
                result_returned=1 if tool_result.content.strip() else 0,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                reason=result.reason,
            )
            return ChildAgentHandle(
                result.session_id, tool_result, runtime, authority
            )
        except BaseException as error:
            failure = error
            child_evidence.update(
                status="failed",
                failure=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            await _fail_unfinished_delegated_task(
                task_manager,
                restricted.task_id,
                agent_id,
            )
            try:
                if self.hooks is not None:
                    await self.hooks.dispatch_post(
                        HookPoint.SUBAGENT_STOP,
                        _stop_hook_payload(restricted, caller, result, failure),
                    )
            finally:
                close = getattr(runtime, "close", None)
                if callable(close):
                    await close()


async def _fail_unfinished_delegated_task(
    manager: object | None,
    task_id: str | None,
    agent_id: str,
) -> None:
    if manager is None or not task_id or not agent_id:
        return
    try:
        task = await manager.get(task_id)
        if (
            getattr(task, "status", None) is TaskStatus.IN_PROGRESS
            and getattr(task, "owner_agent_id", None) == agent_id
        ):
            await manager.fail(task_id, agent_id)
    except Exception:
        return


def _runtime_agent_id(runtime: object, request: ChildAgentRequest) -> str:
    """Resolve the durable execution identity before the first child turn."""
    for value in (
        getattr(runtime, "agent_id", None),
        request.agent_id,
        getattr(runtime, "session_id", None),
    ):
        if isinstance(value, str) and value.strip():
            return value
    raise AgentCreationDenied("child runtime did not provide an agent id")


async def _agent_result_to_tool_result(
    result: AgentResult,
    tool_call_id: str,
    runtime: object,
) -> ToolResult:
    metadata = {
        "session_id": result.session_id,
        "agent_status": result.status,
        "reason": result.reason,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_read_tokens": result.usage.cache_read_tokens,
            "cache_creation_tokens": result.usage.cache_creation_tokens,
            "extensions": dict(result.usage.extensions),
        },
    }
    status = "success" if result.completed else f"agent_{result.status}"
    output = await _latest_assistant_output(runtime, result.session_id)
    content = output or (
        f"Subagent {result.session_id} finished with {result.status}: {result.reason}"
    )
    return ToolResult(tool_call_id, status, content, metadata)


async def _latest_assistant_output(runtime: object, session_id: str) -> str:
    store = getattr(runtime, "store", None)
    load_context = getattr(store, "load_context", None)
    if not callable(load_context):
        return ""
    try:
        context = await load_context(session_id)
        messages = getattr(context, "messages", ())
        for message in reversed(messages):
            if getattr(message, "role", None) != "assistant":
                continue
            parts = []
            for block in getattr(message, "content", ()):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])
            rendered = "".join(parts).strip()
            if rendered:
                return current_secret_redactor().redact_text(rendered)[
                    :_SUBAGENT_OUTPUT_MAX_CHARS
                ]
    except Exception:
        return ""
    return ""


async def resolve_lead_delegation(
    caller: AgentCaller,
    *,
    task_id: str | None,
    workspace_id: str,
    task_manager: TaskManager | None,
) -> AgentCaller:
    """Extend only a trusted lead after durable task verification.

    Child callers are never widened: their authority must already have been
    propagated in ToolContext metadata by their parent runtime.
    """
    if caller.kind != "lead" or task_id is None:
        return caller
    authority = caller.authority
    workspace_known = (
        workspace_id == authority.workspace_id
        or workspace_id in authority.task_workspaces
    )
    if task_id in authority.task_ids and workspace_known:
        return caller
    if task_manager is None:
        raise AgentCreationDenied("lead task delegation is not configured")
    try:
        await task_manager.get(task_id)
    except (TaskManagerError, ValueError) as error:
        raise AgentCreationDenied("delegated task could not be verified") from error
    extended = ChildAuthority(
        tools=authority.tools,
        workspace_id=authority.workspace_id,
        permission_mode=authority.permission_mode,
        task_ids=authority.task_ids | {task_id},
        max_rounds=authority.max_rounds,
        max_tool_calls=authority.max_tool_calls,
        task_workspaces=authority.task_workspaces | {workspace_id},
    )
    return AgentCaller(
        caller.kind,
        caller.session_id,
        extended,
        runtime=caller.runtime,
    )


def _hook_payload(
    request: ChildAgentRequest, caller: AgentCaller
) -> dict[str, object]:
    return {
        "profile": request.profile,
        "objective": request.objective,
        "tool_call_id": request.tool_call_id,
        "task_id": request.task_id,
        "worktree_id": request.worktree_id,
        "caller_kind": caller.kind,
        "caller_session_id": caller.session_id,
        "authority": {
            "tools": sorted(request.authority.tools),
            "workspace_id": request.authority.workspace_id,
            "permission_mode": request.authority.permission_mode,
            "task_ids": sorted(request.authority.task_ids),
            "max_rounds": request.authority.max_rounds,
            "max_tool_calls": request.authority.max_tool_calls,
        },
    }


def _stop_hook_payload(
    request: ChildAgentRequest,
    caller: AgentCaller,
    result: AgentResult | None,
    failure: BaseException | None,
) -> dict[str, object]:
    payload = _hook_payload(request, caller)
    if result is not None:
        payload["session_id"] = result.session_id
        payload["agent_status"] = result.status
        payload["reason"] = result.reason
    elif failure is not None:
        payload["agent_status"] = "failed"
        payload["failure_type"] = type(failure).__name__
    return payload


def _string_set(value: object, field_name: str) -> frozenset[str]:
    if not isinstance(value, (set, frozenset, list, tuple)):
        raise ValueError(f"{field_name} must be a collection of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return frozenset(value)


def _non_empty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _positive(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
