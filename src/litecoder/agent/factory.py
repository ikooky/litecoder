"""Factories for constructing isolated agent runtimes."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol
from litecoder.paths import AppPaths
from litecoder.tools.duplicate_guard import DuplicateGuard
from litecoder.tools.executor import ToolExecutor
from litecoder.tools.models import ToolCall, ToolContext, ToolResult
from litecoder.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from litecoder.tasks.manager import TaskManager
    from litecoder.tasks.worktrees import WorktreeManager
    from litecoder.agent.runtime import AgentRuntime, RuntimeContext
    from litecoder.tasks.subagents import ChildAgentRequest


class AgentRuntimeFactory(Protocol):
    """Protocol describing the agent runtime factory behavior."""
    async def create_child(
        self, request: ChildAgentRequest
    ) -> AgentRuntime: ...


class _ChildExecutor:
    """Internal helper for the child executor."""
    def __init__(
        self,
        executor: object,
        *,
        registry: ToolRegistry,
        allowed_tools: frozenset[str],
        max_calls: int,
        task_manager: TaskManager | None,
        task_id: str | None,
        agent_id: str,
    ) -> None:
        self.executor = executor
        self.registry = registry
        self.allowed_tools = allowed_tools
        self.remaining = max_calls
        self.task_manager = task_manager
        self.task_id = task_id
        self.agent_id = agent_id
        self.lock = asyncio.Lock()

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute the requested tool call."""
        if call.name not in self.allowed_tools:
            return ToolResult(
                call.id, "denied", "Tool is outside delegated authority"
            )
        try:
            tool = self.registry.require(call.name)
        except KeyError:
            return ToolResult(call.id, "unknown_tool", "Unknown tool")
        if tool.spec.mutates_workspace:
            denial = await self._workspace_mutation_denial()
            if denial is not None:
                code, reason = denial
                return ToolResult(
                    call.id,
                    "denied",
                    reason,
                    metadata={
                        "stage": "workspace_mutation",
                        "code": code,
                        "task_id": self.task_id or "",
                        "agent_id": self.agent_id,
                    },
                )
        async with self.lock:
            if self.remaining <= 0:
                return ToolResult(
                    call.id, "denied", "Delegated tool-call budget exhausted"
                )
            self.remaining -= 1
        execute = getattr(self.executor, "execute", None)
        if not callable(execute):
            return ToolResult(call.id, "failed", "Child executor is unavailable")
        return await execute(call, context)

    async def _workspace_mutation_denial(self) -> tuple[str, str] | None:
        if self.task_id is None:
            return (
                "missing_delegated_task",
                "Workspace mutation requires a delegated task and worktree",
            )
        if self.task_manager is None:
            return (
                "task_state_unavailable",
                "Delegated task ownership cannot be verified",
            )
        try:
            task = await self.task_manager.get(self.task_id)
        except Exception:
            return (
                "task_state_unavailable",
                "Delegated task ownership cannot be verified",
            )
        status = getattr(task.status, "value", task.status)
        if task.owner_agent_id != self.agent_id or status != "in_progress":
            return "task_not_assigned", (
                f"Delegated task {self.task_id!r} is not active for this agent"
            )
        return None


class DefaultAgentRuntimeFactory:
    """Build a bounded, isolated child runtime from an active lead runtime."""

    def __init__(
        self,
        parent_runtime: AgentRuntime,
        *,
        worktrees: WorktreeManager | None = None,
        task_manager: TaskManager | None = None,
        parent_session_resolver: Callable[[], str | None] | None = None,
        lease_resolver: Callable[[], object | None] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.parent_runtime = parent_runtime
        self.worktrees = worktrees
        self.task_manager = task_manager
        self.parent_session_resolver = parent_session_resolver
        self.lease_resolver = lease_resolver
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)

    async def create_child(self, request: ChildAgentRequest) -> AgentRuntime:
        """Create the child."""
        parent_session_id = self._parent_session_id()
        paths = await self._child_paths(request)
        child_session_id = self.id_factory()
        if not isinstance(child_session_id, str) or not child_session_id.strip():
            raise RuntimeError("child session id factory returned an invalid value")
        from litecoder.agent.runtime import AgentRuntime

        runtime = AgentRuntime(
            store=self.parent_runtime.store,
            paths=paths,
            provider_name=self.parent_runtime.provider_name,
            model=self.parent_runtime.model,
            loop_factory=lambda provider, model, turn: self._child_loop(
                provider, model, turn, request
            ),
            id_factory=lambda: child_session_id,
            trace_redactor=self.parent_runtime.trace_redactor,
            secret_environment_names=self.parent_runtime.secret_environment_names,
            secret_values=self.parent_runtime.secret_values,
            session_type="child",
            parent_session_id=parent_session_id,
            agent_id=request.agent_id or child_session_id,
            parent_permission_broker=self.parent_runtime.parent_permission_broker,
            permission_mode=self.parent_runtime.current_permission_mode(),
            root_turn_lease=(
                self.lease_resolver() if self.lease_resolver is not None else None
            ),
            owns_store=False,
            declared_session_id=child_session_id,
        )
        return runtime

    def _parent_session_id(self) -> str:
        from litecoder.tasks.subagents import AgentCreationDenied

        resolver = self.parent_session_resolver
        session_id = resolver() if resolver is not None else getattr(
            self.parent_runtime, "active_session_id", None
        )
        if not isinstance(session_id, str) or not session_id.strip():
            raise AgentCreationDenied(
                "child runtimes may only be created during an active lead turn"
            )
        return session_id

    async def _child_paths(self, request: ChildAgentRequest) -> AppPaths:
        from litecoder.tasks.subagents import AgentCreationDenied

        if request.worktree_id is None:
            if request.authority.workspace_id != self.parent_runtime.paths.workspace_id:
                raise AgentCreationDenied("requested workspace is not delegated")
            return self.parent_runtime.paths
        if self.worktrees is None:
            raise AgentCreationDenied("worktree delegation is not configured")
        bindings = await self.worktrees.list()
        binding = next(
            (item for item in bindings if item.id == request.worktree_id), None
        )
        if binding is None:
            raise AgentCreationDenied("worktree binding is not present in Git truth")
        if request.task_id != binding.task_id:
            raise AgentCreationDenied("worktree does not belong to delegated task")
        if request.authority.workspace_id != binding.workspace_id:
            raise AgentCreationDenied("requested workspace is not delegated")
        return replace(
            self.parent_runtime.paths,
            workspace_id=binding.workspace_id,
            workspace_root=binding.workspace_root,
        )

    def _child_loop(
        self,
        provider: str,
        model: str,
        turn: RuntimeContext,
        request: ChildAgentRequest,
    ) -> object:
        loop = self.parent_runtime.loop_factory(provider, model, turn)
        delegated_task_ids = frozenset(request.authority.task_ids)
        loop.delegated_task_ids = delegated_task_ids
        context = getattr(loop, "context", None)
        if context is not None:
            context.prompt_task_ids = delegated_task_ids
            from litecoder.tasks.subagents import profile_instructions
            context.agent_instructions = profile_instructions(request.profile)
        from litecoder.agent.loop import RuntimeBudgets

        registry = ToolRegistry()
        for tool in loop.tools.list():
            if tool.spec.name in request.authority.tools:
                registry.register(tool)
        loop.tools = registry
        duplicates = DuplicateGuard()
        base_executor = loop.executor
        if isinstance(base_executor, ToolExecutor):
            base_executor = base_executor.fork(
                registry=registry,
                duplicates=duplicates,
            )
        loop.executor = _ChildExecutor(
            base_executor,
            registry=registry,
            allowed_tools=request.authority.tools,
            max_calls=request.authority.max_tool_calls,
            task_manager=self.task_manager,
            task_id=request.task_id,
            agent_id=request.agent_id or getattr(turn, "agent_id", "child"),
        )
        loop.duplicates = duplicates
        loop.budgets = RuntimeBudgets(
            max_rounds=request.authority.max_rounds,
            max_tokens=loop.budgets.max_tokens,
        )
        return loop


__all__ = ["AgentRuntimeFactory", "DefaultAgentRuntimeFactory"]
