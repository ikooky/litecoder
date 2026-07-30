"""Background task and notification management."""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from litecoder.common.trace import SecretRedactor, current_secret_redactor
from litecoder.providers._json import snapshot_mapping
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


class BackgroundStatus(StrEnum):
    """Enumeration of the background status values."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class BackgroundHandle:
    """Data model representing the background handle."""
    id: str


@dataclass(frozen=True, slots=True)
class BackgroundState:
    """Data model representing the background state."""
    id: str
    status: BackgroundStatus
    metadata: dict[str, object]
    content: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeNotification:
    """Data model representing the runtime notification."""
    background_id: str
    status: BackgroundStatus
    content: str
    metadata: dict[str, object]

    @property
    def agent_session_id(self) -> str | None:
        """Handle the agent session id operation."""
        value = self.metadata.get("agent_session_id")
        return value if isinstance(value, str) and value else None

    def to_content_block(self) -> dict[str, object]:
        """Convert this object to a content block value."""
        payload = {
            "background_id": self.background_id,
            "status": self.status.value,
            "content": self.content,
            "metadata": self.metadata,
        }
        return {
            "type": "text",
            "text": "[Background notification] "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }


BackgroundRunner = Callable[
    [str, dict[str, object], ToolContext],
    Awaitable[ToolResult],
]


class BackgroundManager:
    """Manager coordinating the background manager."""
    def __init__(
        self,
        *,
        id_factory: Callable[[], str] | None = None,
        close_timeout: float = 1.0,
    ) -> None:
        if close_timeout <= 0:
            raise ValueError("close_timeout must be positive")
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.close_timeout = close_timeout
        self._tasks: dict[str, asyncio.Future[object]] = {}
        self._states: dict[str, BackgroundState] = {}
        self._redactors: dict[str, SecretRedactor] = {}
        self._notifications: list[RuntimeNotification] = []
        self._closed = False

    async def start(
        self,
        awaitable: Awaitable[object],
        metadata: dict[str, object],
    ) -> BackgroundHandle:
        """Start the managed runtime."""
        if self._closed:
            _close_awaitable(awaitable)
            raise RuntimeError("Background manager is closed")
        background_id = self.id_factory()
        if (
            not isinstance(background_id, str)
            or not background_id.strip()
            or background_id in self._tasks
        ):
            _close_awaitable(awaitable)
            raise ValueError("background id is invalid")
        safe_metadata = snapshot_mapping(metadata, "background metadata")
        future = asyncio.ensure_future(awaitable)
        self._tasks[background_id] = future
        self._states[background_id] = BackgroundState(
            background_id,
            BackgroundStatus.RUNNING,
            safe_metadata,
        )
        self._redactors[background_id] = current_secret_redactor()
        future.add_done_callback(
            lambda completed, item_id=background_id: self._finish(
                item_id, completed
            )
        )
        return BackgroundHandle(background_id)

    def status(self, background_id: str) -> BackgroundState:
        """Return the current status."""
        try:
            state = self._states[background_id]
            future = self._tasks[background_id]
        except KeyError:
            raise KeyError(
                f"Background operation {background_id!r} was not found"
            ) from None
        if state.status is BackgroundStatus.RUNNING and future.done():
            self._finish(background_id, future)
        return _state_snapshot(self._states[background_id])

    async def cancel(self, background_id: str) -> BackgroundState:
        """Cancel the pending operation."""
        state = self.status(background_id)
        future = self._tasks[background_id]
        if state.status is BackgroundStatus.RUNNING:
            future.cancel()
            done, pending = await asyncio.wait(
                {future}, timeout=self.close_timeout
            )
            for completed in done:
                self._finish(background_id, completed)
            for item in pending:
                self._mark_cancelled(background_id)
                item.add_done_callback(_consume_future)
        return self.status(background_id)

    async def drain_notifications(
        self, agent_session_id: str | None = None
    ) -> list[RuntimeNotification]:
        """Handle the drain notifications operation."""
        for background_id, future in tuple(self._tasks.items()):
            if future.done():
                self._finish(background_id, future)
        if agent_session_id is None:
            notifications = self._notifications
            self._notifications = []
            return notifications
        selected: list[RuntimeNotification] = []
        remaining: list[RuntimeNotification] = []
        for notification in self._notifications:
            if notification.agent_session_id == agent_session_id:
                selected.append(notification)
            else:
                remaining.append(notification)
        self._notifications = remaining
        return selected

    async def close(self) -> None:
        """Close the managed resource and release any lock."""
        if self._closed:
            return
        self._closed = True
        running = {
            background_id: future
            for background_id, future in self._tasks.items()
            if self._states[background_id].status is BackgroundStatus.RUNNING
        }
        for future in running.values():
            future.cancel()
        if not running:
            return
        done, pending = await asyncio.wait(
            set(running.values()), timeout=self.close_timeout
        )
        future_ids = {future: background_id for background_id, future in running.items()}
        for future in done:
            self._finish(future_ids[future], future)
        for future in pending:
            self._mark_cancelled(future_ids[future])
            future.add_done_callback(_consume_future)

    def _finish(
        self,
        background_id: str,
        completed: asyncio.Future[object],
    ) -> None:
        previous = self._states[background_id]
        if previous.status is not BackgroundStatus.RUNNING:
            return
        redactor = self._redactors.pop(
            background_id, SecretRedactor.with_values(())
        )
        metadata = dict(previous.metadata)
        owner = metadata.get("agent_session_id")
        if completed.cancelled():
            status = BackgroundStatus.CANCELLED
            content = "Background operation cancelled"
        else:
            try:
                result = completed.result()
            except BaseException as error:
                status = BackgroundStatus.FAILED
                content = "Background operation failed"
                metadata["error_type"] = type(error).__name__
            else:
                status, content, result_metadata = _result_payload(result)
                result_metadata.pop("agent_session_id", None)
                metadata.update(result_metadata)
        if isinstance(owner, str) and owner:
            metadata["agent_session_id"] = owner
        self._record_terminal_state(background_id, status, content, metadata, redactor)

    def _mark_cancelled(self, background_id: str) -> None:
        previous = self._states[background_id]
        if previous.status is not BackgroundStatus.RUNNING:
            return
        redactor = self._redactors.pop(
            background_id, SecretRedactor.with_values(())
        )
        self._record_terminal_state(
            background_id,
            BackgroundStatus.CANCELLED,
            "Background operation cancelled",
            dict(previous.metadata),
            redactor,
        )

    def _record_terminal_state(
        self,
        background_id: str,
        status: BackgroundStatus,
        content: str,
        metadata: dict[str, object],
        redactor: SecretRedactor,
    ) -> None:
        safe_content = redactor.redact_text(content)
        safe_metadata = redactor.redact_data(metadata)
        if not isinstance(safe_metadata, dict):
            safe_metadata = {}
        state = BackgroundState(
            background_id,
            status,
            safe_metadata,
            safe_content,
        )
        self._states[background_id] = state
        self._notifications.append(
            RuntimeNotification(
                background_id,
                status,
                safe_content,
                dict(safe_metadata),
            )
        )


class BackgroundStartTool:
    """Component responsible for the background start tool."""
    spec = ToolSpec(
        name="background_start",
        description="Start a registered tool in the background only when its result is independent of the current next step. Record the returned ID and do not claim completion until a completion notification or status confirms it.",
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "input": {"type": "object"},
            },
            "required": ["tool_name", "input"],
            "additionalProperties": False,
        },
        workspace_lock=False,
        mutates_workspace=False,
        permission_risk="safe",
        dedupe_policy="none",
    )

    def __init__(
        self,
        manager: BackgroundManager,
        runner: BackgroundRunner,
    ) -> None:
        self.manager = manager
        self.runner = runner

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        tool_name = call.arguments.get("tool_name")
        arguments = call.arguments.get("input")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ToolFailure("tool_name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ToolFailure("input must be an object")
        if tool_name in _BACKGROUND_TOOL_NAMES:
            raise ToolFailure("background tools cannot start background tools")
        handle = await self.manager.start(
            self.runner(tool_name, arguments, context),
            {
                "tool_name": tool_name,
                "agent_session_id": context.agent_session_id,
            },
        )
        return ToolExecution.success(
            json.dumps(
                {"background_id": handle.id, "status": "running"},
                separators=(",", ":"),
            ),
            metadata={"background_id": handle.id},
        )


class BackgroundStatusTool:
    """Component responsible for the background status tool."""
    spec = ToolSpec(
        name="background_status",
        description="Read the state of a background operation when its result is now needed or the user asks for progress. Do not poll repeatedly without a reason.",
        input_schema={
            "type": "object",
            "properties": {"background_id": {"type": "string"}},
            "required": ["background_id"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        permission_risk="safe",
        workspace_lock=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: BackgroundManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        background_id = _background_id(call)
        state = _safe_status(self.manager, background_id)
        _require_owner(state, context)
        return ToolExecution.success(
            json.dumps(_state_payload(state), ensure_ascii=False),
            metadata={"background_id": background_id},
        )


class BackgroundCancelTool:
    """Component responsible for the background cancel tool."""
    spec = ToolSpec(
        name="background_cancel",
        description="Cancel a running background operation only when the user requests cancellation or its work is definitively obsolete; report the resulting state.",
        input_schema={
            "type": "object",
            "properties": {"background_id": {"type": "string"}},
            "required": ["background_id"],
            "additionalProperties": False,
        },
        mutates_workspace=False,
        permission_risk="safe",
        workspace_lock=False,
        dedupe_policy="none",
    )

    def __init__(self, manager: BackgroundManager) -> None:
        self.manager = manager

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        background_id = _background_id(call)
        state = _safe_status(self.manager, background_id)
        _require_owner(state, context)
        state = await self.manager.cancel(background_id)
        return ToolExecution.success(
            json.dumps(_state_payload(state), ensure_ascii=False),
            metadata={"background_id": background_id},
        )


_BACKGROUND_TOOL_NAMES = {
    "background_start",
    "background_status",
    "background_cancel",
}


def register_background_tools(
    registry: ToolRegistry,
    manager: BackgroundManager,
    runner: BackgroundRunner,
) -> None:
    """Register the background tools."""
    registry.register(BackgroundStartTool(manager, runner))
    registry.register(BackgroundStatusTool(manager))
    registry.register(BackgroundCancelTool(manager))


def _background_id(call: ToolCall) -> str:
    value = call.arguments.get("background_id")
    if not isinstance(value, str) or not value.strip():
        raise ToolFailure("background_id must be a non-empty string")
    return value


def _safe_status(manager: BackgroundManager, background_id: str) -> BackgroundState:
    try:
        return manager.status(background_id)
    except KeyError:
        raise ToolFailure("Background operation was not found") from None


def _require_owner(state: BackgroundState, context: ToolContext) -> None:
    owner = state.metadata.get("agent_session_id")
    if owner != context.agent_session_id:
        raise ToolFailure("Background operation is not owned by this session")


def _state_payload(state: BackgroundState) -> dict[str, object]:
    return {
        "background_id": state.id,
        "status": state.status.value,
        "content": state.content,
        "metadata": dict(state.metadata),
    }


def _state_snapshot(state: BackgroundState) -> BackgroundState:
    return BackgroundState(
        state.id,
        state.status,
        dict(state.metadata),
        state.content,
    )


def _result_payload(
    result: object,
) -> tuple[BackgroundStatus, str, dict[str, object]]:
    if isinstance(result, ToolResult):
        status = (
            BackgroundStatus.COMPLETED
            if result.status == "success"
            else BackgroundStatus.FAILED
        )
        metadata = dict(result.metadata)
        metadata["tool_result_status"] = result.status
        return status, result.content, metadata
    return BackgroundStatus.COMPLETED, str(result), {}


def _consume_future(future: asyncio.Future[object]) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except asyncio.CancelledError:
        pass


def _close_awaitable(awaitable: Awaitable[object]) -> None:
    if inspect.iscoroutine(awaitable):
        awaitable.close()
