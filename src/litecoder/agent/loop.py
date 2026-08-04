"""Agent turn orchestration, recovery, and tool execution."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from litecoder.agent.result import AgentResult
from litecoder.agent.stop import StopPolicy
from litecoder.agent.prompt_policy import (
    CONTINUATION_PROMPT,
    RESPONSE_REPAIR_PROMPT as _RESPONSE_REPAIR_PROMPT,
    TODO_REMINDER_TEXT,
)
from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.common.errors.classifier import ErrorClassifier
from litecoder.common.errors.recovery import (
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
)
from litecoder.common.errors.retry import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONTINUATION_MAX_ATTEMPTS,
    next_output_max_tokens,
)
from litecoder.common.trace import SecretRedactor, TraceContext, TraceSink, bind_secret_redactor
from litecoder.context.manager import ContextManager
from litecoder.context.provider_summary import ProviderContextSummarizer
from litecoder.context.session.models import MessageRecord, SessionStatus
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.hooks import HookManager, HookPoint
from litecoder.memory.coordinator import MemoryCoordinator
from litecoder.memory.extraction import is_explicit_memory_request
from litecoder.memory.service import MemoryService
from litecoder.providers.base import ModelProvider
from litecoder.providers.models import ModelRequest, StopReason, Usage
from litecoder.tools.background import BackgroundManager
from litecoder.tools.models import ToolCall, ToolContext, ToolResult
from litecoder.tools.permission import PermissionMode
from litecoder.tools.registry import ToolRegistry
from litecoder.ui.events import UIEventFactory, UIEventType
from litecoder.ui.sink import RuntimeUISink, emit_ui, flush_ui


class DuplicateWindow(Protocol):
    """Protocol describing the duplicate window behavior."""
    async def start_user_message(self, agent_session_id: str) -> None: ...


class Executor(Protocol):
    """Protocol describing the executor behavior."""
    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult: ...


RecoverySleep = Callable[[float], object]

INITIAL_OUTPUT_MAX_TOKENS = 32_000
MAX_CONTINUATIONS = MODEL_CONTINUATION_MAX_ATTEMPTS
TODO_REMINDER_TOOL_ROUNDS = 3
MAX_CONCURRENT_TOOL_CALLS = 4

_FAILED_MEMORY_EXTRACTION_STATUSES = frozenset({
    "empty",
    "provider_failed",
    "truncated",
    "malformed",
    "failed",
    "timeout",
})


def _is_failed_memory_extraction(payload: Mapping[str, object]) -> bool:
    if payload.get("operation") != "extract":
        return False
    status = payload.get("status")
    return status in _FAILED_MEMORY_EXTRACTION_STATUSES or (
        status == "partial_rejected" and payload.get("written") == 0
    )


def _consume_memory_diagnostics(
    context: object,
) -> tuple[dict[str, object], ...]:
    consume = getattr(context, "consume_memory_diagnostics", None)
    if not callable(consume):
        return ()
    return consume()


def _loaded_memory_count(context: object) -> object:
    return getattr(context, "loaded_memory_count", 0)


def _prompt_telemetry(context: object) -> dict[str, object]:
    """Return bounded prompt telemetry when the context manager provides it."""
    getter = getattr(context, "prompt_telemetry", None)
    if not callable(getter):
        return {}
    try:
        telemetry = getter()
    except Exception:
        return {}
    if not isinstance(telemetry, Mapping):
        return {}
    result: dict[str, object] = {}
    for name in (
        "durable_memory_section_tokens",
        "all_memory_tokens",
        "memory_index_tokens",
        "recalled_memory_tokens",
        "optimized_memory_tokens",
        "memory_context_tokens",
        "memory_catalog_reduction",
    ):
        value = telemetry.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
    recalled_ids = telemetry.get("memory_recalled_ids")
    if isinstance(recalled_ids, (list, tuple)):
        result["memory_recalled_ids"] = [
            item for item in recalled_ids if isinstance(item, str)
        ]
    sections = telemetry.get("prompt_section_tokens")
    if isinstance(sections, Mapping):
        result["prompt_section_tokens"] = {
            str(name): value
            for name, value in sections.items()
            if isinstance(name, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
    return result


@dataclass(frozen=True, slots=True)
class RuntimeBudgets:
    """Data model representing the runtime budgets."""
    max_rounds: int = 96
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_rounds <= 0 or (
            self.max_tokens is not None and self.max_tokens <= 0
        ):
            raise ValueError("runtime budgets must be positive")


@dataclass(slots=True)
class _TurnUsageProgress:
    """Data model representing the turn usage progress."""
    committed: Usage = field(default_factory=lambda: Usage(0, 0))
    active_round: Usage = field(default_factory=lambda: Usage(0, 0))

    def begin_round(self) -> None:
        """Handle the begin round operation."""
        self.active_round = Usage(0, 0)

    def observe_round(self, usage: Usage) -> None:
        """Handle the observe round operation."""
        self.active_round = usage

    def commit_round(self, usage: Usage) -> Usage:
        """Handle the commit round operation."""
        self.committed = _add_usage(self.committed, usage)
        self.active_round = Usage(0, 0)
        return self.committed

    def snapshot(self) -> Usage:
        """Return an immutable snapshot of the current state."""
        return _add_usage(self.committed, self.active_round)


@dataclass(frozen=True, slots=True)
class _ProviderRecoveryOutcome:
    """Data model representing the provider recovery outcome."""
    retry: bool
    terminal: tuple[str, str] | None = None
    feedback_code: str | None = None
    action: RecoveryAction | None = None


class AgentLoop:
    """Component responsible for the agent loop."""
    def __init__(
        self,
        *,
        store: SQLiteSessionStore,
        provider: ModelProvider,
        context: ContextManager,
        tools: ToolRegistry,
        executor: Executor,
        duplicates: DuplicateWindow,
        memory_service: MemoryService | None = None,
        memory_coordinator: MemoryCoordinator | None = None,
        memory_eligible: bool = True,
        background: BackgroundManager | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        team_inbox: object | None = None,
        error_classifier: ErrorClassifier | None = None,
        recovery_sleep: RecoverySleep | None = None,
        budgets: RuntimeBudgets | None = None,
        stop_policy: StopPolicy | None = None,
        ui_sink: RuntimeUISink | None = None,
        hooks: HookManager | None = None,
        trace_recorder: TraceSink | None = None,
        trace_id: str | None = None,
        root_session_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        agent_id: str = "lead",
        parent_permission_broker: object | None = None,
        permission_mode: PermissionMode | str = PermissionMode.ASK,
        permission_mode_resolver: Callable[[], PermissionMode | str] | None = None,
        redactor: SecretRedactor | None = None,
        secret_environment_names: tuple[str, ...] = (),
        secret_values: tuple[str, ...] = (),
        cleanup_timeout: float = 1.0,
        delegated_task_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.store = store
        self.provider = provider
        self.context = context
        self._configure_runtime_compaction()
        self.tools = tools
        self.executor = executor
        self.duplicates = duplicates
        self.memory_service = memory_service
        self.memory_coordinator = memory_coordinator
        self.memory_eligible = memory_eligible
        self.background = background
        self.recovery_policy = recovery_policy
        self.team_inbox = team_inbox
        self.error_classifier = error_classifier or ErrorClassifier()
        self.recovery_sleep = recovery_sleep or asyncio.sleep
        self.budgets = budgets or RuntimeBudgets()
        self.stop_policy = stop_policy or StopPolicy()
        self.ui_sink = ui_sink
        self.hooks = hooks
        self.trace_recorder = trace_recorder
        self.trace_id = trace_id
        self.root_session_id = root_session_id
        self.span_id = span_id or "root"
        self.parent_span_id = parent_span_id
        self.agent_id = agent_id or "lead"
        self.parent_permission_broker = parent_permission_broker
        self.permission_mode = _permission_mode_value(permission_mode)
        self.permission_mode_resolver = permission_mode_resolver
        self.secret_environment_names = tuple(
            dict.fromkeys(secret_environment_names)
        )
        self.secret_values = tuple(
            dict.fromkeys(value for value in secret_values if value)
        )
        if any(
            not isinstance(task_id, str) or not task_id
            for task_id in delegated_task_ids
        ):
            raise ValueError("delegated_task_ids must contain non-empty strings")
        self.delegated_task_ids = frozenset(delegated_task_ids)
        self.redactor = redactor or SecretRedactor.with_values(self.secret_values)
        if cleanup_timeout <= 0:
            raise ValueError("cleanup_timeout must be positive")
        self.cleanup_timeout = cleanup_timeout

    async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
        """Run the turn."""
        # Track usage separately from provider streaming so retries and
        # continuations cannot double-count a partially completed round.
        root_session_id = self.root_session_id or session_id
        usage_progress = _TurnUsageProgress()

        async def execute() -> AgentResult:
            try:
                return await self._run_turn(
                    session_id,
                    prompt,
                    root_session_id,
                    usage_progress,
                )
            except asyncio.CancelledError:
                await self._cancelled(session_id, usage_progress.snapshot())
                raise
            except Exception:
                await self._failed_exception(session_id, usage_progress.snapshot())
                raise

        with bind_secret_redactor(self.redactor):
            if self.trace_recorder is None:
                if self.hooks is not None:
                    raise RuntimeError("hooks require a trace recorder")
                return await execute()
            trace = TraceContext(
                trace_id=self.trace_id or root_session_id,
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                root_session_id=root_session_id,
                session_id=session_id,
                agent_id=self.agent_id,
                recorder=self.trace_recorder,
            )
            with trace.bind():
                return await execute()

    def _ui_factory(self, session_id: str, root_session_id: str) -> UIEventFactory:
        return UIEventFactory(
            session_id=session_id,
            root_session_id=root_session_id,
            trace_id=self.trace_id or root_session_id,
            span_id=self.span_id,
        )

    async def _emit_ui(
        self, factory: UIEventFactory, event_type: UIEventType, **kwargs: object
    ) -> None:
        payload = kwargs.pop("payload", None)
        request_id = kwargs.pop("request_id", None)
        tool_call_id = kwargs.pop("tool_call_id", None)
        tool_name = kwargs.pop("tool_name", None)
        event = factory.next(
            event_type,
            request_id=request_id if isinstance(request_id, str) else None,
            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
            tool_name=tool_name if isinstance(tool_name, str) else None,
            payload=payload if isinstance(payload, dict) else {},
        )
        await emit_ui(self.ui_sink, event)

    async def _record_memory_lifecycle(
        self,
        factory: UIEventFactory,
        payload: Mapping[str, object],
        *,
        visible: bool = False,
    ) -> None:
        """Record the memory lifecycle."""
        if visible and _is_failed_memory_extraction(payload):
            visible_payload = dict(payload)
            visible_payload["visible"] = True
            await self._emit_ui(
                factory,
                UIEventType.DIAGNOSTIC,
                payload={"memory": visible_payload},
            )
        recorder = self.trace_recorder
        if recorder is None:
            return
        record = {
            "event": "memory.lifecycle",
            "trace_id": factory.trace_id,
            "span_id": factory.span_id,
            "parent_span_id": self.parent_span_id,
            "root_session_id": factory.root_session_id,
            "session_id": factory.session_id,
            "agent_id": self.agent_id,
            "attributes": dict(payload),
        }
        try:
            await recorder.record(record)
        except Exception:
            return

    async def _run_turn(
        self,
        session_id: str,
        prompt: str,
        root_session_id: str,
        usage_progress: _TurnUsageProgress,
    ) -> AgentResult:
        """Run the turn."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        turn_started_at = time.monotonic()
        ui = self._ui_factory(session_id, root_session_id)
        await self._emit_ui(ui, UIEventType.TURN_STARTED, payload={"prompt": prompt})
        session_context = await self.store.load_context(session_id)
        session = session_context.session
        turn_start = len(session_context.messages)
        await self.store.mark_status(session_id, SessionStatus.ACTIVE)
        await self.duplicates.start_user_message(session_id)
        if self.hooks is not None:
            submitted = await self.hooks.dispatch_pre(
                HookPoint.USER_PROMPT_SUBMIT,
                {"session_id": session_id, "prompt": prompt},
            )
            if submitted.blocked:
                return await self._finish_with_ui(
                    session_id, "failed", "prompt blocked by hook", Usage(0, 0), ui
                )
            payload = submitted.payload
            if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
                prompt = payload["prompt"]
        await self.store.append_message(
            MessageRecord(
                session_id=session_id,
                role="user",
                content=[{"type": "text", "text": prompt}],
            )
        )
        usage = Usage(0, 0)
        status, reason = "incomplete", "round budget exhausted"
        memory_mutated = False
        output_token_limit = INITIAL_OUTPUT_MAX_TOKENS
        max_tokens_expanded = False
        continuations = 0
        reactive_compaction_attempted = False
        response_repair_feedback: str | None = None
        pending_recovery: RecoveryAction | None = None
        round_number = 1
        rounds_since_todo = 0
        final_todo_reconciliation_attempted = False
        while round_number <= self.budgets.max_rounds:
            await self._inject_background_notifications(session_id)
            await self._inject_team_messages(session_id)
            request = await self.context.build_request(session_id, self.tools)
            if response_repair_feedback is not None:
                request = _with_response_repair_feedback(request)
            for memory_event in _consume_memory_diagnostics(self.context):
                await self._emit_ui(
                    ui,
                    UIEventType.DIAGNOSTIC,
                    payload={"memory": memory_event},
                )
            request, blocked = await self._pre_model(request)
            if blocked:
                status, reason = "failed", "model call blocked by hook"
                break
            request = replace(request, max_tokens=output_token_limit)
            prompt_telemetry = _prompt_telemetry(self.context)
            requested_payload: dict[str, object] = {
                "round_number": round_number,
                "model": request.model,
                "memory_count": _loaded_memory_count(self.context),
            }
            requested_payload.update(prompt_telemetry)
            await self._emit_ui(
                ui,
                UIEventType.MODEL_REQUESTED,
                payload=requested_payload,
            )
            usage_progress.begin_round()
            (
                blocks,
                stop_reason,
                raw_reason,
                round_usage,
                provider_error,
                provider_request_id,
            ) = await self._collect(request, ui, usage_progress)
            retry_after_recovery = False
            if (
                provider_error is None
                and stop_reason is StopReason.CONTEXT_EXHAUSTED
            ):
                provider_error = LiteCoderError(
                    ErrorCode.CONTEXT_OVERFLOW,
                    "Provider context window exhausted",
                )
            if provider_error is None:
                provider_error = _invalid_completed_response_error(
                    blocks, stop_reason, raw_reason
                )
            if provider_error is None and pending_recovery is not None:
                await self._record_recovery_status(
                    pending_recovery,
                    status="recovered",
                    request_id=provider_request_id,
                )
                pending_recovery = None
                response_repair_feedback = None
            terminal_recovery: tuple[str, str] | None = None
            if provider_error is not None:
                recovery = await self._recover_provider_error(
                    session_id,
                    provider_error,
                    ui=ui,
                    request_id=provider_request_id,
                    has_attempted_reactive_compaction=reactive_compaction_attempted,
                )
                if provider_error.code is ErrorCode.CONTEXT_OVERFLOW:
                    reactive_compaction_attempted = True
                if recovery.retry:
                    retry_after_recovery = True
                    pending_recovery = recovery.action
                    if recovery.feedback_code is not None:
                        response_repair_feedback = recovery.feedback_code
                else:
                    terminal_recovery = recovery.terminal
            await flush_ui(self.ui_sink)
            usage = usage_progress.commit_round(round_usage)
            if retry_after_recovery:
                await self._post_model_call(
                    session_id, blocks, stop_reason, round_usage, provider_error
                )
                continue
            if (
                stop_reason is StopReason.MAX_TOKENS
                and terminal_recovery is None
            ):
                assistant_message = MessageRecord(
                    session_id=session_id,
                    role="assistant",
                    content=blocks,
                    token_count=round_usage.output_tokens,
                )
                if _token_budget_exhausted(usage, self.budgets.max_tokens):
                    await self.store.append_message(assistant_message)
                    await self._post_model_call(
                        session_id, blocks, stop_reason, round_usage, provider_error
                    )
                    status, reason = "incomplete", "token budget exhausted"
                    break
                if continuations >= MAX_CONTINUATIONS:
                    await self.store.append_message(assistant_message)
                    await self._post_model_call(
                        session_id, blocks, stop_reason, round_usage, provider_error
                    )
                    status, reason = "incomplete", "continuation budget exhausted"
                    break
                if not max_tokens_expanded:
                    output_token_limit = next_output_max_tokens(
                        output_token_limit,
                        cap=DEFAULT_MAX_OUTPUT_TOKENS,
                    )
                    max_tokens_expanded = True
                continuations += 1
                await self.store.append_messages(
                    [assistant_message, _continuation_message(session_id)]
                )
                await self._post_model_call(
                    session_id, blocks, stop_reason, round_usage, provider_error
                )
                continue
            if terminal_recovery is not None:
                await self._post_model_call(
                    session_id, blocks, stop_reason, round_usage, provider_error
                )
                status, reason = terminal_recovery
                break
            await self.store.append_message(
                MessageRecord(
                    session_id=session_id,
                    role="assistant",
                    content=blocks,
                    token_count=round_usage.output_tokens,
                )
            )
            await self._post_model_call(
                session_id, blocks, stop_reason, round_usage, provider_error
            )
            outcome = self.stop_policy.decide(stop_reason, raw=raw_reason)
            status, reason = outcome.status, raw_reason or stop_reason.value
            if _token_budget_exhausted(usage, self.budgets.max_tokens):
                status, reason = "incomplete", "token budget exhausted"
                break
            if status == "completed":
                todo_items = await self._todo_reminder_items(session_id)
                if (
                    not final_todo_reconciliation_attempted
                    and round_number < self.budgets.max_rounds
                    and _has_open_todos(todo_items)
                ):
                    final_todo_reconciliation_attempted = True
                    await self.store.append_message(
                        _todo_reminder_message(session_id, todo_items)
                    )
                    round_number += 1
                    continue
            if status not in {"continue_tools", "continue_provider"}:
                break
            if outcome.consumes_continuation:
                continuations += 1
                if continuations > MAX_CONTINUATIONS:
                    status, reason = "incomplete", "continuation budget exhausted"
                    break
            if status == "continue_provider":
                continue
            calls = _tool_calls(blocks)
            if not calls:
                status, reason = "failed", "tool_use without tool calls"
                break
            permission_mode = self._current_permission_mode()
            glob_batch_size = sum(call.name == "glob_files" for call in calls)
            tool_context = ToolContext(
                agent_session_id=session_id,
                workspace_id=session.workspace_id,
                workspace_root=Path(session.workspace_path),
                metadata=_tool_metadata(
                    round_number=round_number,
                    root_session_id=root_session_id,
                    project_id=session.project_id,
                    agent_id=self.agent_id,
                    permission_mode=permission_mode,
                    task_ids=self.delegated_task_ids,
                    glob_batch_size=glob_batch_size,
                ),
                secret_environment_names=self.secret_environment_names,
                secret_values=self.secret_values,
                parent_permission_broker=self.parent_permission_broker,
                ui_factory=ui,
            )
            results = await self._execute_tool_calls(calls, tool_context)
            memory_mutated = memory_mutated or _successful_memory_mutation(
                calls,
                results,
            )
            await self.store.append_message(
                MessageRecord(
                    session_id=session_id,
                    role="user",
                    content=[_tool_result_block(result) for result in results],
                )
            )
            todo_items = await self._todo_reminder_items(session_id)
            if _successful_todo_write(calls, results):
                rounds_since_todo = 0
            elif _has_open_todos(todo_items):
                rounds_since_todo += 1
            else:
                rounds_since_todo = 0
            if rounds_since_todo >= TODO_REMINDER_TOOL_ROUNDS:
                await self.store.append_message(
                    _todo_reminder_message(session_id, todo_items)
                )
                rounds_since_todo = 0
            output_token_limit = INITIAL_OUTPUT_MAX_TOKENS
            max_tokens_expanded = False
            continuations = 0
            round_number += 1
        if status in {"continue_tools", "continue_provider"}:
            status, reason = "incomplete", "round budget exhausted"
        if (
            status == "completed"
            and self.memory_eligible
            and self.memory_service is not None
            and self.memory_coordinator is not None
            and not memory_mutated
        ):
            try:
                context = await self.store.load_context(session_id)
                messages = tuple(
                    message
                    for message in context.messages[turn_start:]
                    if message.role in {"user", "assistant"}
                )
                explicit_request = is_explicit_memory_request(messages)
                self.memory_coordinator.submit(
                    self.memory_service,
                    session_id,
                    messages,
                    lambda payload: self._record_memory_lifecycle(
                        ui,
                        payload,
                        visible=explicit_request,
                    ),
                )
            except Exception:
                await self._record_memory_lifecycle(
                    ui,
                    {"operation": "extract", "status": "failed"},
                )
        return await self._finish_with_ui(
            session_id,
            status,
            reason,
            usage,
            ui,
            elapsed_seconds=time.monotonic() - turn_started_at,
        )

    def _current_permission_mode(self) -> str:
        if self.permission_mode_resolver is None:
            return self.permission_mode
        try:
            return _permission_mode_value(self.permission_mode_resolver())
        except (TypeError, ValueError):
            return PermissionMode.ASK.value

    async def _cancel_tool_tasks(
        self, tasks: list[asyncio.Task[ToolResult]]
    ) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        done, pending = await asyncio.wait(
            tasks, timeout=self.cleanup_timeout
        )
        for task in done:
            _consume_task(task)
        for task in pending:
            task.cancel()
            task.add_done_callback(_consume_task)

    async def _execute_tool_calls(
        self, calls: list[ToolCall], context: ToolContext
    ) -> list[ToolResult]:
        """Execute a model tool batch without flooding the event loop."""
        results: list[ToolResult] = []
        for start in range(0, len(calls), MAX_CONCURRENT_TOOL_CALLS):
            batch = calls[start : start + MAX_CONCURRENT_TOOL_CALLS]
            tasks = [
                asyncio.create_task(self.executor.execute(call, context))
                for call in batch
            ]
            try:
                results.extend(await asyncio.gather(*tasks))
            except BaseException:
                await self._cancel_tool_tasks(tasks)
                raise
        return results

    async def _pre_model(self, request: ModelRequest) -> tuple[ModelRequest, bool]:
        if self.hooks is None:
            return request, False
        outcome = await self.hooks.dispatch_pre(
            HookPoint.PRE_MODEL_CALL, {"request": request}
        )
        if outcome.blocked:
            return request, True
        payload = outcome.payload
        if isinstance(payload, dict) and isinstance(payload.get("request"), ModelRequest):
            return payload["request"], False
        return request, False

    async def _post_model_call(
        self,
        session_id: str,
        blocks: list[dict[str, object]],
        stop_reason: StopReason,
        usage: Usage,
        provider_error: LiteCoderError | None,
    ) -> None:
        if self.hooks is None:
            return
        await self.hooks.dispatch_post(
            HookPoint.POST_MODEL_CALL,
            {
                "session_id": session_id,
                "blocks": blocks,
                "stop_reason": stop_reason.value,
                "usage": _usage_payload(usage),
                "error": (
                    provider_error.code.value
                    if provider_error is not None
                    else None
                ),
            },
        )

    async def _collect(
        self,
        request: ModelRequest,
        ui: UIEventFactory,
        usage_progress: _TurnUsageProgress,
    ) -> tuple[
        list[dict[str, object]],
        StopReason,
        str,
        Usage,
        LiteCoderError | None,
        str | None,
    ]:
        blocks: dict[int, dict[str, object]] = {}
        usage = Usage(0, 0)
        stop_reason = StopReason.UNKNOWN
        raw_reason = "missing response.completed"
        provider_error: LiteCoderError | None = None
        provider_request_id: str | None = None
        thinking_started: set[int] = set()
        tool_calls_started: set[str] = set()

        async def emit_thinking_started(
            index: int | None, request_id: str | None
        ) -> None:
            if index is None or index in thinking_started:
                return
            thinking_started.add(index)
            await self._emit_ui(
                ui,
                UIEventType.THINKING_STARTED,
                request_id=request_id,
                payload={"index": index},
            )

        async def emit_tool_call_started(
            call_id: str | None,
            tool_name: str | None,
            request_id: str | None,
            payload: dict[str, object] | None = None,
        ) -> None:
            if call_id is None or call_id in tool_calls_started:
                return
            tool_calls_started.add(call_id)
            kwargs: dict[str, object] = {
                "request_id": request_id,
                "tool_call_id": call_id,
                "payload": payload or {},
            }
            if tool_name:
                kwargs["tool_name"] = tool_name
            await self._emit_ui(ui, UIEventType.TOOL_CALL_STARTED, **kwargs)

        try:
            async for event in self.provider.stream(request):
                if event.type == "response.request_id" and event.request_id is not None:
                    provider_request_id = event.request_id
                    await self._emit_ui(
                        ui,
                        UIEventType.MODEL_REQUEST_ID,
                        request_id=event.request_id,
                        payload={"request_id": event.request_id},
                    )
                elif event.type == "provider.error":
                    provider_error = event.error or LiteCoderError(
                        ErrorCode.INTERNAL,
                        "Provider error",
                        retryable=False,
                    )
                    provider_request_id = event.request_id
                    break
                elif event.type == "content.started" and event.index is not None:
                    block = dict(event.block or {})
                    if _is_thinking_block(block):
                        await emit_thinking_started(event.index, event.request_id)
                    call_id = _tool_call_id_from_block(block)
                    if call_id is not None:
                        await emit_tool_call_started(
                            call_id,
                            _tool_call_name_from_block(block),
                            event.request_id,
                            _tool_call_started_payload(block),
                        )
                elif event.type == "usage" and event.usage is not None:
                    usage = event.usage
                    usage_progress.observe_round(usage)
                    await self._emit_ui(
                        ui,
                        UIEventType.USAGE_UPDATED,
                        request_id=event.request_id,
                        payload=_usage_payload(usage),
                    )
                elif event.type == "content.delta" and isinstance(event.delta, dict):
                    text = _thinking_text_from_delta(event.delta)
                    if text:
                        await emit_thinking_started(event.index, event.request_id)
                        await self._emit_ui(
                            ui,
                            UIEventType.THINKING_DELTA,
                            request_id=event.request_id,
                            payload={"text": text},
                        )
                elif event.type == "text.delta" and isinstance(event.delta, str):
                    await self._emit_ui(
                        ui,
                        UIEventType.ASSISTANT_DELTA,
                        request_id=event.request_id,
                        payload={"text": event.delta},
                    )
                elif event.type == "tool_call.input_delta" and isinstance(event.delta, str):
                    await emit_tool_call_started(
                        event.tool_call_id, None, event.request_id
                    )
                    await self._emit_ui(
                        ui,
                        UIEventType.TOOL_CALL_INPUT_DELTA,
                        request_id=event.request_id,
                        tool_call_id=event.tool_call_id,
                        payload={"text": event.delta},
                    )
                elif event.type == "tool_call.completed" and event.tool_call is not None:
                    await emit_tool_call_started(
                        event.tool_call.call_id,
                        event.tool_call.name,
                        event.request_id,
                        {"arguments": event.tool_call.input},
                    )
                    await self._emit_ui(
                        ui,
                        UIEventType.TOOL_CALL_COMPLETED,
                        request_id=event.request_id,
                        tool_call_id=event.tool_call.call_id,
                        tool_name=event.tool_call.name,
                        payload={"arguments": event.tool_call.input},
                    )
                elif event.type == "content.completed" and event.index is not None:
                    block = dict(event.block or {})
                    blocks[event.index] = block
                    completed_thinking = _thinking_text_from_block(block)
                    if completed_thinking:
                        await emit_thinking_started(event.index, event.request_id)
                        await self._emit_ui(
                            ui,
                            UIEventType.THINKING_COMPLETED,
                            request_id=event.request_id,
                            payload={"text": completed_thinking},
                        )
                    assistant_text = _assistant_text_from_block(block)
                    if assistant_text:
                        await self._emit_ui(
                            ui,
                            UIEventType.ASSISTANT_COMPLETED,
                            request_id=event.request_id,
                            payload={"text": assistant_text},
                        )
                    call_id = _tool_call_id_from_block(block)
                    if call_id is not None:
                        await emit_tool_call_started(
                            call_id,
                            _tool_call_name_from_block(block),
                            event.request_id,
                            _tool_call_started_payload(block),
                        )
                elif event.type == "response.completed":
                    if event.request_id is not None:
                        provider_request_id = event.request_id
                    stop_reason = event.stop_reason or StopReason.UNKNOWN
                    raw_reason = event.raw_stop_reason or stop_reason.value
                    if event.usage is not None:
                        usage = event.usage
                        usage_progress.observe_round(usage)
                    await self._emit_ui(
                        ui,
                        UIEventType.MODEL_COMPLETED,
                        request_id=event.request_id,
                        payload={
                            "stop_reason": stop_reason.value,
                            "raw_stop_reason": raw_reason,
                        },
                    )
        except Exception as error:
            if self.recovery_policy is None:
                raise
            provider_error = self.error_classifier.classify(error)
        return (
            [blocks[index] for index in sorted(blocks)],
            stop_reason,
            raw_reason,
            usage,
            provider_error,
            provider_request_id,
        )


    async def _inject_team_messages(self, session_id: str) -> None:
        inbox = self.team_inbox
        if inbox is None:
            return
        drain = getattr(inbox, "drain_inbox", None)
        if not callable(drain):
            return
        received = drain(self.agent_id)
        messages = await received if inspect.isawaitable(received) else received
        if not isinstance(messages, (list, tuple)):
            return
        rendered: list[str] = []
        for message in messages:
            sender = getattr(message, "sender", None)
            body = getattr(message, "body", None)
            if not isinstance(sender, str) or not isinstance(body, str):
                continue
            rendered.append(f"From teammate {sender}:\n{body}")
        if not rendered:
            return
        await self.store.append_message(
            MessageRecord(
                session_id=session_id,
                role="user",
                content=[{
                    "type": "text",
                    "text": "Team inbox messages:\n\n" + "\n\n".join(rendered),
                }],
            )
        )
    async def _todo_reminder_items(
        self, session_id: str
    ) -> tuple[tuple[str, str], ...]:
        try:
            todos = await self.store.list_todos(session_id)
        except (KeyError, OSError, RuntimeError, ValueError):
            return ()
        items: list[tuple[str, str]] = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            status = todo.get("status")
            content = todo.get("content")
            if not isinstance(status, str) or not isinstance(content, str):
                continue
            items.append((status, content))
        return tuple(items)

    async def _inject_background_notifications(
        self, session_id: str
    ) -> None:
        if self.background is None:
            return
        notifications = await self.background.drain_notifications(session_id)
        if not notifications:
            return
        await self.store.append_message(
            MessageRecord(
                session_id=session_id,
                role="user",
                content=[
                    notification.to_content_block()
                    for notification in notifications
                ],
            )
        )

    def _can_compact_context(self) -> bool:
        return getattr(self.context, "can_compact", False) is True

    def _configure_runtime_compaction(self) -> None:
        if not isinstance(self.context, ContextManager):
            return
        self.context.configure_runtime_compaction(
            ProviderContextSummarizer(self.provider, self.context.model)
        )

    async def _recover_provider_error(
        self,
        session_id: str,
        error: LiteCoderError,
        *,
        ui: UIEventFactory,
        request_id: str | None,
        has_attempted_reactive_compaction: bool,
    ) -> _ProviderRecoveryOutcome:
        if self.recovery_policy is None:
            await self._emit_provider_error(
                error,
                ui=ui,
                request_id=request_id,
                retrying=False,
                recovery_action="fail",
                recovery_reason=error.code.value,
            )
            return _ProviderRecoveryOutcome(
                False, ("failed", error.code.value)
            )
        action = self.recovery_policy.choose(
            error,
            RecoveryContext(
                can_compact=self._can_compact_context(),
                has_attempted_reactive_compaction=has_attempted_reactive_compaction,
            ),
        )
        retrying = action.kind in {
            "retry", "retry_with_feedback", "compact_then_retry"
        }
        await self._emit_provider_error(
            error,
            ui=ui,
            request_id=request_id,
            retrying=retrying,
            recovery_action=action.kind,
            recovery_reason=action.reason,
            attempt=action.attempt,
            max_attempts=action.max_attempts,
            failure_origin=action.failure_origin.value,
            failure_code=action.failure_code,
            recovery_strategy=action.strategy.value,
            delay_seconds=action.delay_seconds,
        )
        if action.kind == "retry":
            await self._sleep_for_recovery(action.delay_seconds)
            return _ProviderRecoveryOutcome(True, action=action)
        if action.kind == "retry_with_feedback":
            return _ProviderRecoveryOutcome(
                True,
                feedback_code=action.feedback_code,
                action=action,
            )
        if action.kind == "compact_then_retry":
            try:
                compact = getattr(self.context, "compact_reactively", None)
                if not callable(compact):
                    compact = getattr(self.context, "compact")
                outcome = compact(session_id)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as recovery_error:
                if self.hooks is not None:
                    await self.hooks.record_runtime_fact(
                        "provider.runtime",
                        {
                            "stage": "recovery",
                            "status": "failed",
                            "error_code": error.code.value,
                            "request_id": request_id,
                            "failure_origin": action.failure_origin.value,
                            "failure_code": action.failure_code,
                            "recovery_strategy": action.strategy.value,
                            "attempt": action.attempt,
                            "max_attempts": action.max_attempts,
                            "delay_seconds": action.delay_seconds,
                            "reason": "context compaction failed",
                            "error_type": type(recovery_error).__name__,
                        },
                    )
                return _ProviderRecoveryOutcome(
                    False, ("failed", "context compaction failed")
                )
            return _ProviderRecoveryOutcome(True, action=action)
        if action.kind == "stop_incomplete":
            return _ProviderRecoveryOutcome(
                False, ("incomplete", action.reason), action=action
            )
        return _ProviderRecoveryOutcome(
            False, ("failed", action.reason), action=action
        )

    async def _emit_provider_error(
        self,
        error: LiteCoderError,
        *,
        ui: UIEventFactory,
        request_id: str | None,
        retrying: bool,
        recovery_action: str,
        recovery_reason: str,
        attempt: int | None = None,
        max_attempts: int | None = None,
        failure_origin: str | None = None,
        failure_code: str | None = None,
        recovery_strategy: str | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        payload: dict[str, object] = {
            "code": error.code.value,
            "message": str(error),
            "retryable": error.retryable,
            "retrying": retrying,
            "request_id": request_id,
            "details": dict(error.details),
            "recovery_action": recovery_action,
            "recovery_reason": recovery_reason,
        }
        if attempt is not None and max_attempts is not None:
            payload.update(attempt=attempt, max_attempts=max_attempts)
        if self.hooks is not None:
            status = (
                "retrying"
                if retrying
                else "incomplete"
                if recovery_action == "stop_incomplete"
                else "failed"
            )
            await self.hooks.record_runtime_fact(
                "provider.runtime",
                {
                    "stage": "recovery",
                    "status": status,
                    "error_code": error.code.value,
                    "request_id": request_id,
                    "failure_origin": failure_origin or "internal",
                    "failure_code": failure_code or error.code.value,
                    "recovery_strategy": recovery_strategy or (
                        "stop"
                        if recovery_action in {"fail", "stop_incomplete"}
                        else recovery_action
                    ),
                    "recovery_reason": recovery_reason,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "delay_seconds": delay_seconds,
                },
            )
        await self._emit_ui(
            ui,
            UIEventType.PROVIDER_ERROR,
            request_id=request_id,
            payload=payload,
        )

    async def _record_recovery_status(
        self,
        action: RecoveryAction,
        *,
        status: str,
        request_id: str | None,
    ) -> None:
        if self.hooks is None:
            return
        await self.hooks.record_runtime_fact(
            "provider.runtime",
            {
                "stage": "recovery",
                "status": status,
                "request_id": request_id,
                "failure_origin": action.failure_origin.value,
                "failure_code": action.failure_code,
                "recovery_strategy": action.strategy.value,
                "attempt": action.attempt,
                "max_attempts": action.max_attempts,
                "delay_seconds": action.delay_seconds,
            },
        )

    async def _sleep_for_recovery(self, delay_seconds: float) -> None:
        outcome = self.recovery_sleep(delay_seconds)
        if inspect.isawaitable(outcome):
            await outcome
    async def _finish_with_ui(
        self,
        session_id: str,
        status: str,
        reason: str,
        usage: Usage,
        ui: UIEventFactory,
        elapsed_seconds: float | None = None,
    ) -> AgentResult:
        result = await self._finish(session_id, status, reason, usage)
        await self._emit_ui(
            ui,
            UIEventType.TURN_FINISHED,
            payload={
                "status": result.status,
                "reason": result.reason,
                "total_tokens": result.usage.total_tokens,
                **({
                    "elapsed_seconds": elapsed_seconds,
                } if elapsed_seconds is not None else {}),
            },
        )
        await flush_ui(self.ui_sink)
        return result

    async def _finish(
        self, session_id: str, status: str, reason: str, usage: Usage
    ) -> AgentResult:
        session_status = {
            "completed": SessionStatus.IDLE,
            "incomplete": SessionStatus.INCOMPLETE,
            "refused": SessionStatus.FAILED,
            "failed": SessionStatus.FAILED,
            "cancelled": SessionStatus.CANCELLED,
        }.get(status, SessionStatus.INCOMPLETE)
        await self.store.mark_status(session_id, session_status)
        result = AgentResult(session_id, status, reason, usage)
        if self.hooks is not None:
            hook = self.hooks.dispatch_post(
                HookPoint.AGENT_STOP, {"result": _result_payload(result)}
            )
            await self._bounded_cleanup(hook)
        if status == "completed":
            await flush_ui(self.ui_sink)
        else:
            await self._bounded_cleanup(flush_ui(self.ui_sink))
        return result

    async def _cancelled(self, session_id: str, usage: Usage) -> None:
        await self._bounded_cleanup(
            self.store.mark_status(session_id, SessionStatus.CANCELLED)
        )
        if self.hooks is not None:
            result = AgentResult(
                session_id, "cancelled", "cancelled", usage
            )
            await self._bounded_cleanup(
                self.hooks.dispatch_post(
                    HookPoint.AGENT_STOP, {"result": _result_payload(result)}
                )
            )
        await self._bounded_cleanup(flush_ui(self.ui_sink))

    async def _failed_exception(self, session_id: str, usage: Usage) -> None:
        await self._bounded_cleanup(
            self.store.mark_status(session_id, SessionStatus.FAILED)
        )
        if self.hooks is not None:
            result = AgentResult(
                session_id, "failed", "internal_error", usage
            )
            await self._bounded_cleanup(
                self.hooks.dispatch_post(
                    HookPoint.AGENT_STOP, {"result": _result_payload(result)}
                )
            )
        await self._bounded_cleanup(flush_ui(self.ui_sink))

    async def _bounded_cleanup(self, awaitable: object) -> None:
        task = asyncio.create_task(awaitable)  # type: ignore[arg-type]
        done, pending = await asyncio.wait(
            {task}, timeout=self.cleanup_timeout
        )
        if done:
            _consume_task(task)
            return
        task.cancel()
        _, pending = await asyncio.wait(
            pending, timeout=min(0.01, self.cleanup_timeout)
        )
        for item in pending:
            item.cancel()
            item.add_done_callback(_consume_task)



def _permission_mode_value(mode: PermissionMode | str) -> str:
    return PermissionMode(str(mode)).value


def _tool_metadata(
    *,
    round_number: int,
    root_session_id: str,
    project_id: str,
    agent_id: str,
    permission_mode: str,
    task_ids: frozenset[str],
    glob_batch_size: int,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "round_number": round_number,
        "root_session_id": root_session_id,
        "project_id": project_id,
        "agent_id": agent_id,
        "permission_mode": permission_mode,
        "task_ids": sorted(task_ids),
    }
    if permission_mode == PermissionMode.BYPASS.value:
        metadata["bypass_authorized"] = True
    if glob_batch_size > 1:
        metadata["glob_batch_size"] = glob_batch_size
    return metadata


def _consume_task(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass

def _usage_payload(usage: Usage) -> dict[str, object]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_creation_tokens": usage.cache_creation_tokens,
        "extensions": dict(usage.extensions),
    }


def _result_payload(result: AgentResult) -> dict[str, object]:
    return {
        "session_id": result.session_id,
        "status": result.status,
        "reason": result.reason,
        "usage": _usage_payload(result.usage),
    }


def _tool_calls(blocks: list[dict[str, object]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for block in blocks:
        if block.get("type") != "tool_call":
            continue
        call_id = block.get("call_id")
        name = block.get("name")
        arguments = block.get("input")
        if (
            isinstance(call_id, str)
            and isinstance(name, str)
            and isinstance(arguments, dict)
        ):
            calls.append(ToolCall(call_id, name, arguments))
    return calls


def _successful_memory_mutation(
    calls: list[ToolCall],
    results: list[ToolResult],
) -> bool:
    return any(
        call.name in {"memory_update", "memory_delete"}
        and result.status == "success"
        and result.metadata.get("changed_workspace") is True
        for call, result in zip(calls, results)
    )
def _successful_todo_write(
    calls: list[ToolCall], results: list[ToolResult]
) -> bool:
    return any(
        call.name == "todo_write" and result.status == "success"
        for call, result in zip(calls, results)
    )


def _has_open_todos(items: tuple[tuple[str, str], ...]) -> bool:
    return any(status != "completed" for status, _ in items)


def _thinking_text_from_delta(delta: dict[str, object]) -> str:
    value = delta.get("thinking")
    if isinstance(value, str):
        return value
    reasoning = delta.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def _thinking_text_from_block(block: dict[str, object]) -> str:
    block_type = block.get("type")
    if block_type == "thinking" and isinstance(block.get("thinking"), str):
        return str(block["thinking"])
    provider = block.get("provider")
    if isinstance(provider, dict):
        merged = provider.get("merged")
        if isinstance(merged, dict) and isinstance(
            merged.get("reasoning_content"), str
        ):
            return str(merged["reasoning_content"])
    return ""


def _is_thinking_block(block: dict[str, object]) -> bool:
    return block.get("type") == "thinking" or bool(_thinking_text_from_block(block))


def _assistant_text_from_block(block: dict[str, object]) -> str:
    block_type = block.get("type")
    if block_type in {"text", "refusal"} and isinstance(block.get("text"), str):
        return str(block["text"])
    return ""


def _tool_call_id_from_block(block: dict[str, object]) -> str | None:
    if block.get("type") not in {"tool_call", "tool_use"}:
        return None
    for key in ("call_id", "id"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _tool_call_name_from_block(block: dict[str, object]) -> str | None:
    value = block.get("name")
    return value if isinstance(value, str) and value else None


def _tool_call_started_payload(block: dict[str, object]) -> dict[str, object]:
    arguments = block.get("input")
    if not isinstance(arguments, dict):
        arguments = block.get("arguments")
    return {"arguments": arguments} if isinstance(arguments, dict) else {}


def _with_response_repair_feedback(request: ModelRequest) -> ModelRequest:
    return replace(
        request,
        system=[
            *request.system,
            {"type": "text", "text": _RESPONSE_REPAIR_PROMPT},
        ],
    )


def _invalid_completed_response_error(
    blocks: list[dict[str, object]],
    stop_reason: StopReason,
    raw_reason: str,
) -> LiteCoderError | None:
    failure_code: str | None = None
    if stop_reason is StopReason.UNKNOWN:
        failure_code = (
            "missing_response_completed"
            if raw_reason == "missing response.completed"
            else "unknown_stop_reason"
        )
    elif stop_reason is StopReason.TOOL_USE and not _tool_calls(blocks):
        failure_code = "tool_use_without_calls"
    elif stop_reason is StopReason.END_TURN and not _has_assistant_output(blocks):
        failure_code = "empty_response"
    if failure_code is None:
        return None
    return LiteCoderError(
        ErrorCode.PROVIDER_INVALID_RESPONSE,
        "Provider returned an invalid response",
        retryable=True,
        details={
            "provider_error_type": failure_code,
        },
    )


def _has_assistant_output(blocks: list[dict[str, object]]) -> bool:
    for block in blocks:
        if block.get("type") in {"text", "refusal"}:
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return True
        if block.get("type") == "tool_call":
            return True
    return False


def _tool_result_block(result: ToolResult) -> dict[str, object]:
    return {
        "type": "tool_result",
        "tool_call_id": result.tool_call_id,
        "status": result.status,
        "content": result.content,
        "metadata": result.metadata,
    }



def _continuation_message(session_id: str) -> MessageRecord:
    return MessageRecord(
        session_id=session_id,
        role="user",
        content=[{"type": "text", "text": CONTINUATION_PROMPT}],
    )

def _todo_reminder_message(
    session_id: str, items: tuple[tuple[str, str], ...]
) -> MessageRecord:
    return _todo_system_reminder_message(session_id, items, TODO_REMINDER_TEXT)


def _todo_system_reminder_message(
    session_id: str,
    items: tuple[tuple[str, str], ...],
    reminder_text: str,
) -> MessageRecord:
    rendered = "\n".join(
        f"{index}. [{status}] {content}"
        for index, (status, content) in enumerate(items, start=1)
    )
    content = reminder_text
    if rendered:
        content = f"{content}\n\nHere are the existing contents of your todo list:\n\n[{rendered}]"
    return MessageRecord(
        session_id=session_id,
        role="user",
        content=[{
            "type": "text",
            "text": f"<system-reminder>\n{content}\n</system-reminder>",
        }],
    )

def _add_usage(left: Usage, right: Usage) -> Usage:
    keys = set(left.extensions) | set(right.extensions)
    return Usage(
        left.input_tokens + right.input_tokens,
        left.output_tokens + right.output_tokens,
        (left.cache_read_tokens or 0) + (right.cache_read_tokens or 0),
        (left.cache_creation_tokens or 0) + (right.cache_creation_tokens or 0),
        {
            key: left.extensions.get(key, 0) + right.extensions.get(key, 0)
            for key in keys
        },
    )


def _token_budget_exhausted(usage: Usage, limit: int | None) -> bool:
    return limit is not None and usage.total_tokens > limit
