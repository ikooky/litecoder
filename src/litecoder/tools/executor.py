"""Tool execution pipeline and concurrency controls."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from itertools import chain, islice
from dataclasses import dataclass
from jsonschema.exceptions import SchemaError, ValidationError

from litecoder.common.locks import NamedFileLock
from litecoder.common.trace import bind_secret_redactor, current_secret_redactor
from litecoder.hooks import HookDiagnostic, HookManager, HookPoint
from litecoder.providers._json import JsonValue
from litecoder.tools.artifacts import (
    ARTIFACT_PREVIEW_BYTES,
    TOOL_RESULT_INLINE_BYTES,
    ArtifactReference,
    ArtifactStore,
)
from litecoder.tools.duplicate_guard import DuplicateGuard, PreparedDuplicate
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolPartialFailure,
    ToolResult,
    ToolSpec,
)
from litecoder.tools.permission import PermissionService
from litecoder.tools.registry import ToolRegistry
from litecoder.tools.workspace_version import WorkspaceState, WorkspaceStateRegistry
from litecoder.ui.events import UIEventFactory, UIEventType
from litecoder.ui.sink import RuntimeUISink, emit_ui


@dataclass(slots=True)
class _PipelineState:
    """Data model representing the pipeline state."""
    decision_call: ToolCall
    pending_version: WorkspaceState | None = None


@asynccontextmanager
async def _workspace_execution_lock(
    tool: object,
    state: WorkspaceState,
) -> AsyncIterator[None]:
    spec = getattr(tool, "spec", None)
    if getattr(spec, "workspace_lock", True) is False:
        yield
        return
    concurrency = getattr(spec, "concurrency", None)
    if concurrency == "traversal":
        # Do not let queued traversals hold read ownership while a writer waits.
        async with state.traversal_lock:
            async with state.lock.read():
                yield
        return
    lock = (
        state.lock.write()
        if getattr(spec, "mutates_workspace", False) or concurrency == "exclusive"
        else state.lock.read()
    )
    async with lock:
        yield

class ToolExecutor:
    """Component responsible for the tool executor."""
    def __init__(
        self,
        registry: ToolRegistry,
        hooks: HookManager,
        duplicates: DuplicateGuard,
        permission: PermissionService,
        workspaces: WorkspaceStateRegistry,
        *,
        error_hook_timeout: float = 1.0,
        artifact_store: ArtifactStore | None = None,
        artifact_store_resolver: (
            Callable[[ToolContext], ArtifactStore] | None
        ) = None,
        ui_sink: RuntimeUISink | None = None,
        ui_factory_resolver: Callable[[ToolContext], UIEventFactory] | None = None,
        workspace_lock_resolver: (
            Callable[[ToolContext], NamedFileLock | None] | None
        ) = None,
    ) -> None:
        if error_hook_timeout <= 0:
            raise ValueError("error_hook_timeout must be positive")
        if artifact_store is not None and artifact_store_resolver is not None:
            raise ValueError(
                "provide artifact_store or artifact_store_resolver, not both"
            )
        self.registry = registry
        self.hooks = hooks
        self.duplicates = duplicates
        self.permission = permission
        self.workspaces = workspaces
        self._error_hook_timeout = error_hook_timeout
        self.artifact_store = artifact_store
        self.artifact_store_resolver = artifact_store_resolver
        self.ui_sink = ui_sink
        self.ui_factory_resolver = ui_factory_resolver
        self.workspace_lock_resolver = workspace_lock_resolver
        self._ui_factories: dict[tuple[str, str, str, int], UIEventFactory] = {}

    def fork(
        self,
        *,
        registry: ToolRegistry,
        duplicates: DuplicateGuard,
    ) -> "ToolExecutor":
        """Create a session-scoped executor over the shared runtime services."""
        return ToolExecutor(
            registry,
            self.hooks,
            duplicates,
            self.permission,
            self.workspaces,
            error_hook_timeout=self._error_hook_timeout,
            artifact_store=self.artifact_store,
            artifact_store_resolver=self.artifact_store_resolver,
            ui_sink=self.ui_sink,
            ui_factory_resolver=self.ui_factory_resolver,
            workspace_lock_resolver=self.workspace_lock_resolver,
        )

    @asynccontextmanager
    async def _workspace_file_lock(
        self,
        tool: object,
        context: ToolContext,
    ) -> AsyncIterator[None]:
        spec = getattr(tool, "spec", None)
        if getattr(spec, "workspace_lock", True) is False:
            yield
            return
        requires_lock = (
            getattr(spec, "mutates_workspace", False) is True
            or getattr(spec, "concurrency", None) == "exclusive"
        )
        if not requires_lock or self.workspace_lock_resolver is None:
            yield
            return
        lock = self.workspace_lock_resolver(context)
        if lock is None:
            yield
            return
        async with lock.acquired_async():
            yield

    def _ui_factory(self, context: ToolContext) -> UIEventFactory | None:
        runtime_factory = getattr(context, "ui_factory", None)
        if isinstance(runtime_factory, UIEventFactory):
            return runtime_factory
        if self.ui_factory_resolver is None:
            return None
        root = context.metadata.get("root_session_id", context.agent_session_id)
        if not isinstance(root, str) or not root.strip():
            root = context.agent_session_id
        key = (
            context.agent_session_id,
            context.workspace_id,
            root,
            _round_number(context),
        )
        factory = self._ui_factories.get(key)
        if factory is None:
            factory = self.ui_factory_resolver(context)
            self._ui_factories[key] = factory
            if len(self._ui_factories) > 512:
                self._ui_factories.pop(next(iter(self._ui_factories)))
        return factory

    async def _ui(
        self,
        context: ToolContext,
        event_type: UIEventType,
        call: ToolCall,
        payload: dict[str, object],
    ) -> None:
        if self.ui_sink is None:
            return
        try:
            factory = self._ui_factory(context)
            if factory is None:
                return
            event = factory.next(
                event_type,
                tool_call_id=call.id,
                tool_name=call.name,
                payload=payload,
            )
            await emit_ui(self.ui_sink, event)
        except Exception:
            return

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute the requested tool call."""
        with bind_secret_redactor(context.redactor):
            original = ToolCall(call.id, call.name, call.arguments)
            pipeline = _PipelineState(original)
            try:
                return await self._execute_pipeline(original, context, pipeline)
            except asyncio.CancelledError:
                if pipeline.pending_version is not None:
                    await self._finish_pending_version(pipeline)
                await self._cancellation_cleanup(pipeline.decision_call)
                raise

    async def _execute_pipeline(
        self,
        original: ToolCall,
        context: ToolContext,
        pipeline: _PipelineState,
    ) -> ToolResult:
        """Execute the pipeline."""
        # Each stage observes the transformed call and workspace version from
        # the preceding stage before a tool is allowed to mutate state.
        original_id, original_name = original.id, original.name
        try:
            tool = self.registry.require(original.name)
        except KeyError:
            await self._fact("registry", original, status="unknown_tool")
            return await self._final(
                ToolResult(original_id, "unknown_tool", "Unknown tool"),
                original,
                context,
            )
        await self._fact("registry", original, status="found")

        pre = await self.hooks.dispatch_pre(
            HookPoint.PRE_TOOL_USE, {"call": original}
        )
        if pre.blocked:
            await self._fact(
                "pre", original, status="blocked", arguments=original.arguments
            )
            return await self._final(
                ToolResult(original_id, "hook_blocked", "Blocked by PreToolUse"),
                original,
                context,
            )
        transformed = _transformed_call(pre.payload, original_id, original_name)
        if transformed is None:
            await self._fact(
                "pre", original, status="invalid", arguments=original.arguments
            )
            return await self._final(
                ToolResult(
                    original_id, "hook_blocked", "Invalid PreToolUse mutation"
                ),
                original,
                context,
            )
        decision_call = ToolCall(
            transformed.id, transformed.name, transformed.arguments
        )
        pipeline.decision_call = decision_call
        await self._fact(
            "pre",
            decision_call,
            status="completed",
            arguments=decision_call.arguments,
        )

        state = self.workspaces.get(context.workspace_id)
        try:
            _validate_tool_arguments(tool.spec, decision_call.arguments)
        except SchemaError:
            await self._fact(
                "arguments", decision_call, status="invalid"
            )
            return await self._final(
                ToolResult(
                    original_id,
                    "tool_error",
                    "Tool input schema is invalid",
                    {"automatic_retry": False},
                ),
                decision_call,
                context,
            )
        except ValidationError as error:
            await self._fact(
                "arguments",
                decision_call,
                status="invalid",
                validation_code=_validation_code(error),
                validation_path=_validation_path(error),
            )
            return await self._final(
                ToolResult(
                    original_id,
                    "invalid_arguments",
                    _safe_validation_message(error),
                    {
                        "automatic_retry": False,
                        "validation_code": _validation_code(error),
                        "validation_path": _validation_path(error),
                    },
                ),
                decision_call,
                context,
            )
        round_number = _round_number(context)
        prepared = self.duplicates.prepare(
            decision_call,
            context.workspace_id,
            tool.spec,
            agent_session_id=context.agent_session_id,
        )
        failure: ToolResult | None = None
        execution: ToolExecution | None = None
        async with self.duplicates.execution_lease(
            context.agent_session_id,
            context.workspace_id,
            call=decision_call,
            spec=tool.spec,
            prepared=prepared,
        ) as leased:
            duplicate = await self.duplicates.check(
                context.agent_session_id,
                context.workspace_id,
                state.version,
                round_number=round_number,
                call=decision_call,
                spec=tool.spec,
                prepared=leased,
            )
            await self._fact(
                "duplicate",
                decision_call,
                status="blocked" if duplicate else "clear",
            )
            if duplicate is not None:
                return await self._final(duplicate, decision_call, context)

            guard_reason = await _tool_hard_guard(tool, decision_call, context)
            if guard_reason is not None:
                await self._fact(
                    "permission", decision_call, status="deny", allowed=False,
                    hard_invariant=True, reason=guard_reason,
                )
                await self._ui(
                    context,
                    UIEventType.TOOL_EXECUTION_DENIED,
                    decision_call,
                    {"reason": guard_reason, "arguments": decision_call.arguments},
                )
                return await self._final(
                    ToolResult(original_id, "denied", guard_reason),
                    decision_call,
                    context,
                )

            permission_payload = _permission_request_payload(
                self.permission, tool.spec, decision_call, context
            )
            if permission_payload is not None:
                await self._ui(
                    context,
                    UIEventType.PERMISSION_REQUESTED,
                    decision_call,
                    permission_payload,
                )
            decision = await self.permission.decide(
                tool.spec, decision_call, context
            )
            if permission_payload is not None:
                resolved_payload = dict(permission_payload)
                resolved_payload.update(
                    allowed=decision.allowed,
                    action=decision.action,
                    reason=decision.reason,
                )
                await self._ui(
                    context,
                    UIEventType.PERMISSION_RESOLVED,
                    decision_call,
                    resolved_payload,
                )
            await self._fact(
                "permission",
                decision_call,
                status=decision.action,
                allowed=decision.allowed,
                reason=decision.reason,
            )
            if not decision.allowed:
                await self._ui(
                    context,
                    UIEventType.TOOL_EXECUTION_DENIED,
                    decision_call,
                    {"reason": decision.reason, "arguments": decision_call.arguments},
                )
                return await self._final(
                    ToolResult(original_id, "denied", decision.reason),
                    decision_call,
                    context,
                )

            before = state.version
            read_side_effect = False
            partial_read_side_effect = False
            error_fact_emitted = False
            async with _workspace_execution_lock(tool, state):
                async with self._workspace_file_lock(tool, context):
                    before = state.version
                    await self._fact(
                        "execute",
                        decision_call,
                        status="started",
                        workspace_version=before,
                    )
                    await self._ui(
                        context,
                        UIEventType.TOOL_EXECUTION_STARTED,
                        decision_call,
                        {
                            "arguments": decision_call.arguments,
                            "workspace_version": before,
                        },
                    )
                    execution_call = ToolCall(
                        decision_call.id,
                        decision_call.name,
                        decision_call.arguments,
                    )
                    try:
                        execution = await tool.execute(execution_call, context)
                        if not isinstance(execution, ToolExecution):
                            raise TypeError("invalid tool execution")
                    except ToolDenied as error:
                        failure = ToolResult(
                            original_id,
                            "denied",
                            error.safe_message,
                            {"automatic_retry": False, "changed_workspace": False},
                        )
                    except ToolFailure as error:
                        metadata = dict(error.metadata)
                        metadata.update(
                            automatic_retry=False, changed_workspace=False
                        )
                        failure = ToolResult(
                            original_id, "tool_error", error.safe_message, metadata
                        )
                    except ToolPartialFailure as error:
                        if error.changed_workspace and tool.spec.mutates_workspace:
                            state.version += 1
                        elif error.changed_workspace:
                            partial_read_side_effect = True
                            pipeline.pending_version = state
                        metadata = dict(error.metadata)
                        metadata.update(
                            automatic_retry=False,
                            changed_workspace=error.changed_workspace,
                        )
                        failure = ToolResult(
                            original_id,
                            "partial_failure",
                            error.safe_message,
                            metadata,
                        )
                    except asyncio.CancelledError:
                        if tool.spec.mutates_workspace:
                            state.version += 1
                        raise
                    except Exception as error:
                        if tool.spec.mutates_workspace:
                            state.version += 1
                        failure = ToolResult(
                            original_id,
                            "tool_error",
                            "Tool execution failed",
                            {"automatic_retry": False},
                        )
                        error_fact_emitted = True
                        await self._fact(
                            "error",
                            decision_call,
                            status="tool_error",
                            automatic_retry=False,
                            error_type=type(error).__name__,
                        )
                    else:
                        if execution.status != "success":
                            if tool.spec.mutates_workspace:
                                state.version += 1
                            elif execution.changed_workspace:
                                read_side_effect = True
                                pipeline.pending_version = state
                            status = (
                                "contract_violation"
                                if not tool.spec.mutates_workspace
                                and execution.changed_workspace
                                else "tool_error"
                            )
                            failure = ToolResult(
                                original_id,
                                status,
                                "Tool execution contract violated",
                                {"automatic_retry": False},
                            )
                        elif not tool.spec.mutates_workspace and execution.changed_workspace:
                            read_side_effect = True
                            pipeline.pending_version = state
                        else:
                            preview = _safe_preview(self.duplicates, execution.preview)
                            await self._commit_success_shielded(
                                state,
                                decision_call,
                                context,
                                round_number,
                                leased,
                                preview,
                                before,
                                execution.changed_workspace,
                            )

            if read_side_effect or partial_read_side_effect:
                await self._finish_pending_version(pipeline)
                if read_side_effect and failure is None:
                    failure = ToolResult(
                        original_id,
                        "contract_violation",
                        "Tool safety contract violated",
                        {"automatic_retry": False},
                    )

            if failure is not None:
                failure.metadata["workspace_version"] = state.version
                await self._ui(
                    context,
                    UIEventType.TOOL_EXECUTION_FAILED,
                    decision_call,
                    {
                        "status": failure.status,
                        "message": _safe_ui_text(failure.content),
                        "metadata": _safe_ui_metadata(failure.metadata),
                    },
                )
                if failure.status == "partial_failure":
                    await self._fact(
                        "partial",
                        decision_call,
                        status="partial_failure",
                        changed_workspace=failure.metadata.get(
                            "changed_workspace", False
                        ),
                    )
                elif not error_fact_emitted:
                    await self._fact(
                        "error",
                        decision_call,
                        status=failure.status,
                        automatic_retry=False,
                    )
            await self._fact(
                "version",
                decision_call,
                status="incremented" if state.version != before else "unchanged",
                workspace_version=state.version,
            )

        if failure is not None:
            await self._fact("post", decision_call, status=failure.status)
            diagnostics = await self.hooks.dispatch_post(
                HookPoint.TOOL_ERROR,
                {"call": decision_call, "result": failure},
            )
            return await self._final(
                _attach_diagnostics(failure, diagnostics), decision_call,
                context,
            )

        assert execution is not None
        await self._fact("post", decision_call, status="started")
        diagnostics = await self.hooks.dispatch_post(
            HookPoint.POST_TOOL_USE,
            {"call": decision_call, "execution": execution},
        )
        result = _attach_diagnostics(
            execution.to_result(original_id), diagnostics
        )
        final_result = await self._final(
            result,
            decision_call,
            context,
            changed_workspace=execution.changed_workspace,
            workspace_version=state.version,
        )
        if final_result.status != "success":
            failed_payload: dict[str, object] = {
                "status": final_result.status,
                "message": _safe_ui_text(final_result.content),
                "metadata": _safe_ui_metadata(final_result.metadata),
                "changed_workspace": execution.changed_workspace,
                "workspace_version": state.version,
            }
            if "artifact_error" in final_result.metadata:
                failed_payload["artifact_error"] = final_result.metadata[
                    "artifact_error"
                ]
            await self._ui(
                context,
                UIEventType.TOOL_EXECUTION_FAILED,
                decision_call,
                failed_payload,
            )
            return final_result

        finished_payload: dict[str, object] = {
            "status": final_result.status,
            "preview": _safe_ui_text(execution.preview),
            "metadata": _safe_ui_metadata(final_result.metadata),
            "changed_workspace": execution.changed_workspace,
            "workspace_version": state.version,
        }
        artifact = final_result.metadata.get("artifact")
        if isinstance(artifact, dict):
            finished_payload["artifact"] = artifact
        await self._ui(
            context,
            UIEventType.TOOL_EXECUTION_FINISHED,
            decision_call,
            finished_payload,
        )
        if decision_call.name == "todo_write":
            todos = final_result.metadata.get("todos")
            if isinstance(todos, list):
                await self._ui(
                    context,
                    UIEventType.TODO_UPDATED,
                    decision_call,
                    {"todos": todos},
                )
        return final_result

    async def _commit_success_shielded(
        self,
        state: WorkspaceState,
        call: ToolCall,
        context: ToolContext,
        round_number: int,
        prepared: PreparedDuplicate | None,
        preview: JsonValue,
        before: int,
        changed_workspace: bool,
    ) -> None:
        async def commit() -> None:
            if changed_workspace:
                state.version += 1
            if prepared is not None:
                await self.duplicates.record_prepared_success(
                    context.agent_session_id,
                    context.workspace_id,
                    before,
                    round_number=round_number,
                    prepared=prepared,
                    preview=preview,
                    post_workspace_version=state.version,
                    round_prevalidated=True,
                )

        task = asyncio.create_task(commit())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise

    async def _finish_pending_version(
        self, pipeline: _PipelineState
    ) -> None:
        state = pipeline.pending_version
        if state is None:
            return
        try:
            await self._increment_exclusive_shielded(state)
        finally:
            pipeline.pending_version = None
    async def _increment_exclusive_shielded(
        self, state: WorkspaceState
    ) -> None:
        async def increment() -> None:
            async with state.lock.write():
                state.version += 1

        task = asyncio.create_task(increment())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
    async def _final(
        self,
        result: ToolResult,
        call: ToolCall,
        context: ToolContext,
        *,
        changed_workspace: bool | None = None,
        workspace_version: int | None = None,
    ) -> ToolResult:
        result = _redacted_bounded_result(result)
        if (
            (
                self.artifact_store is not None
                or self.artifact_store_resolver is not None
            )
            and len(result.content.encode("utf-8")) > TOOL_RESULT_INLINE_BYTES
        ):
            result = await self._offload_result(
                result,
                call,
                context,
                changed_workspace=changed_workspace,
                workspace_version=workspace_version,
            )
        final_facts: dict[str, object] = {"status": result.status}
        if result.status != "success":
            final_facts.update(
                message=result.content,
                metadata=result.metadata,
            )
        await self._fact("final", call, **final_facts)
        return result

    async def _offload_result(
        self,
        result: ToolResult,
        call: ToolCall,
        context: ToolContext,
        *,
        changed_workspace: bool | None,
        workspace_version: int | None,
    ) -> ToolResult:
        try:
            store = self.artifact_store
            if store is None:
                assert self.artifact_store_resolver is not None
                store = self.artifact_store_resolver(context)
            reference = await store.persist(call.id, result.content)
        except Exception:
            metadata = dict(result.metadata)
            metadata.update(artifact_error=True, automatic_retry=False)
            if changed_workspace is not None:
                metadata["changed_workspace"] = changed_workspace
            if workspace_version is not None:
                metadata["workspace_version"] = workspace_version
            metadata = _bounded_metadata(metadata)
            await self._fact(
                "artifact", call, status="failed", automatic_retry=False
            )
            preview = _truncate_utf8(result.content, ARTIFACT_PREVIEW_BYTES)
            content = (
                f"{preview}\n\n" if preview else ""
            ) + "[Full tool output unavailable: artifact persistence failed.]"
            return _redacted_bounded_result(
                ToolResult(
                    result.tool_call_id, "tool_error", content, metadata
                )
            )
        artifact = reference.as_metadata()
        metadata = dict(result.metadata)
        metadata["artifact"] = artifact
        metadata = _bounded_metadata(metadata)
        await self._fact("artifact", call, status="stored", artifact=artifact)
        return _redacted_bounded_result(
            ToolResult(
                result.tool_call_id,
                result.status,
                _artifact_result_content(reference),
                metadata,
            )
        )

    async def _fact(self, stage: str, call: ToolCall, **facts: object) -> None:
        await self.hooks.record_runtime_fact(
            "tool.runtime",
            {
                "stage": stage,
                "tool_call_id": call.id,
                "tool_name": call.name,
                **facts,
            },
        )

    async def _cancellation_cleanup(self, call: ToolCall) -> None:
        await self._bounded_cleanup(
            self._fact(
                "cancel", call, status="cancelled", automatic_retry=False
            )
        )
        await self._bounded_cleanup(
            self.hooks.dispatch_post(
                HookPoint.TOOL_ERROR,
                {"call": call, "status": "cancelled"},
            )
        )
        await self._bounded_cleanup(
            self._fact("final", call, status="cancelled")
        )

    async def _bounded_cleanup(self, awaitable: object) -> None:
        task = asyncio.create_task(awaitable)  # type: ignore[arg-type]
        done, pending = await asyncio.wait(
            {task}, timeout=self._error_hook_timeout
        )
        if done:
            _consume_cleanup_task(task)
            return
        task.cancel()
        _, pending = await asyncio.wait(
            pending, timeout=min(0.01, self._error_hook_timeout)
        )
        for item in pending:
            item.cancel()
            item.add_done_callback(_consume_cleanup_task)


def _consume_cleanup_task(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass

def _transformed_call(
    payload: object, call_id: str, name: str
) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("call")
    if isinstance(value, ToolCall):
        call = value
    elif isinstance(value, Mapping):
        arguments = value.get("arguments")
        if not isinstance(arguments, dict):
            return None
        try:
            call = ToolCall(value.get("id"), value.get("name"), arguments)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return call if call.id == call_id and call.name == name else None


def _validate_tool_arguments(spec: ToolSpec, arguments: dict[str, object]) -> None:
    from jsonschema.validators import validator_for

    validator_class = validator_for(spec.input_schema)
    validator_class.check_schema(spec.input_schema)
    validator_class(spec.input_schema).validate(arguments)


def _safe_validation_message(error: ValidationError) -> str:
    path = _validation_path(error)
    if error.validator == "required":
        missing = _missing_required_property(error)
        if missing is not None:
            return (
                f"Invalid tool arguments: required field {path}.{missing} "
                "is missing"
            )
        return f"Invalid tool arguments: a required field is missing at {path}"
    if error.validator == "type":
        return f"Invalid tool arguments: {path} has the wrong type"
    if error.validator == "additionalProperties":
        return f"Invalid tool arguments: {path} contains unsupported fields"
    if error.validator == "enum":
        return f"Invalid tool arguments: {path} must use an allowed value"
    if error.validator in {
        "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"
    }:
        return f"Invalid tool arguments: {path} is outside the allowed range"
    if error.validator in {"minLength", "maxLength", "pattern", "format"}:
        return f"Invalid tool arguments: {path} has an invalid format"
    return f"Invalid tool arguments at {path}"


def _validation_code(error: ValidationError) -> str:
    validator = error.validator
    return validator if isinstance(validator, str) and validator else "invalid"


def _validation_path(error: ValidationError) -> str:
    parts: list[str] = []
    for value in error.absolute_path:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            parts.append(f"[{value}]")
        elif isinstance(value, str) and value.isidentifier() and len(value) <= 64:
            parts.append(f".{value}")
        else:
            parts.append(".*")
    return "$" + "".join(parts)


def _missing_required_property(error: ValidationError) -> str | None:
    required = error.validator_value
    instance = error.instance
    if not isinstance(required, list) or not isinstance(instance, Mapping):
        return None
    for value in required:
        if (
            isinstance(value, str)
            and value not in instance
            and value.isidentifier()
            and len(value) <= 64
        ):
            return value
    return None


def _permission_request_payload(
    permission: PermissionService,
    spec: ToolSpec,
    call: ToolCall,
    context: ToolContext,
) -> dict[str, object] | None:
    mode = context.metadata.get("permission_mode", "ask")
    mode_value = str(mode)
    try:
        classified = permission.classify(mode_value, spec, context)
    except (TypeError, ValueError):
        return None
    if classified.action != "prompt":
        return None
    try:
        scope = PermissionService._approval_key(spec, call, context)[4]
    except Exception:
        scope = "unknown"
    return {
        "mode": mode_value,
        "risk": str(spec.permission_risk),
        "scope": scope,
        "reason": classified.reason,
    }


def _round_number(context: ToolContext) -> int:
    value = context.metadata.get("round_number", 0)
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        else 0
    )


def _safe_preview(guard: DuplicateGuard, preview: object) -> JsonValue:
    try:
        return guard.freeze_preview(preview)
    except Exception:
        return None


_RESULT_METADATA_BYTES = 16_384
_RESULT_METADATA_STRING_BYTES = 1_000
_RESULT_METADATA_ITEMS = 50
_TOOL_UI_TEXT_BYTES = 1_000


def _safe_ui_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _truncate_utf8(
        current_secret_redactor().redact_text(value), _TOOL_UI_TEXT_BYTES
    )


def _safe_ui_metadata(value: object) -> dict[str, object]:
    redacted = current_secret_redactor().redact_data(value)
    if not isinstance(redacted, dict):
        return {}
    return _bounded_metadata(dict(redacted))

def _redacted_bounded_result(result: ToolResult) -> ToolResult:
    redactor = current_secret_redactor()
    content = redactor.redact_text(result.content)
    metadata = redactor.redact_data(result.metadata)
    if not isinstance(metadata, dict):
        metadata = {}
    return ToolResult(
        result.tool_call_id,
        result.status,
        content,
        _bounded_metadata(metadata),
    )


def _bounded_metadata(metadata: dict[object, object]) -> dict[str, object]:
    bounded: dict[str, object] = {}
    truncated = False
    priority = (
        "artifact",
        "artifact_error",
        "automatic_retry",
        "changed_workspace",
        "workspace_version",
    )
    priority_keys = tuple(key for key in priority if key in metadata)
    remaining_keys = (key for key in metadata if key not in priority)
    for key in chain(priority_keys, remaining_keys):
        if not isinstance(key, str):
            truncated = True
            continue
        value, value_truncated = _bounded_json_value(metadata[key])
        candidate = {**bounded, key: value}
        if len(_json_bytes(candidate)) > _RESULT_METADATA_BYTES:
            truncated = True
            continue
        bounded[key] = value
        truncated = truncated or value_truncated
    if truncated:
        bounded["metadata_truncated"] = True
        while len(_json_bytes(bounded)) > _RESULT_METADATA_BYTES:
            removable = next(
                (
                    key
                    for key in reversed(bounded)
                    if key != "metadata_truncated" and key not in priority
                ),
                None,
            )
            if removable is None:
                removable = next(
                    (
                        key
                        for key in reversed(bounded)
                        if key != "metadata_truncated"
                    ),
                    None,
                )
            if removable is None:
                break
            del bounded[removable]
    return bounded


def _bounded_json_value(value: object) -> tuple[object, bool]:
    if isinstance(value, str):
        rendered = _truncate_utf8(value, _RESULT_METADATA_STRING_BYTES)
        return rendered, rendered != value
    if isinstance(value, list):
        items: list[object] = []
        truncated = len(value) > _RESULT_METADATA_ITEMS
        for item in value[:_RESULT_METADATA_ITEMS]:
            bounded, item_truncated = _bounded_json_value(item)
            items.append(bounded)
            truncated = truncated or item_truncated
        return items, truncated
    if isinstance(value, dict):
        items: dict[str, object] = {}
        truncated = len(value) > _RESULT_METADATA_ITEMS
        for key, item in islice(value.items(), _RESULT_METADATA_ITEMS):
            if not isinstance(key, str):
                truncated = True
                continue
            bounded, item_truncated = _bounded_json_value(item)
            items[key] = bounded
            truncated = truncated or item_truncated
        return items, truncated
    return value, False


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def _artifact_result_content(reference: ArtifactReference) -> str:
    prefix = f"{reference.preview}\n\n" if reference.preview else ""
    return (
        prefix
        + f"[Full tool output saved to {reference.path} "
        + f"({reference.bytes} UTF-8 bytes).]"
    )

def _attach_diagnostics(
    result: ToolResult, diagnostics: list[HookDiagnostic]
) -> ToolResult:
    if not diagnostics:
        return result
    metadata = dict(result.metadata)
    metadata["hook_diagnostics"] = [
        {
            "hook_id": item.hook_id,
            "point": item.point.value,
            "phase": item.phase,
            "kind": item.kind,
            "code": item.code,
            "message": item.message,
        }
        for item in diagnostics
    ]
    return ToolResult(
        result.tool_call_id, result.status, result.content, metadata
    )


async def _tool_hard_guard(
    tool: object, call: ToolCall, context: ToolContext
) -> str | None:
    guard = getattr(tool, "hard_guard", None)
    if guard is None:
        return None
    try:
        outcome = guard(call, context)
        reason = await outcome if inspect.isawaitable(outcome) else outcome
    except Exception:
        return "Mandatory workspace safety guard failed"
    if reason is None:
        return None
    if isinstance(reason, str) and reason:
        return reason
    return "Denied by workspace safety policy"
