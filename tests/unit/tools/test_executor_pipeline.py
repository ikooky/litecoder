from __future__ import annotations
import asyncio
from collections.abc import Mapping
from pathlib import Path
import pytest
from litecoder.tools.artifacts import TOOL_RESULT_INLINE_BYTES
from litecoder.hooks import HookManager, HookOutcome, HookPoint
from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecution, ToolExecutor, ToolPartialFailure, ToolRegistry, ToolSpec, WorkspaceStateRegistry

class RecordingTrace:
    def __init__(self, fail_stage=None): self.facts, self.fail_stage = [], fail_stage
    async def record(self, fact: Mapping[str, object]):
        saved = dict(fact); self.facts.append(saved)
        if self.fail_stage is not None and saved.get("stage") == self.fail_stage: raise RuntimeError("mandatory trace failure")

class FakeTool:
    def __init__(self, spec, execute=None): self.spec, self._execute, self.calls = spec, execute, 0
    async def execute(self, call, context):
        self.calls += 1
        return ToolExecution.success("ok", preview="preview") if self._execute is None else await self._execute(call, context)

def ctx(tmp_path, **metadata):
    values = {"round_number": 1, "permission_mode": "ask", "root_session_id": "root"}; values.update(metadata)
    return ToolContext("agent", "workspace", tmp_path, metadata=values)

def build(tool, trace=None, permission=None):
    registry = ToolRegistry(); registry.register(tool); workspaces = WorkspaceStateRegistry(); hooks = HookManager(trace_hook=trace or RecordingTrace())
    return ToolExecutor(registry, hooks, DuplicateGuard(annotation=lambda **_: None), permission or PermissionService(prompt=lambda _: "Allow once"), workspaces), hooks, workspaces

@pytest.mark.asyncio
async def test_pipeline_order_transformation_and_noop_version(tmp_path):
    seen = []
    async def execute(call, _): seen.append(call.arguments); return ToolExecution.success("done", preview="done")
    trace = RecordingTrace(); executor, hooks, workspaces = build(FakeTool(ToolSpec("write", "write", {}, True), execute), trace)
    async def transform(envelope):
        payload = envelope.payload; payload["call"].arguments["text"] = "changed"; return HookOutcome(payload)
    hooks.register(HookPoint.PRE_TOOL_USE, transform, name="transform")
    result = await executor.execute(ToolCall("original", "write", {"text": "old"}), ctx(tmp_path))
    assert (result.tool_call_id, result.status, seen, workspaces.get("workspace").version) == ("original", "success", [{"text": "changed"}], 0)
    assert [f["stage"] for f in trace.facts if f.get("event") == "tool.runtime"] == ["registry", "pre", "duplicate", "permission", "execute", "version", "post", "final"]
    assert next(f for f in trace.facts if f.get("stage") == "pre")["arguments"] == {"text": "changed"}

@pytest.mark.asyncio
async def test_pre_hook_accepts_json_call_mapping(tmp_path):
    seen = []

    async def execute(call, _):
        seen.append(call.arguments)
        return ToolExecution.success("done")

    tool = FakeTool(
        ToolSpec(
            "write",
            "write",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            True,
        ),
        execute,
    )
    executor, hooks, _ = build(tool)

    async def transform(envelope):
        call = envelope.payload["call"]
        return HookOutcome(
            {
                "call": {
                    "id": call.id,
                    "name": call.name,
                    "arguments": {"text": "changed"},
                }
            }
        )

    hooks.register(HookPoint.PRE_TOOL_USE, transform, name="json-transform")
    result = await executor.execute(
        ToolCall("original", "write", {"text": "old"}), ctx(tmp_path)
    )

    assert result.status == "success"
    assert seen == [{"text": "changed"}]


@pytest.mark.asyncio
async def test_duplicate_prevents_permission_prompt(tmp_path):
    prompts = 0
    async def prompt(_):
        nonlocal prompts; prompts += 1; return "Allow once"
    tool = FakeTool(ToolSpec("external", "external", {}, False, permission_risk="external")); executor, _, _ = build(tool, permission=PermissionService(prompt=prompt))
    call = ToolCall("one", "external", {"query": "same"}); await executor.execute(call, ctx(tmp_path))
    result = await executor.execute(ToolCall("two", "external", call.arguments), ctx(tmp_path))
    assert (result.status, result.tool_call_id, prompts) == ("duplicate_blocked", "two", 1)

@pytest.mark.asyncio
async def test_invalid_pre_identity_mutation_fails_closed(tmp_path):
    tool = FakeTool(ToolSpec("read", "read", {}, False)); executor, hooks, _ = build(tool)
    async def replace(_): return HookOutcome({"call": ToolCall("changed", "read", {})})
    hooks.register(HookPoint.PRE_TOOL_USE, replace, name="replace")
    result = await executor.execute(ToolCall("original", "read", {}), ctx(tmp_path))
    assert (result.tool_call_id, result.status, tool.calls) == ("original", "hook_blocked", 0)

@pytest.mark.asyncio
async def test_unknown_tool_is_safe_result(tmp_path):
    executor = ToolExecutor(ToolRegistry(), HookManager(trace_hook=RecordingTrace()), DuplicateGuard(annotation=lambda **_: None), PermissionService(), WorkspaceStateRegistry())
    result = await executor.execute(ToolCall("missing-id", "missing", {"token": "secret"}), ctx(tmp_path))
    assert (result.tool_call_id, result.status) == ("missing-id", "unknown_tool") and "secret" not in result.content

@pytest.mark.asyncio
async def test_mutation_commits_version_and_cache_before_mandatory_post_failure(tmp_path):
    async def mutate(*_): return ToolExecution.success("written", changed_workspace=True, preview="written")
    trace = RecordingTrace("post"); executor, _, workspaces = build(FakeTool(ToolSpec("write", "write", {}, True), mutate), trace); call = ToolCall("one", "write", {"path": "a"})
    with pytest.raises(RuntimeError, match="mandatory trace failure"): await executor.execute(call, ctx(tmp_path))
    trace.fail_stage = None; duplicate = await executor.execute(ToolCall("two", "write", call.arguments), ctx(tmp_path))
    assert (workspaces.get("workspace").version, duplicate.status) == (1, "duplicate_blocked")

@pytest.mark.asyncio
async def test_post_user_diagnostics_only_attach_metadata(tmp_path):
    executor, hooks, _ = build(FakeTool(ToolSpec("read", "read", {}, False)))
    async def bad(_): raise RuntimeError("user-secret")
    hooks.register(HookPoint.POST_TOOL_USE, bad, name="bad"); result = await executor.execute(ToolCall("one", "read", {}), ctx(tmp_path))
    assert (result.status, result.content, result.metadata["hook_diagnostics"][0]["code"]) == ("success", "ok", "hook_exception") and "user-secret" not in repr(result.metadata)

@pytest.mark.asyncio
@pytest.mark.parametrize(("changed", "version"), [(True, 1), (False, 0)])
async def test_partial_failure_versions_dispatches_error_and_never_caches(tmp_path, changed, version):
    async def partial(*_): raise ToolPartialFailure("Partly applied", changed_workspace=changed)
    tool = FakeTool(ToolSpec("write", "write", {}, True), partial); executor, hooks, workspaces = build(tool); observed = []
    async def error_hook(envelope): observed.append(envelope.point); return HookOutcome(envelope.payload)
    hooks.register(HookPoint.TOOL_ERROR, error_hook, name="error"); result = await executor.execute(ToolCall("one", "write", {}), ctx(tmp_path))
    assert (result.status, result.metadata["automatic_retry"], workspaces.get("workspace").version, observed) == ("partial_failure", False, version, [HookPoint.TOOL_ERROR])
    await executor.execute(ToolCall("two", "write", {}), ctx(tmp_path)); assert tool.calls == 2

@pytest.mark.asyncio
async def test_unexpected_mutation_error_is_conservative_and_sanitized(tmp_path):
    async def fail(*_): raise RuntimeError("raw-api-key-secret")
    trace = RecordingTrace()
    executor, _, workspaces = build(FakeTool(ToolSpec("write", "write", {}, True), fail), trace); result = await executor.execute(ToolCall("one", "write", {"token": "secret"}), ctx(tmp_path))
    assert (result.status, result.metadata["automatic_retry"], workspaces.get("workspace").version) == ("tool_error", False, 1)
    error = next(f for f in trace.facts if f.get("stage") == "error")
    final = next(f for f in trace.facts if f.get("stage") == "final")
    assert error["error_type"] == "RuntimeError"
    assert final["message"] == "Tool execution failed"
    assert final["metadata"]["automatic_retry"] is False
    assert "raw-api-key-secret" not in repr(trace.facts)
    assert "raw-api-key-secret" not in repr(result) and "secret" not in repr(result)

@pytest.mark.asyncio
async def test_read_contract_violation_increments_once_and_is_not_cached(tmp_path):
    async def violate(*_): return ToolExecution.success("changed", changed_workspace=True)
    tool = FakeTool(ToolSpec("read", "read", {}, False), violate); executor, _, workspaces = build(tool)
    assert (await executor.execute(ToolCall("one", "read", {}), ctx(tmp_path))).status == "contract_violation" and workspaces.get("workspace").version == 1
    await executor.execute(ToolCall("two", "read", {}), ctx(tmp_path)); assert tool.calls == 2

@pytest.mark.asyncio
async def test_mutating_cancellation_versions_dispatches_error_and_propagates(tmp_path):
    entered = asyncio.Event(); release = asyncio.Event()
    async def wait(*_): entered.set(); await release.wait(); return ToolExecution.success("never")
    executor, hooks, workspaces = build(FakeTool(ToolSpec("write", "write", {}, True), wait)); error_seen = asyncio.Event()
    async def error_hook(envelope): error_seen.set(); return HookOutcome(envelope.payload)
    hooks.register(HookPoint.TOOL_ERROR, error_hook, name="error"); task = asyncio.create_task(executor.execute(ToolCall("one", "write", {}), ctx(tmp_path))); await entered.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert error_seen.is_set() and workspaces.get("workspace").version == 1

@pytest.mark.asyncio
async def test_permission_denial_preserves_id_without_bookkeeping(tmp_path):
    async def deny(_): return "Deny"
    trace = RecordingTrace()
    tool = FakeTool(ToolSpec("deploy", "deploy", {}, True, permission_risk="high")); executor, _, workspaces = build(tool, trace, permission=PermissionService(prompt=deny)); result = await executor.execute(ToolCall("deny-id", "deploy", {}), ctx(tmp_path))
    assert (result.tool_call_id, result.status, tool.calls, workspaces.get("workspace").version) == ("deny-id", "denied", 0, 0)
    permission = next(f for f in trace.facts if f.get("stage") == "permission")
    final = next(f for f in trace.facts if f.get("stage") == "final")
    assert permission["reason"] == "Permission denied"
    assert final["message"] == "Permission denied"


@pytest.mark.asyncio
async def test_hard_guard_reason_is_recorded_in_permission_and_final_facts(tmp_path):
    trace = RecordingTrace()
    tool = FakeTool(ToolSpec("guarded", "guarded", {}, False))
    tool.hard_guard = lambda *_: "Outside workspace"  # type: ignore[attr-defined]
    executor, _, _ = build(tool, trace)

    result = await executor.execute(ToolCall("guarded-id", "guarded", {}), ctx(tmp_path))

    permission = next(f for f in trace.facts if f.get("stage") == "permission")
    final = next(f for f in trace.facts if f.get("stage") == "final")
    assert result.status == "denied"
    assert permission["hard_invariant"] is True
    assert permission["reason"] == "Outside workspace"
    assert final["message"] == "Outside workspace"

@pytest.mark.asyncio
async def test_executor_emits_tool_ui_events_for_success(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolExecution, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    class Tool:
        spec = ToolSpec("read_file", "Read", {"type": "object"}, False)

        async def execute(self, call, context):
            return ToolExecution.success("content", preview="42 lines")

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("call-1", "read_file", {"path": "README.md"}),
        ToolContext("session-1", "workspace", tmp_path, metadata={"root_session_id": "root"}),
    )

    assert result.status == "success"
    assert [event.type for event in sink.events] == [
        UIEventType.TOOL_EXECUTION_STARTED,
        UIEventType.TOOL_EXECUTION_FINISHED,
    ]
    assert [event.sequence for event in sink.events] == [1, 2]
    assert sink.events[0].tool_name == "read_file"
    assert sink.events[1].payload["preview"] == "42 lines"


@pytest.mark.asyncio
async def test_executor_emits_tool_ui_event_for_denied_permission(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolExecution, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    class Tool:
        spec = ToolSpec("run_shell", "Run", {"type": "object"}, False, permission_risk="external")

        async def execute(self, call, context):
            return ToolExecution.success("not reached")

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(prompt=lambda prompt: "Deny"),
        WorkspaceStateRegistry(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("call-1", "run_shell", {"command": "pytest"}),
        ToolContext("session-1", "workspace", tmp_path, metadata={"root_session_id": "root"}),
    )

    assert result.status == "denied"
    assert [event.type for event in sink.events] == [
        UIEventType.PERMISSION_REQUESTED,
        UIEventType.PERMISSION_RESOLVED,
        UIEventType.TOOL_EXECUTION_DENIED,
    ]
    assert [event.sequence for event in sink.events] == [1, 2, 3]
    assert sink.events[0].payload["risk"] == "external"
    assert sink.events[1].payload["allowed"] is False
    assert sink.events[1].payload["reason"] == "Permission denied"
    denied = sink.events[2]
    assert denied.payload["arguments"] == {"command": "pytest"}


@pytest.mark.asyncio
async def test_executor_emits_tool_ui_event_for_tool_failure(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolFailure, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    class Tool:
        spec = ToolSpec("bad_tool", "Bad", {"type": "object"}, False)

        async def execute(self, call, context):
            raise ToolFailure("safe failure", metadata={"exit_code": 35})

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("call-1", "bad_tool", {}),
        ToolContext("session-1", "workspace", tmp_path, metadata={"root_session_id": "root"}),
    )

    assert result.status == "tool_error"
    assert [event.type for event in sink.events] == [
        UIEventType.TOOL_EXECUTION_STARTED,
        UIEventType.TOOL_EXECUTION_FAILED,
    ]
    failures = [event for event in sink.events if event.type is UIEventType.TOOL_EXECUTION_FAILED]
    assert failures
    assert failures[0].payload["message"] == "safe failure"
    assert failures[0].payload["metadata"]["exit_code"] == 35


@pytest.mark.asyncio
async def test_executor_bounds_and_redacts_failed_ui_message(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolFailure, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    secret = "ui-failure-secret"

    class Tool:
        spec = ToolSpec("bad_tool", "Bad", {"type": "object"}, False)

        async def execute(self, call, context):
            del call, context
            raise ToolFailure(
                f"{secret}-" + ("x" * 5_000),
                metadata={"secret": secret, "exit_code": 35},
            )

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("call-1", "bad_tool", {}),
        ToolContext(
            "session-1",
            "workspace",
            tmp_path,
            metadata={"root_session_id": "root"},
            secret_values=(secret,),
        ),
    )

    assert result.status == "tool_error"
    failures = [event for event in sink.events if event.type is UIEventType.TOOL_EXECUTION_FAILED]
    assert failures
    message = failures[0].payload["message"]
    assert isinstance(message, str)
    assert secret not in message
    assert len(message.encode("utf-8")) <= 1_000
    metadata = failures[0].payload["metadata"]
    assert metadata["exit_code"] == 35
    assert secret not in repr(metadata)


@pytest.mark.asyncio
async def test_executor_bounds_and_redacts_finished_ui_preview(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolExecution, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    secret = "ui-preview-secret"

    class Tool:
        spec = ToolSpec("read_file", "Read", {"type": "object"}, False)

        async def execute(self, call, context):
            del call, context
            return ToolExecution.success(
                "content",
                preview=f"{secret}-" + ("x" * 5_000),
            )

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("call-1", "read_file", {"path": "README.md"}),
        ToolContext(
            "session-1",
            "workspace",
            tmp_path,
            metadata={"root_session_id": "root"},
            secret_values=(secret,),
        ),
    )

    assert result.status == "success"
    assert [event.type for event in sink.events] == [
        UIEventType.TOOL_EXECUTION_STARTED,
        UIEventType.TOOL_EXECUTION_FINISHED,
    ]
    preview = sink.events[1].payload["preview"]
    assert isinstance(preview, str)
    assert secret not in preview
    assert len(preview.encode("utf-8")) <= 1_000


class _FailingArtifactStore:
    async def persist(self, tool_call_id: str, content: str):
        del tool_call_id, content
        raise OSError("artifact failure")


@pytest.mark.asyncio
async def test_executor_emits_failed_ui_event_when_final_artifact_persistence_fails(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolExecution, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    class Tool:
        spec = ToolSpec("large_output", "Large", {"type": "object"}, False)

        async def execute(self, call, context):
            del call, context
            return ToolExecution.success(
                "x" * (TOOL_RESULT_INLINE_BYTES + 1),
                preview="large output",
            )

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        artifact_store=_FailingArtifactStore(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("call-1", "large_output", {}),
        ToolContext("session-1", "workspace", tmp_path, metadata={"root_session_id": "root"}),
    )

    assert result.status == "tool_error"
    assert result.metadata["artifact_error"] is True
    assert [event.type for event in sink.events] == [
        UIEventType.TOOL_EXECUTION_STARTED,
        UIEventType.TOOL_EXECUTION_FAILED,
    ]
    assert sink.events[-1].payload["status"] == "tool_error"
    assert sink.events[-1].payload["artifact_error"] is True


class RecordingAsyncLock:
    def __init__(self) -> None:
        self.acquired = 0

    def acquired_async(self) -> object:
        lock = self

        class Context:
            async def __aenter__(self) -> object:
                lock.acquired += 1
                return lock

            async def __aexit__(self, *exc_info: object) -> None:
                return None

        return Context()


@pytest.mark.asyncio
async def test_mutating_tool_acquires_workspace_file_lock(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(ToolSpec("write", "write", {}, True)))
    workspace_lock = RecordingAsyncLock()
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        workspace_lock_resolver=lambda _context: workspace_lock,
    )
    context = ToolContext(
        "session-1",
        "workspace-1",
        tmp_path,
        metadata={"permission_mode": "bypass", "bypass_authorized": True},
    )

    result = await executor.execute(ToolCall("call-1", "write", {}), context)

    assert result.status == "success"
    assert workspace_lock.acquired == 1


@pytest.mark.asyncio
async def test_read_only_tool_does_not_acquire_workspace_file_lock(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(ToolSpec("read", "read", {}, False)))
    workspace_lock = RecordingAsyncLock()
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        workspace_lock_resolver=lambda _context: workspace_lock,
    )
    context = ToolContext(
        "session-1",
        "workspace-1",
        tmp_path,
        metadata={"permission_mode": "bypass", "bypass_authorized": True},
    )

    result = await executor.execute(ToolCall("call-1", "read", {}), context)

    assert result.status == "success"
    assert workspace_lock.acquired == 0

@pytest.mark.asyncio
async def test_executor_emits_explicit_todo_snapshot_event(tmp_path: Path) -> None:
    from litecoder.hooks import HookManager
    from litecoder.tools import DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecutor, ToolExecution, ToolRegistry, ToolSpec, WorkspaceStateRegistry
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.sink import RecordingUISink

    todos = [
        {
            "content": "Verify UI",
            "active_form": "Verifying UI",
            "status": "in_progress",
        }
    ]

    class Tool:
        spec = ToolSpec("todo_write", "Todo", {"type": "object"}, False)

        async def execute(self, call, context):
            return ToolExecution.success(
                "Replaced session TODO list.",
                metadata={"todos": todos},
                preview=todos,
            )

    sink = RecordingUISink()
    registry = ToolRegistry()
    registry.register(Tool())
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
        ui_sink=sink,
        ui_factory_resolver=lambda context: UIEventFactory(session_id=context.agent_session_id, root_session_id="root"),
    )

    result = await executor.execute(
        ToolCall("todo-1", "todo_write", {"todos": todos}),
        ToolContext("session-1", "workspace", tmp_path, metadata={"root_session_id": "root"}),
    )

    assert result.status == "success"
    assert [event.type for event in sink.events] == [
        UIEventType.TOOL_EXECUTION_STARTED,
        UIEventType.TOOL_EXECUTION_FINISHED,
        UIEventType.TODO_UPDATED,
    ]
    assert sink.events[-1].tool_call_id == "todo-1"
    assert sink.events[-1].payload["todos"] == todos
