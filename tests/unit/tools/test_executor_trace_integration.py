from __future__ import annotations

import asyncio
import json

import pytest

from litecoder.common.trace import (
    SecretRedactor, TraceContext, TraceRecorder, bind_secret_redactor,
    trace_annotation,
)
from litecoder.hooks import HookManager, HookOutcome, HookPoint, TraceHook
from litecoder.tools import (
    DuplicateGuard, PermissionService, ToolCall, ToolContext, ToolExecution,
    ToolFailure,
    ToolExecutor, ToolRegistry, ToolSpec, WorkspaceStateRegistry,
)
from tests.unit.tools.test_executor_pipeline import FakeTool, RecordingTrace, build, ctx


@pytest.mark.asyncio
async def test_real_trace_stage_order_annotation_identifiers_sequence_and_redaction(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(["secret-value"]))
    await recorder.start()
    registry = ToolRegistry()
    registry.register(FakeTool(ToolSpec("read", "read", {}, False)))
    executor = ToolExecutor(registry, HookManager(), DuplicateGuard(), PermissionService(), WorkspaceStateRegistry())
    context = TraceContext.root("trace-1", "root-session", "agent", recorder)
    with context.bind():
        tool_context = ToolContext("agent", "workspace", tmp_path, metadata={"round_number": 1})
        first = ToolCall("one", "read", {"token": "secret-value"})
        assert (await executor.execute(first, tool_context)).status == "success"
        assert (await executor.execute(ToolCall("two", "read", first.arguments), tool_context)).status == "duplicate_blocked"
    await recorder.close()

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    runtime = [row for row in rows if row["event"] == "tool.runtime"]
    assert [row["stage"] for row in runtime] == [
        "registry", "pre", "duplicate", "permission", "execute", "version", "post", "final",
        "registry", "pre", "duplicate", "final",
    ]
    annotation = next(row for row in rows if row["event"] == "trace.annotation")
    duplicate = next(row for row in runtime if row["stage"] == "duplicate" and row["status"] == "blocked")
    assert annotation["sequence"] < duplicate["sequence"]
    assert {row["trace_id"] for row in runtime} == {"trace-1"}
    assert {row["span_id"] for row in runtime} == {"root"}
    assert runtime[1]["arguments"]["token"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_mandatory_registry_fact_failure_is_not_tool_mutation(tmp_path):
    tool = FakeTool(ToolSpec("write", "write", {}, True))
    executor, _, workspaces = build(tool, RecordingTrace("registry"))
    with pytest.raises(RuntimeError, match="mandatory trace failure"):
        await executor.execute(ToolCall("one", "write", {}), ctx(tmp_path))
    assert tool.calls == 0
    assert workspaces.get("workspace").version == 0


@pytest.mark.asyncio
async def test_read_error_does_not_increment_version(tmp_path):
    async def fail(*_):
        raise RuntimeError("secret-error")
    executor, _, workspaces = build(FakeTool(ToolSpec("read", "read", {}, False), fail))
    result = await executor.execute(ToolCall("one", "read", {}), ctx(tmp_path))
    assert result.status == "tool_error"
    assert workspaces.get("workspace").version == 0


_EXPOSURE_MESSAGE = "sensitive value was exposed"


def _assert_secret_absent(secret: str, rendered: str) -> None:
    if secret in rendered:
        pytest.fail(_EXPOSURE_MESSAGE, pytrace=False)


@pytest.mark.asyncio
async def test_tool_trace_scope_redacts_the_entire_pipeline_with_empty_recorder(
    tmp_path,
):
    trace_path = tmp_path / "trace-empty-redactor.jsonl"
    secrets = {
        "write_file": "-".join(("write", "runtime", "secret")),
        "edit_file": "-".join(("edit", "runtime", "secret")),
        "run_shell": "-".join(("shell", "runtime", "secret")),
        "search_text": "-".join(("search", "runtime", "secret")),
        "hook": "-".join(("hook", "runtime", "secret")),
        "annotation": "-".join(("annotation", "runtime", "secret")),
        "duplicate": "-".join(("duplicate", "runtime", "secret")),
        "result": "-".join(("result", "runtime", "secret")),
        "error": "-".join(("error", "runtime", "secret")),
    }
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    registry = ToolRegistry()

    async def execute(call, _context):
        if call.name == "error_tool":
            raise ToolFailure(
                f"failed {secrets['error']}",
                metadata={"detail": secrets["error"]},
            )
        await trace_annotation(
            intent=f"inspect {secrets['annotation']}",
            reason=None,
            attributes={
                "annotation_detail": secrets["annotation"],
                "call_arguments": call.arguments,
            },
        )
        return ToolExecution.success(
            f"result {secrets['result']}",
            metadata={"detail": secrets["result"]},
            preview=secrets["result"],
        )

    for name in ("write_file", "edit_file", "run_shell", "error_tool"):
        registry.register(
            FakeTool(
                ToolSpec(name, name, {}, False, dedupe_policy="none"),
                execute,
            )
        )
    registry.register(
        FakeTool(ToolSpec("search_text", "search_text", {}, False), execute)
    )

    async def duplicate_annotation(**annotation):
        attributes = dict(annotation["attributes"])
        attributes["duplicate_detail"] = secrets["duplicate"]
        await trace_annotation(
            intent=annotation["intent"],
            reason=annotation["reason"],
            attributes=attributes,
        )

    hooks = HookManager()
    executor = ToolExecutor(
        registry,
        hooks,
        DuplicateGuard(annotation=duplicate_annotation),
        PermissionService(),
        WorkspaceStateRegistry(),
    )
    expected_arguments = {
        "write_file": ("content", secrets["write_file"]),
        "edit_file": ("new_text", secrets["edit_file"]),
        "run_shell": ("argv", ["program", secrets["run_shell"]]),
        "search_text": ("query", secrets["search_text"]),
        "error_tool": ("message", secrets["error"]),
    }
    hook_observations: list[bool] = []

    async def pre_hook(envelope):
        call = envelope.payload["call"]
        key, expected = expected_arguments[call.name]
        hook_observations.append(call.arguments[key] == expected)
        call.arguments["hook_detail"] = secrets["hook"]
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.PRE_TOOL_USE, pre_hook, name="secret-observer")
    tool_context = ToolContext(
        "agent",
        "workspace",
        tmp_path,
        metadata={"round_number": 1},
        secret_values=tuple(secrets.values()),
    )
    trace_context = TraceContext.root(
        "trace-scoped", "root-session", "agent", recorder
    )
    calls = [
        ToolCall("write", "write_file", {"content": secrets["write_file"]}),
        ToolCall(
            "edit",
            "edit_file",
            {"old_text": "old", "new_text": secrets["edit_file"]},
        ),
        ToolCall(
            "shell",
            "run_shell",
            {"argv": ["program", secrets["run_shell"]]},
        ),
        ToolCall(
            "search-one",
            "search_text",
            {"query": secrets["search_text"]},
        ),
        ToolCall(
            "search-two",
            "search_text",
            {"query": secrets["search_text"]},
        ),
        ToolCall("error", "error_tool", {"message": secrets["error"]}),
    ]
    with trace_context.bind():
        results = [
            await executor.execute(tool_call, tool_context)
            for tool_call in calls
        ]
    await recorder.close()

    assert hook_observations == [True] * len(calls)
    assert [result.status for result in results] == [
        "success",
        "success",
        "success",
        "success",
        "duplicate_blocked",
        "tool_error",
    ]
    rendered = trace_path.read_text(encoding="utf-8")
    for secret in secrets.values():
        _assert_secret_absent(secret, rendered)
    assert "[REDACTED]" in rendered

@pytest.mark.asyncio
async def test_trace_facts_omit_tool_context_non_repr_secret_fields(tmp_path):
    secret = "-".join(("context", "field", "secret"))
    trace_path = tmp_path / "trace-hidden-fields.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    trace_context = TraceContext.root(
        "trace-fields", "root-session", "agent", recorder
    )
    tool_context = ToolContext(
        "agent",
        "workspace",
        tmp_path,
        secret_values=(secret,),
    )

    with trace_context.bind(), bind_secret_redactor(tool_context.redactor):
        await TraceHook().record({"tool_context": tool_context})
    await recorder.close()

    row = json.loads(trace_path.read_text(encoding="utf-8"))
    serialized = row["tool_context"]
    assert "secret_values" not in serialized
    assert "_redactor" not in serialized
    _assert_secret_absent(secret, repr(row))


@pytest.mark.asyncio
async def test_cancellation_error_hook_does_not_receive_tool_context_secrets(
    tmp_path,
):
    secret = "-".join(("cancel", "context", "secret"))
    trace_path = tmp_path / "trace-cancel-context.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    entered = asyncio.Event()

    async def wait_forever(_call, _context):
        entered.set()
        await asyncio.Event().wait()
        return ToolExecution.success("unreachable")

    registry = ToolRegistry()
    registry.register(
        FakeTool(
            ToolSpec("cancel_tool", "cancel_tool", {}, False),
            wait_forever,
        )
    )
    hooks = HookManager()
    observed_safe_payload = False

    async def error_hook(envelope):
        nonlocal observed_safe_payload
        observed_safe_payload = "context" not in envelope.payload
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.TOOL_ERROR, error_hook, name="cancel-observer")
    executor = ToolExecutor(
        registry,
        hooks,
        DuplicateGuard(),
        PermissionService(),
        WorkspaceStateRegistry(),
    )
    tool_context = ToolContext(
        "agent",
        "workspace",
        tmp_path,
        metadata={"round_number": 1},
        secret_values=(secret,),
    )
    trace_context = TraceContext.root(
        "trace-cancel", "root-session", "agent", recorder
    )

    with trace_context.bind():
        task = asyncio.create_task(
            executor.execute(
                ToolCall("cancel", "cancel_tool", {"value": secret}),
                tool_context,
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await recorder.close()

    assert observed_safe_payload is True
    _assert_secret_absent(secret, trace_path.read_text(encoding="utf-8"))
