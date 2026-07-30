from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from litecoder.common.trace.context import TraceContext
from litecoder.common.trace.emit import trace_annotation
from litecoder.common.trace.recorder import TraceRecorder
from litecoder.common.trace.redaction import SecretRedactor
from litecoder.hooks import (
    HookDiagnostic,
    HookEnvelope,
    HookManager,
    HookOutcome,
    HookPoint,
    TraceHook,
)


class RecordingTraceHook:
    def __init__(self) -> None:
        self.facts: list[dict[str, object]] = []

    async def record(self, fact: Mapping[str, object]) -> None:
        self.facts.append(dict(fact))


class StatefulSnapshot:
    def __init__(self, fail_on: int, secret: str = "snapshot-secret") -> None:
        self.fail_on = fail_on
        self.secret = secret
        self.copies = 0

    def __deepcopy__(self, memo: dict[int, object]) -> StatefulSnapshot:
        self.copies += 1
        if self.copies == self.fail_on:
            raise RuntimeError(self.secret)
        return self


class HeldTraceHook(RecordingTraceHook):
    def __init__(self, event: str, status: str) -> None:
        super().__init__()
        self.event = event
        self.status = status
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._held = False

    async def record(self, fact: Mapping[str, object]) -> None:
        if (
            not self._held
            and fact.get("event") == self.event
            and fact.get("status") == self.status
        ):
            self._held = True
            self.entered.set()
            await self.release.wait()
        await super().record(fact)


def _diagnostic(
    *,
    hook_id: str = "user-check",
    point: HookPoint = HookPoint.PRE_TOOL_USE,
    phase: str = "pre",
    kind: str = "notice",
    code: str = "user_notice",
    message: str = "Hook reported a diagnostic.",
) -> HookDiagnostic:
    return HookDiagnostic(
        hook_id=hook_id,
        point=point,
        phase=phase,
        kind=kind,
        code=code,
        message=message,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hook_points_are_the_approved_nine_values() -> None:
    assert {point.value for point in HookPoint} == {
        "UserPromptSubmit",
        "PreModelCall",
        "PostModelCall",
        "PreToolUse",
        "PostToolUse",
        "ToolError",
        "AgentStop",
        "SubagentStart",
        "SubagentStop",
    }


def test_envelope_and_outcome_take_independent_deep_snapshots() -> None:
    source = {"nested": {"items": ["before"]}}
    envelope = HookEnvelope(
        point=HookPoint.PRE_TOOL_USE,
        payload=source,
        hook_id="rewrite",
        dispatch_id="dispatch-1",
        phase="pre",
    )
    outcome = HookOutcome(payload=envelope.payload)

    source["nested"]["items"].append("source-change")
    envelope.payload["nested"]["items"].append("envelope-change")

    assert envelope.payload == {
        "nested": {"items": ["before", "envelope-change"]}
    }
    assert outcome.payload == {"nested": {"items": ["before"]}}
    with pytest.raises(FrozenInstanceError):
        envelope.hook_id = "replacement"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_pre_hooks_run_in_order_and_feed_snapshotted_mutations_forward() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    caller = {"arguments": {"path": "a.py"}, "aliases": [[]]}
    seen: list[tuple[str, object]] = []
    retained: list[HookOutcome] = []

    async def rewrite(envelope: HookEnvelope) -> HookOutcome:
        seen.append((envelope.hook_id, envelope.payload["arguments"]["path"]))
        envelope.payload["arguments"]["path"] = "src/a.py"
        result = HookOutcome(payload=envelope.payload)
        retained.append(result)
        return result

    async def observe(envelope: HookEnvelope) -> HookOutcome:
        seen.append((envelope.hook_id, envelope.payload["arguments"]["path"]))
        envelope.payload["aliases"][0].append("hook-only")
        return HookOutcome(payload=envelope.payload)

    first = manager.register(HookPoint.PRE_TOOL_USE, rewrite, name="rewrite")
    second = manager.register(HookPoint.PRE_TOOL_USE, observe, name="observe")
    outcome = await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, caller)

    retained[0].payload["arguments"]["path"] = "late-change"
    assert first != second
    assert seen == [("rewrite", "a.py"), ("observe", "src/a.py")]
    assert outcome.payload == {
        "arguments": {"path": "src/a.py"},
        "aliases": [["hook-only"]],
    }
    assert caller == {"arguments": {"path": "a.py"}, "aliases": [[]]}
    assert outcome.blocked is False
    assert [fact["event"] for fact in trace.facts] == [
        "hook.dispatch.start",
        "hook.invocation.start",
        "hook.invocation.end",
        "hook.invocation.start",
        "hook.invocation.end",
        "hook.dispatch.end",
    ]
    assert [
        fact["mutated"]
        for fact in trace.facts
        if fact["event"] == "hook.invocation.end"
    ] == [True, True]


@pytest.mark.asyncio
async def test_pre_block_short_circuits_with_accumulated_diagnostics() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    called: list[str] = []

    async def first(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(payload=envelope.payload, diagnostics=(_diagnostic(),))

    async def blocker(envelope: HookEnvelope) -> HookOutcome:
        envelope.payload["blocked_by"] = envelope.hook_id
        return HookOutcome(
            payload=envelope.payload,
            blocked=True,
            diagnostics=(
                _diagnostic(
                    hook_id="blocker",
                    kind="block",
                    code="policy_block",
                    message="Operation was blocked by policy.",
                ),
            ),
        )

    async def later(envelope: HookEnvelope) -> HookOutcome:
        called.append(envelope.hook_id)
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, first, name="first")
    manager.register(HookPoint.PRE_TOOL_USE, blocker, name="blocker")
    manager.register(HookPoint.PRE_TOOL_USE, later, name="later")

    outcome = await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {"value": 1})

    assert outcome.blocked is True
    assert outcome.payload == {"value": 1, "blocked_by": "blocker"}
    assert [item.code for item in outcome.diagnostics] == [
        "user_notice",
        "policy_block",
    ]
    assert called == []
    assert trace.facts[-1]["status"] == "blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_hook_outcome_error_diagnostic_is_traced_with_its_code(
    phase: str,
) -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    point = HookPoint.PRE_TOOL_USE if phase == "pre" else HookPoint.POST_TOOL_USE

    async def failing(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(
            payload=envelope.payload,
            diagnostics=(
                _diagnostic(
                    point=point,
                    phase=phase,
                    kind="notice",
                    code="before_error",
                ),
                _diagnostic(
                    point=point,
                    phase=phase,
                    kind="error",
                    code="external_failure",
                    message="external diagnostic text",
                ),
                _diagnostic(
                    point=point,
                    phase=phase,
                    kind="error",
                    code="later_error",
                ),
            ),
        )

    manager.register(point, failing, name="failing")

    if phase == "pre":
        outcome = await manager.dispatch_pre(point, {"value": 1})
        assert outcome.blocked is False
    else:
        diagnostics = await manager.dispatch_post(point, {"value": 1})
        assert [item.code for item in diagnostics] == [
            "before_error",
            "external_failure",
            "later_error",
        ]

    invocation_end = next(
        fact for fact in trace.facts if fact["event"] == "hook.invocation.end"
    )
    dispatch_end = trace.facts[-1]
    for fact in (invocation_end, dispatch_end):
        assert fact["error"] is True
        assert fact["error_code"] == "external_failure"
        assert "message" not in fact
        assert "external diagnostic text" not in repr(fact)


@pytest.mark.asyncio
async def test_policy_block_trace_is_not_an_error() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)

    async def blocker(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(
            payload=envelope.payload,
            blocked=True,
            diagnostics=(_diagnostic(kind="block", code="policy_block"),),
        )

    manager.register(HookPoint.PRE_TOOL_USE, blocker, name="blocker")

    outcome = await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {"value": 1})

    assert outcome.blocked is True
    invocation_end = next(
        fact for fact in trace.facts if fact["event"] == "hook.invocation.end"
    )
    dispatch_end = trace.facts[-1]
    for fact in (invocation_end, dispatch_end):
        assert fact["error"] is False
        assert fact["error_code"] is None


@pytest.mark.asyncio
async def test_pre_exception_fails_closed_without_exception_text() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    secret = "hook-exception-secret"

    async def broken(envelope: HookEnvelope) -> HookOutcome:
        raise RuntimeError(secret)

    manager.register(HookPoint.PRE_TOOL_USE, broken, name="broken")
    outcome = await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {"value": 1})

    assert outcome.blocked is True
    assert len(outcome.diagnostics) == 1
    diagnostic = outcome.diagnostics[0]
    assert diagnostic == HookDiagnostic(
        hook_id="broken",
        point=HookPoint.PRE_TOOL_USE,
        phase="pre",
        kind="error",
        code="hook_exception",
        message="Hook execution failed.",
    )
    assert secret not in repr(outcome)
    assert secret not in repr(trace.facts)
    assert trace.facts[-2]["status"] == "failed"
    assert trace.facts[-1]["status"] == "blocked"


@pytest.mark.asyncio
async def test_pre_invalid_return_type_fails_closed() -> None:
    manager = HookManager(trace_hook=RecordingTraceHook())

    async def invalid(envelope: HookEnvelope) -> object:
        return {"payload": envelope.payload}

    manager.register(HookPoint.PRE_MODEL_CALL, invalid, name="invalid")
    outcome = await manager.dispatch_pre(HookPoint.PRE_MODEL_CALL, {"model": "x"})

    assert outcome.blocked is True
    assert outcome.diagnostics[0].code == "invalid_hook_outcome"
    assert outcome.diagnostics[0].message == "Hook returned an invalid outcome."


@pytest.mark.asyncio
async def test_post_mutations_and_blocks_are_ignored_and_reported() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    caller = {"execution": {"status": "success"}}
    observed: list[str] = []

    async def attempted_change(envelope: HookEnvelope) -> HookOutcome:
        envelope.payload["execution"]["status"] = "rewritten"
        return HookOutcome(payload=envelope.payload, blocked=True)

    async def observe_authoritative(envelope: HookEnvelope) -> HookOutcome:
        observed.append(envelope.payload["execution"]["status"])
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.POST_TOOL_USE, attempted_change, name="attempt")
    manager.register(HookPoint.POST_TOOL_USE, observe_authoritative, name="observe")
    diagnostics = await manager.dispatch_post(HookPoint.POST_TOOL_USE, caller)

    assert caller == {"execution": {"status": "success"}}
    assert observed == ["success"]
    assert [item.code for item in diagnostics] == [
        "post_mutation_ignored",
        "post_block_ignored",
    ]
    attempt_end = [
        fact
        for fact in trace.facts
        if fact.get("event") == "hook.invocation.end"
        and fact.get("hook_id") == "attempt"
    ][0]
    assert attempt_end["mutated"] is True
    assert attempt_end["blocked"] is True
    assert trace.facts[-1]["status"] == "completed_with_diagnostics"


@pytest.mark.asyncio
async def test_post_failures_are_sanitized_and_later_hooks_continue() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    called: list[str] = []
    secret = "post-hook-secret"

    async def broken(envelope: HookEnvelope) -> HookOutcome:
        raise ValueError(secret)

    async def invalid(envelope: HookEnvelope) -> object:
        return None

    async def later(envelope: HookEnvelope) -> HookOutcome:
        called.append(envelope.hook_id)
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.POST_MODEL_CALL, broken, name="broken")
    manager.register(HookPoint.POST_MODEL_CALL, invalid, name="invalid")
    manager.register(HookPoint.POST_MODEL_CALL, later, name="later")
    diagnostics = await manager.dispatch_post(
        HookPoint.POST_MODEL_CALL, {"response": "ok"}
    )

    assert called == ["later"]
    assert [item.code for item in diagnostics] == [
        "hook_exception",
        "invalid_hook_outcome",
    ]
    assert secret not in repr(diagnostics)
    assert secret not in repr(trace.facts)


def test_registration_handles_are_stable_and_only_user_hooks_can_be_removed() -> None:
    manager = HookManager(trace_hook=RecordingTraceHook())

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(payload=envelope.payload)

    first = manager.register(HookPoint.PRE_TOOL_USE, hook, name="first")
    second = manager.register(HookPoint.POST_TOOL_USE, hook, name="second")

    assert manager.unregister(first) is True
    assert manager.unregister(first) is False
    with pytest.raises(ValueError, match="reserved"):
        manager.unregister("trace-hook")
    with pytest.raises(ValueError, match="reserved"):
        manager.register(HookPoint.PRE_TOOL_USE, hook, name="trace-hook")
    with pytest.raises(ValueError, match="already registered"):
        manager.register(HookPoint.POST_TOOL_USE, hook, name="second")

    manager.clear()
    assert manager.unregister(second) is False


@pytest.mark.asyncio
async def test_clear_point_does_not_affect_other_points_or_mandatory_trace() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    called: list[str] = []

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        called.append(envelope.hook_id)
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="pre-tool")
    manager.register(HookPoint.PRE_MODEL_CALL, hook, name="pre-model")
    manager.clear(HookPoint.PRE_TOOL_USE)

    await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {})
    await manager.dispatch_pre(HookPoint.PRE_MODEL_CALL, {})

    assert called == ["pre-model"]
    assert sum(fact["event"] == "hook.dispatch.start" for fact in trace.facts) == 2


@pytest.mark.asyncio
async def test_cancellation_is_traced_and_propagates_without_later_hooks() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    entered = asyncio.Event()
    release = asyncio.Event()
    later_called = False

    async def waiting(envelope: HookEnvelope) -> HookOutcome:
        entered.set()
        await release.wait()
        return HookOutcome(payload=envelope.payload)

    async def later(envelope: HookEnvelope) -> HookOutcome:
        nonlocal later_called
        later_called = True
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, waiting, name="waiting")
    manager.register(HookPoint.PRE_TOOL_USE, later, name="later")
    task = asyncio.create_task(manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {}))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert later_called is False
    assert [fact["status"] for fact in trace.facts if "status" in fact][-2:] == [
        "cancelled",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_payload_snapshot_failure_blocks_safely_before_user_hook() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    called = False

    class Unsnapshotable:
        def __deepcopy__(self, memo: dict[int, object]) -> object:
            raise RuntimeError("snapshot-secret")

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        nonlocal called
        called = True
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="never-called")
    outcome = await manager.dispatch_pre(
        HookPoint.PRE_TOOL_USE, {"value": Unsnapshotable()}
    )

    assert called is False
    assert outcome.blocked is True
    assert outcome.payload is None
    assert outcome.diagnostics[0].code == "payload_snapshot_failed"
    assert "snapshot-secret" not in repr(outcome)
    assert "snapshot-secret" not in repr(trace.facts)


@pytest.mark.asyncio
async def test_per_hook_envelope_snapshot_failure_is_traced_and_sanitized() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    called = False

    class FailsOnSecondSnapshot:
        copies = 0

        def __deepcopy__(self, memo: dict[int, object]) -> object:
            self.copies += 1
            if self.copies > 1:
                raise RuntimeError("second-snapshot-secret")
            return self

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        nonlocal called
        called = True
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="never-called")
    value = FailsOnSecondSnapshot()
    outcome = await manager.dispatch_pre(
        HookPoint.PRE_TOOL_USE, {"value": value}
    )

    assert called is False
    assert outcome.blocked is True
    assert outcome.payload == {"value": value}
    assert outcome.diagnostics[0].code == "payload_snapshot_failed"
    assert "second-snapshot-secret" not in repr(outcome)
    assert trace.facts[-1]["status"] == "blocked"


@pytest.mark.asyncio
async def test_default_trace_hook_uses_active_context_recorder_and_redaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    secret = "configured-hook-secret"
    recorder = TraceRecorder(path, SecretRedactor.with_values([secret]))
    await recorder.start()
    context = TraceContext(
        trace_id="trace-1",
        span_id="tool-span",
        parent_span_id="turn-span",
        root_session_id="root-session",
        session_id="child-session",
        agent_id="worker",
        recorder=recorder,
    )
    manager = HookManager()

    with context.bind():
        outcome = await manager.dispatch_pre(
            HookPoint.PRE_TOOL_USE, {"authorization": f"Bearer {secret}"}
        )
        await trace_annotation(
            intent="inspect",
            reason=None,
            attributes={"credential": secret},
        )

    await recorder.close()

    assert outcome.blocked is False
    rendered = path.read_text(encoding="utf-8")
    assert secret not in rendered
    rows = _rows(path)
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["event"] for row in rows] == [
        "hook.dispatch.start",
        "hook.dispatch.end",
        "trace.annotation",
    ]
    for row in rows:
        assert row["trace_id"] == "trace-1"
        assert row["span_id"] == "tool-span"
        assert row["parent_span_id"] == "turn-span"
        assert row["root_session_id"] == "root-session"
        assert row["session_id"] == "child-session"
        assert row["agent_id"] == "worker"


@pytest.mark.asyncio
async def test_default_trace_hook_requires_context_before_user_hooks() -> None:
    manager = HookManager()
    called = False

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        nonlocal called
        called = True
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="not-called")
    with pytest.raises(RuntimeError, match="No active TraceContext"):
        await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {})

    assert called is False


@pytest.mark.asyncio
async def test_mandatory_trace_failure_aborts_dispatch() -> None:
    class FailingSink:
        async def record(self, payload: Mapping[str, object]) -> None:
            raise RuntimeError("recorder unavailable")

    manager = HookManager()
    context = TraceContext.root("trace-1", "session-1", "lead", FailingSink())
    called = False

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        nonlocal called
        called = True
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="not-called")
    with context.bind(), pytest.raises(RuntimeError, match="recorder unavailable"):
        await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {})

    assert called is False


@pytest.mark.asyncio
async def test_trace_failure_after_invocation_is_not_converted_to_diagnostic() -> None:
    class FailingTraceHook(RecordingTraceHook):
        async def record(self, fact: Mapping[str, object]) -> None:
            await super().record(fact)
            if fact["event"] == "hook.invocation.end":
                raise RuntimeError("trace failed")

    trace = FailingTraceHook()
    manager = HookManager(trace_hook=trace)
    called: list[str] = []

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        called.append(envelope.hook_id)
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="first")
    manager.register(HookPoint.PRE_TOOL_USE, hook, name="later")

    with pytest.raises(RuntimeError, match="trace failed"):
        await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {})

    assert called == ["first"]


@pytest.mark.asyncio
async def test_user_hook_envelope_cannot_observe_internal_trace_writes() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    received: list[HookEnvelope] = []

    async def inspect(envelope: HookEnvelope) -> HookOutcome:
        received.append(envelope)
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, inspect, name="inspect")
    caller = {"value": 1}
    await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, caller)

    assert len(received) == 1
    assert received[0].payload == caller
    assert set(received[0].payload) == {"value"}
    assert all(fact is not received[0].payload for fact in trace.facts)


@pytest.mark.asyncio
async def test_deepcopyable_domain_dataclasses_are_supported() -> None:
    @dataclass
    class ToolCall:
        name: str
        arguments: dict[str, object]

    manager = HookManager(trace_hook=RecordingTraceHook())
    caller = ToolCall("read", {"path": "a.py"})

    async def rewrite(envelope: HookEnvelope) -> HookOutcome:
        envelope.payload.arguments["path"] = "src/a.py"
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, rewrite, name="rewrite")
    outcome = await manager.dispatch_pre(HookPoint.PRE_TOOL_USE, caller)

    assert outcome.payload == ToolCall("read", {"path": "src/a.py"})
    assert caller == ToolCall("read", {"path": "a.py"})


def test_trace_hook_is_the_production_mandatory_hook_type() -> None:
    assert TraceHook.__name__ == "TraceHook"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "invalid"])
async def test_pre_failure_returns_last_snapshot_without_copying_again(
    failure: str,
) -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    stateful = StatefulSnapshot(fail_on=3, secret=f"{failure}-return-secret")

    async def hook(envelope: HookEnvelope) -> object:
        if failure == "exception":
            raise RuntimeError("hook-secret")
        return object()

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="failure")
    outcome = await manager.dispatch_pre(
        HookPoint.PRE_TOOL_USE, {"stateful": stateful}
    )

    assert outcome.blocked is True
    assert outcome.payload == {"stateful": stateful}
    assert stateful.copies == 2
    assert "secret" not in repr(outcome)
    assert "secret" not in repr(trace.facts)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True])
async def test_pre_completion_builds_return_before_success_trace_without_extra_copy(
    blocked: bool,
) -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    stateful = StatefulSnapshot(fail_on=5, secret="final-return-secret")

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(payload=envelope.payload, blocked=blocked)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="complete")
    outcome = await manager.dispatch_pre(
        HookPoint.PRE_TOOL_USE, {"stateful": stateful}
    )

    assert outcome.blocked is blocked
    assert outcome.payload == {"stateful": stateful}
    assert stateful.copies == 4
    assert trace.facts[-1]["status"] == ("blocked" if blocked else "completed")


@pytest.mark.asyncio
async def test_pre_hook_payload_snapshot_failure_returns_last_known_good_snapshot() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    stateful = StatefulSnapshot(fail_on=4, secret="hook-payload-secret")

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        envelope.payload["changed"] = True
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="unsnapshotable-result")
    outcome = await manager.dispatch_pre(
        HookPoint.PRE_TOOL_USE, {"stateful": stateful}
    )

    assert outcome.blocked is True
    assert outcome.payload == {"stateful": stateful}
    assert outcome.diagnostics[-1].code == "payload_snapshot_failed"
    assert stateful.copies == 4
    assert "hook-payload-secret" not in repr(outcome)
    assert "hook-payload-secret" not in repr(trace.facts)


@pytest.mark.asyncio
async def test_post_stateful_payload_snapshot_failure_continues_without_leaking() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    stateful = StatefulSnapshot(fail_on=4, secret="post-return-secret")
    called: list[str] = []

    async def failing_snapshot(envelope: HookEnvelope) -> HookOutcome:
        called.append("failing")
        return HookOutcome(payload=envelope.payload)

    async def later(envelope: HookEnvelope) -> HookOutcome:
        called.append("later")
        return HookOutcome(payload={"safe": True})

    manager.register(HookPoint.POST_TOOL_USE, failing_snapshot, name="failing")
    manager.register(HookPoint.POST_TOOL_USE, later, name="later")
    diagnostics = await manager.dispatch_post(
        HookPoint.POST_TOOL_USE, {"stateful": stateful}
    )

    assert called == ["failing", "later"]
    assert [item.code for item in diagnostics] == [
        "payload_snapshot_failed",
        "post_mutation_ignored",
    ]
    assert "post-return-secret" not in repr(diagnostics)
    assert "post-return-secret" not in repr(trace.facts)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_dispatch_membership_is_snapshotted_before_dispatch_start_trace(
    phase: str,
) -> None:
    trace = HeldTraceHook("hook.dispatch.start", "started")
    manager = HookManager(trace_hook=trace)
    called: list[str] = []
    point = HookPoint.PRE_TOOL_USE if phase == "pre" else HookPoint.POST_TOOL_USE

    async def first(envelope: HookEnvelope) -> HookOutcome:
        called.append("first")
        return HookOutcome(payload=envelope.payload)

    async def late(envelope: HookEnvelope) -> HookOutcome:
        called.append("late")
        return HookOutcome(payload=envelope.payload)

    manager.register(point, first, name="first")
    first_dispatch = (
        manager.dispatch_pre(point, {})
        if phase == "pre"
        else manager.dispatch_post(point, {})
    )
    task = asyncio.create_task(first_dispatch)
    await trace.entered.wait()
    manager.register(point, late, name="late")
    trace.release.set()
    await task

    assert called == ["first"]
    second_dispatch = (
        manager.dispatch_pre(point, {})
        if phase == "pre"
        else manager.dispatch_post(point, {})
    )
    await second_dispatch
    assert called == ["first", "first", "late"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "status", "expected_cancelled"),
    [
        ("hook.dispatch.start", "started", ["hook.dispatch.end"]),
        (
            "hook.invocation.start",
            "started",
            ["hook.invocation.end", "hook.dispatch.end"],
        ),
        (
            "hook.invocation.end",
            "completed",
            ["hook.invocation.end", "hook.dispatch.end"],
        ),
        ("hook.dispatch.end", "completed", ["hook.dispatch.end"]),
    ],
)
async def test_pre_cancellation_at_each_trace_phase_is_traced_and_propagated(
    event: str,
    status: str,
    expected_cancelled: list[str],
) -> None:
    trace = HeldTraceHook(event, status)
    manager = HookManager(trace_hook=trace)

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="hook")
    task = asyncio.create_task(manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {}))
    await trace.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [
        fact["event"] for fact in trace.facts if fact.get("status") == "cancelled"
    ] == expected_cancelled


@pytest.mark.asyncio
async def test_post_cancellation_during_user_hook_is_traced_and_propagated() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    entered = asyncio.Event()

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        entered.set()
        await asyncio.Event().wait()
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.POST_TOOL_USE, hook, name="hook")
    task = asyncio.create_task(manager.dispatch_post(HookPoint.POST_TOOL_USE, {}))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [
        fact["event"] for fact in trace.facts if fact.get("status") == "cancelled"
    ] == ["hook.invocation.end", "hook.dispatch.end"]


@pytest.mark.asyncio
async def test_cancellation_cleanup_timeout_cancels_and_drains_trace_task() -> None:
    class CleanupBlockingTrace(RecordingTraceHook):
        def __init__(self) -> None:
            super().__init__()
            self.user_entered = asyncio.Event()
            self.cleanup_entered = asyncio.Event()
            self.cleanup_drained = asyncio.Event()

        async def record(self, fact: Mapping[str, object]) -> None:
            if fact.get("status") == "cancelled":
                self.cleanup_entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cleanup_drained.set()
            await super().record(fact)

    trace = CleanupBlockingTrace()
    manager = HookManager(trace_hook=trace, cancellation_trace_timeout=0.01)

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        trace.user_entered.set()
        await asyncio.Event().wait()
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="hook")
    task = asyncio.create_task(manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {}))
    await trace.user_entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert trace.cleanup_entered.is_set()
    assert trace.cleanup_drained.is_set()

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_user_hook_cannot_suppress_dispatch_cancellation(phase: str) -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    entered = asyncio.Event()

    async def suppressing(envelope: HookEnvelope) -> HookOutcome:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, suppressing, name="suppressing")
    dispatch = (
        manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {})
        if phase == "pre"
        else manager.dispatch_post(HookPoint.PRE_TOOL_USE, {})
    )
    task = asyncio.create_task(dispatch)
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [
        fact["event"] for fact in trace.facts if fact.get("status") == "cancelled"
    ] == ["hook.invocation.end", "hook.dispatch.end"]

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "status", "expected_cancelled"),
    [
        ("hook.dispatch.start", "started", ["hook.dispatch.end"]),
        (
            "hook.invocation.start",
            "started",
            ["hook.invocation.end", "hook.dispatch.end"],
        ),
        (
            "hook.invocation.end",
            "completed",
            ["hook.invocation.end", "hook.dispatch.end"],
        ),
        ("hook.dispatch.end", "completed", ["hook.dispatch.end"]),
    ],
)
async def test_post_cancellation_at_each_trace_phase_is_traced_and_propagated(
    event: str,
    status: str,
    expected_cancelled: list[str],
) -> None:
    trace = HeldTraceHook(event, status)
    manager = HookManager(trace_hook=trace)

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.POST_TOOL_USE, hook, name="hook")
    task = asyncio.create_task(manager.dispatch_post(HookPoint.POST_TOOL_USE, {}))
    await trace.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [
        fact["event"] for fact in trace.facts if fact.get("status") == "cancelled"
    ] == expected_cancelled


@pytest.mark.asyncio
async def test_cancellation_cleanup_trace_failure_preserves_original_cancellation() -> None:
    class FailingCancellationTrace(RecordingTraceHook):
        async def record(self, fact: Mapping[str, object]) -> None:
            if fact.get("status") == "cancelled":
                raise RuntimeError("cleanup-secret")
            await super().record(fact)

    trace = FailingCancellationTrace()
    manager = HookManager(trace_hook=trace)
    entered = asyncio.Event()

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        entered.set()
        await asyncio.Event().wait()
        return HookOutcome(payload=envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, hook, name="hook")
    task = asyncio.create_task(manager.dispatch_pre(HookPoint.PRE_TOOL_USE, {}))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_user_hook_uncancel_cannot_clear_parent_cancellation(phase: str) -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    entered = asyncio.Event()
    child_handled = asyncio.Event()
    later_called = False

    async def uncancelling(envelope: HookEnvelope) -> HookOutcome:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            asyncio.current_task().uncancel()  # type: ignore[union-attr]
            child_handled.set()
            return HookOutcome(payload=envelope.payload)

    async def later(envelope: HookEnvelope) -> HookOutcome:
        nonlocal later_called
        later_called = True
        return HookOutcome(payload=envelope.payload)

    point = HookPoint.PRE_TOOL_USE if phase == "pre" else HookPoint.POST_TOOL_USE
    manager.register(point, uncancelling, name="uncancelling")
    manager.register(point, later, name="later")
    dispatch = (
        manager.dispatch_pre(point, {})
        if phase == "pre"
        else manager.dispatch_post(point, {})
    )
    task = asyncio.create_task(dispatch)
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.1)

    assert child_handled.is_set()
    assert later_called is False
    assert [
        fact["event"] for fact in trace.facts if fact.get("status") == "cancelled"
    ] == ["hook.invocation.end", "hook.dispatch.end"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_trace_hook_uncancel_cannot_clear_parent_cancellation(phase: str) -> None:
    class UncancellingTrace(RecordingTraceHook):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.child_handled = asyncio.Event()

        async def record(self, fact: Mapping[str, object]) -> None:
            if fact.get("event") == "hook.dispatch.start":
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    asyncio.current_task().uncancel()  # type: ignore[union-attr]
                    self.child_handled.set()
                    return
            await super().record(fact)

    trace = UncancellingTrace()
    manager = HookManager(trace_hook=trace)
    hook_called = False
    point = HookPoint.PRE_TOOL_USE if phase == "pre" else HookPoint.POST_TOOL_USE

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        nonlocal hook_called
        hook_called = True
        return HookOutcome(payload=envelope.payload)

    manager.register(point, hook, name="hook")
    dispatch = (
        manager.dispatch_pre(point, {})
        if phase == "pre"
        else manager.dispatch_post(point, {})
    )
    task = asyncio.create_task(dispatch)
    await trace.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.1)

    assert trace.child_handled.is_set()
    assert hook_called is False
    assert any(fact.get("status") == "cancelled" for fact in trace.facts)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_stubborn_cleanup_is_detached_after_hard_timeout(phase: str) -> None:
    class StubbornCleanupTrace(RecordingTraceHook):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_entered = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def record(self, fact: Mapping[str, object]) -> None:
            if fact.get("status") == "cancelled":
                self.cleanup_entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    asyncio.current_task().uncancel()  # type: ignore[union-attr]
                    await self.release.wait()
                finally:
                    self.finished.set()
                return
            await super().record(fact)

    trace = StubbornCleanupTrace()
    manager = HookManager(trace_hook=trace, cancellation_trace_timeout=0.01)
    entered = asyncio.Event()
    point = HookPoint.PRE_TOOL_USE if phase == "pre" else HookPoint.POST_TOOL_USE

    async def hook(envelope: HookEnvelope) -> HookOutcome:
        entered.set()
        await asyncio.Event().wait()
        return HookOutcome(payload=envelope.payload)

    manager.register(point, hook, name="hook")
    dispatch = (
        manager.dispatch_pre(point, {})
        if phase == "pre"
        else manager.dispatch_post(point, {})
    )
    task = asyncio.create_task(dispatch)
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.1)

    assert trace.cleanup_entered.is_set()
    assert manager._background_tasks
    trace.release.set()
    await asyncio.wait_for(trace.finished.wait(), timeout=0.1)
    pending = tuple(manager._background_tasks)
    if pending:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=0.1
        )
    # Done callbacks that discard finished tasks from the tracked set run on
    # the next loop tick; yield once so _background_tasks is observed empty.
    await asyncio.sleep(0)
    assert manager._background_tasks == set()

@pytest.mark.asyncio
async def test_runtime_fact_facade_is_not_visible_and_snapshots_redacts() -> None:
    trace = RecordingTraceHook()
    manager = HookManager(trace_hook=trace)
    user_seen = False

    async def user_hook(envelope: HookEnvelope) -> HookOutcome:
        nonlocal user_seen
        user_seen = True
        return HookOutcome(envelope.payload)

    manager.register(HookPoint.PRE_TOOL_USE, user_hook, name="runtime-user")
    facts = {"stage": "registry", "token": "secret"}
    await manager.record_runtime_fact("tool.runtime", facts)
    facts["stage"] = "changed"
    assert user_seen is False
    assert trace.facts[-1] == {
        "event": "tool.runtime", "stage": "registry", "token": "[REDACTED]"
    }


@pytest.mark.asyncio
async def test_runtime_fact_mandatory_failure_propagates() -> None:
    class Failure:
        async def record(self, fact: Mapping[str, object]) -> None:
            raise RuntimeError("trace-failed")

    with pytest.raises(RuntimeError, match="trace-failed"):
        await HookManager(trace_hook=Failure()).record_runtime_fact(
            "tool.runtime", {"stage": "registry"}
        )
