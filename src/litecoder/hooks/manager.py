"""Context assembly, persistence, and compaction coordination."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

from litecoder.common.trace import TraceSink
from litecoder.common.trace.redaction import current_secret_redactor
from litecoder.hooks.builtin import TraceHook
from litecoder.hooks.models import (
    HookDiagnostic,
    HookEnvelope,
    HookOutcome,
    HookPoint,
)


_TRACE_HANDLE = "trace-hook"
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")

# MandatoryTraceHook is the same contract as TraceSink: a non-user-interceptable
# recorder. Kept as an alias so the HookManager signature reads intentfully.
MandatoryTraceHook = TraceSink

UserHook = Callable[[HookEnvelope], Awaitable[HookOutcome]]


@dataclass(frozen=True, slots=True)
class _Registration:
    """Data model representing the registration."""
    handle: str
    hook_id: str
    hook: UserHook


@dataclass(slots=True)
class _DispatchState:
    """Data model representing the dispatch state."""
    point: HookPoint
    phase: str
    dispatch_id: str
    dispatch_started: float
    hook_id: str | None = None
    invocation_started: float | None = None
    active_child: asyncio.Task[object] | None = None

    def begin_invocation(self, hook_id: str, started: float) -> None:
        """Handle the begin invocation operation."""
        self.hook_id = hook_id
        self.invocation_started = started

    def finish_invocation(self) -> None:
        """Handle the finish invocation operation."""
        self.hook_id = None
        self.invocation_started = None


_CURRENT_DISPATCH: ContextVar[_DispatchState | None] = ContextVar(
    "litecoder_hook_dispatch", default=None
)


class HookManager:
    """Ordered user-hook dispatch with mandatory, non-user-interceptable tracing."""

    def __init__(
        self,
        trace_hook: MandatoryTraceHook | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        cancellation_trace_timeout: float = 1.0,
    ) -> None:
        if cancellation_trace_timeout <= 0:
            raise ValueError("cancellation_trace_timeout must be positive")
        self._trace_hook = trace_hook if trace_hook is not None else TraceHook()
        self._clock = clock
        self._cancellation_trace_timeout = cancellation_trace_timeout
        self._hooks: dict[HookPoint, list[_Registration]] = defaultdict(list)
        self._registrations: dict[str, tuple[HookPoint, _Registration]] = {}
        self._names: set[str] = set()
        self._next_registration = 1
        self._next_dispatch = 1
        self._background_tasks: set[asyncio.Task[object]] = set()

    def register(
        self,
        point: HookPoint,
        hook: UserHook,
        *,
        name: str | None = None,
    ) -> str:
        """Register the requested operation."""
        self._validate_point(point)
        if not callable(hook):
            raise TypeError("hook must be callable")
        handle = f"user-hook-{self._next_registration:06d}"
        hook_id = handle if name is None else name
        if hook_id == _TRACE_HANDLE:
            raise ValueError("trace-hook is reserved for the mandatory TraceHook")
        if not _SAFE_NAME.fullmatch(hook_id):
            raise ValueError("hook name must be a safe identifier")
        if hook_id in self._names:
            raise ValueError(f"hook name is already registered: {hook_id}")

        self._next_registration += 1
        registration = _Registration(handle=handle, hook_id=hook_id, hook=hook)
        self._hooks[point].append(registration)
        self._registrations[handle] = (point, registration)
        self._names.add(hook_id)
        return handle

    def unregister(self, handle: str) -> bool:
        """Remove a hook registration."""
        if handle == _TRACE_HANDLE:
            raise ValueError("trace-hook is reserved and cannot be removed")
        located = self._registrations.pop(handle, None)
        if located is None:
            return False
        point, registration = located
        self._hooks[point].remove(registration)
        self._names.remove(registration.hook_id)
        return True

    def clear(self, point: HookPoint | None = None) -> None:
        """Clear the requested operation."""
        if point is not None:
            self._validate_point(point)
            registrations = tuple(self._hooks.pop(point, ()))
        else:
            registrations = tuple(
                registration
                for values in self._hooks.values()
                for registration in values
            )
            self._hooks.clear()
        for registration in registrations:
            self._registrations.pop(registration.handle, None)
            self._names.discard(registration.hook_id)

    async def record_runtime_fact(
        self, event: str, facts: Mapping[str, object]
    ) -> None:
        """Record a mandatory runtime fact without invoking user hooks."""
        if not isinstance(event, str) or not _SAFE_NAME.fullmatch(event):
            raise ValueError("event must be a safe identifier")
        try:
            snapshot = _snapshot(dict(facts))
        except Exception:
            snapshot = {"snapshot_error": True}
        if not isinstance(snapshot, dict):
            snapshot = {"snapshot_error": True}
        snapshot["event"] = event
        sanitized = current_secret_redactor().redact_data(snapshot)
        state = _DispatchState(
            point=HookPoint.TOOL_ERROR,
            phase="post",
            dispatch_id="runtime-fact",
            dispatch_started=self._clock(),
        )
        await self._await_isolated(self._trace_hook.record(sanitized), state)

    async def dispatch_pre(self, point: HookPoint, payload: object) -> HookOutcome:
        """Handle the dispatch pre operation."""
        self._validate_point(point)
        registrations = tuple(self._hooks[point])
        state = _DispatchState(
            point=point,
            phase="pre",
            dispatch_id=self._new_dispatch_id(),
            dispatch_started=self._clock(),
        )
        token = _CURRENT_DISPATCH.set(state)
        try:
            await self._trace(
                event="hook.dispatch.start",
                point=point.value,
                phase="pre",
                dispatch_id=state.dispatch_id,
                status="started",
                payload=payload,
            )

            try:
                current = _snapshot(payload)
            except Exception:
                diagnostic = _manager_diagnostic(
                    point,
                    "pre",
                    "payload_snapshot_failed",
                    "Payload snapshot failed.",
                )
                outcome = HookOutcome._from_trusted_snapshot(
                    None, blocked=True, diagnostics=(diagnostic,)
                )
                await self._trace_dispatch_end(
                    point=point,
                    phase="pre",
                    dispatch_id=state.dispatch_id,
                    status="blocked",
                    started=state.dispatch_started,
                    blocked=True,
                    error=True,
                    error_code=diagnostic.code,
                    diagnostic_count=1,
                    payload=None,
                )
                return outcome

            diagnostics: list[HookDiagnostic] = []
            for registration in registrations:
                invocation_started = self._clock()
                state.begin_invocation(registration.hook_id, invocation_started)
                await self._trace(
                    event="hook.invocation.start",
                    point=point.value,
                    phase="pre",
                    dispatch_id=state.dispatch_id,
                    hook_id=registration.hook_id,
                    status="started",
                )
                try:
                    envelope = HookEnvelope(
                        point=point,
                        payload=current,
                        hook_id=registration.hook_id,
                        dispatch_id=state.dispatch_id,
                        phase="pre",
                    )
                except Exception:
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "pre",
                        "payload_snapshot_failed",
                        "Payload snapshot failed.",
                    )
                    diagnostics.append(diagnostic)
                    outcome = HookOutcome._from_trusted_snapshot(
                        current, blocked=True, diagnostics=tuple(diagnostics)
                    )
                    await self._trace_invocation_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    await self._trace_dispatch_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        status="blocked",
                        started=state.dispatch_started,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                        diagnostic_count=len(diagnostics),
                        payload=current,
                    )
                    return outcome

                try:
                    result = await self._await_isolated(
                        registration.hook(envelope), state
                    )
                except Exception:
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "pre",
                        "hook_exception",
                        "Hook execution failed.",
                    )
                    diagnostics.append(diagnostic)
                    outcome = HookOutcome._from_trusted_snapshot(
                        current, blocked=True, diagnostics=tuple(diagnostics)
                    )
                    await self._trace_invocation_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    await self._trace_dispatch_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        status="blocked",
                        started=state.dispatch_started,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                        diagnostic_count=len(diagnostics),
                        payload=current,
                    )
                    return outcome

                if not isinstance(result, HookOutcome):
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "pre",
                        "invalid_hook_outcome",
                        "Hook returned an invalid outcome.",
                    )
                    diagnostics.append(diagnostic)
                    outcome = HookOutcome._from_trusted_snapshot(
                        current, blocked=True, diagnostics=tuple(diagnostics)
                    )
                    await self._trace_invocation_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    await self._trace_dispatch_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        status="blocked",
                        started=state.dispatch_started,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                        diagnostic_count=len(diagnostics),
                        payload=current,
                    )
                    return outcome

                try:
                    next_payload = _snapshot(result.payload)
                except Exception:
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "pre",
                        "payload_snapshot_failed",
                        "Payload snapshot failed.",
                    )
                    diagnostics.append(diagnostic)
                    outcome = HookOutcome._from_trusted_snapshot(
                        current, blocked=True, diagnostics=tuple(diagnostics)
                    )
                    await self._trace_invocation_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    await self._trace_dispatch_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        status="blocked",
                        started=state.dispatch_started,
                        blocked=True,
                        error=True,
                        error_code=diagnostic.code,
                        diagnostic_count=len(diagnostics),
                        payload=current,
                    )
                    return outcome

                mutated = _changed(current, next_payload)
                current = next_payload
                diagnostics.extend(result.diagnostics)
                error_code = _first_error_code(result.diagnostics)
                outcome = HookOutcome._from_trusted_snapshot(
                    current,
                    blocked=result.blocked,
                    diagnostics=tuple(diagnostics),
                )
                await self._trace_invocation_end(
                    point=point,
                    phase="pre",
                    dispatch_id=state.dispatch_id,
                    hook_id=registration.hook_id,
                    status="blocked" if result.blocked else "completed",
                    started=invocation_started,
                    mutated=mutated,
                    blocked=result.blocked,
                    error=error_code is not None,
                    error_code=error_code,
                )
                state.finish_invocation()
                if result.blocked:
                    error_code = _first_error_code(diagnostics)
                    await self._trace_dispatch_end(
                        point=point,
                        phase="pre",
                        dispatch_id=state.dispatch_id,
                        status="blocked",
                        started=state.dispatch_started,
                        blocked=True,
                        error=error_code is not None,
                        error_code=error_code,
                        diagnostic_count=len(diagnostics),
                        payload=current,
                    )
                    return outcome

            outcome = HookOutcome._from_trusted_snapshot(
                current, diagnostics=tuple(diagnostics)
            )
            status = "completed_with_diagnostics" if diagnostics else "completed"
            error_code = _first_error_code(diagnostics)
            await self._trace_dispatch_end(
                point=point,
                phase="pre",
                dispatch_id=state.dispatch_id,
                status=status,
                started=state.dispatch_started,
                blocked=False,
                error=error_code is not None,
                error_code=error_code,
                diagnostic_count=len(diagnostics),
                payload=current,
            )
            return outcome
        except asyncio.CancelledError:
            try:
                await self._trace_cancellation(state)
            except asyncio.CancelledError:
                pass
            raise
        finally:
            _CURRENT_DISPATCH.reset(token)

    async def dispatch_post(
        self, point: HookPoint, payload: object
    ) -> list[HookDiagnostic]:
        """Handle the dispatch post operation."""
        self._validate_point(point)
        registrations = tuple(self._hooks[point])
        state = _DispatchState(
            point=point,
            phase="post",
            dispatch_id=self._new_dispatch_id(),
            dispatch_started=self._clock(),
        )
        token = _CURRENT_DISPATCH.set(state)
        try:
            await self._trace(
                event="hook.dispatch.start",
                point=point.value,
                phase="post",
                dispatch_id=state.dispatch_id,
                status="started",
                payload=payload,
            )

            diagnostics: list[HookDiagnostic] = []
            try:
                authoritative = _snapshot(payload)
            except Exception:
                diagnostic = _manager_diagnostic(
                    point,
                    "post",
                    "payload_snapshot_failed",
                    "Payload snapshot failed.",
                )
                diagnostics.append(diagnostic)
                await self._trace_dispatch_end(
                    point=point,
                    phase="post",
                    dispatch_id=state.dispatch_id,
                    status="completed_with_diagnostics",
                    started=state.dispatch_started,
                    blocked=False,
                    error=True,
                    error_code=diagnostic.code,
                    diagnostic_count=1,
                    payload=None,
                )
                return diagnostics

            for registration in registrations:
                invocation_started = self._clock()
                state.begin_invocation(registration.hook_id, invocation_started)
                await self._trace(
                    event="hook.invocation.start",
                    point=point.value,
                    phase="post",
                    dispatch_id=state.dispatch_id,
                    hook_id=registration.hook_id,
                    status="started",
                )
                try:
                    envelope = HookEnvelope(
                        point=point,
                        payload=authoritative,
                        hook_id=registration.hook_id,
                        dispatch_id=state.dispatch_id,
                        phase="post",
                    )
                except Exception:
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "post",
                        "payload_snapshot_failed",
                        "Payload snapshot failed.",
                    )
                    diagnostics.append(diagnostic)
                    await self._trace_invocation_end(
                        point=point,
                        phase="post",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=False,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    continue

                try:
                    result = await self._await_isolated(
                        registration.hook(envelope), state
                    )
                except Exception:
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "post",
                        "hook_exception",
                        "Hook execution failed.",
                    )
                    diagnostics.append(diagnostic)
                    await self._trace_invocation_end(
                        point=point,
                        phase="post",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=False,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    continue

                if not isinstance(result, HookOutcome):
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "post",
                        "invalid_hook_outcome",
                        "Hook returned an invalid outcome.",
                    )
                    diagnostics.append(diagnostic)
                    await self._trace_invocation_end(
                        point=point,
                        phase="post",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=False,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    continue

                try:
                    returned_payload = _snapshot(result.payload)
                except Exception:
                    diagnostic = _hook_diagnostic(
                        registration.hook_id,
                        point,
                        "post",
                        "payload_snapshot_failed",
                        "Payload snapshot failed.",
                    )
                    diagnostics.append(diagnostic)
                    await self._trace_invocation_end(
                        point=point,
                        phase="post",
                        dispatch_id=state.dispatch_id,
                        hook_id=registration.hook_id,
                        status="failed",
                        started=invocation_started,
                        mutated=False,
                        blocked=False,
                        error=True,
                        error_code=diagnostic.code,
                    )
                    state.finish_invocation()
                    continue

                mutated = _changed(authoritative, returned_payload)
                diagnostics.extend(result.diagnostics)
                error_code = _first_error_code(result.diagnostics)
                if mutated:
                    diagnostics.append(
                        _hook_diagnostic(
                            registration.hook_id,
                            point,
                            "post",
                            "post_mutation_ignored",
                            "Post-hook payload mutation was ignored.",
                            kind="ignored",
                        )
                    )
                if result.blocked:
                    diagnostics.append(
                        _hook_diagnostic(
                            registration.hook_id,
                            point,
                            "post",
                            "post_block_ignored",
                            "Post-hook block request was ignored.",
                            kind="ignored",
                        )
                    )
                await self._trace_invocation_end(
                    point=point,
                    phase="post",
                    dispatch_id=state.dispatch_id,
                    hook_id=registration.hook_id,
                    status=(
                        "completed_with_ignored_changes"
                        if mutated or result.blocked
                        else "completed"
                    ),
                    started=invocation_started,
                    mutated=mutated,
                    blocked=result.blocked,
                    error=error_code is not None,
                    error_code=error_code,
                )
                state.finish_invocation()

            status = "completed_with_diagnostics" if diagnostics else "completed"
            error_code = _first_error_code(diagnostics)
            await self._trace_dispatch_end(
                point=point,
                phase="post",
                dispatch_id=state.dispatch_id,
                status=status,
                started=state.dispatch_started,
                blocked=False,
                error=error_code is not None,
                error_code=error_code,
                diagnostic_count=len(diagnostics),
                payload=authoritative,
            )
            return diagnostics
        except asyncio.CancelledError:
            try:
                await self._trace_cancellation(state)
            except asyncio.CancelledError:
                pass
            raise
        finally:
            _CURRENT_DISPATCH.reset(token)

    async def _trace_invocation_end(
        self,
        *,
        point: HookPoint,
        phase: str,
        dispatch_id: str,
        hook_id: str,
        status: str,
        started: float,
        mutated: bool,
        blocked: bool,
        error: bool,
        error_code: str | None,
    ) -> None:
        await self._trace(
            event="hook.invocation.end",
            point=point.value,
            phase=phase,
            dispatch_id=dispatch_id,
            hook_id=hook_id,
            status=status,
            mutated=mutated,
            blocked=blocked,
            error=error,
            error_code=error_code,
            duration_ms=_duration_ms(self._clock() - started),
        )

    async def _trace_dispatch_end(
        self,
        *,
        point: HookPoint,
        phase: str,
        dispatch_id: str,
        status: str,
        started: float,
        blocked: bool,
        error: bool,
        error_code: str | None,
        diagnostic_count: int,
        payload: object,
    ) -> None:
        await self._trace(
            event="hook.dispatch.end",
            point=point.value,
            phase=phase,
            dispatch_id=dispatch_id,
            status=status,
            blocked=blocked,
            error=error,
            error_code=error_code,
            diagnostic_count=diagnostic_count,
            duration_ms=_duration_ms(self._clock() - started),
            payload=payload,
        )

    async def _trace_cancellation(self, state: _DispatchState) -> None:
        facts: list[dict[str, object]] = []
        if state.hook_id is not None and state.invocation_started is not None:
            facts.append(
                {
                    "event": "hook.invocation.end",
                    "point": state.point.value,
                    "phase": state.phase,
                    "dispatch_id": state.dispatch_id,
                    "hook_id": state.hook_id,
                    "status": "cancelled",
                    "mutated": False,
                    "blocked": False,
                    "error": False,
                    "error_code": None,
                    "duration_ms": _duration_ms(
                        self._clock() - state.invocation_started
                    ),
                }
            )
        facts.append(
            {
                "event": "hook.dispatch.end",
                "point": state.point.value,
                "phase": state.phase,
                "dispatch_id": state.dispatch_id,
                "status": "cancelled",
                "blocked": False,
                "error": False,
                "error_code": None,
                "diagnostic_count": 0,
                "duration_ms": _duration_ms(
                    self._clock() - state.dispatch_started
                ),
                "payload": None,
            }
        )

        tasks: list[asyncio.Task[object]] = []
        active = state.active_child
        state.active_child = None
        if active is not None and not active.done():
            active.cancel()
            tasks.append(active)
        tasks.extend(
            self._spawn_tracked(self._trace_hook.record(fact)) for fact in facts
        )

        try:
            _, pending = await asyncio.wait(
                tasks, timeout=self._cancellation_trace_timeout
            )
        except asyncio.CancelledError:
            pending = {task for task in tasks if not task.done()}

        if not pending:
            return
        for task in pending:
            task.cancel()
        grace = min(0.01, self._cancellation_trace_timeout)
        try:
            await asyncio.wait(pending, timeout=grace)
        except asyncio.CancelledError:
            pass

    async def _trace(self, **fact: object) -> None:
        state = _CURRENT_DISPATCH.get()
        if state is None:
            raise RuntimeError("Hook Trace write requires an active dispatch")
        await self._await_isolated(self._trace_hook.record(fact), state)

    async def _await_isolated(
        self, awaitable: Awaitable[object], state: _DispatchState
    ) -> object:
        task = self._spawn_tracked(awaitable)
        state.active_child = task
        try:
            return await asyncio.shield(task)
        finally:
            if state.active_child is task and task.done():
                state.active_child = None

    def _spawn_tracked(
        self, awaitable: Awaitable[object]
    ) -> asyncio.Task[object]:
        if asyncio.iscoroutine(awaitable):
            task = asyncio.create_task(awaitable)
        else:
            async def bridge() -> object:
                """Bridge events into the configured hook manager."""
                return await awaitable

            task = asyncio.create_task(bridge())
        self._background_tasks.add(task)
        task.add_done_callback(self._consume_background_task)
        return task

    def _consume_background_task(self, task: asyncio.Task[object]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    def _new_dispatch_id(self) -> str:
        value = f"dispatch-{self._next_dispatch:08d}"
        self._next_dispatch += 1
        return value

    @staticmethod
    def _validate_point(point: HookPoint) -> None:
        if not isinstance(point, HookPoint):
            raise TypeError("point must be a HookPoint")


def _snapshot(payload: object) -> object:
    return copy.deepcopy(payload)


def _changed(before: object, after: object) -> bool:
    try:
        comparison = before == after
        return not bool(comparison)
    except Exception:
        return True


def _duration_ms(seconds: float) -> float:
    return round(max(0.0, seconds) * 1000.0, 3)


def _first_error_code(diagnostics: Iterable[HookDiagnostic]) -> str | None:
    return next(
        (diagnostic.code for diagnostic in diagnostics if diagnostic.kind == "error"),
        None,
    )


def _manager_diagnostic(
    point: HookPoint, phase: str, code: str, message: str
) -> HookDiagnostic:
    return HookDiagnostic(
        hook_id="hook-manager",
        point=point,
        phase=phase,
        kind="error",
        code=code,
        message=message,
    )


def _hook_diagnostic(
    hook_id: str,
    point: HookPoint,
    phase: str,
    code: str,
    message: str,
    *,
    kind: str = "error",
) -> HookDiagnostic:
    return HookDiagnostic(
        hook_id=hook_id,
        point=point,
        phase=phase,
        kind=kind,
        code=code,
        message=message,
    )
