"""Evaluation case execution and cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from litecoder.agent.loop import RuntimeBudgets
from litecoder.agent.runtime import AgentRuntime
from litecoder.eval.domain import (
    AgentExecution,
    CasePaths,
    CaseSpec,
    ExecutionCandidate,
    ExecutionFailure,
    ExecutionPolicy,
    Metric,
)
from litecoder.hooks import HookEnvelope, HookManager, HookOutcome, HookPoint
from litecoder.providers.registry import close_default_async_clients
from litecoder.tasks.models import TaskCreate, TaskStatus
from litecoder.tools.permission import PermissionPrompt, PromptChoice
from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.sink import CompositeUISink, RecordingUISink, RuntimeUISink, flush_ui


RuntimeBuilder = Callable[..., Awaitable[AgentRuntime]]
LiveUISinkFactory = Callable[[Path], RuntimeUISink]


@dataclass(frozen=True, slots=True)
class ExecutedCase:
    """Data model representing the executed case."""
    execution: AgentExecution
    events: tuple[RuntimeUIEvent, ...]


class CaseExecutor(Protocol):
    """Protocol describing the case executor behavior."""
    async def execute(
        self,
        spec: CaseSpec,
        paths: CasePaths,
        policy: ExecutionPolicy,
        candidate: ExecutionCandidate | None = None,
    ) -> ExecutedCase: ...


class RuntimeCaseExecutor:
    """Component responsible for the runtime case executor."""
    def __init__(
        self,
        runtime_builder: RuntimeBuilder,
        *,
        live_ui_sink: RuntimeUISink | None = None,
        live_ui_sink_factory: LiveUISinkFactory | None = None,
    ) -> None:
        if live_ui_sink is not None and live_ui_sink_factory is not None:
            raise ValueError("configure either a live UI sink or a sink factory")
        self.runtime_builder = runtime_builder
        self.live_ui_sink = live_ui_sink
        self.live_ui_sink_factory = live_ui_sink_factory

    async def execute(
        self,
        spec: CaseSpec,
        paths: CasePaths,
        policy: ExecutionPolicy,
        candidate: ExecutionCandidate | None = None,
    ) -> ExecutedCase:
        """Execute the requested tool call."""
        selected = candidate or ExecutionCandidate("primary", spec.prompt())
        outcome, events, runtime_state = await self._run_once(
            selected.prompt,
            paths.solution.parent,
            paths.trace,
            policy,
            mode=spec.mode,
            candidate=selected,
        )
        metrics = dict(outcome.metrics)
        metrics["candidate_name"] = Metric("candidate_name", selected.name)
        metrics.update(
            {
                name: Metric(name, value)
                for name, value in runtime_state.items()
            }
        )
        return ExecutedCase(
            AgentExecution(
                outcome.status,
                outcome.reason,
                paths.solution.read_text(encoding="utf-8"),
                outcome.input_tokens,
                outcome.output_tokens,
                outcome.elapsed_seconds,
                metrics,
                outcome.failure,
            ),
            events,
        )

    async def _run_once(
        self,
        prompt: str,
        workspace: Path,
        trace_path: Path,
        policy: ExecutionPolicy,
        *,
        mode: str,
        candidate: ExecutionCandidate | None = None,
    ) -> tuple[
        AgentExecution,
        tuple[RuntimeUIEvent, ...],
        dict[str, float | int | str],
    ]:
        """Run the once."""
        selected = candidate or ExecutionCandidate("primary", prompt)
        recording = RecordingUISink()
        live_ui_sink = (
            self.live_ui_sink_factory(workspace)
            if self.live_ui_sink_factory is not None
            else self.live_ui_sink
        )
        ui_sink: RuntimeUISink = (
            CompositeUISink(recording, _RootSink(live_ui_sink))
            if live_ui_sink is not None
            else recording
        )
        runtime_options: dict[str, object] = {
            "ui_sink": ui_sink,
            "isolated_workspace": True,
            "permission_prompt": _permission_prompt_for(mode),
            "hook_registrar": (
                _register_eval_hooks if mode == "tools-hooks" else None
            ),
            "tool_allowlist": (
                None if "*" in policy.allowed_tools else policy.allowed_tools
            ),
        }
        if selected.context_compaction != "default":
            runtime_options["context_compaction"] = (
                selected.context_compaction == "enabled"
            )
        if selected.context_budget_tokens is not None:
            runtime_options["context_budget_tokens"] = (
                selected.context_budget_tokens
            )
        if selected.memory_recall != "default":
            runtime_options["memory_recall"] = selected.memory_recall == "enabled"
        if policy.max_rounds is not None or policy.max_tokens is not None:
            defaults = RuntimeBudgets()
            runtime_options["runtime_budgets"] = RuntimeBudgets(
                max_rounds=(
                    policy.max_rounds
                    if policy.max_rounds is not None
                    else defaults.max_rounds
                ),
                max_tokens=policy.max_tokens,
            )
        runtime = await self.runtime_builder(workspace, **runtime_options)
        result = None
        setup_result = None
        run_error: Exception | None = None
        runtime_state: dict[str, float | int | str] = {}
        final_event_start = 0
        started = time.perf_counter()
        try:
            try:
                await _ensure_runtime_started(runtime)
                if selected.task_recovery:
                    runtime_state.update(
                        await _stage_task_recovery(runtime, workspace)
                    )
                    await runtime.close()
                    runtime = await self.runtime_builder(workspace, **runtime_options)
                    await _ensure_runtime_started(runtime)
                    runtime_state.update(
                        await _resume_task_recovery(
                            runtime,
                            workspace,
                            str(runtime_state["checkpoint_artifact_sha256"]),
                        )
                    )
                if selected.setup_prompt:
                    setup_result = await runtime.run(selected.setup_prompt)
                    if getattr(setup_result, "status", "error") == "completed":
                        session_id = getattr(setup_result, "session_id", None)
                        if (
                            not selected.restart_after_setup
                            and (not isinstance(session_id, str) or not session_id)
                        ):
                            raise RuntimeError(
                                "setup turn did not return a resumable session"
                            )
                        if selected.restart_after_setup:
                            await runtime.close()
                            runtime = await self.runtime_builder(
                                workspace, **runtime_options
                            )
                            await _ensure_runtime_started(runtime)
                            runtime_state["runtime_restart_count"] = 1
                        final_event_start = len(recording.events)
                        if selected.restart_after_setup:
                            result = await runtime.run(prompt)
                            runtime_state["fresh_session_continuation"] = 1
                        else:
                            result = await runtime.resume(str(session_id), prompt)
                    else:
                        result = setup_result
                else:
                    final_event_start = len(recording.events)
                    result = await runtime.run(prompt)
                if selected.task_recovery:
                    runtime_state.update(
                        await _finish_task_recovery(runtime, result)
                    )
                runtime_state.update(
                    await _runtime_experiment_state(
                        runtime,
                        result,
                        recording.events[final_event_start:],
                        selected,
                    )
                )
            except Exception as error:
                run_error = error
        finally:
            session_ids = _trace_session_ids(runtime, result, recording.events)
            try:
                await runtime.close()
            finally:
                try:
                    _copy_trace(runtime, session_ids, trace_path)
                finally:
                    try:
                        await close_default_async_clients()
                    finally:
                        await flush_ui(ui_sink)
                        if live_ui_sink is not None:
                            await flush_ui(live_ui_sink)
        elapsed = time.perf_counter() - started
        if result is not None:
            status = result.status
            reason = "" if result.status == "completed" else result.reason
            setup_input, setup_output = _result_usage(setup_result)
            result_input, result_output = _result_usage(result)
            if result is setup_result:
                setup_input = setup_output = 0
            input_tokens = setup_input + result_input
            output_tokens = setup_output + result_output
            has_continuation = setup_result is not None and result is not setup_result
            setup_events = (
                recording.events[:final_event_start] if has_continuation else []
            )
            continuation_events = (
                recording.events[final_event_start:]
                if has_continuation
                else recording.events
            )
            setup_total_input = _sum_usage_input_tokens(setup_events)
            continuation_total_input = _sum_usage_input_tokens(
                continuation_events
            )
            runtime_state.update(
                {
                    "setup_input_tokens": setup_input,
                    "setup_output_tokens": setup_output,
                    "setup_total_input_tokens": setup_total_input,
                    "setup_first_request_input_tokens": (
                        _first_usage_input_tokens(setup_events)
                    ),
                    "continuation_input_tokens": result_input,
                    "continuation_output_tokens": result_output,
                    "continuation_total_input_tokens": continuation_total_input,
                    "continuation_first_request_input_tokens": (
                        _first_usage_input_tokens(continuation_events)
                    ),
                    "total_recorded_input_tokens": (
                        setup_total_input + continuation_total_input
                    ),
                    "experiment_turn_count": 1 + int(setup_result is not None),
                }
            )
        else:
            status = "error"
            reason = _runtime_error_reason(run_error)
            input_tokens, output_tokens = _usage_from_events(recording.events)
        failure = _execution_failure(
            status,
            reason,
            run_error,
            recording.events,
        )
        budget_exhausted = int(
            failure is not None and failure.stage == "budget"
        )
        metrics = {
            "input_tokens": Metric("input_tokens", input_tokens, "tokens"),
            "output_tokens": Metric("output_tokens", output_tokens, "tokens"),
            "wall_clock_seconds": Metric("wall_clock_seconds", elapsed, "seconds"),
            "budget_exhausted": Metric("budget_exhausted", budget_exhausted),
        }
        provider_name = getattr(runtime, "provider_name", None)
        model = getattr(runtime, "model", None)
        if isinstance(provider_name, str) and provider_name:
            metrics["runtime_provider"] = Metric(
                "runtime_provider", provider_name
            )
        if isinstance(model, str) and model:
            metrics["runtime_model"] = Metric("runtime_model", model)
        return (
            AgentExecution(
                status,
                reason,
                "",
                input_tokens,
                output_tokens,
                elapsed,
                metrics,
                failure,
            ),
            tuple(recording.events),
            runtime_state,
        )


async def _ensure_runtime_started(runtime: object) -> None:
    start = getattr(runtime, "start", None)
    if callable(start):
        outcome = start()
        if asyncio.iscoroutine(outcome):
            await outcome


async def _stage_task_recovery(
    runtime: object, workspace: Path
) -> dict[str, float | int | str]:
    manager = getattr(runtime, "task_manager", None)
    if manager is None:
        raise RuntimeError("task-state evaluation requires TaskManager")
    await manager.create(
        TaskCreate(
            "eval-edit",
            "Implement solution",
            "Implement the EvalPlus solution after recovery.",
        )
    )
    await manager.create(
        TaskCreate(
            "eval-validate",
            "Validate solution",
            "Validate only after the edit task completes.",
            dependencies=("eval-edit",),
        )
    )
    await manager.claim("eval-edit", "eval-agent")
    solution = workspace / "solution.py"
    return {
        "interruption_checkpoint": "after-task-claim-before-agent-turn",
        "checkpoint_artifact_sha256": _file_sha256(solution),
        "runtime_restart_count": 1,
    }


async def _resume_task_recovery(
    runtime: object, workspace: Path, expected_artifact_sha256: str
) -> dict[str, float | int | str]:
    manager = getattr(runtime, "task_manager", None)
    if manager is None:
        raise RuntimeError("task-state evaluation requires TaskManager")
    tasks = await manager.list()
    by_id = {getattr(task, "id", ""): task for task in tasks}
    edit = by_id.get("eval-edit")
    validate = by_id.get("eval-validate")
    recovered = int(
        edit is not None
        and getattr(edit, "status", None) is TaskStatus.INTERRUPTED
    )
    dependencies_preserved = int(
        validate is not None
        and tuple(getattr(validate, "dependencies", ())) == ("eval-edit",)
    )
    duplicate_steps = len(tasks) - len(by_id)
    await manager.resume("eval-edit", "eval-agent")
    artifact_sha256 = _file_sha256(workspace / "solution.py")
    return {
        "recovered": recovered,
        "dependencies_preserved": dependencies_preserved,
        "duplicate_steps": duplicate_steps,
        "artifact_present_after_restart": int((workspace / "solution.py").exists()),
        "artifact_preserved_after_restart": int(
            bool(expected_artifact_sha256)
            and artifact_sha256 == expected_artifact_sha256
        ),
    }


async def _finish_task_recovery(
    runtime: object, result: object | None
) -> dict[str, float | int | str]:
    manager = getattr(runtime, "task_manager", None)
    if manager is None:
        raise RuntimeError("task-state evaluation requires TaskManager")
    if getattr(result, "status", None) == "completed":
        await manager.complete("eval-edit", "eval-agent")
        await manager.claim("eval-validate", "eval-agent")
        await manager.complete("eval-validate", "eval-agent")
    tasks = await manager.list()
    by_id = {getattr(task, "id", ""): task for task in tasks}
    completed = sum(
        getattr(task, "status", None) is TaskStatus.COMPLETED for task in tasks
    )
    return {
        "recovery_workflow_completed": int(completed == 2),
        "recovered_task_count": len(tasks),
        "duplicate_steps": len(tasks) - len(by_id),
    }


async def _runtime_experiment_state(
    runtime: object,
    result: object | None,
    final_events: list[RuntimeUIEvent],
    candidate: ExecutionCandidate,
) -> dict[str, float | int | str]:
    state: dict[str, float | int | str] = {}
    if candidate.context_compaction != "default":
        state["context_compaction_enabled"] = int(
            candidate.context_compaction == "enabled"
        )
        state["context_compaction_count"] = await _context_summary_count(
            runtime, getattr(result, "session_id", None)
        )
    if candidate.memory_recall != "default":
        state["memory_recall_enabled"] = int(
            candidate.memory_recall == "enabled"
        )
        state["memory_recalled_items"] = _recalled_memory_count(final_events)
    return state


async def _context_summary_count(runtime: object, session_id: object) -> int:
    store = getattr(runtime, "store", None)
    if store is None or not isinstance(session_id, str) or not session_id:
        return 0
    try:
        context = await store.load_context(session_id)
    except Exception:
        return 0
    return sum(
        block.get("type") == "context_summary"
        for message in getattr(context, "messages", ())
        for block in getattr(message, "content", ())
        if isinstance(block, dict)
    )


def _recalled_memory_count(events: list[RuntimeUIEvent]) -> int:
    counts = [
        event.payload.get("memory_count")
        for event in events
        if event.type is UIEventType.MODEL_REQUESTED
        and isinstance(event.payload.get("memory_count"), int)
    ]
    return max(counts, default=0)


def _result_usage(result: object | None) -> tuple[int, int]:
    usage = getattr(result, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    return (
        input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens if isinstance(output_tokens, int) else 0,
    )


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()

class _RootSink:
    """Internal helper for the root sink."""
    def __init__(self, sink: RuntimeUISink) -> None:
        self.sink = sink

    def emit(self, event: RuntimeUIEvent) -> object:
        """Emit the supplied event."""
        if event.span_id not in {None, "root"}:
            return None
        return self.sink.emit(event)

    def flush(self) -> None:
        """Flush pending output."""
        return None


def _allow_eval_permission(prompt: object) -> PromptChoice:
    if not isinstance(prompt, PermissionPrompt):
        return PromptChoice.DENY
    if prompt.tool_name in {"write_file", "edit_file"} and prompt.risk == "workspace":
        path = prompt.arguments.get("path")
        return (
            PromptChoice.ALLOW_ONCE
            if isinstance(path, str) and Path(path).as_posix() == "solution.py"
            else PromptChoice.DENY
        )
    if prompt.tool_name in {"read_file", "glob_files", "search_text"} and prompt.risk in {
        "safe",
        "workspace",
    }:
        return PromptChoice.ALLOW_ONCE
    if prompt.tool_name == "run_shell" and prompt.risk == "high" and _allowed_test_shell(prompt):
        return PromptChoice.ALLOW_ONCE
    return PromptChoice.DENY


def _permission_prompt_for(mode: str) -> Callable[[object], PromptChoice]:
    if mode == "memory":
        return _allow_memory_eval_permission
    return _allow_eval_permission


def _allow_memory_eval_permission(prompt: object) -> PromptChoice:
    if (
        isinstance(prompt, PermissionPrompt)
        and prompt.tool_name == "memory_update"
        and prompt.risk == "workspace"
    ):
        return PromptChoice.ALLOW_ONCE
    return _allow_eval_permission(prompt)


def _allowed_test_shell(prompt: PermissionPrompt) -> bool:
    argv = prompt.arguments.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        return False
    root = _resolved_root(prompt.workspace_root)
    if root is None or not _cwd_in_root(prompt, root):
        return False
    command = Path(argv[0]).name.casefold().removesuffix(".exe")
    if command.startswith("python") or command == "py":
        if len(argv) < 3 or argv[1:3] not in (["-m", "pytest"], ["-m", "unittest"]):
            return False
        args = argv[3:]
    elif command in {"pytest", "py.test"}:
        args = argv[1:]
    else:
        return False
    return _test_args_in_root(args, root)


def _resolved_root(value: str) -> Path | None:
    try:
        return Path(value).resolve() if value else None
    except OSError:
        return None


def _cwd_in_root(prompt: PermissionPrompt, root: Path) -> bool:
    cwd = prompt.arguments.get("cwd", ".")
    return isinstance(cwd, str) and _path_in_root(root, cwd)


def _test_args_in_root(args: list[str], root: Path) -> bool:
    """Test the args in root."""
    value_next = False
    for arg in args:
        if value_next:
            value_next = False
            if not arg or "\n" in arg or "\r" in arg:
                return False
            continue
        if arg in {"-q", "-v", "-vv", "-x", "--quiet", "--verbose", "--disable-warnings", "--no-header", "--no-summary", "--cache-clear"}:
            continue
        if arg in {"-k", "-m"}:
            value_next = True
            continue
        if arg.startswith("--maxfail=") and arg.removeprefix("--maxfail=").isdigit():
            continue
        if arg.startswith("--tb=") and arg.removeprefix("--tb=") in {"auto", "short", "long", "line", "native", "no"}:
            continue
        if arg.startswith("-") or not _path_in_root(root, arg.split("::", 1)[0] or "."):
            return False
    return not value_next


def _path_in_root(root: Path, value: str) -> bool:
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _register_eval_hooks(hooks: HookManager) -> None:
    async def pass_through(envelope: HookEnvelope) -> HookOutcome:
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.PRE_TOOL_USE, pass_through, name="eval-pre-tool-use")
    hooks.register(HookPoint.POST_TOOL_USE, pass_through, name="eval-post-tool-use")
    hooks.register(HookPoint.TOOL_ERROR, pass_through, name="eval-tool-error")


def _trace_session_ids(
    runtime: object, result: object | None, events: list[RuntimeUIEvent]
) -> tuple[str, ...]:
    values: list[str] = []
    for candidate in (
        getattr(result, "session_id", None),
        *(event.root_session_id for event in reversed(events)),
        *(event.session_id for event in reversed(events)),
    ):
        if isinstance(candidate, str) and candidate.strip() and candidate not in values:
            values.append(candidate)
    return tuple(values)


def _copy_trace(runtime: object, session_ids: tuple[str, ...], target: Path) -> None:
    paths = getattr(runtime, "paths", None)
    user_dir = getattr(paths, "user_dir", None)
    project_id = getattr(paths, "project_id", None)
    if not isinstance(user_dir, Path) or not isinstance(project_id, str):
        return
    trace_dir = user_dir / "projects" / project_id / "traces"
    lines: list[str] = []
    seen_records: set[tuple[object, ...]] = set()
    seen_sources: set[Path] = set()
    for session_id in session_ids:
        source = trace_dir / f"{session_id}.jsonl"
        if source in seen_sources or not source.exists() or not source.stat().st_size:
            continue
        seen_sources.add(source)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            identity = _trace_record_identity(line)
            if identity in seen_records:
                continue
            seen_records.add(identity)
            lines.append(line)
    if lines:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trace_record_identity(line: str) -> tuple[object, ...]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return ("raw", line)
    if not isinstance(value, dict):
        return ("raw", line)
    trace_id = value.get("trace_id")
    sequence = value.get("sequence")
    if isinstance(trace_id, str) and isinstance(sequence, int):
        return ("trace", trace_id, sequence)
    return ("json", json.dumps(value, ensure_ascii=False, sort_keys=True))


def _usage_from_events(events: list[RuntimeUIEvent]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for event in events:
        if event.type is not UIEventType.USAGE_UPDATED:
            continue
        input_value = event.payload.get("input_tokens")
        output_value = event.payload.get("output_tokens")
        if isinstance(input_value, int):
            input_tokens = max(input_tokens, input_value)
        if isinstance(output_value, int):
            output_tokens = max(output_tokens, output_value)
    return input_tokens, output_tokens


def _first_usage_input_tokens(events: list[RuntimeUIEvent]) -> int:
    for event in events:
        if event.type is not UIEventType.USAGE_UPDATED:
            continue
        value = event.payload.get("input_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _sum_usage_input_tokens(events: list[RuntimeUIEvent]) -> int:
    """Return the sum of per-request input usage events."""
    total = 0
    for event in events:
        if event.type is not UIEventType.USAGE_UPDATED:
            continue
        value = event.payload.get("input_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
    return total


def _runtime_error_reason(error: Exception | None) -> str:
    if error is None:
        return "RuntimeError: runtime ended without a result"
    message = str(error).strip() or "runtime execution failed"
    return f"{type(error).__name__}: {message}"


def _execution_failure(
    status: str,
    reason: str,
    run_error: Exception | None,
    events: list[RuntimeUIEvent],
) -> ExecutionFailure | None:
    if status == "completed":
        return None
    if run_error is not None:
        return ExecutionFailure(
            "runtime",
            "exception",
            reason,
            error_type=type(run_error).__name__,
        )
    if reason in {
        "round budget exhausted",
        "token budget exhausted",
        "continuation budget exhausted",
    }:
        return ExecutionFailure(
            "budget",
            reason.replace(" ", "_"),
            reason,
        )
    provider = next(
        (
            event
            for event in reversed(events)
            if event.type is UIEventType.PROVIDER_ERROR
            and event.payload.get("retrying") is not True
        ),
        None,
    )
    if provider is not None:
        payload = provider.payload
        message = payload.get("message")
        code = payload.get("code")
        details = {
            name: value
            for name in (
                "recovery_action",
                "recovery_reason",
                "attempt",
                "max_attempts",
                "request_id",
            )
            if isinstance((value := payload.get(name)), (str, int, float))
            and not isinstance(value, bool)
        }
        provider_details = payload.get("details")
        if isinstance(provider_details, dict):
            details.update(
                {
                    str(name): value
                    for name, value in provider_details.items()
                    if isinstance(value, (str, int, float))
                    and not isinstance(value, bool)
                }
            )
        return ExecutionFailure(
            "provider",
            "provider_error",
            message if isinstance(message, str) and message.strip() else reason,
            error_type=code if isinstance(code, str) else "",
            details=details,
        )
    normalized = status.replace(" ", "_") if status else "unknown"
    return ExecutionFailure(
        "agent",
        normalized,
        reason or f"agent ended with status {status}",
    )
