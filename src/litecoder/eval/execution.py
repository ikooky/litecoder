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
from litecoder.tasks.worktrees import run_git
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
        if selected.topology != "default":
            await _ensure_eval_git_workspace(paths.solution.parent)
        outcome, events, runtime_state = await self._run_once(
            selected.prompt,
            paths.solution.parent,
            paths.trace,
            policy,
            mode=spec.mode,
            topology=selected.topology,
            candidate=selected,
        )
        metrics = dict(outcome.metrics)
        metrics["candidate_name"] = Metric("candidate_name", selected.name)
        metrics["candidate_topology"] = Metric(
            "candidate_topology", selected.topology
        )
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
        topology: str = "default",
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
            "permission_prompt": _permission_prompt_for(mode, topology),
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
            if topology != "default":
                runtime_state.update(
                    await _runtime_multi_agent_state(runtime, topology)
                )
        finally:
            session_ids = _trace_session_ids(runtime, result, recording.events)
            try:
                await runtime.close()
            finally:
                try:
                    try:
                        _copy_trace(runtime, session_ids, trace_path)
                    finally:
                        await _cleanup_eval_worktrees(runtime)
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
            runtime_state.update(
                {
                    "setup_input_tokens": setup_input,
                    "setup_output_tokens": setup_output,
                    "setup_first_request_input_tokens": (
                        _first_usage_input_tokens(setup_events)
                    ),
                    "continuation_input_tokens": result_input,
                    "continuation_output_tokens": result_output,
                    "continuation_first_request_input_tokens": (
                        _first_usage_input_tokens(continuation_events)
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


def _permission_prompt_for(
    mode: str, topology: str
) -> Callable[[object], PromptChoice]:
    if topology != "default":
        return _allow_multi_agent_permission
    if mode == "memory":
        return _allow_memory_eval_permission
    return _allow_eval_permission


def _allow_multi_agent_permission(prompt: object) -> PromptChoice:
    if not isinstance(prompt, PermissionPrompt):
        return PromptChoice.DENY
    if prompt.tool_name == "worktree_create" and prompt.risk == "workspace":
        return PromptChoice.ALLOW_ONCE
    if prompt.tool_name in {"spawn_subagent", "team_create"} and prompt.risk == "high":
        return PromptChoice.ALLOW_ONCE
    return _allow_eval_permission(prompt)


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


async def _ensure_eval_git_workspace(workspace: Path) -> None:
    if (workspace / ".git").exists():
        return
    for arguments in (
        ("init",),
        ("config", "user.email", "litecoder-eval@example.invalid"),
        ("config", "user.name", "LiteCoder Eval"),
        ("add", "--", "solution.py"),
        ("commit", "-m", "Initialize multi-agent evaluation workspace"),
    ):
        result = await run_git(workspace, *arguments)
        if result.returncode != 0:
            raise RuntimeError("multi-agent evaluation requires a Git workspace")


async def _cleanup_eval_worktrees(runtime: object) -> None:
    """Remove worktrees created by a completed evaluation case only."""
    manager = getattr(runtime, "worktree_manager", None)
    if manager is None:
        return
    list_worktrees = getattr(manager, "list", None)
    remove_worktree = getattr(manager, "remove", None)
    if not callable(list_worktrees) or not callable(remove_worktree):
        return
    bindings = list_worktrees()
    if asyncio.iscoroutine(bindings):
        bindings = await bindings
    if not isinstance(bindings, (list, tuple)):
        return
    for binding in bindings:
        removed = remove_worktree(binding, discard=True)
        if asyncio.iscoroutine(removed):
            await removed


async def _runtime_multi_agent_state(
    runtime: object, topology: str
) -> dict[str, float | int | str]:
    if topology == "subagent":
        return await _runtime_subagent_state(runtime)
    if topology == "team":
        return await _runtime_team_state(runtime)
    return {"closed_loop_valid": 0}


async def _runtime_subagent_state(runtime: object) -> dict[str, float | int | str]:
    manager = getattr(runtime, "subagent_manager", None)
    history = getattr(manager, "spawn_history", ())
    if not isinstance(history, (list, tuple)):
        history = ()
    tasks = await _service_list(getattr(runtime, "task_manager", None))
    worktrees = await _service_list(getattr(runtime, "worktree_manager", None))
    agent_ids = {
        item.get("agent_id")
        for item in history
        if isinstance(item, dict)
        and isinstance(item.get("agent_id"), str)
        and item.get("agent_id")
    }
    completed = [task for task in tasks if _task_status(task) == "completed"]
    owned_completed = [
        task
        for task in completed
        if getattr(task, "owner_agent_id", None) in agent_ids
        and isinstance(getattr(task, "worktree_id", None), str)
    ]
    returned = sum(
        item.get("result_returned") == 1 for item in history if isinstance(item, dict)
    )
    closed = bool(history and returned and owned_completed and worktrees)
    return {
        "agent_count": len(history),
        "task_count": len(tasks),
        "completed_task_count": len(completed),
        "owned_completed_task_count": len(owned_completed),
        "worktree_count": len(worktrees),
        "results_returned": returned,
        "closed_loop_valid": int(closed),
    }


async def _runtime_team_state(runtime: object) -> dict[str, float | int | str]:
    team = getattr(runtime, "team_manager", None)
    tasks = await _service_list(getattr(runtime, "task_manager", None))
    worktrees = await _service_list(getattr(runtime, "worktree_manager", None))
    members = getattr(team, "last_turn_members", ())
    if not isinstance(members, (list, tuple)):
        members = ()
    member_ids = {
        member.get("agent_id")
        for member in members
        if isinstance(member, dict) and isinstance(member.get("agent_id"), str)
    }
    completed_tasks = [task for task in tasks if _task_status(task) == "completed"]
    failed_tasks = [task for task in tasks if _task_status(task) == "failed"]
    owned_completed = [
        task
        for task in completed_tasks
        if getattr(task, "owner_agent_id", None) in member_ids
        and isinstance(getattr(task, "worktree_id", None), str)
    ]
    distinct_owners = {
        getattr(task, "owner_agent_id", None) for task in owned_completed
    }
    sent = getattr(team, "last_turn_messages_sent", 0)
    received = getattr(team, "last_turn_messages_received", 0)
    sent = sent if isinstance(sent, int) else 0
    received = received if isinstance(received, int) else 0
    worker_sent = getattr(team, "last_turn_worker_results_sent", 0)
    worker_delivered = getattr(team, "last_turn_worker_results_delivered", 0)
    peer_sent = getattr(team, "last_turn_peer_messages_sent", 0)
    worker_sent = worker_sent if isinstance(worker_sent, int) else 0
    worker_delivered = worker_delivered if isinstance(worker_delivered, int) else 0
    peer_sent = peer_sent if isinstance(peer_sent, int) else 0
    peer_valid = int(peer_sent > 0)
    closed = bool(
        len(members) >= 2
        and len(distinct_owners) >= 2
        and len(worktrees) >= 2
        and sent
        and received
        and worker_sent >= 2
        and worker_delivered >= 2
        and peer_valid
    )
    return {
        "agent_count": len(members),
        "task_count": len(tasks),
        "completed_task_count": len(completed_tasks),
        "failed_task_count": len(failed_tasks),
        "owned_completed_task_count": len(owned_completed),
        "worktree_count": len(worktrees),
        "messages_sent": sent,
        "messages_received": received,
        "worker_results_sent": worker_sent,
        "worker_results_delivered": worker_delivered,
        "peer_messages_sent": peer_sent,
        "peer_communication_valid": peer_valid,
        "closed_loop_valid": int(closed),
    }


async def _service_list(service: object) -> tuple[object, ...]:
    method = getattr(service, "list", None)
    if not callable(method):
        return ()
    try:
        value = method()
        if asyncio.iscoroutine(value):
            value = await value
    except Exception:
        return ()
    return tuple(value) if isinstance(value, (list, tuple)) else ()


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


def _task_status(task: object) -> object:
    status = getattr(task, "status", None)
    return getattr(status, "value", status)
