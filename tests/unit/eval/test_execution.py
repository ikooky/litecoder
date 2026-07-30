from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from litecoder.agent.result import AgentResult
from litecoder.eval.artifacts import prepare_case
from litecoder.eval.domain import (
    AgentExecution,
    CaseSpec,
    ExecutionCandidate,
    ExecutionFailure,
    ExecutionPolicy,
)
from litecoder.eval.execution import (
    RuntimeCaseExecutor,
    _allow_eval_permission,
    _allow_multi_agent_permission,
    _cleanup_eval_worktrees,
    _copy_trace,
    _execution_failure,
    _path_in_root,
)
from litecoder.providers.models import Usage
from litecoder.tasks.manager import TaskManager
from litecoder.tasks.store import TaskStore
from litecoder.tools.permission import PermissionPrompt, PromptChoice
from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.sink import emit_ui


def _prompt(
    tool: str,
    risk: str,
    arguments: dict[str, object],
    workspace: Path,
) -> PermissionPrompt:
    return PermissionPrompt(
        tool,
        risk,
        f"{risk}:eval",
        arguments,  # type: ignore[arg-type]
        workspace_root=str(workspace),
    )


def test_eval_permissions_only_allow_solution_and_test_runners(tmp_path: Path) -> None:
    assert _allow_eval_permission(
        _prompt("write_file", "workspace", {"path": "solution.py"}, tmp_path)
    ) is PromptChoice.ALLOW_ONCE
    assert _allow_eval_permission(
        _prompt("write_file", "workspace", {"path": "tests/test_solution.py"}, tmp_path)
    ) is PromptChoice.DENY
    assert _allow_eval_permission(
        _prompt(
            "run_shell",
            "high",
            {"argv": ["python", "-m", "pytest", "-q"], "cwd": "."},
            tmp_path,
        )
    ) is PromptChoice.ALLOW_ONCE
    assert _allow_eval_permission(
        _prompt(
            "run_shell",
            "high",
            {"argv": ["python", "-c", "print('probe')"], "cwd": "."},
            tmp_path,
        )
    ) is PromptChoice.DENY


def test_multi_agent_permissions_add_only_required_coordination(tmp_path: Path) -> None:
    assert _allow_multi_agent_permission(
        _prompt(
            "spawn_subagent",
            "high",
            {"task_id": "worker", "worktree_id": "worker"},
            tmp_path,
        )
    ) is PromptChoice.ALLOW_ONCE
    assert _allow_multi_agent_permission(
        _prompt(
            "worktree_create",
            "workspace",
            {"task_id": "worker", "branch": "eval-worker"},
            tmp_path,
        )
    ) is PromptChoice.ALLOW_ONCE
    assert _allow_multi_agent_permission(
        _prompt(
            "team_create",
            "high",
            {
                "display_name": "worker",
                "objective": "solve",
                "tools": ["read_file"],
                "budget": {"max_rounds": 2, "max_tool_calls": 4},
            },
            tmp_path,
        )
    ) is PromptChoice.ALLOW_ONCE
    assert _allow_multi_agent_permission(
        _prompt("worktree_remove", "high", {"id": "worker"}, tmp_path)
    ) is PromptChoice.DENY


def test_candidate_workspace_cannot_traverse_to_sibling(tmp_path: Path) -> None:
    team_workspace = tmp_path / "candidates" / "team" / "execution"
    team_workspace.mkdir(parents=True)

    assert not _path_in_root(team_workspace, "../../subagent/execution/solution.py")


@pytest.mark.asyncio
async def test_cleanup_eval_worktrees_removes_only_managed_bindings() -> None:
    removed: list[tuple[object, bool]] = []
    bindings = (object(), object())

    class Worktrees:
        async def list(self) -> tuple[object, ...]:
            return bindings

        async def remove(self, binding: object, *, discard: bool) -> None:
            removed.append((binding, discard))

    await _cleanup_eval_worktrees(SimpleNamespace(worktree_manager=Worktrees()))

    assert removed == [(binding, True) for binding in bindings]


def test_copy_trace_merges_root_and_child_session_traces(tmp_path: Path) -> None:
    trace_dir = tmp_path / "user" / "projects" / "project" / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "root.jsonl").write_text(
        '{"trace_id":"trace","sequence":1,"tool_call_id":"root-call"}\n',
        encoding="utf-8",
    )
    (trace_dir / "child.jsonl").write_text(
        '{"trace_id":"trace","sequence":1,"tool_call_id":"root-call"}\n'
        '{"trace_id":"trace","sequence":2,"tool_call_id":"child-call"}\n',
        encoding="utf-8",
    )
    runtime = SimpleNamespace(
        paths=SimpleNamespace(user_dir=tmp_path / "user", project_id="project")
    )
    target = tmp_path / "captured" / "trace.jsonl"

    _copy_trace(runtime, ("root", "child"), target)

    assert target.read_text(encoding="utf-8").splitlines() == [
        '{"trace_id":"trace","sequence":1,"tool_call_id":"root-call"}',
        '{"trace_id":"trace","sequence":2,"tool_call_id":"child-call"}',
    ]


@pytest.mark.asyncio
async def test_runtime_executor_applies_policy_and_records_runtime_events(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, workspace: Path, sink: object) -> None:
            self.workspace = workspace
            self.sink = sink
            self.paths = SimpleNamespace(
                user_dir=tmp_path / "user",
                project_id="eval-project",
            )

        async def run(self, prompt: str) -> AgentResult:
            captured["prompt"] = prompt
            (self.workspace / "solution.py").write_text(
                "def answer():\n    return 42\n", encoding="utf-8"
            )
            await emit_ui(
                self.sink,
                RuntimeUIEvent(UIEventType.MODEL_REQUESTED, 1, 0.0),
            )
            return AgentResult("session-1", "completed", "", Usage(12, 3))

        async def close(self) -> None:
            captured["closed"] = True

    async def builder(workspace: Path, **kwargs: object) -> FakeRuntime:
        captured.update(kwargs)
        return FakeRuntime(workspace, kwargs["ui_sink"])

    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "agent-benchmark",
    )
    paths = prepare_case(tmp_path / "run", spec)
    policy = ExecutionPolicy(
        frozenset({"read_file", "edit_file"}),
        max_rounds=7,
        max_tokens=321,
    )

    executed = await RuntimeCaseExecutor(builder).execute(spec, paths, policy)

    budgets = captured["runtime_budgets"]
    assert getattr(budgets, "max_rounds") == 7
    assert getattr(budgets, "max_tokens") == 321
    assert captured["tool_allowlist"] == policy.allowed_tools
    assert captured["isolated_workspace"] is True
    assert "timeout_seconds" not in policy.to_json()
    assert captured["closed"] is True
    assert executed.execution.solution.endswith("return 42\n")
    assert executed.execution.input_tokens == 12
    assert [event.type for event in executed.events] == [UIEventType.MODEL_REQUESTED]


@pytest.mark.asyncio
async def test_runtime_executor_builds_live_sink_for_candidate_workspace(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class LiveSink:
        def emit(self, event: RuntimeUIEvent) -> None:
            captured["live_event"] = event

        def flush(self) -> None:
            captured["live_flushed"] = True

    class FakeRuntime:
        def __init__(self, workspace: Path, sink: object) -> None:
            self.workspace = workspace
            self.sink = sink
            self.paths = SimpleNamespace(
                user_dir=tmp_path / "user",
                project_id="eval-project",
            )

        async def run(self, prompt: str) -> AgentResult:
            del prompt
            (self.workspace / "solution.py").write_text(
                "def answer():\n    return 42\n", encoding="utf-8"
            )
            await emit_ui(
                self.sink,
                RuntimeUIEvent(UIEventType.MODEL_REQUESTED, 1, 0.0),
            )
            return AgentResult("session-1", "completed", "", Usage(1, 1))

        async def close(self) -> None:
            return None

    async def builder(workspace: Path, **kwargs: object) -> FakeRuntime:
        return FakeRuntime(workspace, kwargs["ui_sink"])

    def sink_factory(workspace: Path) -> LiveSink:
        captured["live_workspace"] = workspace
        return LiveSink()

    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "agent-benchmark",
    )
    paths = prepare_case(tmp_path / "run", spec)

    await RuntimeCaseExecutor(
        builder, live_ui_sink_factory=sink_factory
    ).execute(spec, paths, ExecutionPolicy(frozenset({"read_file"})))

    assert captured["live_workspace"] == paths.solution.parent
    assert getattr(captured["live_event"], "type") is UIEventType.MODEL_REQUESTED
    assert captured["live_flushed"] is True


@pytest.mark.asyncio
async def test_runtime_executor_omits_unbounded_eval_budget_override(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace
            self.paths = SimpleNamespace(
                user_dir=tmp_path / "user",
                project_id="eval-project",
            )

        async def run(self, prompt: str) -> AgentResult:
            return AgentResult("session-1", "completed", "", Usage(1, 1))

        async def close(self) -> None:
            return None

    async def builder(workspace: Path, **kwargs: object) -> FakeRuntime:
        captured.update(kwargs)
        return FakeRuntime(workspace)

    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "multi-agent",
    )
    paths = prepare_case(tmp_path / "run", spec)
    policy = ExecutionPolicy(
        frozenset({"*"}),
        max_rounds=None,
        max_tokens=None,
    )

    await RuntimeCaseExecutor(builder).execute(spec, paths, policy)

    assert "runtime_budgets" not in captured


@pytest.mark.asyncio
async def test_runtime_executor_restarts_cross_session_memory_candidate(
    tmp_path: Path,
) -> None:
    builds: list[dict[str, object]] = []

    class FakeRuntime:
        def __init__(self, workspace: Path, sink: object) -> None:
            self.workspace = workspace
            self.sink = sink
            self.paths = SimpleNamespace(
                user_dir=tmp_path / "user",
                project_id="eval-project",
            )

        async def run(self, prompt: str) -> AgentResult:
            if "memory_update" in prompt:
                await emit_ui(
                    self.sink,
                    RuntimeUIEvent(
                        UIEventType.USAGE_UPDATED,
                        1,
                        0.0,
                        payload={"input_tokens": 5, "output_tokens": 2},
                    ),
                )
                return AgentResult("memory-setup", "completed", "", Usage(5, 2))
            assert "Continue" in prompt
            (self.workspace / "solution.py").write_text(
                "def answer():\n    return 42\n", encoding="utf-8"
            )
            await emit_ui(
                self.sink,
                RuntimeUIEvent(
                    UIEventType.USAGE_UPDATED,
                    1,
                    0.0,
                    payload={"input_tokens": 11, "output_tokens": 3},
                ),
            )
            await emit_ui(
                self.sink,
                RuntimeUIEvent(
                    UIEventType.MODEL_REQUESTED,
                    2,
                    0.1,
                    payload={"memory_count": 1},
                ),
            )
            return AgentResult("memory-continuation", "completed", "", Usage(11, 3))

        async def resume(self, session_id: str, prompt: str) -> AgentResult:
            raise AssertionError("cross-session memory must start a fresh session")

        async def close(self) -> None:
            return None

    async def builder(workspace: Path, **kwargs: object) -> FakeRuntime:
        builds.append(dict(kwargs))
        return FakeRuntime(workspace, kwargs["ui_sink"])

    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "memory",
    )
    paths = prepare_case(tmp_path / "run", spec)
    candidate = ExecutionCandidate(
        "treatment",
        "Continue the remembered task.",
        setup_prompt="Use memory_update to remember the task.",
        restart_after_setup=True,
        memory_recall="enabled",
    )

    executed = await RuntimeCaseExecutor(builder).execute(
        spec,
        paths,
        ExecutionPolicy(frozenset({"read_file", "memory_update"})),
        candidate,
    )

    assert len(builds) == 2
    assert all(item["memory_recall"] is True for item in builds)
    assert executed.execution.input_tokens == 16
    assert executed.execution.metrics["runtime_restart_count"].value == 1
    assert executed.execution.metrics["memory_recalled_items"].value == 1
    assert executed.execution.metrics["fresh_session_continuation"].value == 1
    assert executed.execution.metrics["setup_first_request_input_tokens"].value == 5
    assert (
        executed.execution.metrics["continuation_first_request_input_tokens"].value
        == 11
    )


@pytest.mark.asyncio
async def test_runtime_executor_recreates_services_and_recovers_task_state(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "durable-tasks"

    class FakeRuntime:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace
            self.task_manager = TaskManager(TaskStore(task_root))
            self.paths = SimpleNamespace(
                user_dir=tmp_path / "user",
                project_id="eval-project",
            )

        async def start(self) -> None:
            await self.task_manager.recover_interrupted()

        async def run(self, prompt: str) -> AgentResult:
            assert "Implement" in prompt
            (self.workspace / "solution.py").write_text(
                "def answer():\n    return 42\n", encoding="utf-8"
            )
            return AgentResult("task-session", "completed", "", Usage(9, 2))

        async def close(self) -> None:
            return None

    async def builder(workspace: Path, **kwargs: object) -> FakeRuntime:
        del kwargs
        return FakeRuntime(workspace)

    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "task-state",
    )
    paths = prepare_case(tmp_path / "run", spec)

    executed = await RuntimeCaseExecutor(builder).execute(
        spec,
        paths,
        ExecutionPolicy(frozenset({"read_file", "edit_file"})),
        ExecutionCandidate("recovery", spec.prompt(), task_recovery=True),
    )

    metrics = executed.execution.metrics
    assert metrics["recovered"].value == 1
    assert metrics["dependencies_preserved"].value == 1
    assert metrics["artifact_preserved_after_restart"].value == 1
    assert metrics["duplicate_steps"].value == 0
    assert metrics["recovery_workflow_completed"].value == 1


def test_execution_failure_records_budget_kind() -> None:
    failure = _execution_failure(
        "incomplete",
        "round budget exhausted",
        None,
        [],
    )

    assert failure is not None
    assert failure.stage == "budget"
    assert failure.kind == "round_budget_exhausted"
    assert failure.message == "round budget exhausted"


def test_execution_failure_records_terminal_provider_details() -> None:
    event = RuntimeUIEvent(
        UIEventType.PROVIDER_ERROR,
        1,
        0.0,
        payload={
            "code": "transient_provider",
            "message": "Provider request failed temporarily",
            "retrying": False,
            "recovery_action": "fail",
            "recovery_reason": "retry budget exhausted",
            "attempt": 2,
            "max_attempts": 2,
        },
    )

    failure = _execution_failure(
        "failed",
        "transient_provider retry budget exhausted",
        None,
        [event],
    )

    assert failure is not None
    assert failure.stage == "provider"
    assert failure.kind == "provider_error"
    assert failure.error_type == "transient_provider"
    assert failure.details["attempt"] == 2


def test_execution_failure_preserves_provider_adapter_reason() -> None:
    event = RuntimeUIEvent(
        UIEventType.PROVIDER_ERROR,
        1,
        0.0,
        payload={
            "code": "internal",
            "message": "Provider returned invalid streaming data",
            "retrying": False,
            "details": {
                "provider_error_type": "invalid_provider_data",
                "provider_data_reason": "provider content index is invalid",
            },
        },
    )

    failure = _execution_failure("failed", "invalid stream", None, [event])

    assert failure is not None
    assert failure.details["provider_error_type"] == "invalid_provider_data"
    assert failure.details["provider_data_reason"] == (
        "provider content index is invalid"
    )


@pytest.mark.asyncio
async def test_runtime_executor_preserves_failure_when_copying_outcome(
    tmp_path: Path,
) -> None:
    failure = ExecutionFailure(
        "provider",
        "provider_error",
        "Provider request failed",
        error_type="internal",
    )

    class FailedExecutor(RuntimeCaseExecutor):
        async def _run_once(self, *args: object, **kwargs: object):
            del args, kwargs
            return (
                AgentExecution(
                    "failed",
                    "internal",
                    "",
                    0,
                    0,
                    1.0,
                    {},
                    failure,
                ),
                (),
                {},
            )

    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "agent-benchmark",
    )
    paths = prepare_case(tmp_path, spec)

    executed = await FailedExecutor(lambda *args, **kwargs: None).execute(
        spec,
        paths,
        ExecutionPolicy(frozenset({"read_file"})),
    )

    assert executed.execution.failure == failure
