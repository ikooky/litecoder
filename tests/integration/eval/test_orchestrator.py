from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.eval.domain import (
    AgentExecution,
    ExecutionFailure,
    Metric,
    RunSpec,
    ValidationResult,
)
from litecoder.eval.evalplus import EvalPlusTask
from litecoder.eval.execution import ExecutedCase
from litecoder.eval.orchestrator import EvalOrchestrator
from litecoder.eval.validation import ValidationCapture
from litecoder.ui.events import RuntimeUIEvent, UIEventType


class FakeExecutor:
    async def execute(self, spec, paths, policy, candidate):
        del spec, policy, candidate
        solution = "def answer():\n    return 42\n"
        paths.solution.write_text(solution, encoding="utf-8")
        execution = AgentExecution(
            "completed",
            "",
            solution,
            10,
            5,
            1.0,
            {
                "input_tokens": Metric("input_tokens", 10),
                "output_tokens": Metric("output_tokens", 5),
                "wall_clock_seconds": Metric("wall_clock_seconds", 1.0),
                "budget_exhausted": Metric("budget_exhausted", 0),
            },
        )
        return ExecutedCase(
            execution,
            (RuntimeUIEvent(UIEventType.MODEL_REQUESTED, 1, 0.0),),
        )


class PassingValidator:
    def validate(self, spec, solution):
        assert spec.task_id == "HumanEval/0"
        assert "return 42" in solution
        return ValidationCapture(
            ValidationResult(True, "pass", "pass", 0, None, 0.1),
            "passed\n",
        )


class BudgetExecutor:
    async def execute(self, spec, paths, policy, candidate):
        del spec, policy, candidate
        solution = paths.solution.read_text(encoding="utf-8")
        failure = ExecutionFailure(
            "budget",
            "round_budget_exhausted",
            "round budget exhausted",
        )
        return ExecutedCase(
            AgentExecution(
                "incomplete",
                "round budget exhausted",
                solution,
                10,
                5,
                1.0,
                {
                    "input_tokens": Metric("input_tokens", 10),
                    "output_tokens": Metric("output_tokens", 5),
                    "wall_clock_seconds": Metric("wall_clock_seconds", 1.0),
                    "budget_exhausted": Metric("budget_exhausted", 1),
                },
                failure,
            ),
            (),
        )


class FailingValidator:
    def validate(self, spec, solution):
        del spec, solution
        return ValidationCapture(
            ValidationResult(
                False,
                "fail",
                "fail",
                2,
                0,
                0.1,
                "EvalPlus base and plus failed",
            ),
            "failed\n",
        )


class RaisingExecutor:
    async def execute(self, spec, paths, policy, candidate):
        del spec, paths, policy, candidate
        raise RuntimeError("runtime construction failed")


class ProviderFailureExecutor:
    def __init__(self, *, recover: bool) -> None:
        self.recover = recover
        self.calls = 0

    async def execute(self, spec, paths, policy, candidate):
        del spec, policy, candidate
        self.calls += 1
        if self.recover and self.calls > 1:
            solution = "def answer():\n    return 42\n"
            paths.solution.write_text(solution, encoding="utf-8")
            return ExecutedCase(
                AgentExecution(
                    "completed",
                    "",
                    solution,
                    10,
                    5,
                    1.0,
                    {"budget_exhausted": Metric("budget_exhausted", 0)},
                ),
                (),
            )
        failure = ExecutionFailure(
            "provider",
            "provider_error",
            "Provider returned an invalid response",
            error_type="provider_invalid_response",
        )
        return ExecutedCase(
            AgentExecution(
                "incomplete",
                "provider invalid response",
                paths.solution.read_text(encoding="utf-8"),
                10,
                5,
                1.0,
                {"budget_exhausted": Metric("budget_exhausted", 0)},
                failure,
            ),
            (),
        )


class MultiCandidateExecutor:
    def __init__(self, *, subagent_closed: int = 1) -> None:
        self.subagent_closed = subagent_closed
        self.workspaces: dict[str, Path] = {}

    async def execute(self, spec, paths, policy, candidate):
        del spec, policy
        self.workspaces[candidate.name] = paths.solution.parent
        solution = (
            "def answer():\n"
            f"    return {42 if candidate.name == 'team' else 41}\n"
        )
        paths.solution.write_text(solution, encoding="utf-8")
        closed = self.subagent_closed if candidate.name == "subagent" else 1
        metrics = {
            "input_tokens": Metric("input_tokens", 10),
            "output_tokens": Metric("output_tokens", 5),
            "wall_clock_seconds": Metric("wall_clock_seconds", 1.0),
            "budget_exhausted": Metric("budget_exhausted", 0),
            "candidate_name": Metric("candidate_name", candidate.name),
            "candidate_topology": Metric(
                "candidate_topology", candidate.topology
            ),
            "closed_loop_valid": Metric("closed_loop_valid", closed),
        }
        if candidate.name == "team":
            metrics["peer_communication_valid"] = Metric(
                "peer_communication_valid", 1
            )
        return ExecutedCase(
            AgentExecution(
                "completed", "", solution, 10, 5, 1.0, metrics
            ),
            (),
        )


class MultiPassingValidator:
    def validate(self, spec, solution):
        del spec
        assert "return 4" in solution
        return ValidationCapture(
            ValidationResult(True, "pass", "pass", 0, None, 0.1),
            "passed\n",
        )


@pytest.mark.asyncio
async def test_orchestrator_runs_all_pipeline_stages(tmp_path: Path) -> None:
    report = await EvalOrchestrator(FakeExecutor(), PassingValidator()).run(
        RunSpec("run-1", "agent-benchmark", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    case = report.cases[0]
    assert case.status == "passed"
    assert case.stage.value == "scored"
    assert case.paths.diff.stat().st_size > 0
    assert case.paths.local_tests.read_text(encoding="utf-8") == "No local tests executed.\n"
    assert case.metrics["local_test_attempted"].value == 0
    assert case.paths.validation_result.exists()
    assert case.paths.mode_evidence.exists()
    assert report.guardrails["artifact_evidence_ready"].value == 1
    assert report.guardrails["artifact_evidence_ready_rate"].value == 1.0


@pytest.mark.asyncio
async def test_orchestrator_separates_mixed_dataset_case_ids(tmp_path: Path) -> None:
    tasks = (
        EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),
        EvalPlusTask("Mbpp/1", "def answer():\n", "answer", "mbpp"),
    )

    report = await EvalOrchestrator(FakeExecutor(), PassingValidator()).run(
        RunSpec("run-1", "agent-benchmark", ("humaneval", "mbpp"), tmp_path),
        tasks,
    )

    assert [case.spec.case_id for case in report.cases] == [
        "humaneval-0001",
        "mbpp-0001",
    ]


@pytest.mark.asyncio
async def test_orchestrator_records_execution_and_validation_failures(
    tmp_path: Path,
) -> None:
    report = await EvalOrchestrator(BudgetExecutor(), FailingValidator()).run(
        RunSpec("run-1", "agent-benchmark", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    case = report.cases[0]
    assert case.status == "failed"
    assert "execution (budget/round_budget_exhausted)" in case.failure_reason
    assert "validation: EvalPlus base and plus failed" in case.failure_reason
    assert case.execution.to_json()["failure"] == {
        "stage": "budget",
        "kind": "round_budget_exhausted",
        "message": "round budget exhausted",
        "error_type": "",
        "details": {},
    }


@pytest.mark.asyncio
async def test_orchestrator_records_executor_exception_type(tmp_path: Path) -> None:
    report = await EvalOrchestrator(RaisingExecutor(), FailingValidator()).run(
        RunSpec("run-1", "agent-benchmark", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    failure = report.cases[0].execution.failure
    assert failure is not None
    assert failure.stage == "orchestrator"
    assert failure.kind == "executor_exception"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "RuntimeError: runtime construction failed"
    assert report.cases[0].status == "infra_error"
    assert report.status == "completed_with_infra_errors"


@pytest.mark.asyncio
async def test_orchestrator_does_not_retry_provider_infrastructure_failure(
    tmp_path: Path,
) -> None:
    executor = ProviderFailureExecutor(recover=True)
    report = await EvalOrchestrator(executor, PassingValidator()).run(
        RunSpec("run-1", "agent-benchmark", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    case = report.cases[0]
    assert executor.calls == 1
    assert case.status == "infra_error"
    assert {
        "attempt_count",
        "infra_retry_count",
        "selected_attempt",
    }.isdisjoint(case.metrics)
    assert not (case.paths.root / "attempts").exists()
    assert "max_infra_retries" not in report.metadata
    assert report.status == "completed_with_infra_errors"


@pytest.mark.asyncio
async def test_orchestrator_excludes_unrecovered_infrastructure_case_from_rate(
    tmp_path: Path,
) -> None:
    executor = ProviderFailureExecutor(recover=False)
    report = await EvalOrchestrator(executor, PassingValidator()).run(
        RunSpec("run-1", "agent-benchmark", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    assert executor.calls == 1
    assert report.cases[0].status == "infra_error"
    assert report.status == "completed_with_infra_errors"
    assert report.primary_metrics["task_pass_rate"].value == 0.0
    assert report.guardrails["scoreable_case_count"].value == 0
    assert report.guardrails["infra_error_case_count"].value == 1


@pytest.mark.asyncio
async def test_multi_agent_runs_isolated_subagent_and_team_candidates(
    tmp_path: Path,
) -> None:
    executor = MultiCandidateExecutor()
    report = await EvalOrchestrator(executor, MultiPassingValidator()).run(
        RunSpec("run-1", "multi-agent", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    case = report.cases[0]
    assert case.status == "passed"
    assert set(case.candidates) == {"subagent", "team"}
    assert case.candidates["subagent"].validation.passed
    assert case.candidates["team"].validation.passed
    assert executor.workspaces["subagent"] != executor.workspaces["team"]
    assert "subagent" in executor.workspaces["subagent"].parts
    assert "team" in executor.workspaces["team"].parts
    assert executor.workspaces["subagent"].parent.name == "subagent"
    assert executor.workspaces["team"].parent.name == "team"
    assert case.metrics["subagent_closed_loop_valid"].value == 1
    assert case.metrics["team_peer_communication_valid"].value == 1
    assert case.execution.solution.endswith("return 42\n")
    assert case.paths.solution.read_text(encoding="utf-8").endswith("return 42\n")


@pytest.mark.asyncio
async def test_multi_agent_failure_names_the_failed_candidate_stage(
    tmp_path: Path,
) -> None:
    report = await EvalOrchestrator(
        MultiCandidateExecutor(subagent_closed=0), MultiPassingValidator()
    ).run(
        RunSpec("run-1", "multi-agent", "humaneval", tmp_path),
        (EvalPlusTask("HumanEval/0", "def answer():\n", "answer", "humaneval"),),
    )

    case = report.cases[0]
    assert case.status == "failed"
    assert "subagent.closed_loop" in case.failure_reason
