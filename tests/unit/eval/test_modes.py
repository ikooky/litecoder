from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from litecoder.context.token_budget import estimate_tokens
from litecoder.eval.artifacts import prepare_case
from litecoder.eval.domain import (
    AgentExecution,
    CandidateReport,
    CaseReport,
    CaseSpec,
    CaseStage,
    Metric,
    ValidationResult,
)
from litecoder.eval.memory_fixture import fixture_for
from litecoder.eval.modes.agent_benchmark import AgentBenchmarkMode
from litecoder.eval.modes.context_manager import ContextManagerMode
from litecoder.eval.modes.memory import MemoryMode
from litecoder.eval.modes.task_state import TaskStateMode
from litecoder.eval.modes.tools_hooks import ToolsHooksMode


def test_agent_benchmark_aggregates_process_guardrails(tmp_path: Path) -> None:
    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "agent-benchmark",
    )
    paths = prepare_case(tmp_path, spec)
    metrics = {
        "wall_clock_seconds": Metric("wall_clock_seconds", 2.0),
        "input_tokens": Metric("input_tokens", 10),
        "output_tokens": Metric("output_tokens", 5),
        "model_rounds": Metric("model_rounds", 2),
        "tool_calls": Metric("tool_calls", 2),
        "tool_successful_calls": Metric("tool_successful_calls", 2),
        "tool_failed_calls": Metric("tool_failed_calls", 0),
        "permission_denied_calls": Metric("permission_denied_calls", 0),
        "duplicate_blocked_calls": Metric("duplicate_blocked_calls", 0),
        "tool_outcome_coverage": Metric("tool_outcome_coverage", 1),
        "budget_exhausted": Metric("budget_exhausted", 0),
        "diff_valid": Metric("diff_valid", 1),
        "validation_evidence_ready": Metric("validation_evidence_ready", 1),
        "mode_evidence_ready": Metric("mode_evidence_ready", 1),
    }
    execution = AgentExecution("completed", "", "def answer(): return 1", 10, 5, 2, metrics)
    case = CaseReport(
        spec,
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        ValidationResult(True, "pass", "pass", 0, None, 0.1),
        metrics,
    )

    result = AgentBenchmarkMode().aggregate((case,))

    assert result.primary["task_pass_rate"].value == 1.0
    assert result.guardrails["artifact_evidence_ready"].value == 1
    assert result.guardrails["budget_compliance_rate"].value == 1.0
    assert result.guardrails["tool_execution_success_rate"].value == 1.0


def _mode_case(tmp_path: Path, mode: str) -> tuple[CaseSpec, object, AgentExecution]:
    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        mode,
    )
    paths = prepare_case(tmp_path / mode, spec)
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
            "closed_loop_valid": Metric("closed_loop_valid", 1),
            "peer_communication_valid": Metric(
                "peer_communication_valid", 1
            ),
        },
    )
    return spec, paths, execution


@pytest.mark.asyncio
async def test_context_memory_and_task_state_modes_define_production_experiments(
    tmp_path: Path,
) -> None:
    context_spec, context_paths, context_execution = _mode_case(
        tmp_path, "context-manager"
    )
    context_mode = ContextManagerMode()
    context_candidates = context_mode.candidates(context_spec)
    assert [candidate.name for candidate in context_candidates] == [
        "control",
        "treatment",
    ]
    assert context_candidates[0].setup_prompt == context_candidates[1].setup_prompt
    assert context_candidates[0].context_compaction == "disabled"
    assert context_candidates[0].context_budget_tokens is None
    assert context_candidates[1].context_compaction == "enabled"
    assert context_candidates[1].context_budget_tokens == 8_000
    assert context_candidates[1].setup_prompt.count("diagnostic-") == 500
    assert estimate_tokens(context_candidates[1].setup_prompt) > 7_500
    assert "## Setup turn" in context_candidates[1].artifact_prompt()
    assert "## Execution turn" in context_candidates[1].artifact_prompt()
    context_marker = re.search(
        r"continuation_marker=(LITECODER_EVAL_CONTEXT_[0-9a-f]+)",
        context_candidates[1].setup_prompt,
    )
    assert context_marker is not None
    assert context_marker.group(1) not in context_candidates[1].prompt
    context_execution = AgentExecution(
        "completed",
        "",
        context_execution.solution + f"# {context_marker.group(1)}\n",
        10,
        5,
        1.0,
        {
            "candidate_name": Metric("candidate_name", "treatment"),
            "context_compaction_enabled": Metric(
                "context_compaction_enabled", 1
            ),
            "context_compaction_count": Metric("context_compaction_count", 1),
            "continuation_first_request_input_tokens": Metric(
                "continuation_first_request_input_tokens", 80
            ),
        },
    )
    context = await context_mode.measure(
        context_spec, context_paths, context_execution
    )
    assert context.metrics["context_compaction_count"].value == 1
    assert context.metrics["continuation_constraint_retained"].value == 1

    memory_spec, memory_paths, memory_execution = _mode_case(tmp_path, "memory")
    memory_mode = MemoryMode()
    memory_candidates = memory_mode.candidates(memory_spec)
    assert [candidate.name for candidate in memory_candidates] == ["primary"]
    assert memory_candidates[0].restart_after_setup is False
    assert memory_candidates[0].setup_prompt == ""
    assert memory_candidates[0].memory_recall == "enabled"
    memory_mode.prepare_candidate(memory_spec, memory_paths, memory_candidates[0])
    fixture = fixture_for(memory_spec)
    assert len(fixture.entries) == 9
    assert len(list((memory_paths.solution.parent / ".memory").glob("*.md"))) == 10
    other_spec = CaseSpec(
        "case-0002",
        memory_spec.task_id,
        memory_spec.dataset,
        memory_spec.entry_point,
        memory_spec.starter_code,
        memory_spec.mode,
    )
    assert fixture.fixture_id != fixture_for(other_spec).fixture_id

    task_spec, task_paths, task_execution = _mode_case(tmp_path, "task-state")
    task_mode = TaskStateMode()
    assert task_mode.candidates(task_spec)[0].task_recovery is True
    task_execution = AgentExecution(
        "completed",
        "",
        task_execution.solution,
        10,
        5,
        1.0,
        {
            "recovered": Metric("recovered", 1),
            "dependencies_preserved": Metric("dependencies_preserved", 1),
            "artifact_preserved_after_restart": Metric(
                "artifact_preserved_after_restart", 1
            ),
            "recovery_workflow_completed": Metric(
                "recovery_workflow_completed", 1
            ),
            "duplicate_steps": Metric("duplicate_steps", 0),
        },
    )
    task = await task_mode.measure(task_spec, task_paths, task_execution)
    assert task.metrics["recovered"].value == 1
    assert task.metrics["duplicate_steps"].value == 0
    assert task.metrics["lost_artifacts"].value == 0


def test_context_mode_excludes_unexercised_compaction(tmp_path: Path) -> None:
    spec, paths, execution = _mode_case(tmp_path, "context-manager")
    validation = ValidationResult(True, "pass", "pass", 0, None, 0.1)
    control = CandidateReport(
        "control",
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        validation,
        {
            "continuation_first_request_input_tokens": Metric(
                "continuation_first_request_input_tokens", 100
            ),
            "context_compaction_count": Metric("context_compaction_count", 0),
        },
    )
    treatment = CandidateReport(
        "treatment",
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        validation,
        {
            "continuation_first_request_input_tokens": Metric(
                "continuation_first_request_input_tokens", 50
            ),
            "context_compaction_count": Metric("context_compaction_count", 0),
        },
    )
    mode = ContextManagerMode()

    combined = mode.combine_metrics({"control": control, "treatment": treatment})
    status, reason = mode.score({"control": control, "treatment": treatment})

    assert combined["paired_input_token_reduction"].value == 0.0
    assert combined["compaction_exercised"].value == 0
    assert status == "invalid"
    assert reason == "context treatment did not exercise production compaction"


@pytest.mark.asyncio
async def test_tools_hooks_mode_correlates_dispatch_trace_to_tool_call(
    tmp_path: Path,
) -> None:
    spec, paths, execution = _mode_case(tmp_path, "tools-hooks")
    rows = [
        {
            "event": "tool.runtime",
            "tool_call_id": "call-1",
            "stage": stage,
            "status": (
                "allow"
                if stage == "permission"
                else "success" if stage == "final" else "ok"
            ),
        }
        for stage in ("registry", "pre", "permission", "execute", "final")
    ]
    rows.extend(
        {
            "event": "hook.dispatch.end",
            "point": point,
            "payload": {"call": {"id": "call-1"}},
        }
        for point in ("PreToolUse", "PostToolUse")
    )
    paths.trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    measurement = await ToolsHooksMode().measure(spec, paths, execution)

    assert measurement.metrics["total_tool_calls"].value == 1
    assert measurement.metrics["executed_calls_fully_traced"].value == 1
    assert measurement.metrics["executed_calls_with_hooks"].value == 1


@pytest.mark.asyncio
async def test_tools_hooks_mode_uses_terminal_specific_lifecycles(
    tmp_path: Path,
) -> None:
    spec, paths, execution = _mode_case(tmp_path, "tools-hooks")
    lifecycles = {
        "success": ("registry", "pre", "permission", "execute", "final"),
        "denied": ("registry", "pre", "permission", "final"),
        "duplicate_blocked": ("registry", "pre", "duplicate", "final"),
        "invalid_arguments": ("registry", "pre", "arguments", "final"),
    }
    rows = []
    for index, (status, stages) in enumerate(lifecycles.items(), start=1):
        call_id = f"call-{index}"
        rows.extend(
            {
                "event": "tool.runtime",
                "tool_call_id": call_id,
                "stage": stage,
                "status": status if stage == "final" else "ok",
            }
            for stage in stages
        )
        rows.append(
            {
                "event": "hook.dispatch.end",
                "point": "PreToolUse",
                "payload": {"call": {"id": call_id}},
            }
        )
        if status == "success":
            rows.append(
                {
                    "event": "hook.dispatch.end",
                    "point": "PostToolUse",
                    "payload": {"call": {"id": call_id}},
                }
            )
    paths.trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    mode = ToolsHooksMode()
    measurement = await mode.measure(spec, paths, execution)

    assert measurement.metrics["terminal_outcomes_traced"].value == 4
    assert measurement.metrics["executed_calls_fully_traced"].value == 1
    assert measurement.metrics["denied_calls_fully_traced"].value == 1
    assert measurement.metrics["duplicate_calls_fully_traced"].value == 1
    assert measurement.metrics["invalid_calls_fully_traced"].value == 1
    candidate = CandidateReport(
        "primary",
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        ValidationResult(True, "pass", "pass", 0, None, 0.1),
        measurement.metrics,
    )
    assert mode.score({"primary": candidate}) == ("passed", "")
    assert "empty path" in mode.candidates(spec)[0].setup_prompt

    case_metrics = {
        **measurement.metrics,
        "diff_valid": Metric("diff_valid", 1),
        "tool_outcome_coverage": Metric("tool_outcome_coverage", 1),
        "validation_evidence_ready": Metric("validation_evidence_ready", 1),
        "mode_evidence_ready": Metric("mode_evidence_ready", 1),
        "budget_exhausted": Metric("budget_exhausted", 0),
        "permission_denied_calls": Metric("permission_denied_calls", 1),
        "tool_successful_calls": Metric("tool_successful_calls", 1),
        "tool_failed_calls": Metric("tool_failed_calls", 3),
    }
    case = CaseReport(
        spec,
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        candidate.validation,
        case_metrics,
        candidates={"primary": candidate},
    )

    aggregate = mode.aggregate((case,))

    assert aggregate.guardrails["task_pass_rate"].value == 1.0
    assert aggregate.guardrails["artifact_evidence_ready"].value == 1
    assert "permission_clean_case_rate" not in aggregate.guardrails
    assert "tool_execution_success_rate" not in aggregate.guardrails


@pytest.mark.asyncio
async def test_tools_hooks_mode_rejects_unexercised_terminal_path(
    tmp_path: Path,
) -> None:
    spec, paths, execution = _mode_case(tmp_path, "tools-hooks")
    paths.trace.write_text(
        "".join(
            json.dumps(
                {
                    "event": "tool.runtime",
                    "tool_call_id": "call-1",
                    "stage": stage,
                    "status": "success" if stage == "final" else "ok",
                }
            )
            + "\n"
            for stage in ("registry", "pre", "permission", "execute", "final")
        ),
        encoding="utf-8",
    )
    mode = ToolsHooksMode()
    measurement = await mode.measure(spec, paths, execution)
    candidate = CandidateReport(
        "primary",
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        ValidationResult(True, "pass", "pass", 0, None, 0.1),
        measurement.metrics,
    )

    status, reason = mode.score({"primary": candidate})

    assert status == "invalid"
    assert reason == "tools/hooks probe did not exercise: denied, duplicate, invalid"


@pytest.mark.asyncio
async def test_memory_mode_scores_selection_and_marker_metrics(
    tmp_path: Path,
) -> None:
    spec, paths, execution = _mode_case(tmp_path, "memory")
    mode = MemoryMode()
    candidates = mode.candidates(spec)
    fixture = fixture_for(spec)
    mode.prepare_candidate(spec, paths, candidates[0])
    marked_execution = AgentExecution(
        execution.status,
        execution.reason,
        execution.solution + f"# {fixture.marker}\n",
        execution.input_tokens,
        execution.output_tokens,
        execution.elapsed_seconds,
        {
            "candidate_name": Metric("candidate_name", "primary"),
            "memory_recalled_items": Metric("memory_recalled_items", 4),
            "memory_recalled_ids": Metric(
                "memory_recalled_ids",
                json.dumps(
                    sorted(fixture.relevant_names | {"evalplus-other-task-a"})
                ),
            ),
        },
    )
    measurement = await mode.measure(spec, paths, marked_execution)
    assert measurement.metrics["memory_correct_marker_written"].value == 1
    assert measurement.metrics["memory_relevant_count"].value == 3
    assert measurement.metrics["memory_true_positive_count"].value == 3
    assert measurement.metrics["memory_false_positive_count"].value == 1
    assert measurement.metrics["memory_recall_rate"].value == 1.0
    assert measurement.metrics["memory_precision_rate"].value == 0.75
    validation = ValidationResult(True, "pass", "pass", 0, None, 0.1)
    reports = {
        "primary": CandidateReport(
            "primary",
            "passed",
            CaseStage.SCORED,
            paths,
            marked_execution,
            validation,
            measurement.metrics,
        )
    }
    combined = mode.combine_metrics(reports)

    assert combined["primary_memory_success"].value == 1
    assert combined["primary_memory_true_positive_count"].value == 3
    assert "memory_hit_at_1" not in combined
    assert mode.score(reports) == ("passed", "")

    aggregate_case = CaseReport(
        spec,
        "passed",
        CaseStage.SCORED,
        paths,
        execution,
        validation,
        {
            **combined,
            "primary_all_memory_tokens": Metric(
                "primary_all_memory_tokens", 10_000
            ),
            "primary_optimized_memory_tokens": Metric(
                "primary_optimized_memory_tokens", 3_500
            ),
        },
    )
    aggregate = mode.aggregate((aggregate_case,))
    assert aggregate.primary["memory_context_reduction_rate"].value == 0.65
    assert aggregate.primary["memory_recall_rate"].value == 1.0
    assert aggregate.primary["memory_precision_rate"].value == 0.75
    assert aggregate.primary["memory_false_positive_count"].value == 1

    contaminated = CandidateReport(
        "primary",
        "passed",
        CaseStage.SCORED,
        paths,
        AgentExecution(
            marked_execution.status,
            marked_execution.reason,
            marked_execution.solution
            + f"# {next(iter(fixture.adversarial_markers))}\n",
            marked_execution.input_tokens,
            marked_execution.output_tokens,
            marked_execution.elapsed_seconds,
            marked_execution.metrics,
        ),
        validation,
        {
            **measurement.metrics,
            "memory_wrong_marker_written": Metric(
                "memory_wrong_marker_written", 1
            ),
        },
    )
    assert mode.score({"primary": contaminated})[0] == "failed"

    unscoreable = CaseReport(
        spec,
        "infra_error",
        CaseStage.SCORED,
        paths,
        execution,
        validation,
        {},
    )
    aggregate = mode.aggregate((unscoreable,))
    assert aggregate.primary["memory_task_success_rate"].value == "N/A"
    assert aggregate.primary["memory_precision_rate"].value == "N/A"
    assert aggregate.guardrails["task_pass_rate"].value == "N/A"
