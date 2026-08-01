"""Multi-agent evaluation mode."""

from __future__ import annotations

from collections.abc import Mapping

from litecoder.eval.domain import (
    AgentExecution,
    CandidateReport,
    CasePaths,
    CaseReport,
    CaseSpec,
    ExecutionCandidate,
    Metric,
    ModeMeasurement,
)
from litecoder.eval.modes.base import (
    EvalModePlugin,
    RunMeasurement,
    average,
    capability_cases,
    combined_candidate_metrics,
    metric,
    numeric,
    numeric_metric,
    process_guardrails,
    task_guardrail,
    text_metric,
)


class MultiAgentMode(EvalModePlugin):
    """Component responsible for the multi agent mode."""
    name = "multi-agent"

    _MIN_TIME_COMPARABLE_CASES = 3

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return candidate implementations for evaluation."""
        return (
            ExecutionCandidate("subagent", _subagent_prompt(spec.prompt()), "subagent"),
            ExecutionCandidate("team", _team_prompt(spec.prompt()), "team"),
        )

    def summary_candidate(self, candidates: Mapping[str, CandidateReport]) -> str:
        """Return the candidate used for summary reporting."""
        del candidates
        return "team"

    async def measure(
        self, spec: CaseSpec, paths: CasePaths, execution: AgentExecution
    ) -> ModeMeasurement:
        """Measure the selected evaluation case."""
        del spec, paths
        candidate_name = text_metric(execution.metrics, "candidate_name", "unknown")
        values = {
            name: item.value
            for name, item in execution.metrics.items()
            if name not in {"candidate_name", "candidate_topology"}
        }
        return ModeMeasurement(
            metrics={
                name: Metric(name, item.value, item.unit)
                for name, item in execution.metrics.items()
                if name not in {"candidate_name", "candidate_topology"}
            },
            evidence={
                "source": "production-runtime",
                "candidate": candidate_name,
                "topology": text_metric(
                    execution.metrics, "candidate_topology", "unknown"
                ),
                "runtime": values,
            },
        )

    def combine_metrics(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, Metric]:
        """Handle the combine metrics operation."""
        result = combined_candidate_metrics(candidates)
        for candidate_name, candidate in candidates.items():
            result[f"{candidate_name}_success_rate"] = metric(
                f"{candidate_name}_success_rate", float(candidate.status == "passed")
            )
            result[f"{candidate_name}_code_validation_passed"] = metric(
                f"{candidate_name}_code_validation_passed",
                int(candidate.validation.passed),
            )
            result[f"{candidate_name}_workflow_completed"] = metric(
                f"{candidate_name}_workflow_completed",
                int(numeric_metric(candidate.metrics, "closed_loop_valid") == 1),
            )
            peer_completed = int(
                candidate_name != "team"
                or numeric_metric(candidate.metrics, "peer_communication_valid") == 1
            )
            result[f"{candidate_name}_peer_communication_completed"] = metric(
                f"{candidate_name}_peer_communication_completed", peer_completed
            )
            result[f"{candidate_name}_full_case_success"] = metric(
                f"{candidate_name}_full_case_success",
                int(
                    candidate.status == "passed"
                    and numeric_metric(candidate.metrics, "closed_loop_valid") == 1
                    and peer_completed == 1
                ),
            )
            result[f"{candidate_name}_time_seconds"] = metric(
                f"{candidate_name}_time_seconds",
                candidate.execution.elapsed_seconds,
                "seconds",
            )

        return result

    def case_evidence(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, object]:
        """Handle the case evidence operation."""
        return {
            "source": "isolated-production-runtime-candidates",
            "candidates": {
                name: {
                    "status": candidate.status,
                    "failure_reason": candidate.failure_reason,
                    "workspace": str(candidate.paths.root),
                    "validation": candidate.validation.to_json(),
                    "closed_loop_valid": numeric_metric(
                        candidate.metrics, "closed_loop_valid"
                    ),
                }
                for name, candidate in candidates.items()
            },
        }

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score the evaluation candidate."""
        missing = [name for name in ("subagent", "team") if name not in candidates]
        if missing:
            return "invalid", f"missing multi-agent candidates: {', '.join(missing)}"
        infrastructure = [
            f"{name}: {candidate.failure_reason or 'candidate error'}"
            for name, candidate in candidates.items()
            if candidate.status == "infra_error"
        ]
        if infrastructure:
            return "infra_error", "; ".join(infrastructure)
        invalid = [
            f"{name}: {candidate.failure_reason or 'candidate invalid'}"
            for name, candidate in candidates.items()
            if candidate.status == "invalid"
        ]
        if invalid:
            return "invalid", "; ".join(invalid)
        failures: list[str] = []
        for name in ("subagent", "team"):
            candidate = candidates[name]
            if candidate.status != "passed":
                failures.append(
                    f"{name}: {candidate.failure_reason or 'candidate failed'}"
                )
            if numeric_metric(candidate.metrics, "closed_loop_valid") != 1:
                failures.append(f"{name}.closed_loop: lifecycle validation failed")
        return ("failed", "; ".join(failures)) if failures else ("passed", "")

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        eligible = capability_cases(cases)
        comparable = tuple(
            case
            for case in eligible
            if numeric(case, "subagent_full_case_success") == 1
            and numeric(case, "team_full_case_success") == 1
        )
        enough_time_samples = len(comparable) >= self._MIN_TIME_COMPARABLE_CASES
        subagent_time = average(comparable, "subagent_time_seconds")
        team_time = average(comparable, "team_time_seconds")
        return RunMeasurement(
            primary={
                "team_vs_subagent_time_reduction": metric(
                    "team_vs_subagent_time_reduction",
                    (
                        1 - team_time / subagent_time
                        if enough_time_samples and subagent_time
                        else "N/A"
                    ),
                ),
                "team_full_case_success_rate": metric(
                    "team_full_case_success_rate",
                    average(eligible, "team_full_case_success"),
                ),
                "team_code_validation_pass_rate": metric(
                    "team_code_validation_pass_rate",
                    average(eligible, "team_code_validation_passed"),
                ),
                "team_workflow_completion_rate": metric(
                    "team_workflow_completion_rate",
                    average(eligible, "team_workflow_completed"),
                ),
            },
            supporting={
                "subagent_full_case_success_rate": metric(
                    "subagent_full_case_success_rate",
                    average(eligible, "subagent_full_case_success"),
                ),
                "subagent_code_validation_pass_rate": metric(
                    "subagent_code_validation_pass_rate",
                    average(eligible, "subagent_code_validation_passed"),
                ),
                "subagent_workflow_completion_rate": metric(
                    "subagent_workflow_completion_rate",
                    average(eligible, "subagent_workflow_completed"),
                ),
                "subagent_closed_loop_rate": metric(
                    "subagent_closed_loop_rate",
                    average(eligible, "subagent_closed_loop_valid"),
                ),
                "team_closed_loop_rate": metric(
                    "team_closed_loop_rate",
                    average(eligible, "team_closed_loop_valid"),
                ),
                "team_peer_communication_rate": metric(
                    "team_peer_communication_rate",
                    average(eligible, "team_peer_communication_completed"),
                ),
                "time_comparable_case_count": metric(
                    "time_comparable_case_count", len(comparable)
                ),
                "time_comparison_minimum_case_count": metric(
                    "time_comparison_minimum_case_count",
                    self._MIN_TIME_COMPARABLE_CASES,
                ),
            },
            guardrails={**task_guardrail(cases), **process_guardrails(cases)},
        )


def _subagent_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Exercise the production single-subagent workflow. Create a durable pending "
        "task with task_create, create and bind its worktree with worktree_create, "
        "then call spawn_subagent with that task_id and worktree_id. Do not create a "
        "team. The child must explicitly task_claim before editing and task_complete "
        "after finishing. Review the returned result and integrate the correct "
        "implementation into this candidate's root solution.py."
    )


def _team_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Exercise the production team workflow. Create at least two independent "
        "pending tasks and a bound worktree for each, then create at least two team "
        "members with team_create. Each member must explicitly task_claim its own "
        "task, do substantive work, and task_complete it. Require at least one "
        "member-to-member team_send exchange as well as substantive results sent to "
        "the lead. The lead must receive and review the results, then integrate the "
        "selected implementation into this candidate's root solution.py."
    )
