"""Task-state evaluation mode."""

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
    metric,
    numeric_metric,
    process_guardrails,
    task_guardrail,
)


class TaskStateMode(EvalModePlugin):
    """Component responsible for the task state mode."""
    name = "task-state"

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return candidate implementations for evaluation."""
        return (
            ExecutionCandidate(
                "recovery",
                spec.prompt(),
                task_recovery=True,
            ),
        )

    def summary_candidate(self, candidates: Mapping[str, CandidateReport]) -> str:
        """Return the candidate used for summary reporting."""
        del candidates
        return "recovery"

    async def measure(
        self, spec: CaseSpec, paths: CasePaths, execution: AgentExecution
    ) -> ModeMeasurement:
        """Measure the selected evaluation case."""
        del spec, paths
        metrics = {
            name: Metric(name, item.value, item.unit)
            for name, item in execution.metrics.items()
            if name != "candidate_name"
        }
        metrics["lost_artifacts"] = metric(
            "lost_artifacts",
            int(
                numeric_metric(
                    execution.metrics, "artifact_preserved_after_restart"
                )
                != 1
            ),
        )
        return ModeMeasurement(
            metrics=metrics,
            evidence={
                "source": "production-runtime-startup-recovery",
                "interruption_checkpoint": text_metric(
                    execution.metrics,
                    "interruption_checkpoint",
                    "unknown",
                ),
                "runtime_restart_count": numeric_metric(
                    execution.metrics, "runtime_restart_count"
                ),
                "recovered": numeric_metric(execution.metrics, "recovered"),
                "dependencies_preserved": numeric_metric(
                    execution.metrics, "dependencies_preserved"
                ),
                "artifact_preserved_after_restart": numeric_metric(
                    execution.metrics, "artifact_preserved_after_restart"
                ),
                "recovery_workflow_completed": numeric_metric(
                    execution.metrics, "recovery_workflow_completed"
                ),
            },
        )

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score the evaluation candidate."""
        candidate = candidates.get("recovery")
        if candidate is None:
            return "invalid", "missing task-state recovery candidate"
        if candidate.status in {"infra_error", "invalid"}:
            return candidate.status, candidate.failure_reason
        failures: list[str] = []
        if candidate.status != "passed":
            failures.append(candidate.failure_reason or "solution failed")
        for name in (
            "recovered",
            "dependencies_preserved",
            "artifact_preserved_after_restart",
            "recovery_workflow_completed",
        ):
            if numeric_metric(candidate.metrics, name) != 1:
                failures.append(f"{name} was not satisfied")
        if numeric_metric(candidate.metrics, "duplicate_steps") != 0:
            failures.append("duplicate durable steps were observed")
        return ("failed", "; ".join(failures)) if failures else ("passed", "")

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        eligible = capability_cases(cases)
        return RunMeasurement(
            primary={
                "recovery_success_rate": metric(
                    "recovery_success_rate", average(eligible, "recovered")
                ),
                "continuation_success_rate": metric(
                    "continuation_success_rate",
                    average(eligible, "recovery_workflow_completed"),
                ),
                "artifact_loss_rate": metric(
                    "artifact_loss_rate", average(eligible, "lost_artifacts")
                ),
            },
            supporting={
                "dependency_retention_rate": metric(
                    "dependency_retention_rate",
                    average(eligible, "dependencies_preserved"),
                ),
                "duplicate_step_rate": metric(
                    "duplicate_step_rate", average(eligible, "duplicate_steps")
                ),
            },
            guardrails={**task_guardrail(cases), **process_guardrails(cases)},
        )


def text_metric(metrics: Mapping[str, Metric], name: str, default: str) -> str:
    item = metrics.get(name)
    return item.value if item is not None and isinstance(item.value, str) else default
