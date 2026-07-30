"""Agent benchmark evaluation mode."""

from __future__ import annotations

from collections import Counter

from litecoder.eval.domain import AgentExecution, CasePaths, CaseReport, CaseSpec, ModeMeasurement
from litecoder.eval.modes.base import (
    EvalModePlugin,
    RunMeasurement,
    average,
    capability_cases,
    metric,
    passed,
    process_guardrails,
    ratio,
    total,
)


class AgentBenchmarkMode(EvalModePlugin):
    """Component responsible for the agent benchmark mode."""
    name = "agent-benchmark"

    async def measure(
        self, spec: CaseSpec, paths: CasePaths, execution: AgentExecution
    ) -> ModeMeasurement:
        """Measure the selected evaluation case."""
        del spec, paths
        return ModeMeasurement(
            metrics=execution.metrics,
            evidence={
                "source": "runtime-execution",
                "status": execution.status,
                "reason": execution.reason,
                "input_tokens": execution.input_tokens,
                "output_tokens": execution.output_tokens,
                "elapsed_seconds": execution.elapsed_seconds,
                "failure": (
                    execution.failure.to_json() if execution.failure else None
                ),
            },
        )

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        eligible = capability_cases(cases)
        failures = Counter(
            case.failure_reason for case in eligible if case.status != "passed"
        )
        failure_text = ", ".join(
            f"{reason or 'unknown'}:{count}"
            for reason, count in sorted(failures.items())
        )
        return RunMeasurement(
            primary={
                "task_pass_rate": metric(
                    "task_pass_rate", ratio(passed(eligible), len(eligible))
                )
            },
            supporting={
                "passed_total": metric(
                    "passed_total", f"{passed(eligible)}/{len(eligible)}"
                ),
                "average_time_seconds": metric(
                    "average_time_seconds", average(eligible, "wall_clock_seconds"), "seconds"
                ),
                "average_tokens": metric(
                    "average_tokens",
                    average(eligible, "input_tokens") + average(eligible, "output_tokens"),
                    "tokens",
                ),
                "average_model_rounds": metric(
                    "average_model_rounds", average(eligible, "model_rounds")
                ),
                "permission_denied_call_count": metric(
                    "permission_denied_call_count",
                    total(cases, "permission_denied_calls"),
                ),
                "tool_failed_call_count": metric(
                    "tool_failed_call_count", total(cases, "tool_failed_calls")
                ),
                "duplicate_blocked_call_count": metric(
                    "duplicate_blocked_call_count",
                    total(cases, "duplicate_blocked_calls"),
                ),
                "tool_successful_call_count": metric(
                    "tool_successful_call_count",
                    total(cases, "tool_successful_calls"),
                ),
                "budget_exhausted_case_count": metric(
                    "budget_exhausted_case_count", total(cases, "budget_exhausted")
                ),
                "failure_reason_distribution": metric(
                    "failure_reason_distribution", failure_text
                ),
            },
            guardrails=process_guardrails(cases),
        )
