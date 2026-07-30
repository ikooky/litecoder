"""Base interfaces for evaluation modes or providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from litecoder.eval.domain import (
    AgentExecution,
    CandidateReport,
    CasePaths,
    CaseReport,
    CaseSpec,
    ExecutionCandidate,
    ExecutionPolicy,
    Metric,
    ModeMeasurement,
)
from litecoder.eval.policies import policy_for_mode


PROCESS_MINIMUMS = (
    "diff_valid",
    "tool_outcome_coverage",
    "validation_evidence_ready",
    "mode_evidence_ready",
)
PROCESS_SUMS = (
    "tool_calls",
    "tool_successful_calls",
    "permission_requests",
    "permission_denied_calls",
    "duplicate_blocked_calls",
    "tool_failed_calls",
    "undispatched_tool_calls",
)

@dataclass(frozen=True, slots=True)
class RunMeasurement:
    """Data model representing the run measurement."""
    primary: Mapping[str, Metric]
    supporting: Mapping[str, Metric]
    guardrails: Mapping[str, Metric]


class EvalModePlugin(ABC):
    """Component responsible for the eval mode plugin."""
    name: str

    def policy(self, spec: CaseSpec) -> ExecutionPolicy:
        """Return the evaluation policy."""
        return policy_for_mode(spec.mode)

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return candidate implementations for evaluation."""
        return (ExecutionCandidate("primary", spec.prompt()),)

    def summary_candidate(self, candidates: Mapping[str, CandidateReport]) -> str:
        """Return the candidate used for summary reporting."""
        if "primary" in candidates:
            return "primary"
        return next(iter(candidates))

    def combine_metrics(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, Metric]:
        """Handle the combine metrics operation."""
        return candidates[self.summary_candidate(candidates)].metrics

    def case_evidence(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, object]:
        """Handle the case evidence operation."""
        return {
            "candidates": {
                name: {
                    "status": candidate.status,
                    "failure_reason": candidate.failure_reason,
                    "workspace": str(candidate.paths.root),
                }
                for name, candidate in candidates.items()
            }
        }

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score the evaluation candidate."""
        candidate = candidates[self.summary_candidate(candidates)]
        return candidate.status, candidate.failure_reason

    @abstractmethod
    async def measure(
        self,
        spec: CaseSpec,
        paths: CasePaths,
        execution: AgentExecution,
    ) -> ModeMeasurement:
        """Measure the selected evaluation case."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        raise NotImplementedError


def metric(name: str, value: float | int | str, unit: str = "") -> Metric:
    """Return the named evaluation metric."""
    return Metric(name, value, unit)


def numeric(case: CaseReport, name: str) -> float:
    """Return the numeric metric value."""
    value = case.metrics.get(name)
    if value is None or isinstance(value.value, str):
        return 0.0
    return float(value.value)


def total(cases: tuple[CaseReport, ...], name: str) -> float:
    """Return the total metric value."""
    return sum(numeric(case, name) for case in cases)


def average(cases: tuple[CaseReport, ...], name: str) -> float:
    """Return the average metric value."""
    return total(cases, name) / len(cases) if cases else 0.0


def ratio(numerator: float, denominator: float) -> float:
    """Return the metric ratio."""
    return numerator / denominator if denominator else 0.0


def passed(cases: tuple[CaseReport, ...]) -> int:
    """Return whether the evaluation passed."""
    return sum(case.status == "passed" for case in cases)


def capability_cases(cases: tuple[CaseReport, ...]) -> tuple[CaseReport, ...]:
    """Handle the capability cases operation."""
    return tuple(case for case in cases if case.status in {"passed", "failed"})


def artifact_evidence_ready(case: CaseReport) -> bool:
    """Handle the artifact evidence ready operation."""
    return (
        numeric(case, "diff_valid") == 1
        and numeric(case, "tool_outcome_coverage") == 1
        and numeric(case, "validation_evidence_ready") == 1
    )


def task_guardrail(cases: tuple[CaseReport, ...]) -> dict[str, Metric]:
    """Handle the task guardrail operation."""
    eligible = capability_cases(cases)
    return {
        "task_pass_rate": metric(
            "task_pass_rate",
            ratio(passed(eligible), len(eligible)) if eligible else "N/A",
        )
    }


def process_guardrails(cases: tuple[CaseReport, ...]) -> dict[str, Metric]:
    """Process the guardrails."""
    count = len(cases)
    executed_tool_calls = total(cases, "tool_successful_calls") + total(
        cases, "tool_failed_calls"
    )
    artifact_ready = sum(artifact_evidence_ready(case) for case in cases)
    artifact_missing = ", ".join(
        case.spec.case_id for case in cases if not artifact_evidence_ready(case)
    )
    return {
        "artifact_evidence_ready": metric(
            "artifact_evidence_ready",
            int(bool(cases) and artifact_ready == count),
        ),
        "artifact_evidence_ready_rate": metric(
            "artifact_evidence_ready_rate", ratio(artifact_ready, count)
        ),
        "artifact_evidence_missing_cases": metric(
            "artifact_evidence_missing_cases", artifact_missing
        ),
        "mode_evidence_ready": metric(
            "mode_evidence_ready",
            int(bool(cases) and all(numeric(case, "mode_evidence_ready") == 1 for case in cases)),
        ),
        "budget_compliance_rate": metric(
            "budget_compliance_rate",
            ratio(count - total(cases, "budget_exhausted"), count),
        ),
        "permission_clean_case_rate": metric(
            "permission_clean_case_rate",
            ratio(
                sum(numeric(case, "permission_denied_calls") == 0 for case in cases),
                count,
            ),
        ),
        "tool_execution_success_rate": metric(
            "tool_execution_success_rate",
            ratio(total(cases, "tool_successful_calls"), executed_tool_calls),
        ),
        "scoreable_case_count": metric(
            "scoreable_case_count", len(capability_cases(cases))
        ),
        "infra_error_case_count": metric(
            "infra_error_case_count",
            sum(case.status == "infra_error" for case in cases),
        ),
        "invalid_case_count": metric(
            "invalid_case_count", sum(case.status == "invalid" for case in cases)
        ),
    }


def combined_candidate_metrics(
    candidates: Mapping[str, CandidateReport],
) -> dict[str, Metric]:
    """Handle the combined candidate metrics operation."""
    result: dict[str, Metric] = {}
    for candidate_name, candidate in candidates.items():
        for name, item in candidate.metrics.items():
            combined_name = f"{candidate_name}_{name}"
            result[combined_name] = Metric(combined_name, item.value, item.unit)
        result[f"{candidate_name}_validation_passed"] = metric(
            f"{candidate_name}_validation_passed",
            int(candidate.validation.passed),
        )
        result[f"{candidate_name}_candidate_passed"] = metric(
            f"{candidate_name}_candidate_passed",
            int(candidate.status == "passed"),
        )
    for name in PROCESS_MINIMUMS:
        result[name] = metric(
            name,
            min(numeric_metric(candidate.metrics, name) for candidate in candidates.values()),
        )
    for name in PROCESS_SUMS:
        result[name] = metric(
            name,
            sum(numeric_metric(candidate.metrics, name) for candidate in candidates.values()),
        )
    result["budget_exhausted"] = metric(
        "budget_exhausted",
        max(
            numeric_metric(candidate.metrics, "budget_exhausted")
            for candidate in candidates.values()
        ),
    )
    return result


def numeric_metric(metrics: Mapping[str, Metric], name: str) -> float:
    """Read a numeric metric, returning 0.0 when missing or text-valued."""
    item = metrics.get(name)
    if item is None or isinstance(item.value, str):
        return 0.0
    return float(item.value)


def text_metric(
    metrics: Mapping[str, Metric], name: str, default: str
) -> str:
    """Read a text metric, returning ``default`` when missing or non-string."""
    item = metrics.get(name)
    return item.value if item is not None and isinstance(item.value, str) else default
