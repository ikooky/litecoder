"""Context-manager evaluation mode."""

from __future__ import annotations

import hashlib
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
    numeric_metric,
    process_guardrails,
    task_guardrail,
    text_metric,
)


_CONTEXT_BUDGET_TOKENS = 4_096
_DIAGNOSTIC_LINE_COUNT = 360


class ContextManagerMode(EvalModePlugin):
    """Component responsible for the context manager mode."""
    name = "context-manager"

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return candidate implementations for evaluation."""
        setup = _setup_prompt(spec)
        prompt = _continuation_prompt(spec)
        return (
            ExecutionCandidate(
                "control",
                prompt,
                setup_prompt=setup,
                context_compaction="disabled",
            ),
            ExecutionCandidate(
                "treatment",
                prompt,
                setup_prompt=setup,
                context_compaction="enabled",
                context_budget_tokens=_CONTEXT_BUDGET_TOKENS,
            ),
        )

    def summary_candidate(self, candidates: Mapping[str, CandidateReport]) -> str:
        """Return the candidate used for summary reporting."""
        del candidates
        return "treatment"

    async def measure(
        self, spec: CaseSpec, paths: CasePaths, execution: AgentExecution
    ) -> ModeMeasurement:
        """Measure the selected evaluation case."""
        del paths
        candidate = text_metric(execution.metrics, "candidate_name", "unknown")
        retained = int(_context_marker(spec) in execution.solution)
        metrics = {
            name: Metric(name, item.value, item.unit)
            for name, item in execution.metrics.items()
            if name not in {"candidate_name", "candidate_topology"}
        }
        metrics["continuation_constraint_retained"] = metric(
            "continuation_constraint_retained", retained
        )
        return ModeMeasurement(
            metrics=metrics,
            evidence={
                "source": "production-runtime-context-ab",
                "candidate": candidate,
                "context_compaction_enabled": numeric_metric(
                    execution.metrics, "context_compaction_enabled"
                ),
                "context_compaction_count": numeric_metric(
                    execution.metrics, "context_compaction_count"
                ),
                "continuation_constraint_retained": retained,
            },
        )

    def combine_metrics(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, Metric]:
        """Handle the combine metrics operation."""
        result = combined_candidate_metrics(candidates)
        control_first_tokens = numeric_metric(
            candidates["control"].metrics,
            "continuation_first_request_input_tokens",
        )
        treatment_first_tokens = numeric_metric(
            candidates["treatment"].metrics,
            "continuation_first_request_input_tokens",
        )
        control_continuation_tokens = _input_metric(
            candidates["control"].metrics,
            "continuation_total_input_tokens",
            "continuation_input_tokens",
        )
        treatment_continuation_tokens = _input_metric(
            candidates["treatment"].metrics,
            "continuation_total_input_tokens",
            "continuation_input_tokens",
        )
        control_full_run_tokens = _input_metric(
            candidates["control"].metrics,
            "total_recorded_input_tokens",
            "input_tokens",
        )
        treatment_full_run_tokens = _input_metric(
            candidates["treatment"].metrics,
            "total_recorded_input_tokens",
            "input_tokens",
        )
        compaction_exercised = int(
            numeric_metric(
                candidates["treatment"].metrics,
                "context_compaction_count",
            )
            >= 1
        )
        first_reduction = (
            1 - treatment_first_tokens / control_first_tokens
            if compaction_exercised and control_first_tokens
            else 0.0
        )
        continuation_reduction = (
            1 - treatment_continuation_tokens / control_continuation_tokens
            if compaction_exercised and control_continuation_tokens
            else 0.0
        )
        full_run_reduction = (
            1 - treatment_full_run_tokens / control_full_run_tokens
            if compaction_exercised and control_full_run_tokens
            else 0.0
        )
        result["first_request_input_token_reduction"] = metric(
            "first_request_input_token_reduction", first_reduction
        )
        result["continuation_input_token_reduction"] = metric(
            "continuation_input_token_reduction", continuation_reduction
        )
        result["full_run_input_token_reduction"] = metric(
            "full_run_input_token_reduction", full_run_reduction
        )
        # Keep the old key as a compatibility alias. New reports use the
        # full-run metric so a single request cannot overstate savings.
        result["paired_input_token_reduction"] = metric(
            "paired_input_token_reduction", first_reduction
        )
        result["compaction_exercised"] = metric(
            "compaction_exercised",
            compaction_exercised,
        )
        return result

    def case_evidence(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, object]:
        """Handle the case evidence operation."""
        return {
            "source": "paired-production-runtime-context-ab",
            "same_setup_prompt": True,
            "same_continuation_prompt": True,
            "context_budget_tokens": _CONTEXT_BUDGET_TOKENS,
            "candidates": {
                name: {
                    "status": candidate.status,
                    "validation_passed": candidate.validation.passed,
                    "continuation_first_request_input_tokens": numeric_metric(
                        candidate.metrics,
                        "continuation_first_request_input_tokens",
                    ),
                    "continuation_total_input_tokens": _input_metric(
                        candidate.metrics,
                        "continuation_total_input_tokens",
                        "continuation_input_tokens",
                    ),
                    "total_recorded_input_tokens": _input_metric(
                        candidate.metrics,
                        "total_recorded_input_tokens",
                        "input_tokens",
                    ),
                    "context_compaction_count": numeric_metric(
                        candidate.metrics, "context_compaction_count"
                    ),
                }
                for name, candidate in candidates.items()
            },
        }

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score the evaluation candidate."""
        missing = [name for name in ("control", "treatment") if name not in candidates]
        if missing:
            return "invalid", f"missing context A/B candidates: {', '.join(missing)}"
        for status in ("infra_error", "invalid"):
            affected = [name for name, item in candidates.items() if item.status == status]
            if affected:
                return status, f"unscoreable context A/B candidate(s): {', '.join(affected)}"
        treatment = candidates["treatment"]
        if numeric_metric(treatment.metrics, "context_compaction_count") < 1:
            return "invalid", "context treatment did not exercise production compaction"
        if numeric_metric(
            treatment.metrics, "continuation_constraint_retained"
        ) != 1:
            return "failed", "context treatment did not retain the setup constraint"
        if treatment.status != "passed":
            return "failed", treatment.failure_reason or "context treatment failed"
        return "passed", ""

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        eligible = capability_cases(cases)
        return RunMeasurement(
            primary={
                "treatment_task_success_rate": metric(
                    "treatment_task_success_rate",
                    average(eligible, "treatment_candidate_passed"),
                ),
                "control_task_success_rate": metric(
                    "control_task_success_rate",
                    average(eligible, "control_candidate_passed"),
                ),
                "average_paired_input_token_reduction": metric(
                    "average_paired_input_token_reduction",
                    average(eligible, "paired_input_token_reduction"),
                ),
                "average_full_run_input_token_reduction": metric(
                    "average_full_run_input_token_reduction",
                    average(eligible, "full_run_input_token_reduction"),
                ),
            },
            supporting={
                "average_first_request_input_token_reduction": metric(
                    "average_first_request_input_token_reduction",
                    average(eligible, "first_request_input_token_reduction"),
                ),
                "average_continuation_input_token_reduction": metric(
                    "average_continuation_input_token_reduction",
                    average(eligible, "continuation_input_token_reduction"),
                ),
                "treatment_constraint_retention_rate": metric(
                    "treatment_constraint_retention_rate",
                    average(
                        eligible,
                        "treatment_continuation_constraint_retained",
                    ),
                ),
                "compaction_exercise_rate": metric(
                    "compaction_exercise_rate",
                    average(eligible, "compaction_exercised"),
                ),
                "treatment_budget_exhaustion_rate": metric(
                    "treatment_budget_exhaustion_rate",
                    average(eligible, "treatment_budget_exhausted"),
                ),
            },
            guardrails={**task_guardrail(cases), **process_guardrails(cases)},
        )


def _setup_prompt(spec: CaseSpec) -> str:
    marker = _context_marker(spec)
    diagnostic = "\n".join(
        f"diagnostic-{index:03d}: unrelated historical tool output for {spec.dataset}"
        for index in range(_DIAGNOSTIC_LINE_COUNT)
    )
    return (
        "This is the setup turn for a continuation task. Do not edit files yet. "
        "Retain these constraints for the next turn: only solution.py may change; "
        f"task_id={spec.task_id}; entry_point={spec.entry_point}; "
        f"dataset={spec.dataset}; continuation_marker={marker}. In the next turn, "
        "add that exact marker as a Python comment in solution.py. Reply with a "
        "brief acknowledgement only.\n\n"
        f"Historical diagnostic context follows:\n{diagnostic}"
    )


def _continuation_prompt(spec: CaseSpec) -> str:
    return (
        "Continue the task from the previous turn. Apply the retained constraints "
        "and implement the requested function now.\n\n"
        f"{spec.prompt()}"
    )


def _context_marker(spec: CaseSpec) -> str:
    digest = hashlib.sha256(
        f"{spec.dataset}:{spec.task_id}:{spec.entry_point}:context".encode("utf-8")
    ).hexdigest()[:16]
    return f"LITECODER_EVAL_CONTEXT_{digest}"


def _input_metric(metrics: Mapping[str, Metric], preferred: str, fallback: str) -> float:
    value = numeric_metric(metrics, preferred)
    return value if value > 0 else numeric_metric(metrics, fallback)
