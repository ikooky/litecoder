"""Memory evaluation mode and memory-specific helpers."""

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
)


class MemoryMode(EvalModePlugin):
    """Component responsible for the memory mode."""
    name = "memory"

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return candidate implementations for evaluation."""
        setup = _setup_prompt(spec, distractors=False)
        continuation = _continuation_prompt()
        return (
            ExecutionCandidate(
                "control",
                continuation,
                setup_prompt=setup,
                restart_after_setup=True,
                memory_recall="disabled",
            ),
            ExecutionCandidate(
                "treatment",
                continuation,
                setup_prompt=setup,
                restart_after_setup=True,
                memory_recall="enabled",
            ),
            ExecutionCandidate(
                "distractor",
                continuation,
                setup_prompt=_setup_prompt(spec, distractors=True),
                restart_after_setup=True,
                memory_recall="enabled",
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
        marker_retained = int(_memory_marker(spec) in execution.solution)
        distractor_rejected = int(_distractor_marker(spec) not in execution.solution)
        metrics = {
            name: Metric(name, item.value, item.unit)
            for name, item in execution.metrics.items()
            if name not in {"candidate_name", "candidate_topology"}
        }
        metrics["memory_marker_retained"] = metric(
            "memory_marker_retained", marker_retained
        )
        metrics["distractor_marker_rejected"] = metric(
            "distractor_marker_rejected", distractor_rejected
        )
        return ModeMeasurement(
            metrics=metrics,
            evidence={
                "source": "production-runtime-cross-session-memory",
                "candidate": candidate,
                "restart_count": numeric_metric(
                    execution.metrics, "runtime_restart_count"
                ),
                "memory_recalled_items": numeric_metric(
                    execution.metrics, "memory_recalled_items"
                ),
                "memory_marker_retained": marker_retained,
                "distractor_marker_rejected": distractor_rejected,
            },
        )

    def combine_metrics(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, Metric]:
        """Handle the combine metrics operation."""
        result = combined_candidate_metrics(candidates)
        for name in ("control", "treatment", "distractor"):
            candidate = candidates[name]
            result[f"{name}_recall_exercised"] = metric(
                f"{name}_recall_exercised",
                int(numeric_metric(candidate.metrics, "memory_recalled_items") >= 1),
            )
            result[f"{name}_memory_success"] = metric(
                f"{name}_memory_success",
                int(
                    candidate.status == "passed"
                    and numeric_metric(
                        candidate.metrics, "memory_marker_retained"
                    )
                    == 1
                    and (
                        name == "control"
                        or numeric_metric(
                            candidate.metrics, "memory_recalled_items"
                        )
                        >= 1
                    )
                    and (
                        name != "distractor"
                        or numeric_metric(
                            candidate.metrics, "distractor_marker_rejected"
                        )
                        == 1
                    )
                ),
            )
        result["treatment_uplift"] = metric(
            "treatment_uplift",
            numeric_metric(result, "treatment_memory_success")
            - numeric_metric(result, "control_memory_success"),
        )
        result["distractor_resistance"] = metric(
            "distractor_resistance",
            numeric_metric(result, "distractor_memory_success"),
        )
        return result

    def case_evidence(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, object]:
        """Handle the case evidence operation."""
        return {
            "source": "recreated-production-runtime-memory-comparison",
            "fresh_runtime_for_continuation": True,
            "fresh_session_for_continuation": True,
            "candidates": {
                name: {
                    "status": candidate.status,
                    "validation_passed": candidate.validation.passed,
                    "memory_recalled_items": numeric_metric(
                        candidate.metrics, "memory_recalled_items"
                    ),
                    "runtime_restart_count": numeric_metric(
                        candidate.metrics, "runtime_restart_count"
                    ),
                    "memory_marker_retained": numeric_metric(
                        candidate.metrics, "memory_marker_retained"
                    ),
                    "distractor_marker_rejected": numeric_metric(
                        candidate.metrics, "distractor_marker_rejected"
                    ),
                }
                for name, candidate in candidates.items()
            },
        }

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score the evaluation candidate."""
        expected = ("control", "treatment", "distractor")
        missing = [name for name in expected if name not in candidates]
        if missing:
            return "invalid", f"missing memory candidates: {', '.join(missing)}"
        for status in ("infra_error", "invalid"):
            affected = [name for name, item in candidates.items() if item.status == status]
            if affected:
                return status, f"unscoreable memory candidate(s): {', '.join(affected)}"
        control = candidates["control"]
        if numeric_metric(control.metrics, "memory_recalled_items") != 0:
            return "invalid", "control unexpectedly recalled production memory"
        if numeric_metric(control.metrics, "memory_marker_retained") != 0:
            return "invalid", "control retained the marker without memory recall"
        failures: list[str] = []
        for name in ("treatment", "distractor"):
            candidate = candidates[name]
            if candidate.status != "passed":
                failures.append(f"{name}: {candidate.failure_reason or 'failed'}")
            if numeric_metric(candidate.metrics, "memory_recalled_items") < 1:
                failures.append(f"{name}: production memory recall was not exercised")
            if numeric_metric(candidate.metrics, "memory_marker_retained") != 1:
                failures.append(f"{name}: remembered marker was not retained")
        if numeric_metric(
            candidates["distractor"].metrics, "distractor_marker_rejected"
        ) != 1:
            failures.append("distractor: unrelated marker was incorrectly applied")
        return ("failed", "; ".join(failures)) if failures else ("passed", "")

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        eligible = capability_cases(cases)
        return RunMeasurement(
            primary={
                "treatment_memory_success_rate": metric(
                    "treatment_memory_success_rate",
                    _average_or_na(eligible, "treatment_memory_success"),
                ),
                "cross_session_treatment_success_rate": metric(
                    "cross_session_treatment_success_rate",
                    _average_or_na(eligible, "treatment_memory_success"),
                ),
                "memory_success_uplift": metric(
                    "memory_success_uplift",
                    _average_or_na(eligible, "treatment_uplift"),
                ),
                "distractor_resistance_rate": metric(
                    "distractor_resistance_rate",
                    _average_or_na(eligible, "distractor_resistance"),
                ),
                "treatment_task_success_rate": metric(
                    "treatment_task_success_rate",
                    _average_or_na(eligible, "treatment_candidate_passed"),
                ),
            },
            supporting={
                "control_memory_success_rate": metric(
                    "control_memory_success_rate",
                    _average_or_na(eligible, "control_memory_success"),
                ),
                "control_success_rate": metric(
                    "control_success_rate",
                    _average_or_na(eligible, "control_memory_success"),
                ),
                "control_task_success_rate": metric(
                    "control_task_success_rate",
                    _average_or_na(eligible, "control_candidate_passed"),
                ),
                "treatment_recall_exercise_rate": metric(
                    "treatment_recall_exercise_rate",
                    _average_or_na(eligible, "treatment_recall_exercised"),
                ),
                "treatment_budget_exhaustion_rate": metric(
                    "treatment_budget_exhaustion_rate",
                    _average_or_na(eligible, "treatment_budget_exhausted"),
                ),
                "distractor_recall_exercise_rate": metric(
                    "distractor_recall_exercise_rate",
                    _average_or_na(eligible, "distractor_recall_exercised"),
                ),
                "distractor_task_success_rate": metric(
                    "distractor_task_success_rate",
                    _average_or_na(eligible, "distractor_candidate_passed"),
                ),
                "distractor_budget_exhaustion_rate": metric(
                    "distractor_budget_exhaustion_rate",
                    _average_or_na(eligible, "distractor_budget_exhausted"),
                ),
            },
            guardrails={**task_guardrail(cases), **process_guardrails(cases)},
        )


def _setup_prompt(spec: CaseSpec, *, distractors: bool) -> str:
    marker = _memory_marker(spec)
    relevant = (
        "Use memory_update now to store a project memory named evalplus-current-task. "
        "Its description must say Current EvalPlus continuation constraint and its "
        "body must preserve exactly: "
        f"task_id={spec.task_id}; entry_point={spec.entry_point}; "
        f"dataset={spec.dataset}; only modify solution.py; marker={marker}. "
    )
    if not distractors:
        return relevant + "After storing it, reply briefly and do not edit files."
    return (
        relevant
        + "Also use memory_update to store a project memory named unrelated-task "
        "whose description is Unrelated documentation preference and whose body is "
        "For documentation work, write terse bullet lists and use marker="
        f"{_distractor_marker(spec)}; this does not apply to coding tasks. After "
        "storing both memories, reply briefly and do not edit files."
    )


def _continuation_prompt() -> str:
    return (
        "Continue the benchmark task I asked you to remember in the previous session. "
        "Retrieve and use the relevant durable memory named evalplus-current-task; "
        "do not assume its contents if it was not recalled. Inspect the existing "
        "solution.py as needed and implement the solution now. Add the exact "
        "remembered marker as a Python comment in solution.py, but do not apply "
        "markers from unrelated memories."
    )


def _memory_marker(spec: CaseSpec) -> str:
    digest = hashlib.sha256(
        f"{spec.dataset}:{spec.task_id}:{spec.entry_point}:relevant".encode("utf-8")
    ).hexdigest()[:16]
    return f"LITECODER_EVAL_MEMORY_{digest}"


def _distractor_marker(spec: CaseSpec) -> str:
    digest = hashlib.sha256(
        f"{spec.dataset}:{spec.task_id}:{spec.entry_point}:distractor".encode("utf-8")
    ).hexdigest()[:16]
    return f"LITECODER_EVAL_DISTRACTOR_{digest}"


def text_metric(metrics: Mapping[str, Metric], name: str, default: str) -> str:
    item = metrics.get(name)
    return item.value if item is not None and isinstance(item.value, str) else default


def _average_or_na(cases: tuple[CaseReport, ...], name: str) -> float | str:
    return average(cases, name) if cases else "N/A"
