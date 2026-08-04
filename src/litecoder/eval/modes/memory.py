"""Memory selection evaluation mode and memory-specific measurements."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

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
from litecoder.eval.memory_fixture import (
    fixture_for,
    memory_marker,
    write_memory_fixture,
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
    ratio,
    task_guardrail,
    text_metric,
)
from litecoder.memory.store import MemoryStore


class MemoryMode(EvalModePlugin):
    """Evaluate selection from a pre-seeded, noisy memory catalog."""

    name = "memory"

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return one normal memory-enabled selector candidate."""
        del spec
        return (
            ExecutionCandidate(
                "primary",
                _continuation_prompt(),
                memory_recall="enabled",
            ),
        )

    def prepare_candidate(
        self,
        spec: CaseSpec,
        paths: CasePaths,
        candidate: ExecutionCandidate,
    ) -> None:
        """Create the independent nine-entry catalog for this case."""
        if candidate.name != "primary":
            raise ValueError(f"unexpected memory candidate: {candidate.name}")
        write_memory_fixture(paths.solution.parent, spec)

    def summary_candidate(self, candidates: Mapping[str, CandidateReport]) -> str:
        """Return the selector candidate used for summary reporting."""
        del candidates
        return "primary"

    async def measure(
        self, spec: CaseSpec, paths: CasePaths, execution: AgentExecution
    ) -> ModeMeasurement:
        """Measure retrieval quality, marker use, task success, and lifecycle."""
        candidate = text_metric(execution.metrics, "candidate_name", "unknown")
        fixture = fixture_for(spec)
        recalled_names = _memory_ids(execution.metrics)
        true_positive = len(recalled_names & fixture.relevant_names)
        false_negative = len(fixture.relevant_names - recalled_names)
        false_positive = len(recalled_names - fixture.relevant_names)
        correct_marker_written = int(fixture.marker in execution.solution)
        wrong_marker_written = int(
            any(marker in execution.solution for marker in fixture.adversarial_markers)
        )
        lifecycle = _memory_lifecycle(paths.trace)
        final_count = _final_memory_count(paths.solution.parent)
        metrics = {
            name: Metric(name, item.value, item.unit)
            for name, item in execution.metrics.items()
            if name != "candidate_name"
        }
        metrics.update(
            {
                "memory_fixture_entry_count": metric(
                    "memory_fixture_entry_count", len(fixture.entries)
                ),
                "memory_initial_count": metric(
                    "memory_initial_count", len(fixture.entries)
                ),
                "memory_final_count": metric("memory_final_count", final_count),
                "memory_correct_marker_written": metric(
                    "memory_correct_marker_written", correct_marker_written
                ),
                # Compatibility alias for existing report consumers.
                "memory_marker_retained": metric(
                    "memory_marker_retained", correct_marker_written
                ),
                "memory_wrong_marker_written": metric(
                    "memory_wrong_marker_written", wrong_marker_written
                ),
                "memory_relevant_count": metric(
                    "memory_relevant_count", len(fixture.relevant_names)
                ),
                "memory_retrieved_count": metric(
                    "memory_retrieved_count", len(recalled_names)
                ),
                "memory_true_positive_count": metric(
                    "memory_true_positive_count", true_positive
                ),
                "memory_false_negative_count": metric(
                    "memory_false_negative_count", false_negative
                ),
                "memory_false_positive_count": metric(
                    "memory_false_positive_count", false_positive
                ),
                "memory_recall_rate": metric(
                    "memory_recall_rate", ratio(true_positive, len(fixture.relevant_names))
                ),
                "memory_precision_rate": metric(
                    "memory_precision_rate", ratio(true_positive, len(recalled_names))
                ),
                "memory_false_positive_rate": metric(
                    "memory_false_positive_rate", ratio(false_positive, len(recalled_names))
                ),
                # Compatibility alias; the experiment now reports precision explicitly.
                "memory_accuracy": metric(
                    "memory_accuracy", ratio(true_positive, len(recalled_names))
                ),
                "memory_dream_triggered": metric(
                    "memory_dream_triggered", int(lifecycle["dream_count"] > 0)
                ),
                "memory_dream_count": metric(
                    "memory_dream_count", lifecycle["dream_count"]
                ),
                "memory_extract_count": metric(
                    "memory_extract_count", lifecycle["extract_count"]
                ),
                "memory_dream_statuses": metric(
                    "memory_dream_statuses", json.dumps(lifecycle["dream_statuses"])
                ),
            }
        )
        return ModeMeasurement(
            metrics=metrics,
            evidence={
                "source": "production-runtime-memory-selection",
                "candidate": candidate,
                "fixture_id": fixture.fixture_id,
                "fixture_entry_count": len(fixture.entries),
                "fixture_relevant_names": sorted(fixture.relevant_names),
                "fixture_same_topic_names": sorted(fixture.same_topic_names),
                "fixture_unrelated_names": sorted(fixture.unrelated_names),
                "fixture_stale_conflict_names": sorted(
                    fixture.stale_conflict_names
                ),
                "fixture_adversarial_markers": sorted(
                    fixture.adversarial_markers
                ),
                "memory_recalled_items": numeric_metric(
                    execution.metrics, "memory_recalled_items"
                ),
                "memory_recalled_ids": sorted(recalled_names),
                "memory_initial_count": len(fixture.entries),
                "memory_final_count": final_count,
                "memory_true_positive_count": true_positive,
                "memory_false_negative_count": false_negative,
                "memory_false_positive_count": false_positive,
                "memory_recall_rate": ratio(
                    true_positive, len(fixture.relevant_names)
                ),
                "memory_precision_rate": ratio(true_positive, len(recalled_names)),
                "memory_correct_marker_written": correct_marker_written,
                "memory_wrong_marker_written": wrong_marker_written,
                "memory_dream_triggered": int(lifecycle["dream_count"] > 0),
                "memory_dream_count": lifecycle["dream_count"],
                "memory_dream_statuses": lifecycle["dream_statuses"],
                "restart_count": numeric_metric(
                    execution.metrics, "runtime_restart_count"
                ),
            },
        )

    def combine_metrics(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, Metric]:
        """Combine the selector candidate without an A/B uplift."""
        result = combined_candidate_metrics(candidates)
        candidate = candidates["primary"]
        recalled = numeric_metric(candidate.metrics, "memory_recalled_items")
        correct_marker = numeric_metric(
            candidate.metrics, "memory_correct_marker_written"
        )
        wrong_marker = numeric_metric(
            candidate.metrics, "memory_wrong_marker_written"
        )
        result["primary_recall_exercised"] = metric(
            "primary_recall_exercised", int(recalled >= 1)
        )
        result["primary_memory_success"] = metric(
            "primary_memory_success",
            int(
                candidate.status == "passed"
                and correct_marker == 1
                and wrong_marker == 0
            ),
        )
        return result

    def case_evidence(
        self, candidates: Mapping[str, CandidateReport]
    ) -> Mapping[str, object]:
        """Handle the case evidence operation."""
        return {
            "source": "production-runtime-memory-selection",
            "fresh_catalog_for_each_case": True,
            "initial_catalog_size": 9,
            "maximum_retrieval_items": 5,
            "candidates": {
                name: {
                    "status": candidate.status,
                    "validation_passed": candidate.validation.passed,
                    "memory_recalled_items": numeric_metric(
                        candidate.metrics, "memory_recalled_items"
                    ),
                    "memory_recalled_ids": _memory_ids_json(candidate.metrics),
                    "memory_initial_count": numeric_metric(
                        candidate.metrics, "memory_initial_count"
                    ),
                    "memory_final_count": numeric_metric(
                        candidate.metrics, "memory_final_count"
                    ),
                    "memory_relevant_count": numeric_metric(
                        candidate.metrics, "memory_relevant_count"
                    ),
                    "memory_retrieved_count": numeric_metric(
                        candidate.metrics, "memory_retrieved_count"
                    ),
                    "memory_true_positive_count": numeric_metric(
                        candidate.metrics, "memory_true_positive_count"
                    ),
                    "memory_false_negative_count": numeric_metric(
                        candidate.metrics, "memory_false_negative_count"
                    ),
                    "memory_false_positive_count": numeric_metric(
                        candidate.metrics, "memory_false_positive_count"
                    ),
                    "memory_recall_rate": numeric_metric(
                        candidate.metrics, "memory_recall_rate"
                    ),
                    "memory_precision_rate": numeric_metric(
                        candidate.metrics, "memory_precision_rate"
                    ),
                    "memory_correct_marker_written": numeric_metric(
                        candidate.metrics, "memory_correct_marker_written"
                    ),
                    "memory_wrong_marker_written": numeric_metric(
                        candidate.metrics, "memory_wrong_marker_written"
                    ),
                    "memory_dream_triggered": numeric_metric(
                        candidate.metrics, "memory_dream_triggered"
                    ),
                    "memory_dream_count": numeric_metric(
                        candidate.metrics, "memory_dream_count"
                    ),
                    "durable_memory_section_tokens": numeric_metric(
                        candidate.metrics, "durable_memory_section_tokens"
                    ),
                    "all_memory_tokens": numeric_metric(
                        candidate.metrics, "all_memory_tokens"
                    ),
                    "memory_index_tokens": numeric_metric(
                        candidate.metrics, "memory_index_tokens"
                    ),
                    "recalled_memory_tokens": numeric_metric(
                        candidate.metrics, "recalled_memory_tokens"
                    ),
                    "optimized_memory_tokens": numeric_metric(
                        candidate.metrics, "optimized_memory_tokens"
                    ),
                    "memory_context_tokens": numeric_metric(
                        candidate.metrics, "memory_context_tokens"
                    ),
                    "memory_context_input_tokens": numeric_metric(
                        candidate.metrics, "memory_context_input_tokens"
                    ),
                    "memory_context_share": numeric_metric(
                        candidate.metrics, "memory_context_share"
                    ),
                    "memory_catalog_reduction": numeric_metric(
                        candidate.metrics, "memory_catalog_reduction"
                    ),
                }
                for name, candidate in candidates.items()
            },
        }

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score task execution while keeping retrieval failures measurable."""
        if "primary" not in candidates:
            return "invalid", "missing memory selector candidate"
        candidate = candidates["primary"]
        if candidate.status in {"infra_error", "invalid"}:
            return candidate.status, candidate.failure_reason or "unscoreable memory selector"
        failures: list[str] = []
        if candidate.status != "passed":
            failures.append(candidate.failure_reason or "task execution failed")
        if numeric_metric(candidate.metrics, "memory_correct_marker_written") != 1:
            failures.append("correct memory marker was not written")
        if numeric_metric(candidate.metrics, "memory_wrong_marker_written") != 0:
            failures.append("adversarial memory marker was written")
        return ("failed", "; ".join(failures)) if failures else ("passed", "")

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate selector quality and normal memory lifecycle outcomes."""
        eligible = capability_cases(cases)
        true_positive = _sum_case_metric(
            eligible, "primary_memory_true_positive_count"
        )
        false_negative = _sum_case_metric(
            eligible, "primary_memory_false_negative_count"
        )
        false_positive = _sum_case_metric(
            eligible, "primary_memory_false_positive_count"
        )
        retrieved = _sum_case_metric(eligible, "primary_memory_retrieved_count")
        all_memory_tokens = _sum_case_metric(eligible, "primary_all_memory_tokens")
        optimized_memory_tokens = _sum_case_metric(
            eligible, "primary_optimized_memory_tokens"
        )
        return RunMeasurement(
            primary={
                "memory_recall_rate": metric(
                    "memory_recall_rate",
                    _ratio_or_na(true_positive, true_positive + false_negative),
                ),
                "memory_precision_rate": metric(
                    "memory_precision_rate", _ratio_or_na(true_positive, retrieved)
                ),
                "memory_false_positive_count": metric(
                    "memory_false_positive_count", false_positive
                ),
                "memory_task_success_rate": metric(
                    "memory_task_success_rate",
                    _average_or_na(eligible, "primary_memory_success"),
                ),
                "memory_correct_marker_rate": metric(
                    "memory_correct_marker_rate",
                    _average_or_na(
                        eligible, "primary_memory_correct_marker_written"
                    ),
                ),
                "memory_context_reduction_rate": metric(
                    "memory_context_reduction_rate",
                    _ratio_or_na(
                        all_memory_tokens - optimized_memory_tokens,
                        all_memory_tokens,
                    ),
                ),
            },
            supporting={
                "memory_false_positive_rate": metric(
                    "memory_false_positive_rate",
                    _ratio_or_na(false_positive, retrieved),
                ),
                "memory_recall_exercise_rate": metric(
                    "memory_recall_exercise_rate",
                    _average_or_na(eligible, "primary_recall_exercised"),
                ),
                "memory_wrong_marker_rate": metric(
                    "memory_wrong_marker_rate",
                    _average_or_na(
                        eligible, "primary_memory_wrong_marker_written"
                    ),
                ),
                "average_memory_initial_count": metric(
                    "average_memory_initial_count",
                    _average_or_na(eligible, "primary_memory_initial_count"),
                ),
                "average_memory_final_count": metric(
                    "average_memory_final_count",
                    _average_or_na(eligible, "primary_memory_final_count"),
                ),
                "dream_trigger_rate": metric(
                    "dream_trigger_rate",
                    _average_or_na(eligible, "primary_memory_dream_triggered"),
                ),
                "average_memory_context_tokens": metric(
                    "average_memory_context_tokens",
                    _average_or_na(eligible, "primary_memory_context_tokens"),
                ),
                "average_memory_context_share": metric(
                    "average_memory_context_share",
                    _average_or_na(eligible, "primary_memory_context_share"),
                ),
                "primary_budget_exhaustion_rate": metric(
                    "primary_budget_exhaustion_rate",
                    _average_or_na(eligible, "primary_budget_exhausted"),
                ),
                "memory_runtime_task_success_rate": metric(
                    "memory_runtime_task_success_rate",
                    _average_or_na(eligible, "primary_candidate_passed"),
                ),
            },
            guardrails={**task_guardrail(cases), **process_guardrails(cases)},
        )


def _continuation_prompt() -> str:
    return (
        "Continue the benchmark coding task. Use durable memory only as supporting "
        "context, select the memories that apply to the active task, and inspect "
        "solution.py as needed. Implement the requested function now. Add the exact "
        "current-task marker from the relevant memory as a Python comment in "
        "solution.py, and do not apply markers from stale, conflicting, or unrelated "
        "memories."
    )


def _memory_ids(metrics: Mapping[str, Metric]) -> set[str]:
    item = metrics.get("memory_recalled_ids")
    if item is None or not isinstance(item.value, str):
        return set()
    try:
        value = json.loads(item.value)
    except (TypeError, ValueError):
        return set()
    if not isinstance(value, list):
        return set()
    return {name for name in value if isinstance(name, str) and name}


def _memory_ids_json(metrics: Mapping[str, Metric]) -> list[str]:
    return sorted(_memory_ids(metrics))


def _memory_lifecycle(path: Path) -> dict[str, object]:
    extract_count = 0
    dream_statuses: list[str] = []
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(record, dict) or record.get("event") != "memory.lifecycle":
                continue
            attributes = record.get("attributes")
            if not isinstance(attributes, dict):
                continue
            operation = attributes.get("operation")
            status = attributes.get("status")
            if operation == "extract":
                extract_count += 1
            elif operation == "dream" and isinstance(status, str):
                dream_statuses.append(status)
    return {
        "extract_count": extract_count,
        "dream_count": len(dream_statuses),
        "dream_statuses": dream_statuses,
    }


def _final_memory_count(workspace: Path) -> int:
    try:
        return len(MemoryStore(workspace / ".memory").scan())
    except (OSError, ValueError):
        return 0


def _sum_case_metric(cases: tuple[CaseReport, ...], name: str) -> float:
    return sum(numeric_metric(case.metrics, name) for case in cases)


def _ratio_or_na(numerator: float, denominator: float) -> float | str:
    return numerator / denominator if denominator else "N/A"


def _average_or_na(cases: tuple[CaseReport, ...], name: str) -> float | str:
    return average(cases, name) if cases else "N/A"


# Kept as a small compatibility shim for callers that used the old helper.
def _memory_marker(spec: CaseSpec) -> str:
    return memory_marker(spec)
