"""Evaluation solution validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from litecoder.eval.domain import CaseSpec, ValidationResult
from litecoder.eval.evalplus import EvalPlusCaseEvaluation, evaluate_solution


@dataclass(frozen=True, slots=True)
class ValidationCapture:
    """Data model representing the validation capture."""
    result: ValidationResult
    output: str


class SolutionValidator(Protocol):
    """Protocol describing the solution validator behavior."""
    def validate(self, spec: CaseSpec, solution: str) -> ValidationCapture: ...


class EvalPlusValidator:
    """Component responsible for the eval plus validator."""
    def validate(self, spec: CaseSpec, solution: str) -> ValidationCapture:
        """Validate the supplied value."""
        started = time.perf_counter()
        evaluation = evaluate_solution(spec.dataset, spec.task_id, solution)
        elapsed = time.perf_counter() - started
        return _capture(evaluation, elapsed)


def _capture(
    evaluation: EvalPlusCaseEvaluation,
    elapsed: float,
) -> ValidationCapture:
    result = ValidationResult(
        evaluation.passed,
        evaluation.base_status,
        evaluation.plus_status,
        evaluation.failed_test_count,
        evaluation.first_failed_index,
        elapsed,
        evaluation.failure_reason,
    )
    lines = [
        f"task_id: {evaluation.task_id}",
        f"passed: {evaluation.passed}",
        f"base_status: {evaluation.base_status}",
        f"plus_status: {evaluation.plus_status}",
        f"failed_test_count: {evaluation.failed_test_count}",
        f"first_failed_index: {evaluation.first_failed_index}",
        f"elapsed_seconds: {elapsed}",
    ]
    if evaluation.failure_reason:
        lines.append(f"reason: {evaluation.failure_reason}")
    return ValidationCapture(result, "\n".join(lines) + "\n")
