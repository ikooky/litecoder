"""Evaluation solution validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from litecoder.eval.domain import CaseSpec, ValidationResult
from litecoder.eval.evalplus import (
    EvalPlusCaseEvaluation,
    EvalPlusExecutionError,
    evaluate_solution,
)


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

    def __init__(
        self,
        *,
        max_infra_retries: int = 1,
    ) -> None:
        if max_infra_retries < 0:
            raise ValueError("max_infra_retries must be non-negative")
        self.max_infra_retries = max_infra_retries

    def validate(self, spec: CaseSpec, solution: str) -> ValidationCapture:
        """Validate the supplied value."""
        started = time.perf_counter()
        evaluation: EvalPlusCaseEvaluation | None = None
        attempts = 0
        last_error: EvalPlusExecutionError | None = None
        for attempts in range(1, self.max_infra_retries + 2):
            try:
                evaluation = self._evaluate(spec, solution)
                break
            except EvalPlusExecutionError as error:
                last_error = error
                if attempts > self.max_infra_retries:
                    raise EvalPlusExecutionError(
                        "EvalPlus validation failed after "
                        f"{attempts} attempt(s): {error}"
                    ) from error
        if evaluation is None:
            raise EvalPlusExecutionError(
                "EvalPlus validation did not produce a result"
            ) from last_error
        elapsed = time.perf_counter() - started
        return _capture(evaluation, elapsed, attempts=attempts)

    def _evaluate(self, spec: CaseSpec, solution: str) -> EvalPlusCaseEvaluation:
        return evaluate_solution(spec.dataset, spec.task_id, solution)


def _capture(
    evaluation: EvalPlusCaseEvaluation,
    elapsed: float,
    *,
    attempts: int = 1,
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
        f"attempts: {attempts}",
    ]
    if evaluation.failure_reason:
        lines.append(f"reason: {evaluation.failure_reason}")
    return ValidationCapture(result, "\n".join(lines) + "\n")
