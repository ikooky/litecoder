"""Evaluation solution validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
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
        isolated_modes: frozenset[str] = frozenset({"multi-agent"}),
        max_infra_retries: int = 1,
    ) -> None:
        if max_infra_retries < 0:
            raise ValueError("max_infra_retries must be non-negative")
        self.isolated_modes = isolated_modes
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
        if spec.mode in self.isolated_modes:
            return _evaluate_isolated(spec.dataset, spec.task_id, solution)
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


def _evaluate_isolated(
    dataset: str,
    task_id: str,
    solution: str,
) -> EvalPlusCaseEvaluation:
    payload = {"dataset": dataset, "task_id": task_id, "solution": solution}
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[2])
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (package_root, existing_path) if value
    )
    timeout = _isolated_validation_timeout()
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-m", "litecoder.eval.validation_worker"],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
            cwd=tempfile.gettempdir(),
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise EvalPlusExecutionError(
            f"isolated EvalPlus validation timed out after {timeout:.1f}s"
        ) from error
    except OSError as error:
        raise EvalPlusExecutionError(
            f"could not start isolated EvalPlus validation: {error}"
        ) from error

    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    response: object = None
    if output_lines:
        try:
            response = json.loads(output_lines[-1])
        except json.JSONDecodeError:
            response = None
    if (
        completed.returncode != 0
        or not isinstance(response, dict)
        or response.get("ok") is not True
    ):
        detail = ""
        if isinstance(response, dict):
            detail = str(response.get("error") or response.get("traceback") or "")
        if not detail:
            detail = completed.stderr.strip() or completed.stdout.strip()
        if not detail:
            detail = f"worker exited with code {completed.returncode}"
        raise EvalPlusExecutionError(
            "isolated EvalPlus validation failed: " + detail[-4_000:]
        )
    raw_evaluation = response.get("evaluation")
    if not isinstance(raw_evaluation, dict):
        raise EvalPlusExecutionError(
            "isolated EvalPlus validation returned no evaluation payload"
        )
    try:
        response_task_id = raw_evaluation["task_id"]
        response_passed = raw_evaluation["passed"]
        failed_test_count = raw_evaluation.get("failed_test_count", 0)
        if not isinstance(response_task_id, str) or response_task_id != task_id:
            raise ValueError("isolated validation returned the wrong task")
        if not isinstance(response_passed, bool):
            raise ValueError("isolated validation returned an invalid pass flag")
        if isinstance(failed_test_count, bool):
            raise ValueError("isolated validation returned an invalid failure count")
        return EvalPlusCaseEvaluation(
            task_id=response_task_id,
            passed=response_passed,
            failure_reason=str(raw_evaluation.get("failure_reason", "")),
            base_status=str(raw_evaluation.get("base_status", "unknown")),
            plus_status=str(raw_evaluation.get("plus_status", "unknown")),
            failed_test_count=int(failed_test_count),
            first_failed_index=(
                None
                if raw_evaluation.get("first_failed_index") is None
                else int(raw_evaluation["first_failed_index"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvalPlusExecutionError(
            "isolated EvalPlus validation returned an invalid evaluation payload"
        ) from error


def _isolated_validation_timeout() -> float:
    try:
        value = float(os.getenv("LITECODER_EVALPLUS_VALIDATION_TIMEOUT", "180"))
    except ValueError:
        return 180.0
    return value if value > 0 else 180.0
