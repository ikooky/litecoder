"""Evaluation scheduling and result orchestration."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

from litecoder.eval.artifacts import (
    capture_execution,
    prepare_candidate,
    prepare_case,
    write_mode_evidence,
    write_validation,
)
from litecoder.eval.domain import (
    AgentExecution,
    CandidateReport,
    CasePaths,
    CaseReport,
    CaseSpec,
    CaseStage,
    ExecutionCandidate,
    ExecutionFailure,
    Metric,
    ModeMeasurement,
    RunReport,
    RunSpec,
    RunStatus,
    ValidationResult,
)
from litecoder.eval.evalplus import EvalPlusTask
from litecoder.eval.execution import CaseExecutor, ExecutedCase
from litecoder.eval.modes import mode_plugin
from litecoder.eval.validation import SolutionValidator


class EvalProgress(Protocol):
    """Protocol describing the eval progress behavior."""
    def case_started(
        self, *, index: int, total: int, task_id: str, workspace: object
    ) -> object: ...

    def case_finished(
        self,
        *,
        index: int,
        total: int,
        task_id: str,
        status: str,
        failure_reason: str,
    ) -> object: ...


class EvalOrchestrator:
    """Component responsible for the eval orchestrator."""
    def __init__(
        self,
        executor: CaseExecutor,
        validator: SolutionValidator,
        *,
        progress: EvalProgress | None = None,
    ) -> None:
        self.executor = executor
        self.validator = validator
        self.progress = progress

    async def run(
        self,
        run_spec: RunSpec,
        tasks: tuple[EvalPlusTask, ...],
    ) -> RunReport:
        """Run the requested operation."""
        started_at = datetime.now(UTC)
        if not tasks:
            raise ValueError("tasks must not be empty")
        configured = set(run_spec.selected_datasets)
        if any(task.dataset not in configured for task in tasks):
            raise ValueError("tasks contain a dataset outside the run specification")
        run_spec.output_dir.mkdir(parents=True, exist_ok=True)
        plugin = mode_plugin(run_spec.mode)
        cases: list[CaseReport] = []
        dataset_numbers = {dataset: 0 for dataset in run_spec.selected_datasets}
        run_status: RunStatus = "completed"
        for index, task in enumerate(tasks, start=1):
            dataset_numbers[task.dataset] += 1
            case_id = (
                f"{task.dataset}-{dataset_numbers[task.dataset]:04d}"
                if len(run_spec.selected_datasets) > 1
                else f"case-{index:04d}"
            )
            spec = CaseSpec(
                case_id,
                task.task_id,
                task.dataset,
                task.entry_point,
                task.prompt,
                run_spec.mode,
            )
            paths = prepare_case(run_spec.output_dir, spec)
            _notify(
                getattr(self.progress, "case_started", None),
                index=index,
                total=len(tasks),
                task_id=task.task_id,
                workspace=paths.root,
            )
            try:
                case = await self._run_case(run_spec, spec, paths, plugin)
            except KeyboardInterrupt:
                raise
            except asyncio.CancelledError:
                run_status = "interrupted"
                break
            cases.append(case)
            _notify(
                getattr(self.progress, "case_finished", None),
                index=index,
                total=len(tasks),
                task_id=task.task_id,
                status=case.status,
                failure_reason=case.failure_reason,
            )
        case_tuple = tuple(cases)
        if run_status == "completed" and any(
            case.status == "infra_error" for case in case_tuple
        ):
            run_status = "completed_with_infra_errors"
        aggregate = plugin.aggregate(case_tuple)
        providers = _runtime_metric_values(case_tuple, "runtime_provider")
        models = _runtime_metric_values(case_tuple, "runtime_model")
        finished_at = datetime.now(UTC)
        return RunReport(
            run_spec.run_id,
            run_spec.mode,
            run_spec.selected_datasets,
            run_spec.output_dir,
            case_tuple,
            status=run_status,
            primary_metrics=aggregate.primary,
            supporting_metrics=aggregate.supporting,
            guardrails=aggregate.guardrails,
            metadata={
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": (finished_at - started_at).total_seconds(),
                "execution_policy": plugin.policy(spec).to_json(),
                "providers": providers,
                "models": models,
                "case_status_counts": {
                    status: sum(case.status == status for case in case_tuple)
                    for status in (
                        "passed",
                        "failed",
                        "infra_error",
                        "invalid",
                    )
                },
            },
        )

    async def _run_case(
        self,
        run_spec: RunSpec,
        spec: CaseSpec,
        paths: CasePaths,
        plugin: object,
    ) -> CaseReport:
        candidates = tuple(plugin.candidates(spec))
        _validate_candidate_plan(candidates)
        reports: dict[str, CandidateReport] = {}
        measurement_evidence: dict[str, object] = {}
        for candidate in candidates:
            candidate_paths = (
                paths
                if len(candidates) == 1 and candidate.name == "primary"
                else prepare_candidate(paths, spec, candidate)
            )
            report, evidence = await self._run_candidate_once(
                run_spec, spec, candidate_paths, plugin, candidate
            )
            measurement_evidence[candidate.name] = evidence
            reports[candidate.name] = report

        metrics = dict(plugin.combine_metrics(reports))
        root_evidence = dict(plugin.case_evidence(reports))
        root_evidence["measurements"] = measurement_evidence
        write_mode_evidence(paths, root_evidence)
        status, failure_reason = plugin.score(reports)
        summary_name = plugin.summary_candidate(reports)
        summary = reports[summary_name]
        if summary.paths != paths:
            _copy_summary_artifacts(paths, summary.paths)
        case = CaseReport(
            spec,
            status,
            CaseStage.SCORED,
            paths,
            summary.execution,
            summary.validation,
            metrics,
            failure_reason,
            reports,
        )
        paths.manifest.write_text(
            json.dumps(
                case.to_json(run_spec.output_dir),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return case

    async def _run_candidate_once(
        self,
        run_spec: RunSpec,
        spec: CaseSpec,
        candidate_paths: CasePaths,
        plugin: object,
        candidate: ExecutionCandidate,
    ) -> tuple[CandidateReport, dict[str, object]]:
        executed = await _execute_candidate(
            self.executor,
            spec,
            candidate_paths,
            plugin.policy(spec),
            candidate,
        )
        process_values = capture_execution(candidate_paths, spec, executed.events)
        execution_metrics = dict(executed.execution.metrics)
        execution_metrics.update(
            {name: Metric(name, value) for name, value in process_values.items()}
        )
        execution_metrics["runtime_status"] = Metric(
            "runtime_status", executed.execution.status
        )
        execution = replace(
            executed.execution,
            metrics=execution_metrics,
            failure=_candidate_failure(candidate.name, executed.execution.failure),
        )
        measurement = await _measure(plugin, spec, candidate_paths, execution)
        measurement_error = "measurement_error" in measurement.evidence
        metrics = {**execution.metrics, **measurement.metrics}
        metrics["mode_evidence_ready"] = Metric(
            "mode_evidence_ready",
            int(bool(measurement.evidence) and not measurement_error),
        )
        write_mode_evidence(candidate_paths, dict(measurement.evidence))
        validation, validation_output, validation_error = _validate(
            self.validator, spec, execution.solution
        )
        write_validation(candidate_paths, validation, validation_output)
        metrics["validation_evidence_ready"] = Metric(
            "validation_evidence_ready",
            int(
                candidate_paths.validation_result.stat().st_size > 0
                and candidate_paths.validation_output.stat().st_size > 0
            ),
        )
        status, failure_reason = _score_candidate(
            candidate.name,
            execution,
            validation,
            validation_error,
            measurement_error,
        )
        report = CandidateReport(
            candidate.name,
            status,
            CaseStage.SCORED,
            candidate_paths,
            execution,
            validation,
            metrics,
            failure_reason,
        )
        candidate_paths.manifest.write_text(
            json.dumps(
                report.to_json(run_spec.output_dir),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return report, dict(measurement.evidence)


async def _execute_candidate(
    executor: CaseExecutor,
    spec: CaseSpec,
    paths: CasePaths,
    policy: object,
    candidate: ExecutionCandidate,
) -> ExecutedCase:
    try:
        return await executor.execute(spec, paths, policy, candidate)
    except Exception as error:
        reason = _exception_text(error)
        execution = AgentExecution(
            "error",
            reason,
            paths.solution.read_text(encoding="utf-8"),
            0,
            0,
            0.0,
            {"budget_exhausted": Metric("budget_exhausted", 0)},
            ExecutionFailure(
                (
                    "orchestrator"
                    if candidate.name == "primary"
                    else f"{candidate.name}.orchestrator"
                ),
                "executor_exception",
                reason,
                error_type=type(error).__name__,
            ),
        )
        return ExecutedCase(execution, ())


async def _measure(plugin, spec, paths, execution) -> ModeMeasurement:
    try:
        return await plugin.measure(spec, paths, execution)
    except Exception as error:
        return ModeMeasurement(
            evidence={
                "measurement_error": _exception_text(error),
                "measurement_error_type": type(error).__name__,
            }
        )


def _validate(
    validator: SolutionValidator,
    spec: CaseSpec,
    solution: str,
) -> tuple[ValidationResult, str, bool]:
    started = time.perf_counter()
    try:
        capture = validator.validate(spec, solution)
        return capture.result, capture.output, False
    except Exception as error:
        elapsed = time.perf_counter() - started
        result = ValidationResult(
            False,
            "error",
            "error",
            0,
            None,
            elapsed,
            _exception_text(error),
        )
        return result, f"validation_error: {_exception_text(error)}\n", True


def _score_candidate(
    name: str,
    execution: AgentExecution,
    validation: ValidationResult,
    validation_error: bool,
    measurement_error: bool,
) -> tuple[str, str]:
    if validation_error:
        return "infra_error", _failure_reason(
            execution, f"{name}.validation: {validation.reason}"
        )
    if _infrastructure_failure(execution.failure):
        return "infra_error", _failure_reason(
            execution, f"{name}.execution: runtime status {execution.status}"
        )
    if measurement_error:
        return "invalid", _failure_reason(
            execution, f"{name}.measurement: measurement failed"
        )
    reasons: list[str] = []
    if execution.status != "completed":
        reasons.append(
            f"{name}.execution: runtime status {execution.status}"
        )
    if not validation.passed:
        reasons.append(
            f"{name}.validation: {validation.reason or 'validation failed'}"
        )
    if not reasons:
        return "passed", ""
    return "failed", _failure_reason(
        execution, "; ".join(reasons)
    )


def _infrastructure_failure(failure: ExecutionFailure | None) -> bool:
    if failure is None:
        return False
    stage = failure.stage.rsplit(".", 1)[-1]
    return stage in {"provider", "orchestrator", "runtime"} or failure.kind in {
        "provider_error",
        "executor_exception",
    }


def _candidate_failure(
    candidate_name: str, failure: ExecutionFailure | None
) -> ExecutionFailure | None:
    if failure is None or candidate_name == "primary":
        return failure
    if failure.stage.startswith(f"{candidate_name}."):
        return failure
    return replace(failure, stage=f"{candidate_name}.{failure.stage}")


def _failure_reason(execution: AgentExecution, stage_reason: str) -> str:
    parts: list[str] = []
    failure = execution.failure
    if failure is not None:
        parts.append(
            f"execution ({failure.stage}/{failure.kind}): {failure.message}"
        )
    if stage_reason:
        parts.append(stage_reason)
    return "; ".join(parts)


def _validate_candidate_plan(candidates: tuple[ExecutionCandidate, ...]) -> None:
    if not candidates:
        raise ValueError("evaluation mode must provide at least one candidate")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("evaluation candidate names must be unique")


def _copy_summary_artifacts(target: CasePaths, source: CasePaths) -> None:
    for name in (
        "solution",
        "diff",
        "trace",
        "events",
        "local_tests",
        "validation_result",
        "validation_output",
    ):
        source_path = getattr(source, name)
        target_path = getattr(target, name)
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)


def _runtime_metric_values(
    cases: tuple[CaseReport, ...], name: str
) -> list[str]:
    values: set[str] = set()
    for case in cases:
        for candidate in case.candidates.values():
            metric = candidate.metrics.get(name)
            if metric is not None and isinstance(metric.value, str) and metric.value:
                values.add(metric.value)
    return sorted(values)


def _exception_text(error: Exception) -> str:
    message = str(error).strip() or "operation failed"
    return f"{type(error).__name__}: {message}"


def _notify(callback: object, **kwargs: object) -> None:
    if not callable(callback):
        return
    try:
        callback(**kwargs)
    except Exception:
        return
