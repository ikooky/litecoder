"""Evaluation domain models and case definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Mapping


DatasetName = Literal["humaneval", "mbpp"]
DatasetSelection = DatasetName | tuple[DatasetName, ...]
CaseStatus = Literal["passed", "failed", "infra_error", "invalid"]
RunStatus = Literal["completed", "completed_with_infra_errors", "interrupted"]
MetricScalar = float | int | str
CandidateTopology = Literal["default", "subagent", "team"]
FeatureToggle = Literal["default", "enabled", "disabled"]
CASE_STATUSES = {"passed", "failed", "infra_error", "invalid"}
RUN_STATUSES = {"completed", "completed_with_infra_errors", "interrupted"}


class EvalMode(StrEnum):
    """Enumeration of the eval mode values."""
    AGENT_BENCHMARK = "agent-benchmark"
    CONTEXT_MANAGER = "context-manager"
    TOOLS_HOOKS = "tools-hooks"
    MEMORY = "memory"
    TASK_STATE = "task-state"
    MULTI_AGENT = "multi-agent"


class CaseStage(StrEnum):
    """Enumeration of the case stage values."""
    PREPARED = "prepared"
    EXECUTED = "executed"
    CAPTURED = "captured"
    VALIDATED = "validated"
    MEASURED = "measured"
    SCORED = "scored"


VALID_DATASETS: tuple[str, ...] = ("humaneval", "mbpp")


def validate_mode(value: str) -> str:
    """Validate the mode."""
    try:
        return EvalMode(value).value
    except ValueError as error:
        raise ValueError(f"Unknown evaluation mode: {value}") from error


def validate_dataset(value: str) -> DatasetName:
    """Validate the dataset."""
    if value not in VALID_DATASETS:
        raise ValueError(f"Unsupported EvalPlus dataset: {value}")
    return value  # type: ignore[return-value]


def validate_datasets(value: object) -> DatasetSelection:
    """Validate the datasets."""
    if isinstance(value, str):
        return validate_dataset(value)
    if not isinstance(value, tuple) or not value:
        raise ValueError("datasets must contain at least one EvalPlus dataset")
    datasets = tuple(validate_dataset(item) for item in value)
    if len(set(datasets)) != len(datasets):
        raise ValueError("datasets must not contain duplicates")
    return datasets


def selected_datasets(value: DatasetSelection) -> tuple[DatasetName, ...]:
    """Handle the selected datasets operation."""
    return (value,) if isinstance(value, str) else value


@dataclass(frozen=True, slots=True)
class Metric:
    """Data model representing the metric."""
    name: str
    value: MetricScalar
    unit: str = ""

    def __post_init__(self) -> None:
        _non_empty(self.name, "metric name")
        if isinstance(self.value, bool) or not isinstance(
            self.value, (float, int, str)
        ):
            raise ValueError("metric value must be a number or string")

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {"value": self.value, "unit": self.unit}

    @classmethod
    def from_json(cls, name: str, value: object) -> "Metric":
        """Construct a value from json data."""
        data = _mapping(value, f"metric {name}")
        scalar = data.get("value")
        if isinstance(scalar, bool) or not isinstance(scalar, (float, int, str)):
            raise ValueError(f"metric {name} value is invalid")
        unit = data.get("unit", "")
        if not isinstance(unit, str):
            raise ValueError(f"metric {name} unit is invalid")
        return cls(name, scalar, unit)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Data model representing the execution policy."""
    allowed_tools: frozenset[str]
    max_rounds: int | None = 24
    max_tokens: int | None = 100_000

    def __post_init__(self) -> None:
        if not self.allowed_tools:
            raise ValueError("execution policy must allow at least one tool")
        if self.max_rounds is not None and self.max_rounds <= 0:
            raise ValueError("execution policy budgets must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("execution policy budgets must be positive")

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "allowed_tools": sorted(self.allowed_tools),
            "max_rounds": self.max_rounds,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True, slots=True)
class ExecutionCandidate:
    """Data model representing the execution candidate."""
    name: str
    prompt: str
    topology: CandidateTopology = "default"
    setup_prompt: str = ""
    restart_after_setup: bool = False
    context_compaction: FeatureToggle = "default"
    context_budget_tokens: int | None = None
    memory_recall: FeatureToggle = "default"
    task_recovery: bool = False

    def __post_init__(self) -> None:
        _non_empty(self.name, "candidate name")
        _non_empty(self.prompt, "candidate prompt")
        if self.topology not in {"default", "subagent", "team"}:
            raise ValueError(f"unsupported candidate topology: {self.topology}")
        if not isinstance(self.setup_prompt, str):
            raise ValueError("candidate setup prompt must be text")
        if self.setup_prompt and not self.setup_prompt.strip():
            raise ValueError("candidate setup prompt must not be blank")
        for name in ("restart_after_setup", "task_recovery"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"candidate {name} must be a bool")
        for name in ("context_compaction", "memory_recall"):
            if getattr(self, name) not in {"default", "enabled", "disabled"}:
                raise ValueError(f"unsupported candidate {name}: {getattr(self, name)}")
        if self.context_budget_tokens is not None and (
            isinstance(self.context_budget_tokens, bool)
            or not isinstance(self.context_budget_tokens, int)
            or self.context_budget_tokens <= 0
        ):
            raise ValueError("candidate context budget must be a positive integer")
        if self.context_compaction != "enabled" and self.context_budget_tokens is not None:
            raise ValueError(
                "candidate context budget requires enabled context compaction"
            )
        if self.restart_after_setup and not self.setup_prompt:
            raise ValueError("candidate restart_after_setup requires a setup prompt")
        if self.task_recovery and self.setup_prompt:
            raise ValueError("task recovery setup is controlled by the evaluation harness")

    def artifact_prompt(self) -> str:
        """Handle the artifact prompt operation."""
        sections: list[str] = []
        if self.setup_prompt:
            sections.extend(("## Setup turn", "", self.setup_prompt, ""))
        if self.task_recovery:
            sections.extend(
                (
                    "## Harness interruption",
                    "",
                    "Checkpoint: after-task-claim-before-agent-turn",
                    "",
                )
            )
        sections.extend(("## Execution turn", "", self.prompt))
        return "\n".join(sections)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Data model representing the case spec."""
    case_id: str
    task_id: str
    dataset: DatasetName
    entry_point: str
    starter_code: str
    mode: str

    def __post_init__(self) -> None:
        for name in ("case_id", "task_id", "entry_point", "starter_code"):
            _non_empty(getattr(self, name), name)
        object.__setattr__(self, "dataset", validate_dataset(self.dataset))
        object.__setattr__(self, "mode", validate_mode(self.mode))

    def prompt(self) -> str:
        """Handle the prompt operation."""
        return (
            "Implement the EvalPlus function in solution.py. Only modify "
            "solution.py. Do not create or edit tests, inspect harness artifacts, "
            "probe the Python environment, or use shell commands except to run an "
            "existing pytest or unittest suite. If a command is denied, do not try "
            "alternative shell commands. Preserve the requested entry point.\n\n"
            f"Task id: {self.task_id}\n"
            f"Entry point: {self.entry_point}\n\n"
            "Starter code:\n"
            f"{self.starter_code}"
        )

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "case_id": self.case_id,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "entry_point": self.entry_point,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class CasePaths:
    """Data model representing the case paths."""
    root: Path
    prompt: Path
    starter: Path
    solution: Path
    diff: Path
    trace: Path
    events: Path
    local_tests: Path
    validation_result: Path
    validation_output: Path
    mode_evidence: Path
    manifest: Path

    def artifacts(self) -> dict[str, Path]:
        """Handle the artifacts operation."""
        return {
            "prompt": self.prompt,
            "starter": self.starter,
            "solution": self.solution,
            "diff": self.diff,
            "trace": self.trace,
            "events": self.events,
            "local_tests": self.local_tests,
            "validation_result": self.validation_result,
            "validation_output": self.validation_output,
            "mode_evidence": self.mode_evidence,
            "manifest": self.manifest,
        }


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    """Data model representing the execution failure."""
    stage: str
    kind: str
    message: str
    error_type: str = ""
    details: Mapping[str, MetricScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("stage", "kind", "message"):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.error_type, str):
            raise ValueError("error_type must be text")
        normalized: dict[str, MetricScalar] = {}
        for name, value in self.details.items():
            if isinstance(value, bool) or not isinstance(value, (float, int, str)):
                raise ValueError("failure details must contain scalar values")
            normalized[str(name)] = value
        object.__setattr__(self, "details", normalized)

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "stage": self.stage,
            "kind": self.kind,
            "message": self.message,
            "error_type": self.error_type,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class AgentExecution:
    """Data model representing the agent execution."""
    status: str
    reason: str
    solution: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    failure: ExecutionFailure | None = None

    def __post_init__(self) -> None:
        _non_empty(self.status, "execution status")
        if self.input_tokens < 0 or self.output_tokens < 0 or self.elapsed_seconds < 0:
            raise ValueError("execution usage must be non-negative")
        object.__setattr__(self, "metrics", _metric_mapping(self.metrics))
        if self.failure is not None and not isinstance(self.failure, ExecutionFailure):
            raise ValueError("failure must be ExecutionFailure or None")

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "status": self.status,
            "reason": self.reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": _metrics_to_json(self.metrics),
            "failure": self.failure.to_json() if self.failure else None,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Data model representing the validation result."""
    passed: bool
    base_status: str
    plus_status: str
    failed_test_count: int
    first_failed_index: int | None
    elapsed_seconds: float
    reason: str = ""

    def __post_init__(self) -> None:
        if self.failed_test_count < 0 or self.elapsed_seconds < 0:
            raise ValueError("validation values must be non-negative")
        if self.first_failed_index is not None and self.first_failed_index < 0:
            raise ValueError("first_failed_index must be non-negative")

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "passed": self.passed,
            "base_status": self.base_status,
            "plus_status": self.plus_status,
            "failed_test_count": self.failed_test_count,
            "first_failed_index": self.first_failed_index,
            "elapsed_seconds": self.elapsed_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ModeMeasurement:
    """Data model representing the mode measurement."""
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _metric_mapping(self.metrics))
        object.__setattr__(self, "evidence", dict(self.evidence))


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Data model representing the candidate report."""
    name: str
    status: CaseStatus
    stage: CaseStage
    paths: CasePaths
    execution: AgentExecution
    validation: ValidationResult
    metrics: Mapping[str, Metric]
    failure_reason: str = ""

    def __post_init__(self) -> None:
        _non_empty(self.name, "candidate name")
        if self.status not in CASE_STATUSES:
            raise ValueError(f"unsupported candidate status: {self.status}")
        object.__setattr__(self, "metrics", _metric_mapping(self.metrics))

    def to_json(self, output_dir: Path) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "name": self.name,
            "status": self.status,
            "stage": self.stage.value,
            "workspace": _relative(self.paths.root, output_dir),
            "artifacts": {
                name: _relative(path, output_dir)
                for name, path in self.paths.artifacts().items()
            },
            "execution": self.execution.to_json(),
            "validation": self.validation.to_json(),
            "metrics": _metrics_to_json(self.metrics),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class CaseReport:
    """Data model representing the case report."""
    spec: CaseSpec
    status: CaseStatus
    stage: CaseStage
    paths: CasePaths
    execution: AgentExecution
    validation: ValidationResult | None
    metrics: Mapping[str, Metric]
    failure_reason: str = ""
    candidates: Mapping[str, CandidateReport] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in CASE_STATUSES:
            raise ValueError(f"unsupported case status: {self.status}")
        object.__setattr__(self, "metrics", _metric_mapping(self.metrics))
        normalized: dict[str, CandidateReport] = {}
        for name, candidate in self.candidates.items():
            if not isinstance(candidate, CandidateReport):
                raise ValueError("candidates must contain CandidateReport values")
            normalized[str(name)] = candidate
        object.__setattr__(self, "candidates", normalized)

    def to_json(self, output_dir: Path) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            **self.spec.to_json(),
            "status": self.status,
            "stage": self.stage.value,
            "workspace": _relative(self.paths.root, output_dir),
            "artifacts": {
                name: _relative(path, output_dir)
                for name, path in self.paths.artifacts().items()
            },
            "execution": self.execution.to_json(),
            "validation": self.validation.to_json() if self.validation else None,
            "metrics": _metrics_to_json(self.metrics),
            "failure_reason": self.failure_reason,
            "candidates": {
                name: candidate.to_json(output_dir)
                for name, candidate in sorted(self.candidates.items())
            },
        }


@dataclass(frozen=True, slots=True)
class RunReport:
    """Data model representing the run report."""
    run_id: str
    mode: str
    datasets: tuple[DatasetName, ...]
    output_dir: Path
    cases: tuple[CaseReport, ...]
    status: RunStatus = "completed"
    primary_metrics: Mapping[str, Metric] = field(default_factory=dict)
    supporting_metrics: Mapping[str, Metric] = field(default_factory=dict)
    guardrails: Mapping[str, Metric] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.run_id, "run_id")
        object.__setattr__(self, "mode", validate_mode(self.mode))
        object.__setattr__(self, "datasets", tuple(validate_dataset(x) for x in self.datasets))
        if not self.datasets:
            raise ValueError("run must include datasets")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unsupported run status: {self.status}")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in ("primary_metrics", "supporting_metrics", "guardrails"):
            object.__setattr__(self, name, _metric_mapping(getattr(self, name)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "datasets": list(self.datasets),
            "status": self.status,
            "output_dir": str(self.output_dir),
            "primary_metrics": _metrics_to_json(self.primary_metrics),
            "supporting_metrics": _metrics_to_json(self.supporting_metrics),
            "guardrails": _metrics_to_json(self.guardrails),
            "metadata": dict(self.metadata),
            "cases": [case.to_json(self.output_dir) for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Data model representing the run spec."""
    run_id: str
    mode: str
    datasets: DatasetSelection
    output_dir: Path

    def __post_init__(self) -> None:
        _non_empty(self.run_id, "run_id")
        object.__setattr__(self, "mode", validate_mode(self.mode))
        object.__setattr__(self, "datasets", validate_datasets(self.datasets))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def selected_datasets(self) -> tuple[DatasetName, ...]:
        """Handle the selected datasets operation."""
        return selected_datasets(self.datasets)


def _non_empty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _metric_mapping(value: Mapping[str, Metric]) -> dict[str, Metric]:
    result: dict[str, Metric] = {}
    for name, metric in value.items():
        if not isinstance(metric, Metric):
            raise ValueError("metrics must contain Metric values")
        result[str(name)] = metric
    return result


def _metrics_to_json(metrics: Mapping[str, Metric]) -> dict[str, object]:
    return {name: metric.to_json() for name, metric in sorted(metrics.items())}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
