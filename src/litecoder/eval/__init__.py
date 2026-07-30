"""Public interfaces for the eval package."""

from litecoder.eval.domain import (
    CandidateReport,
    CaseReport,
    CaseSpec,
    EvalMode,
    ExecutionCandidate,
    ExecutionFailure,
    Metric,
    RunReport,
    RunSpec,
    validate_dataset,
    validate_mode,
)

__all__ = [
    "CandidateReport",
    "CaseReport",
    "CaseSpec",
    "EvalMode",
    "ExecutionCandidate",
    "ExecutionFailure",
    "Metric",
    "RunReport",
    "RunSpec",
    "validate_dataset",
    "validate_mode",
]
