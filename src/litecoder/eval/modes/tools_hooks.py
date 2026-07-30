"""Tool and hook evaluation mode."""

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
    ModeMeasurement,
)
from litecoder.eval.modes.base import (
    EvalModePlugin,
    RunMeasurement,
    capability_cases,
    metric,
    numeric,
    numeric_metric,
    process_guardrails,
    ratio,
    task_guardrail,
    total,
)


class ToolsHooksMode(EvalModePlugin):
    """Component responsible for the tools hooks mode."""
    name = "tools-hooks"

    def candidates(self, spec: CaseSpec) -> tuple[ExecutionCandidate, ...]:
        """Return candidate implementations for evaluation."""
        return (
            ExecutionCandidate(
                "primary",
                spec.prompt(),
                setup_prompt=_hook_probe_prompt(),
            ),
        )

    async def measure(
        self, spec: CaseSpec, paths: CasePaths, execution: AgentExecution
    ) -> ModeMeasurement:
        """Measure the selected evaluation case."""
        del spec, execution
        values = _trace_metrics(paths.trace)
        return ModeMeasurement(
            metrics={name: metric(name, value) for name, value in values.items()},
            evidence={"source": "runtime-trace", **values},
        )

    def score(
        self, candidates: Mapping[str, CandidateReport]
    ) -> tuple[str, str]:
        """Score the evaluation candidate."""
        candidate = candidates.get("primary")
        if candidate is None:
            return "invalid", "missing tools/hooks candidate"
        if candidate.status in {"infra_error", "invalid"}:
            return candidate.status, candidate.failure_reason
        missing = [
            label
            for label, name in (
                ("executed", "executed_tool_calls"),
                ("denied", "denied_tool_calls"),
                ("duplicate", "duplicate_blocked_tool_calls"),
                ("invalid", "invalid_tool_calls"),
            )
            if numeric_metric(candidate.metrics, name) < 1
        ]
        if missing:
            return "invalid", (
                "tools/hooks probe did not exercise: " + ", ".join(missing)
            )
        incomplete = [
            label
            for label, numerator, denominator in (
                (
                    "terminal outcomes",
                    "terminal_outcomes_traced",
                    "terminal_tool_calls",
                ),
                (
                    "executed traces",
                    "executed_calls_fully_traced",
                    "executed_tool_calls",
                ),
                (
                    "executed hooks",
                    "executed_calls_with_hooks",
                    "executed_tool_calls",
                ),
                ("denied traces", "denied_calls_fully_traced", "denied_tool_calls"),
                (
                    "duplicate traces",
                    "duplicate_calls_fully_traced",
                    "duplicate_blocked_tool_calls",
                ),
                ("invalid traces", "invalid_calls_fully_traced", "invalid_tool_calls"),
            )
            if numeric_metric(candidate.metrics, numerator)
            != numeric_metric(candidate.metrics, denominator)
        ]
        if incomplete:
            return (
                "failed",
                "incomplete tools/hooks evidence: " + ", ".join(incomplete),
            )
        if candidate.status != "passed":
            return "failed", candidate.failure_reason or "tools/hooks candidate failed"
        return "passed", ""

    def aggregate(self, cases: tuple[CaseReport, ...]) -> RunMeasurement:
        """Aggregate case metrics into a summary."""
        eligible = capability_cases(cases)
        terminal_calls = total(eligible, "terminal_tool_calls")
        executed_calls = total(eligible, "executed_tool_calls")
        denied_calls = total(eligible, "denied_tool_calls")
        duplicate_calls = total(eligible, "duplicate_blocked_tool_calls")
        rejected_calls = total(eligible, "invalid_tool_calls")
        guardrails = {
            name: value
            for name, value in process_guardrails(cases).items()
            if name
            not in {
                "permission_clean_case_rate",
                "tool_execution_success_rate",
            }
        }
        return RunMeasurement(
            primary={
                "terminal_outcome_coverage": metric(
                    "terminal_outcome_coverage",
                    ratio(total(eligible, "terminal_outcomes_traced"), terminal_calls),
                ),
                "executed_trace_coverage": metric(
                    "executed_trace_coverage",
                    ratio(total(eligible, "executed_calls_fully_traced"), executed_calls),
                )
            },
            supporting={
                "executed_hook_coverage": metric(
                    "executed_hook_coverage",
                    ratio(total(eligible, "executed_calls_with_hooks"), executed_calls),
                ),
                "denied_trace_coverage": metric(
                    "denied_trace_coverage",
                    ratio(total(eligible, "denied_calls_fully_traced"), denied_calls),
                ),
                "duplicate_block_trace_coverage": metric(
                    "duplicate_block_trace_coverage",
                    ratio(
                        total(eligible, "duplicate_calls_fully_traced"), duplicate_calls
                    ),
                ),
                "invalid_call_trace_coverage": metric(
                    "invalid_call_trace_coverage",
                    ratio(total(eligible, "invalid_calls_fully_traced"), rejected_calls),
                ),
                "duplicate_call_block_count": metric(
                    "duplicate_call_block_count", duplicate_calls
                ),
                "permission_denied_call_count": metric(
                    "permission_denied_call_count", denied_calls
                ),
                "hook_scenario_exercise_rate": metric(
                    "hook_scenario_exercise_rate",
                    ratio(
                        sum(
                            numeric(case, name) >= 1
                            for case in eligible
                            for name in (
                                "executed_tool_calls",
                                "denied_tool_calls",
                                "duplicate_blocked_tool_calls",
                                "invalid_tool_calls",
                            )
                        ),
                        len(eligible) * 4,
                    ),
                ),
            },
            guardrails={**task_guardrail(cases), **guardrails},
        )


def _trace_metrics(path: Path) -> dict[str, int]:
    rows = _jsonl(path)
    calls: dict[str, set[str]] = {}
    hooks: dict[str, set[str]] = {}
    final_statuses: dict[str, str] = {}
    for row in rows:
        call_id = _call_id(row)
        if not call_id:
            continue
        event = row.get("event")
        if event == "tool.runtime":
            stage = row.get("stage")
            if isinstance(stage, str):
                calls.setdefault(call_id, set()).add(stage)
            if stage == "final" and isinstance(row.get("status"), str):
                final_statuses[call_id] = str(row["status"])
        elif event == "hook.dispatch.end":
            point = row.get("point")
            if isinstance(point, str):
                hooks.setdefault(call_id, set()).add(point)
    executed = {
        call_id
        for call_id, status in final_statuses.items()
        if status in {"success", "tool_error"}
    }
    denied = {call_id for call_id, status in final_statuses.items() if status == "denied"}
    duplicates = {
        call_id
        for call_id, status in final_statuses.items()
        if status == "duplicate_blocked"
    }
    invalid = {
        call_id
        for call_id, status in final_statuses.items()
        if status == "invalid_arguments"
    }

    def traced(call_id: str, required: set[str]) -> bool:
        return required <= calls.get(call_id, set())

    def hooked(call_id: str) -> bool:
        expected = (
            "PostToolUse"
            if final_statuses.get(call_id) == "success"
            else "ToolError"
        )
        return {"PreToolUse", expected} <= hooks.get(call_id, set())

    return {
        "total_tool_calls": len(calls),
        "terminal_tool_calls": len(calls),
        "terminal_outcomes_traced": len(final_statuses),
        "executed_tool_calls": len(executed),
        "executed_calls_fully_traced": sum(
            traced(call_id, {"registry", "pre", "permission", "execute", "final"})
            for call_id in executed
        ),
        "executed_calls_with_hooks": sum(hooked(call_id) for call_id in executed),
        "denied_tool_calls": len(denied),
        "denied_calls_fully_traced": sum(
            traced(call_id, {"registry", "pre", "permission", "final"})
            for call_id in denied
        ),
        "duplicate_blocked_tool_calls": len(duplicates),
        "duplicate_calls_fully_traced": sum(
            traced(call_id, {"registry", "pre", "duplicate", "final"})
            for call_id in duplicates
        ),
        "invalid_tool_calls": len(invalid),
        "invalid_calls_fully_traced": sum(
            traced(call_id, {"registry", "pre", "arguments", "final"})
            for call_id in invalid
        ),
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _call_id(row: dict[str, object]) -> str:
    value = row.get("tool_call_id")
    if isinstance(value, str):
        return value
    payload = row.get("payload")
    call = payload.get("call") if isinstance(payload, dict) else None
    value = call.get("id") if isinstance(call, dict) else None
    return value if isinstance(value, str) else ""


def _hook_probe_prompt() -> str:
    return (
        "This is a tools/hooks lifecycle probe before the coding turn. Do not edit "
        "solution.py yet. Perform each requested call exactly as written and do not "
        "retry a rejected call: (1) call read_file with path solution.py; (2) call "
        "read_file again with the identical path solution.py so duplicate protection "
        "is exercised; (3) call write_file with path forbidden-probe.txt and content "
        "probe, which is expected to be denied; (4) call read_file with an empty path "
        "string, which is expected to be rejected as invalid arguments. After all four "
        "calls, reply with a brief acknowledgement only."
    )
