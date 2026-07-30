"""Evaluation result reporting and serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping

from litecoder.eval.domain import Metric, RunReport


def render_json(report: RunReport) -> str:
    """Render the json."""
    return json.dumps(report.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_markdown(report: RunReport) -> str:
    """Render the markdown."""
    return render_markdown_payload(report.to_json())


def render_markdown_payload(payload: Mapping[str, object]) -> str:
    """Render the markdown payload."""
    datasets = payload.get("datasets")
    dataset_text = ", ".join(str(item) for item in datasets) if isinstance(datasets, list) else ""
    cases = payload.get("cases")
    case_items = cases if isinstance(cases, list) else []
    lines = [
        f"# {payload.get('mode', 'evaluation')}",
        "",
        f"Run: {payload.get('run_id', '')}",
        f"Status: {payload.get('status', '')}",
        f"Datasets: {dataset_text}",
        f"Cases: {len(case_items)}",
        "",
        "## Run Metadata",
        "",
        *_metadata_table(payload.get("metadata")),
        "",
        "## Primary Metrics",
        "",
        *_metric_table(payload.get("primary_metrics")),
        "",
        "## Supporting Metrics",
        "",
        *_metric_table(payload.get("supporting_metrics")),
        "",
        "## Quality Guardrails",
        "",
        *_metric_table(payload.get("guardrails")),
        "",
        "## Cases",
        "",
        "| Case | Task | Status | Stage | Failure class | Failure | Candidate outcomes | Workspace |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in case_items:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| {case_id} | {task_id} | {status} | {stage} | {failure_class} | {failure} | {candidates} | {workspace} |".format(
                case_id=item.get("case_id", ""),
                task_id=item.get("task_id", ""),
                status=item.get("status", ""),
                stage=item.get("stage", ""),
                failure_class=_failure_class(item.get("failure_reason", "")),
                failure=_table_text(item.get("failure_reason", "")),
                candidates=_candidate_outcomes(item.get("candidates")),
                workspace=item.get("workspace", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _metric_table(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["None."]
    lines = ["| Metric | Value | Unit |", "|---|---:|---|"]
    for name in sorted(value):
        item = value[name]
        if isinstance(item, Metric):
            metric_value = item.value
            unit = item.unit
        elif isinstance(item, Mapping):
            metric_value = item.get("value", "")
            unit = item.get("unit", "")
        else:
            metric_value = item
            unit = ""
        lines.append(f"| {name} | {metric_value} | {unit} |")
    return lines


def _metadata_table(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["None."]
    lines = ["| Field | Value |", "|---|---|"]
    for name in sorted(value):
        item = value[name]
        rendered = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(
            item, (dict, list)
        ) else str(item)
        lines.append(f"| {name} | {_table_text(rendered)} |")
    return lines


def _table_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _failure_class(value: object) -> str:
    text = str(value).casefold()
    if not text:
        return ""
    if "budget exhausted" in text:
        return "budget_exhausted"
    if "closed_loop" in text or "lifecycle validation" in text:
        return "workflow_not_closed"
    if ".validation" in text or "evalplus failed" in text:
        return "validation_failed"
    if "infra" in text or "runtime status error" in text:
        return "infrastructure"
    return "other"


def _candidate_outcomes(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    outcomes: list[str] = []
    for name in sorted(value):
        candidate = value[name]
        if not isinstance(candidate, Mapping):
            continue
        execution = candidate.get("execution")
        runtime_status = (
            execution.get("status", "") if isinstance(execution, Mapping) else ""
        )
        validation = candidate.get("validation")
        validation_text = ""
        if isinstance(validation, Mapping) and "passed" in validation:
            validation_text = "pass" if validation.get("passed") else "fail"
        details = [str(candidate.get("status", ""))]
        if runtime_status:
            details.append(f"runtime={runtime_status}")
        if validation_text:
            details.append(f"validation={validation_text}")
        reason = candidate.get("failure_reason", "")
        if reason:
            details.append(_table_text(reason))
        outcomes.append(f"{name}: {', '.join(details)}")
    return _table_text("; ".join(outcomes))
