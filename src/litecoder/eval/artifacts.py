"""Evaluation artifact collection and storage."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Iterable

from litecoder.eval.domain import (
    CasePaths,
    CaseSpec,
    ExecutionCandidate,
    ValidationResult,
)
from litecoder.ui.events import RuntimeUIEvent, UIEventType


def prepare_case(output_dir: Path, spec: CaseSpec) -> CasePaths:
    """Prepare the case."""
    root = output_dir / "cases" / spec.case_id
    return _prepare_paths(root, spec, spec.prompt())


def prepare_candidate(
    case_paths: CasePaths,
    spec: CaseSpec,
    candidate: ExecutionCandidate,
) -> CasePaths:
    """Prepare the candidate."""
    root = case_paths.root / "candidates" / candidate.name
    return _prepare_paths(root, spec, candidate.artifact_prompt())


def _prepare_paths(root: Path, spec: CaseSpec, prompt: str) -> CasePaths:
    input_dir = root / "input"
    execution_dir = root / "execution"
    validation_dir = root / "validation"
    evidence_dir = root / "evidence"
    for directory in (input_dir, execution_dir, validation_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=False)
    paths = CasePaths(
        root=root,
        prompt=input_dir / "prompt.txt",
        starter=input_dir / "starter.py",
        solution=execution_dir / "solution.py",
        diff=execution_dir / "diff.patch",
        trace=execution_dir / "trace.jsonl",
        events=execution_dir / "events.jsonl",
        local_tests=execution_dir / "local-tests.txt",
        validation_result=validation_dir / "result.json",
        validation_output=validation_dir / "output.txt",
        mode_evidence=evidence_dir / f"{spec.mode}.json",
        manifest=root / "case.json",
    )
    paths.prompt.write_text(prompt, encoding="utf-8")
    paths.starter.write_text(spec.starter_code, encoding="utf-8")
    paths.solution.write_text(spec.starter_code, encoding="utf-8")
    paths.manifest.write_text(
        json.dumps(spec.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def capture_execution(
    paths: CasePaths,
    spec: CaseSpec,
    events: Iterable[RuntimeUIEvent],
) -> dict[str, int]:
    """Handle the capture execution operation."""
    event_list = tuple(events)
    solution = paths.solution.read_text(encoding="utf-8")
    if not paths.trace.exists():
        paths.trace.write_text("", encoding="utf-8")
    paths.diff.write_text(_solution_diff(spec.starter_code, solution), encoding="utf-8")
    paths.events.write_text(_events_jsonl(event_list), encoding="utf-8")
    local_test_output, local_test_evidence = _local_test_output(event_list)
    paths.local_tests.write_text(local_test_output, encoding="utf-8")
    return {
        **_execution_evidence(spec.starter_code, solution, event_list, paths),
        **local_test_evidence,
    }


def write_validation(
    paths: CasePaths,
    result: ValidationResult,
    output: str,
) -> None:
    """Write the validation."""
    paths.validation_result.write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.validation_output.write_text(
        output if output.endswith("\n") else output + "\n",
        encoding="utf-8",
    )


def write_mode_evidence(paths: CasePaths, evidence: dict[str, object]) -> None:
    """Write the mode evidence."""
    paths.mode_evidence.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _solution_diff(starter: str, solution: str) -> str:
    return "".join(
        difflib.unified_diff(
            starter.splitlines(keepends=True),
            solution.splitlines(keepends=True),
            fromfile="a/solution.py",
            tofile="b/solution.py",
        )
    )


def _events_jsonl(events: Iterable[RuntimeUIEvent]) -> str:
    lines = []
    for event in events:
        lines.append(
            json.dumps(
                {
                    "type": event.type.value,
                    "sequence": event.sequence,
                    "timestamp": event.timestamp,
                    "session_id": event.session_id,
                    "root_session_id": event.root_session_id,
                    "trace_id": event.trace_id,
                    "span_id": event.span_id,
                    "request_id": event.request_id,
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "payload": dict(event.payload),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _local_test_output(
    events: tuple[RuntimeUIEvent, ...],
) -> tuple[str, dict[str, int]]:
    commands: dict[str, list[str]] = {}
    sections: list[str] = []
    completed = 0
    passed = 0
    for event in events:
        call_id = event.tool_call_id
        if not call_id or event.tool_name != "run_shell":
            continue
        if event.type is UIEventType.TOOL_EXECUTION_STARTED:
            arguments = event.payload.get("arguments")
            argv = arguments.get("argv") if isinstance(arguments, dict) else None
            if isinstance(argv, list) and _is_test_command(argv):
                commands[call_id] = [str(item) for item in argv]
            continue
        if call_id not in commands or event.type not in {
            UIEventType.TOOL_EXECUTION_FINISHED,
            UIEventType.TOOL_EXECUTION_FAILED,
            UIEventType.TOOL_EXECUTION_DENIED,
        }:
            continue
        completed += 1
        if event.type is UIEventType.TOOL_EXECUTION_FINISHED and _test_passed(event):
            passed += 1
        sections.append(_render_test_event(commands[call_id], event))
    evidence = {
        "local_test_attempted": int(bool(commands)),
        "local_test_completed": int(bool(commands) and completed == len(commands)),
        "local_test_passed": int(bool(commands) and passed == len(commands)),
        "local_test_output_present": int(bool(sections)),
    }
    if not sections:
        return "No local tests executed.\n", evidence
    return "\n\n".join(sections) + "\n", evidence


def _test_passed(event: RuntimeUIEvent) -> bool:
    payload = dict(event.payload)
    metadata = payload.get("metadata")
    details = metadata if isinstance(metadata, dict) else payload
    exit_code = details.get("exit_code")
    status = details.get("status", payload.get("status"))
    if isinstance(exit_code, int):
        return exit_code == 0
    return status in {"success", "passed"}


def _is_test_command(argv: list[object]) -> bool:
    values = [str(item).casefold() for item in argv]
    if not values:
        return False
    command = Path(values[0]).name.removesuffix(".exe")
    if command in {"pytest", "py.test"}:
        return True
    return len(values) >= 3 and command.startswith(("python", "py")) and values[1:3] in (
        ["-m", "pytest"],
        ["-m", "unittest"],
    )


def _render_test_event(argv: list[str], event: RuntimeUIEvent) -> str:
    payload = dict(event.payload)
    metadata = payload.get("metadata")
    details = metadata if isinstance(metadata, dict) else payload
    lines = [f"$ {' '.join(argv)}", f"event: {event.type.value}"]
    for name in ("status", "exit_code", "stdout", "stderr", "reason", "message"):
        value = details.get(name)
        if value not in (None, ""):
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


def _execution_evidence(
    starter: str,
    solution: str,
    events: tuple[RuntimeUIEvent, ...],
    paths: CasePaths,
) -> dict[str, int]:
    counts = {event_type.value: 0 for event_type in UIEventType}
    for event in events:
        counts[event.type.value] += 1
    outcomes, outcome_coverage = _tool_outcomes(paths.trace, events)
    emitted = _event_call_ids(events, UIEventType.TOOL_CALL_COMPLETED)
    dispatched = {
        event.tool_call_id
        for event in events
        if event.type
        in {
            UIEventType.TOOL_EXECUTION_STARTED,
            UIEventType.TOOL_EXECUTION_FINISHED,
            UIEventType.TOOL_EXECUTION_FAILED,
            UIEventType.TOOL_EXECUTION_DENIED,
        }
        and event.tool_call_id
    }
    changed = int(starter != solution)
    diff_valid = int(bool(paths.diff.stat().st_size) == bool(changed))
    return {
        "model_rounds": counts[UIEventType.MODEL_REQUESTED.value],
        "tool_calls": len(outcomes),
        "tool_successful_calls": sum(status == "success" for status in outcomes.values()),
        "permission_requests": counts[UIEventType.PERMISSION_REQUESTED.value],
        "permission_denied_calls": sum(status == "denied" for status in outcomes.values()),
        "duplicate_blocked_calls": sum(
            status == "duplicate_blocked" for status in outcomes.values()
        ),
        "tool_failed_calls": sum(
            status not in {"success", "denied", "duplicate_blocked"}
            for status in outcomes.values()
        ),
        "tool_outcome_coverage": outcome_coverage,
        "undispatched_tool_calls": len(emitted - dispatched),
        "solution_changed": changed,
        "diff_valid": diff_valid,
    }


def _tool_outcomes(
    trace_path: Path,
    events: tuple[RuntimeUIEvent, ...],
) -> tuple[dict[str, str], int]:
    dispatched = {
        event.tool_call_id
        for event in events
        if event.type
        in {
            UIEventType.TOOL_EXECUTION_STARTED,
            UIEventType.TOOL_EXECUTION_FINISHED,
            UIEventType.TOOL_EXECUTION_FAILED,
            UIEventType.TOOL_EXECUTION_DENIED,
        }
        and event.tool_call_id
    }
    traced = _trace_tool_outcomes(trace_path)
    if traced:
        coverage = int(dispatched.issubset(traced))
        for call_id in dispatched:
            traced.setdefault(call_id, "unresolved")
        return traced, coverage
    outcomes: dict[str, str] = {}
    for event in events:
        call_id = event.tool_call_id
        if not call_id:
            continue
        if event.type is UIEventType.TOOL_EXECUTION_FINISHED:
            outcomes[call_id] = "success"
        elif event.type is UIEventType.TOOL_EXECUTION_FAILED:
            outcomes[call_id] = "tool_error"
        elif event.type is UIEventType.TOOL_EXECUTION_DENIED:
            outcomes[call_id] = "denied"
    for call_id in dispatched:
        outcomes.setdefault(call_id, "unresolved")
    return outcomes, int(
        all(outcomes.get(call_id) != "unresolved" for call_id in dispatched)
    )


def _event_call_ids(
    events: tuple[RuntimeUIEvent, ...], event_type: UIEventType
) -> set[str]:
    return {
        event.tool_call_id
        for event in events
        if event.type is event_type and event.tool_call_id
    }


def _trace_tool_outcomes(path: Path) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return outcomes
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("event") != "tool.runtime" or row.get("stage") != "final":
            continue
        call_id = row.get("tool_call_id")
        status = row.get("status")
        if isinstance(call_id, str) and isinstance(status, str):
            outcomes[call_id] = status
    return outcomes
