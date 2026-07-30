from __future__ import annotations

import json
from pathlib import Path

from litecoder.eval.artifacts import capture_execution, prepare_case
from litecoder.eval.domain import CaseSpec
from litecoder.ui.events import RuntimeUIEvent, UIEventType


def _spec() -> CaseSpec:
    return CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "agent-benchmark",
    )


def test_prepare_case_exposes_only_solution_to_execution_workspace(
    tmp_path: Path,
) -> None:
    paths = prepare_case(tmp_path, _spec())

    assert [path.name for path in paths.solution.parent.iterdir()] == ["solution.py"]
    assert paths.starter.read_text(encoding="utf-8") == "def answer():\n"


def test_capture_execution_writes_real_diff_and_test_output(tmp_path: Path) -> None:
    spec = _spec()
    paths = prepare_case(tmp_path, spec)
    paths.solution.write_text("def answer():\n    return 42\n", encoding="utf-8")
    events = (
        RuntimeUIEvent(
            UIEventType.MODEL_REQUESTED,
            1,
            0.0,
        ),
        RuntimeUIEvent(
            UIEventType.TOOL_CALL_STARTED,
            2,
            0.1,
            tool_call_id="test-1",
            tool_name="run_shell",
        ),
        RuntimeUIEvent(
            UIEventType.TOOL_EXECUTION_STARTED,
            3,
            0.2,
            tool_call_id="test-1",
            tool_name="run_shell",
            payload={"arguments": {"argv": ["python", "-m", "pytest", "-q"]}},
        ),
        RuntimeUIEvent(
            UIEventType.TOOL_EXECUTION_FINISHED,
            4,
            0.3,
            tool_call_id="test-1",
            tool_name="run_shell",
            payload={
                "status": "success",
                "metadata": {"exit_code": 0, "stdout": "1 passed"},
            },
        ),
    )

    evidence = capture_execution(paths, spec, events)

    assert "+    return 42" in paths.diff.read_text(encoding="utf-8")
    assert "1 passed" in paths.local_tests.read_text(encoding="utf-8")
    assert evidence["diff_valid"] == 1
    assert evidence["model_rounds"] == 1
    assert evidence["tool_calls"] == 1
    assert evidence["tool_successful_calls"] == 1
    assert evidence["tool_outcome_coverage"] == 1
    assert evidence["local_test_attempted"] == 1
    assert evidence["local_test_completed"] == 1
    assert evidence["local_test_passed"] == 1
    assert evidence["local_test_output_present"] == 1


def test_capture_execution_records_no_local_tests(tmp_path: Path) -> None:
    spec = _spec()
    paths = prepare_case(tmp_path, spec)

    evidence = capture_execution(paths, spec, ())

    assert paths.local_tests.read_text(encoding="utf-8") == "No local tests executed.\n"
    assert evidence["local_test_attempted"] == 0
    assert evidence["local_test_completed"] == 0
    assert evidence["local_test_passed"] == 0
    assert evidence["local_test_output_present"] == 0


def test_capture_execution_records_failed_local_test(tmp_path: Path) -> None:
    spec = _spec()
    paths = prepare_case(tmp_path, spec)
    events = (
        RuntimeUIEvent(
            UIEventType.TOOL_EXECUTION_STARTED,
            1,
            0.0,
            tool_call_id="test-1",
            tool_name="run_shell",
            payload={"arguments": {"argv": ["pytest", "-q"]}},
        ),
        RuntimeUIEvent(
            UIEventType.TOOL_EXECUTION_FINISHED,
            2,
            0.1,
            tool_call_id="test-1",
            tool_name="run_shell",
            payload={
                "status": "success",
                "metadata": {"exit_code": 1, "stdout": "1 failed"},
            },
        ),
    )

    evidence = capture_execution(paths, spec, events)

    assert evidence["local_test_attempted"] == 1
    assert evidence["local_test_completed"] == 1
    assert evidence["local_test_passed"] == 0


def test_capture_execution_separates_emitted_from_dispatched_calls(
    tmp_path: Path,
) -> None:
    spec = _spec()
    paths = prepare_case(tmp_path, spec)
    events = (
        RuntimeUIEvent(
            UIEventType.TOOL_CALL_COMPLETED,
            1,
            0.0,
            tool_call_id="abandoned",
            tool_name="read_file",
        ),
    )

    evidence = capture_execution(paths, spec, events)

    assert evidence["undispatched_tool_calls"] == 1
    assert evidence["tool_outcome_coverage"] == 1


def test_capture_execution_uses_trace_final_tool_statuses(tmp_path: Path) -> None:
    spec = _spec()
    paths = prepare_case(tmp_path, spec)
    statuses = (
        ("read", "success"),
        ("edit", "tool_error"),
        ("duplicate", "duplicate_blocked"),
        ("write", "success"),
        ("shell", "denied"),
    )
    paths.trace.write_text(
        "".join(
            json.dumps(
                {
                    "event": "tool.runtime",
                    "stage": "final",
                    "tool_call_id": call_id,
                    "status": status,
                }
            )
            + "\n"
            for call_id, status in statuses
        ),
        encoding="utf-8",
    )

    evidence = capture_execution(paths, spec, ())

    assert evidence["tool_calls"] == 5
    assert evidence["tool_successful_calls"] == 2
    assert evidence["tool_failed_calls"] == 1
    assert evidence["duplicate_blocked_calls"] == 1
    assert evidence["permission_denied_calls"] == 1
    assert evidence["tool_outcome_coverage"] == 1
