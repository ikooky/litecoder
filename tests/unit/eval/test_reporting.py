from __future__ import annotations

from litecoder.eval.reporting import render_markdown_payload


def test_report_renders_candidate_outcomes_and_failure_class() -> None:
    report = render_markdown_payload(
        {
            "mode": "multi-agent",
            "cases": [
                {
                    "case_id": "case-0001",
                    "task_id": "HumanEval/0",
                    "status": "failed",
                    "stage": "scored",
                    "workspace": "cases/case-0001",
                    "failure_reason": "team.closed_loop: lifecycle validation failed",
                    "candidates": {
                        "team": {
                            "status": "passed",
                            "execution": {"status": "completed"},
                            "validation": {"passed": True},
                            "failure_reason": "",
                        },
                        "subagent": {
                            "status": "failed",
                            "execution": {"status": "incomplete"},
                            "validation": {"passed": True},
                            "failure_reason": "round budget exhausted",
                        },
                    },
                }
            ],
        }
    )

    assert "workflow_not_closed" in report
    assert "team: passed, runtime=completed, validation=pass" in report
    assert "subagent: failed, runtime=incomplete, validation=pass, round budget exhausted" in report
