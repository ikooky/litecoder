from __future__ import annotations

import json
import subprocess

from litecoder.eval.domain import CaseSpec
from litecoder.eval.validation import EvalPlusValidator


def test_multi_agent_validation_isolated_worker_retries_infrastructure_error(
    monkeypatch,
) -> None:
    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "multi-agent",
    )
    responses = [
        subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout=json.dumps({"ok": False, "error": "spawn failed"}),
            stderr="",
        ),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "evaluation": {
                        "task_id": "HumanEval/0",
                        "passed": True,
                        "failure_reason": "",
                        "base_status": "pass",
                        "plus_status": "pass",
                        "failed_test_count": 0,
                        "first_failed_index": None,
                    },
                }
            ),
            stderr="",
        ),
    ]
    calls: list[dict[str, object]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, **kwargs})
        return responses.pop(0)

    monkeypatch.setattr("litecoder.eval.validation.subprocess.run", fake_run)

    capture = EvalPlusValidator(max_infra_retries=1).validate(
        spec,
        "def answer():\n    return 1\n",
    )

    assert capture.result.passed is True
    assert "attempts: 2" in capture.output
    assert len(calls) == 2
    assert calls[0]["cwd"]
    assert calls[0]["args"][0][2:] == ["-m", "litecoder.eval.validation_worker"]
