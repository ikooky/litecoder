"""Subprocess entry point for isolated EvalPlus validation."""

from __future__ import annotations

import json
import sys
import traceback

from litecoder.eval.evalplus import evaluate_solution


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        dataset = payload["dataset"]
        task_id = payload["task_id"]
        solution = payload["solution"]
        if not all(isinstance(value, str) for value in (dataset, task_id, solution)):
            raise ValueError("validation payload fields must be strings")
    except BaseException as error:
        _write_error(error)
        return 2

    try:
        evaluation = evaluate_solution(dataset, task_id, solution)
    except BaseException as error:
        _write_error(error)
        return 2
    print(
        json.dumps(
            {"ok": True, "evaluation": evaluation.to_json()},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _write_error(error: BaseException) -> None:
    print(
        json.dumps(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
