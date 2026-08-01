from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from litecoder.eval.evalplus import EvalPlusTask


def _eval_suite_module():
    try:
        return importlib.import_module("litecoder.eval.suite")
    except ModuleNotFoundError:
        raise AssertionError("litecoder.eval.suite is not implemented") from None


def test_suite_accepts_seed_option() -> None:
    eval_suite = _eval_suite_module()

    parser = eval_suite._parser()

    assert [action.dest for action in parser._actions] == ["help", "seed"]
    assert parser.parse_args([]).seed == eval_suite.SUITE_SEED
    assert parser.parse_args(["--seed", "2026"]).seed == 2026


def test_run_plan_uses_fixed_shared_core_case_counts() -> None:
    eval_suite = _eval_suite_module()

    first = eval_suite.build_run_plan(eval_suite.SUITE_SEED)
    second = eval_suite.build_run_plan(eval_suite.SUITE_SEED)
    different_seed = eval_suite.build_run_plan(eval_suite.SUITE_SEED + 1)

    assert first == second
    assert first == different_seed
    assert len(first) == len(eval_suite.RUN_SPECS)
    assert sum(run.limit for run in first) == sum(limit for _, limit in eval_suite.RUN_SPECS)
    for run, (mode, expected_total) in zip(first, eval_suite.RUN_SPECS, strict=True):
        assert run.mode == mode
        assert run.limit == expected_total
        assert run.datasets == eval_suite.SHARED_DATASET_LIMITS
        assert sum(limit for _, limit in run.datasets) == expected_total


def test_full_suite_uses_one_root_and_one_run_per_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_suite = _eval_suite_module()
    calls: list[dict[str, object]] = []
    fixed = datetime(2026, 7, 22, 21, 25, 44, tzinfo=timezone(timedelta(hours=8)))

    def fake_loader(dataset: str, *, limit: int, seed: int) -> tuple[EvalPlusTask, ...]:
        prefix = "HumanEval" if dataset == "humaneval" else "Mbpp"
        return tuple(
            EvalPlusTask(
                f"{prefix}/{index}",
                f"def answer_{index}():\n",
                f"answer_{index}",
                dataset,
            )
            for index in range(limit)
        )

    def fake_mode_runner(**kwargs: object) -> int:
        calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True)
        (output_dir / "run.json").write_text("{}", encoding="utf-8")
        (output_dir / "report.md").write_text("summary", encoding="utf-8")
        return 0

    monkeypatch.chdir(tmp_path)
    result = eval_suite.run_suite(
        seed=2026,
        task_loader=fake_loader,
        mode_runner=fake_mode_runner,
        now=fixed,
        suffix="9e6dd0b6",
    )

    suite_root = tmp_path / "eval-runs" / "2026-07-22" / "212544-9e6dd0b6"
    assert result == 0
    assert len(calls) == len(eval_suite.RUN_SPECS)
    assert [call["mode"] for call in calls] == [mode for mode, _ in eval_suite.RUN_SPECS]
    assert all(
        Path(call["output_dir"]).resolve() == (suite_root / str(call["mode"])).resolve()
        for call in calls
    )
    assert sum(len(call["tasks"]) for call in calls) == 72
    assert all(call["tasks"] == calls[0]["tasks"] for call in calls)
    assert all(call["datasets"] == ("humaneval", "mbpp") for call in calls)
    assert all((suite_root / mode / "run.json").exists() for mode, _ in eval_suite.RUN_SPECS)
    assert all((suite_root / mode / "report.md").exists() for mode, _ in eval_suite.RUN_SPECS)
    assert (suite_root / "suite.json").exists()
    assert (suite_root / "suite-report.md").exists()
    assert not (suite_root / "humaneval").exists()
    assert not (suite_root / "mbpp").exists()

    plan = json.loads((suite_root / "eval-suite-plan.json").read_text(encoding="utf-8"))
    assert plan["seed"] == 2026
    assert "schema_version" not in plan
    assert plan["task_scope"] == "shared-core"
    assert plan["unique_task_count"] == 12
    assert len(plan["shared_task_ids"]) == 12
    assert plan["total_cases"] == 72
    assert len(plan["runs"]) == len(eval_suite.RUN_SPECS)
    assert all(set(run) == {"datasets", "limit", "mode"} for run in plan["runs"])

    suite = json.loads((suite_root / "suite.json").read_text(encoding="utf-8"))
    assert "schema_version" not in suite
    assert suite["task_scope"] == "shared-core"
    assert suite["seed"] == 2026
    assert suite["unique_task_count"] == 12
    assert suite["shared_task_ids"] == plan["shared_task_ids"]
    assert suite["status"] == "completed"
    assert len(suite["runs"]) == len(eval_suite.RUN_SPECS)


def test_suite_stops_after_first_failed_mode(tmp_path: Path) -> None:
    eval_suite = _eval_suite_module()
    calls: list[str] = []

    def fake_loader(dataset: str, *, limit: int, seed: int) -> tuple[EvalPlusTask, ...]:
        return tuple(
            EvalPlusTask(f"HumanEval/{index}", "def answer():\n", "answer", dataset)
            for index in range(limit)
        )

    def failing_runner(**kwargs: object) -> int:
        calls.append(str(kwargs["mode"]))
        return 7 if len(calls) == 2 else 0

    result = eval_suite.run_suite(
        output_root=tmp_path,
        task_loader=fake_loader,
        mode_runner=failing_runner,
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        suffix="a1b2c3d4",
    )

    assert result == 7
    assert len(calls) == 2


def test_suite_returns_infra_exit_code_when_a_mode_has_infra_cases(
    tmp_path: Path,
) -> None:
    eval_suite = _eval_suite_module()

    def fake_loader(dataset: str, *, limit: int, seed: int) -> tuple[EvalPlusTask, ...]:
        return tuple(
            EvalPlusTask(
                f"HumanEval/{index}",
                "def answer():\n",
                "answer",
                dataset,
            )
            for index in range(limit)
        )

    def infra_runner(**kwargs: object) -> int:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True)
        (output_dir / "run.json").write_text(
            json.dumps(
                {
                    "status": "completed_with_infra_errors",
                    "cases": [{"status": "infra_error"}],
                }
            ),
            encoding="utf-8",
        )
        return 0

    result = eval_suite.run_suite(
        output_root=tmp_path,
        task_loader=fake_loader,
        mode_runner=infra_runner,
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        suffix="f1e2d3c4",
    )

    assert result == eval_suite.SUITE_INFRA_EXIT_CODE
    suite_path = tmp_path / "2026-07-22" / "000000-f1e2d3c4" / "suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert suite["status"] == "completed_with_infra_errors"
    assert "agent-benchmark: 1 case(s)" in suite["error"]
