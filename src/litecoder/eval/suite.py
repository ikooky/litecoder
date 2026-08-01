"""Evaluation suite construction and selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from litecoder.eval.cli import execute_eval_run, write_eval_reports
from litecoder.eval.evalplus import (
    EvalPlusExecutionError,
    EvalPlusTask,
    EvalPlusUnavailable,
    load_evalplus_tasks,
)
from litecoder.eval.domain import DatasetName
from litecoder.eval.provenance import collect_provenance


RUN_SPECS: tuple[tuple[str, int], ...] = (
    ("agent-benchmark", 12),
    ("context-manager", 12),
    ("tools-hooks", 12),
    ("memory", 12),
    ("task-state", 12),
)
DATASETS: tuple[DatasetName, ...] = ("humaneval", "mbpp")
SHARED_DATASET_LIMITS: tuple[tuple[DatasetName, int], ...] = (
    ("humaneval", 6),
    ("mbpp", 6),
)
SUITE_SEED = 2026
SUITE_INFRA_EXIT_CODE = 3


@dataclass(frozen=True, slots=True)
class SuiteRun:
    """Data model representing the suite run."""
    mode: str
    datasets: tuple[tuple[DatasetName, int], ...]

    @property
    def limit(self) -> int:
        """Handle the limit operation."""
        return sum(limit for _, limit in self.datasets)


TaskLoader = Callable[..., tuple[EvalPlusTask, ...]]
ModeRunner = Callable[..., int]


def run_suite(
    *,
    seed: int = SUITE_SEED,
    output_root: Path = Path("eval-runs"),
    task_loader: TaskLoader = load_evalplus_tasks,
    mode_runner: ModeRunner | None = None,
    now: datetime | None = None,
    suffix: str | None = None,
) -> int:
    """Run the suite."""
    started_at = datetime.now(UTC)
    plan = build_run_plan(seed)
    local_now, run_suffix, suite_root = make_suite_location(
        output_root,
        now=now,
        suffix=suffix,
    )
    provenance = collect_provenance()
    run_summaries: list[dict[str, object]] = []
    shared_task_ids: list[str] = []

    def finish(code: int, status: str, error: str = "") -> int:
        _write_suite_reports(
            suite_root,
            plan,
            seed,
            status=status,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            provenance=provenance,
            runs=run_summaries,
            shared_task_ids=shared_task_ids,
            error=error,
        )
        return code

    try:
        shared_tasks = _load_mode_tasks(plan[0], seed, task_loader)
    except (ValueError, EvalPlusUnavailable) as error:
        print(str(error), file=sys.stderr)
        return finish(2, "failed", str(error))
    shared_task_ids.extend(task.task_id for task in shared_tasks)
    _write_plan(
        plan,
        seed,
        suite_root / "eval-suite-plan.json",
        shared_task_ids=shared_task_ids,
    )

    _write_suite_reports(
        suite_root,
        plan,
        seed,
        status="running",
        started_at=started_at,
        finished_at=None,
        provenance=provenance,
        runs=run_summaries,
        shared_task_ids=shared_task_ids,
    )
    runner = mode_runner or _run_mode
    print(f"Evaluation seed: {seed}", flush=True)
    print(f"Suite: {suite_root}", flush=True)

    for run in plan:
        tasks = shared_tasks
        print(
            f"\n=== Running {run.mode}: {run.limit} mixed case(s) ===",
            flush=True,
        )
        run_id = (
            f"{local_now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{run_suffix}-{run.mode}"
        )
        try:
            return_code = runner(
                mode=run.mode,
                datasets=tuple(dataset for dataset, _ in run.datasets),
                tasks=tasks,
                output_dir=suite_root / run.mode,
                run_id=run_id,
            )
        except EvalPlusExecutionError as error:
            print(str(error), file=sys.stderr)
            return finish(3, "failed", str(error))
        run_summaries.append(
            _read_run_summary(suite_root, run.mode, suite_root / run.mode / "run.json")
        )
        if return_code != 0:
            print(
                f"Evaluation failed: {run.mode} (exit {return_code})",
                file=sys.stderr,
            )
            return finish(
                return_code,
                "interrupted" if return_code == 130 else "failed",
                f"{run.mode} exited with {return_code}",
            )

    print(f"\nAll evaluations completed. Reports: {suite_root}", flush=True)
    infra_runs = [item for item in run_summaries if _run_has_infra_errors(item)]
    if infra_runs:
        return finish(
            SUITE_INFRA_EXIT_CODE,
            "completed_with_infra_errors",
            _suite_infra_summary(infra_runs),
        )
    return finish(0, "completed")


def build_run_plan(seed: int) -> tuple[SuiteRun, ...]:
    """Build the run plan."""
    del seed
    return tuple(SuiteRun(mode, SHARED_DATASET_LIMITS) for mode, _ in RUN_SPECS)


def make_suite_location(
    output_root: Path,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> tuple[datetime, str, Path]:
    """Build the suite location."""
    local_now = now if now is not None else datetime.now().astimezone()
    if local_now.tzinfo is None or local_now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    run_suffix = suffix if suffix is not None else uuid.uuid4().hex[:8]
    if len(run_suffix) != 8 or any(
        character not in "0123456789abcdefABCDEF" for character in run_suffix
    ):
        raise ValueError("suffix must be 8 hexadecimal characters")
    suite_root = (
        Path(output_root)
        / local_now.strftime("%Y-%m-%d")
        / f"{local_now.strftime('%H%M%S')}-{run_suffix}"
    )
    return local_now, run_suffix, suite_root


def _load_mode_tasks(
    run: SuiteRun,
    seed: int,
    task_loader: TaskLoader,
) -> tuple[EvalPlusTask, ...]:
    tasks: list[EvalPlusTask] = []
    for dataset, limit in run.datasets:
        tasks.extend(
            task_loader(
                dataset,
                limit=limit,
                seed=_selection_seed(seed, "shared-core", dataset),
            )
        )
    random.Random(_selection_seed(seed, "shared-core", "mixed")).shuffle(tasks)
    return tuple(tasks)


def _run_mode(**kwargs: object) -> int:
    result = execute_eval_run(**kwargs)  # type: ignore[arg-type]
    result_path = write_eval_reports(result)
    print(str(result_path), flush=True)
    return 130 if result.status == "interrupted" else 0


def _write_plan(
    plan: tuple[SuiteRun, ...],
    seed: int,
    path: Path,
    *,
    shared_task_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "task_scope": "shared-core",
        "unique_task_count": len(shared_task_ids),
        "shared_task_ids": shared_task_ids,
        "total_cases": sum(run.limit for run in plan),
        "runs": [
            {
                "mode": run.mode,
                "limit": run.limit,
                "datasets": {dataset: limit for dataset, limit in run.datasets},
            }
            for run in plan
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_run_summary(
    suite_root: Path, mode: str, path: Path
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    cases = payload.get("cases") if isinstance(payload, dict) else None
    case_items = cases if isinstance(cases, list) else []
    counts = {
        status: sum(
            isinstance(case, dict) and case.get("status") == status
            for case in case_items
        )
        for status in ("passed", "failed", "infra_error", "invalid")
    }
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    return {
        "mode": mode,
        "path": str(path.relative_to(suite_root)),
        "run_id": payload.get("run_id", "") if isinstance(payload, dict) else "",
        "status": payload.get("status", "completed") if isinstance(payload, dict) else "completed",
        "cases": len(case_items),
        "case_status_counts": counts,
        "models": metadata.get("models", []) if isinstance(metadata, dict) else [],
        "providers": metadata.get("providers", []) if isinstance(metadata, dict) else [],
        "primary_metrics": payload.get("primary_metrics", {}) if isinstance(payload, dict) else {},
    }


def _run_has_infra_errors(summary: dict[str, object]) -> bool:
    if summary.get("status") == "completed_with_infra_errors":
        return True
    counts = summary.get("case_status_counts")
    return isinstance(counts, dict) and int(counts.get("infra_error", 0) or 0) > 0


def _suite_infra_summary(runs: list[dict[str, object]]) -> str:
    details: list[str] = []
    for item in runs:
        counts = item.get("case_status_counts")
        infra_count = (
            int(counts.get("infra_error", 0) or 0)
            if isinstance(counts, dict)
            else 0
        )
        if infra_count:
            details.append(f"{item.get('mode', 'unknown')}: {infra_count} case(s)")
        elif item.get("status") == "completed_with_infra_errors":
            details.append(f"{item.get('mode', 'unknown')}: infra errors reported")
    suffix = "; ".join(details) if details else "see per-mode reports"
    return "Suite completed with infrastructure errors; " + suffix


def _write_suite_reports(
    suite_root: Path,
    plan: tuple[SuiteRun, ...],
    seed: int,
    *,
    status: str,
    started_at: datetime,
    finished_at: datetime | None,
    provenance: dict[str, object],
    runs: list[dict[str, object]],
    shared_task_ids: list[str],
    error: str = "",
) -> None:
    payload = {
        "suite_id": suite_root.name,
        "seed": seed,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if finished_at else None,
        "error": error,
        "provenance": provenance,
        "task_scope": "shared-core",
        "unique_task_count": len(shared_task_ids),
        "shared_task_ids": shared_task_ids,
        "plan": [
            {
                "mode": run.mode,
                "limit": run.limit,
                "datasets": {dataset: limit for dataset, limit in run.datasets},
            }
            for run in plan
        ],
        "runs": list(runs),
    }
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "suite.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (suite_root / "suite-report.md").write_text(
        _render_suite_markdown(payload), encoding="utf-8"
    )


def _render_suite_markdown(payload: dict[str, object]) -> str:
    """Render the suite markdown."""
    lines = [
        "# LiteCoder Evaluation Suite",
        "",
        f"Status: {payload.get('status', '')}",
        f"Seed: {payload.get('seed', '')}",
        f"Task scope: {payload.get('task_scope', '')}",
        f"Unique tasks: {payload.get('unique_task_count', '')}",
        f"Started: {payload.get('started_at', '')}",
        f"Finished: {payload.get('finished_at', '')}",
        "",
        "## Runs",
        "",
        "| Mode | Status | Cases | Passed | Failed | Infra | Invalid | Models |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    runs = payload.get("runs")
    if isinstance(runs, list):
        for item in runs:
            if not isinstance(item, dict):
                continue
            counts = item.get("case_status_counts")
            values = counts if isinstance(counts, dict) else {}
            models = item.get("models")
            model_text = (
                ", ".join(str(value) for value in models)
                if isinstance(models, list)
                else ""
            )
            lines.append(
                (
                    "| {mode} | {status} | {cases} | {passed} | {failed} | "
                    "{infra} | {invalid} | {models} |"
                ).format(
                    mode=item.get("mode", ""),
                    status=item.get("status", ""),
                    cases=item.get("cases", 0),
                    passed=values.get("passed", 0),
                    failed=values.get("failed", 0),
                    infra=values.get("infra_error", 0),
                    invalid=values.get("invalid", 0),
                    models=model_text,
                )
            )
    error = payload.get("error")
    if isinstance(error, str) and error:
        lines.extend(["", "## Error", "", error])
    return "\n".join(lines) + "\n"


def _selection_seed(seed: int, mode: str, dataset: str) -> int:
    value = f"{seed}:{mode}:{dataset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFFFFFF


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete LiteCoder EvalPlus suite."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SUITE_SEED,
        help=f"random task selection seed (default: {SUITE_SEED})",
    )
    return parser


def main() -> None:
    """Run the command-line entry point."""
    options = _parser().parse_args()
    raise SystemExit(run_suite(seed=options.seed))


if __name__ == "__main__":
    main()
