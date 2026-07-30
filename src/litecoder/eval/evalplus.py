"""EvalPlus benchmark integration."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from litecoder.eval.domain import DatasetName, validate_dataset


class EvalPlusUnavailable(RuntimeError):
    """Component responsible for the eval plus unavailable."""
    pass


class EvalPlusExecutionError(RuntimeError):
    """Raised when the eval plus execution error conditions occur."""
    pass


@dataclass(frozen=True, slots=True)
class EvalPlusTask:
    """Data model representing the eval plus task."""
    task_id: str
    prompt: str
    entry_point: str
    dataset: DatasetName

    def __post_init__(self) -> None:
        _required_str(self.task_id, "task_id")
        _required_str(self.prompt, "prompt")
        _required_str(self.entry_point, "entry_point")
        object.__setattr__(self, "dataset", validate_dataset(self.dataset))


@dataclass(frozen=True, slots=True)
class EvalPlusSample:
    """Data model representing the eval plus sample."""
    task_id: str
    solution: str

    def __post_init__(self) -> None:
        _required_str(self.task_id, "task_id")
        if not isinstance(self.solution, str):
            raise ValueError("solution must be a string")

    def to_json(self) -> dict[str, str]:
        """Convert this object to a JSON-compatible value."""
        return {"task_id": self.task_id, "solution": self.solution}


@dataclass(frozen=True, slots=True)
class EvalPlusCaseEvaluation:
    """Data model representing the eval plus case evaluation."""
    task_id: str
    passed: bool
    failure_reason: str = ""
    base_status: str = "unknown"
    plus_status: str = "unknown"
    failed_test_count: int = 0
    first_failed_index: int | None = None

    def __post_init__(self) -> None:
        _required_str(self.task_id, "task_id")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a bool")
        if not isinstance(self.failure_reason, str):
            raise ValueError("failure_reason must be a string")
        if self.failed_test_count < 0:
            raise ValueError("failed_test_count must be non-negative")
        if self.first_failed_index is not None and self.first_failed_index < 0:
            raise ValueError("first_failed_index must be non-negative")

    def to_json(self) -> dict[str, object]:
        """Convert this object to a JSON-compatible value."""
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "base_status": self.base_status,
            "plus_status": self.plus_status,
            "failed_test_count": self.failed_test_count,
            "first_failed_index": self.first_failed_index,
        }


class OfficialEvalPlusEvaluator:
    """Component responsible for the official eval plus evaluator."""
    def evaluate(
        self,
        dataset: str,
        samples_path: Path,
        task_ids: tuple[str, ...],
    ) -> dict[str, EvalPlusCaseEvaluation]:
        """Handle the evaluate operation."""
        return evaluate_samples(dataset, samples_path, task_ids)


TaskLoader = Callable[[str], Mapping[str, Mapping[str, object]]]


def load_evalplus_tasks(
    dataset: str,
    *,
    limit: int | None = None,
    seed: int | None = None,
    loader: TaskLoader | None = None,
) -> tuple[EvalPlusTask, ...]:
    """Load the evalplus tasks."""
    selected = validate_dataset(dataset)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    try:
        raw_tasks = (loader or _default_loader)(selected)
    except ModuleNotFoundError as error:
        raise EvalPlusUnavailable(
            "EvalPlus is required. Install the eval extra with: "
            "python -m pip install -e \".[eval]\""
        ) from error
    task_ids = sorted(raw_tasks)
    if seed is not None:
        sample_size = len(task_ids) if limit is None else min(limit, len(task_ids))
        task_ids = random.Random(seed).sample(task_ids, sample_size)
    elif limit is not None:
        task_ids = task_ids[:limit]

    tasks: list[EvalPlusTask] = []
    for task_id in task_ids:
        raw = raw_tasks[task_id]
        tasks.append(
            EvalPlusTask(
                task_id=task_id,
                prompt=_required_str(raw.get("prompt"), "prompt"),
                entry_point=_required_str(raw.get("entry_point"), "entry_point"),
                dataset=selected,
            )
        )
    return tuple(tasks)


def build_sample(task: EvalPlusTask, solution: str) -> EvalPlusSample:
    """Build the sample."""
    if not isinstance(task, EvalPlusTask):
        raise ValueError("task must be an EvalPlusTask")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("solution must not be empty")
    return EvalPlusSample(task.task_id, solution)


def write_samples_jsonl(path: Path, samples: Iterable[EvalPlusSample]) -> Path:
    """Write the samples jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
    return path


def evaluate_samples(
    dataset: str,
    samples_path: Path,
    task_ids: tuple[str, ...],
    *,
    write_results: bool = True,
) -> dict[str, EvalPlusCaseEvaluation]:
    """Handle the evaluate samples operation."""
    selected = validate_dataset(dataset)
    try:
        problems, expected_output = _evalplus_problem_set(selected)
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness
    except (ImportError, ModuleNotFoundError) as error:
        raise EvalPlusUnavailable(
            "EvalPlus is required. Install the eval extra with: "
            "python -m pip install -e \".[eval]\""
        ) from error

    samples = _read_samples(samples_path)
    results: dict[str, EvalPlusCaseEvaluation] = {}
    for task_id in task_ids:
        solution = samples.get(task_id)
        if solution is None:
            results[task_id] = EvalPlusCaseEvaluation(task_id, False, "missing sample")
            continue
        results[task_id] = _evaluate_loaded_solution(
            check_correctness,
            selected,
            problems,
            expected_output,
            task_id,
            solution,
            PASS,
        )
    if write_results:
        _write_evalplus_results(
            samples_path.parent / "evalplus-normalized-results.json",
            selected,
            results,
        )
    return results


def evaluate_solution(
    dataset: str,
    task_id: str,
    solution: str,
) -> EvalPlusCaseEvaluation:
    """Handle the evaluate solution operation."""
    selected = validate_dataset(dataset)
    try:
        problems, expected_output = _evalplus_problem_set(selected)
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness
    except (ImportError, ModuleNotFoundError) as error:
        raise EvalPlusUnavailable(
            "EvalPlus is required. Install the eval extra with: "
            'python -m pip install -e ".[eval]"'
        ) from error
    return _evaluate_loaded_solution(
        check_correctness,
        selected,
        problems,
        expected_output,
        task_id,
        solution,
        PASS,
    )


def _evaluate_loaded_solution(
    check_correctness: Callable[..., Mapping[str, object]],
    dataset: DatasetName,
    problems: Mapping[str, Mapping[str, object]],
    expected_output: Mapping[str, Mapping[str, object]],
    task_id: str,
    solution: str,
    pass_status: str,
) -> EvalPlusCaseEvaluation:
    try:
        outcome = _check_evalplus_correctness(
            check_correctness,
            dataset,
            0,
            problems[task_id],
            solution,
            expected_output[task_id],
            base_only=False,
            fast_check=True,
            identifier=task_id,
        )
    except EvalPlusExecutionError:
        raise
    except Exception as error:
        return EvalPlusCaseEvaluation(task_id, False, str(error))
    passed = _evalplus_outcome_passed(outcome, pass_status)
    base = _group_summary(outcome.get("base"), pass_status)
    plus = _group_summary(outcome.get("plus"), pass_status)
    failed_count = base[1] + plus[1]
    first_indices = [value for value in (base[2], plus[2]) if value is not None]
    return EvalPlusCaseEvaluation(
        task_id,
        passed,
        "" if passed else _evalplus_failure_reason(outcome, pass_status),
        base_status=base[0],
        plus_status=plus[0],
        failed_test_count=failed_count,
        first_failed_index=min(first_indices) if first_indices else None,
    )


def _check_evalplus_correctness(
    official_check: Callable[..., Mapping[str, object]],
    dataset: DatasetName,
    completion_id: int,
    problem: Mapping[str, object],
    solution: str,
    expected_output: Mapping[str, object],
    *,
    base_only: bool,
    fast_check: bool,
    identifier: str,
) -> Mapping[str, object]:
    if _requires_windows_evalplus_adapter():
        return _windows_check_correctness(
            dataset,
            completion_id,
            problem,
            solution,
            expected_output,
            base_only=base_only,
            fast_check=fast_check,
            identifier=identifier,
        )
    return official_check(
        dataset,
        completion_id,
        problem,
        solution,
        expected_output,
        base_only=base_only,
        fast_check=fast_check,
        identifier=identifier,
    )


def _requires_windows_evalplus_adapter() -> bool:
    return os.name == "nt"


def _windows_check_correctness(
    dataset: DatasetName,
    completion_id: int,
    problem: Mapping[str, object],
    solution: str,
    expected_output: Mapping[str, object],
    *,
    base_only: bool,
    fast_check: bool,
    identifier: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    result["base"] = _windows_untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        str(problem["entry_point"]),
        expected_output["base"],
        problem["atol"],
        expected_output["base_time"],
        fast_check=fast_check,
    )
    if not base_only:
        result["plus"] = _windows_untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            str(problem["entry_point"]),
            expected_output["plus"],
            problem["atol"],
            expected_output["plus_time"],
            fast_check=fast_check,
        )
    return result


def _windows_untrusted_check(
    dataset: DatasetName,
    code: str,
    inputs: object,
    entry_point: str,
    expected: object,
    atol: object,
    ref_time: object,
    *,
    fast_check: bool,
) -> tuple[str, list[bool]]:
    from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
    from evalplus.eval import FAIL, PASS, TIMEOUT

    input_values = list(inputs)  # type: ignore[arg-type]
    reference_times = list(ref_time)  # type: ignore[arg-type]
    time_limits = [
        max(DEFAULT_MIN_TIME_LIMIT, DEFAULT_GT_TIME_LIMIT_FACTOR * float(value))
        for value in reference_times
    ]
    timeout_cap = _evalplus_timeout_cap()
    timeout = min(timeout_cap, sum(time_limits)) + 1
    if not fast_check:
        timeout += 1

    context = multiprocessing.get_context("spawn")
    progress = context.Value("i", 0)
    status = context.Value("i", 3)
    details = context.Array("b", [False for _ in input_values])
    supervision, worker_supervision = context.Pipe(duplex=False)
    process = context.Process(
        target=_windows_evalplus_worker,
        args=(
            dataset,
            entry_point,
            code,
            input_values,
            expected,
            time_limits,
            atol,
            fast_check,
            status,
            details,
            progress,
            worker_supervision,
        ),
    )
    try:
        try:
            process.start()
        except Exception as error:
            raise EvalPlusExecutionError(
                "EvalPlus Windows worker failed to start"
            ) from error
        finally:
            worker_supervision.close()
        try:
            input_timed_out, overall_timed_out = _monitor_windows_evalplus_worker(
                process,
                supervision,
                timeout + 1,
            )
        except Exception as error:
            _stop_process(process)
            raise EvalPlusExecutionError(
                "EvalPlus Windows worker supervision failed"
            ) from error
        if input_timed_out or overall_timed_out:
            _stop_process(process)
        else:
            process.join(timeout=1)
            if process.is_alive():
                _stop_process(process)
                raise EvalPlusExecutionError(
                    "EvalPlus Windows worker did not exit after closing supervision"
                )
    finally:
        supervision.close()
    if input_timed_out:
        detail_count = min(len(input_values), progress.value + 1)
        return FAIL, [bool(value) for value in details[:detail_count]]
    if overall_timed_out:
        return TIMEOUT, [bool(value) for value in details[: progress.value]]
    if process.exitcode not in (0, None):
        raise EvalPlusExecutionError(
            f"EvalPlus Windows worker exited with code {process.exitcode}"
        )

    status_name = {0: PASS, 1: FAIL, 2: TIMEOUT}.get(status.value)
    if status_name is None:
        raise EvalPlusExecutionError(
            "EvalPlus Windows worker exited without a result"
        )
    detail_values = [bool(value) for value in details[: progress.value]]
    if status_name == PASS and (
        len(detail_values) != len(input_values) or not all(detail_values)
    ):
        status_name = FAIL
    return status_name, detail_values


def _evalplus_timeout_cap() -> float:
    try:
        value = float(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", "60"))
    except ValueError:
        return 60.0
    return value if value > 0 else 60.0


def _monitor_windows_evalplus_worker(
    process: multiprocessing.Process,
    supervision: object,
    overall_timeout: float,
) -> tuple[bool, bool]:
    overall_deadline = time.perf_counter() + overall_timeout
    input_deadline: float | None = None
    while process.is_alive():
        now = time.perf_counter()
        if input_deadline is not None and now >= input_deadline:
            return True, False
        if now >= overall_deadline:
            return False, True
        next_deadline = min(
            overall_deadline,
            input_deadline if input_deadline is not None else overall_deadline,
        )
        wait_seconds = max(0.0, min(0.05, next_deadline - now))
        try:
            has_event = supervision.poll(wait_seconds)  # type: ignore[attr-defined]
        except (EOFError, OSError):
            return _closed_supervision_result(process)
        if has_event:
            while True:
                try:
                    event, value = supervision.recv()  # type: ignore[attr-defined]
                except (EOFError, OSError):
                    return _closed_supervision_result(process)
                if event == "start":
                    input_deadline = float(value)
                elif event == "end":
                    if (
                        input_deadline is not None
                        and float(value) > input_deadline
                    ):
                        return True, False
                    input_deadline = None
                try:
                    has_more = supervision.poll(0)  # type: ignore[attr-defined]
                except (EOFError, OSError):
                    return _closed_supervision_result(process)
                if not has_more:
                    break
    return False, False


def _closed_supervision_result(
    process: multiprocessing.Process,
) -> tuple[bool, bool]:
    process.join(timeout=1)
    if process.is_alive():
        raise OSError(
            "EvalPlus supervision pipe closed while the worker was still running"
        )
    return False, False


def _stop_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def _windows_evalplus_worker(
    dataset: DatasetName,
    entry_point: str,
    code: str,
    inputs: object,
    expected: object,
    time_limits: list[float],
    atol: object,
    fast_check: bool,
    status: object,
    details: object,
    progress: object,
    supervision: object,
) -> None:
    import evalplus.eval as eval_module
    from evalplus.eval import utils as eval_utils

    def reliability_guard_without_resource(maximum_memory_bytes: int | None = None) -> None:
        del maximum_memory_bytes
        eval_utils.reliability_guard(None)

    @contextmanager
    def supervised_time_limit(seconds: float):
        deadline = time.perf_counter() + float(seconds)
        supervision.send(("start", deadline))  # type: ignore[attr-defined]
        try:
            yield
        finally:
            supervision.send(("end", time.perf_counter()))  # type: ignore[attr-defined]

    eval_module.reliability_guard = reliability_guard_without_resource
    eval_module.time_limit = supervised_time_limit
    try:
        eval_module.unsafe_execute(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            status,
            details,
            progress,
        )
    finally:
        supervision.close()  # type: ignore[attr-defined]


def _default_loader(dataset: str) -> Mapping[str, Mapping[str, object]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


def _evalplus_problem_set(dataset: DatasetName) -> tuple[Mapping[str, Mapping[str, object]], object]:
    from evalplus.evaluate import get_groundtruth

    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash

        problems = get_human_eval_plus()
        return problems, get_groundtruth(problems, get_human_eval_plus_hash(), [])
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    problems = get_mbpp_plus()
    return problems, get_groundtruth(
        problems,
        get_mbpp_plus_hash(),
        MBPP_OUTPUT_NOT_NONE_TASKS,
    )


def _read_samples(path: Path) -> dict[str, str]:
    """Read the samples."""
    samples: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            task_id = _required_str(value.get("task_id"), "task_id")
            solution = _required_str(value.get("solution"), "solution")
            samples[task_id] = solution
    return samples


def _evalplus_outcome_passed(outcome: Mapping[str, object], pass_status: str) -> bool:
    return all(
        _test_group_passed(outcome.get(group_name), pass_status)
        for group_name in ("base", "plus")
    )


def _test_group_passed(group: object, pass_status: str) -> bool:
    """Test the group passed."""
    if isinstance(group, (list, tuple)) and group:
        first = group[0]
        if isinstance(first, str):
            return first == pass_status
        if isinstance(group, list):
            return all(_test_group_passed(item, pass_status) for item in group)
    return False


def _evalplus_failure_reason(outcome: Mapping[str, object], pass_status: str) -> str:
    failures: list[str] = []
    for group_name in ("base", "plus"):
        status, count, first = _group_summary(outcome.get(group_name), pass_status)
        if status == pass_status:
            continue
        details = [f"status={status}"]
        if count:
            details.append(f"failed_tests={count}")
        if first is not None:
            details.append(f"first_failed_index={first}")
        failures.append(f"{group_name} ({', '.join(details)})")
    return "EvalPlus failed: " + ", ".join(failures or ["unknown"])


def _group_summary(
    group: object,
    pass_status: str,
) -> tuple[str, int, int | None]:
    if not isinstance(group, (list, tuple)) or not group:
        return "unknown", 0, None
    first = group[0]
    if not isinstance(first, str):
        nested = [_group_summary(item, pass_status) for item in group]
        failed = [item for item in nested if item[0] != pass_status]
        return (
            pass_status if not failed else "fail",
            sum(item[1] for item in failed),
            min(
                (item[2] for item in failed if item[2] is not None),
                default=None,
            ),
        )
    details = group[1] if len(group) > 1 else None
    if not isinstance(details, (list, tuple)):
        return first, int(first != pass_status), 0 if first != pass_status else None
    failed_indices = [
        index
        for index, value in enumerate(details)
        if value is False or (isinstance(value, str) and value != pass_status)
    ]
    if first != pass_status and not failed_indices:
        failed_indices = [0]
    return first, len(failed_indices), failed_indices[0] if failed_indices else None


def _write_evalplus_results(
    path: Path,
    dataset: DatasetName,
    results: dict[str, EvalPlusCaseEvaluation],
) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "results": {
                    task_id: result.to_json()
                    for task_id, result in sorted(results.items())
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _required_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value
