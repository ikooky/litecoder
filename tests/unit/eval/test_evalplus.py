from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from litecoder.eval import evalplus as evalplus_adapter
from litecoder.eval.evalplus import (
    EvalPlusSample,
    EvalPlusTask,
    EvalPlusUnavailable,
    build_sample,
    load_evalplus_tasks,
    write_samples_jsonl,
    evaluate_samples,
    _evalplus_outcome_passed,
)


def test_load_evalplus_tasks_uses_injected_loader_and_limit() -> None:
    raw = {
        "HumanEval/1": {
            "prompt": "def sub(a, b):\n",
            "entry_point": "sub",
        },
        "HumanEval/0": {
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
        },
    }

    tasks = load_evalplus_tasks("humaneval", limit=1, loader=lambda dataset: raw)

    assert tasks == (
        EvalPlusTask(
            task_id="HumanEval/0",
            prompt="def add(a, b):\n",
            entry_point="add",
            dataset="humaneval",
        ),
    )


def test_load_evalplus_tasks_samples_reproducibly_with_seed() -> None:
    raw_tasks = {
        f"HumanEval/{index}": {
            "prompt": f"def answer_{index}():\n",
            "entry_point": f"answer_{index}",
        }
        for index in range(10)
    }

    first = load_evalplus_tasks("humaneval", limit=4, seed=1234, loader=lambda _: raw_tasks)
    second = load_evalplus_tasks("humaneval", limit=4, seed=1234, loader=lambda _: raw_tasks)
    different = load_evalplus_tasks("humaneval", limit=4, seed=5678, loader=lambda _: raw_tasks)

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert [task.task_id for task in first] != [task.task_id for task in different]


def test_load_evalplus_tasks_rejects_non_evalplus_dataset() -> None:
    with pytest.raises(ValueError, match="Unsupported EvalPlus dataset"):
        load_evalplus_tasks("unsupported-dataset", loader=lambda dataset: {})


def test_load_evalplus_tasks_reports_missing_evalplus() -> None:
    def missing_loader(dataset: str):
        raise ModuleNotFoundError("evalplus")

    with pytest.raises(EvalPlusUnavailable, match="Install the eval extra"):
        load_evalplus_tasks("humaneval", loader=missing_loader)


def test_evaluate_samples_calls_evalplus_check_correctness_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id = "HumanEval/0"
    problem = {"prompt": "def answer():\n", "entry_point": "answer"}
    expected = {"base": [42], "plus": [42]}
    calls: list[tuple[object, ...]] = []

    evalplus_package = types.ModuleType("evalplus")
    eval_module = types.ModuleType("evalplus.eval")
    eval_module.PASS = "pass"
    data_module = types.ModuleType("evalplus.data")
    data_module.get_human_eval_plus = lambda: {task_id: problem}
    data_module.get_human_eval_plus_hash = lambda: "humaneval-hash"
    data_module.get_mbpp_plus = lambda: {}
    data_module.get_mbpp_plus_hash = lambda: "mbpp-hash"
    evaluate_module = types.ModuleType("evalplus.evaluate")

    def fake_get_groundtruth(problems: object, task_hash: str, subset: list[object]) -> dict[str, object]:
        assert problems == {task_id: problem}
        assert task_hash == "humaneval-hash"
        assert subset == []
        return {task_id: expected}

    def fake_check_correctness(
        dataset: str,
        completion_id: int,
        received_problem: object,
        solution: str,
        received_expected: object,
        *,
        base_only: bool,
        fast_check: bool,
        identifier: str,
    ) -> dict[str, object]:
        calls.append(
            (
                dataset,
                completion_id,
                received_problem,
                solution,
                received_expected,
                base_only,
                fast_check,
                identifier,
            )
        )
        return {"base": ("pass", [True]), "plus": ("pass", [True])}

    evaluate_module.get_groundtruth = fake_get_groundtruth
    evaluate_module.check_correctness = fake_check_correctness
    monkeypatch.setattr(
        evalplus_adapter,
        "_requires_windows_evalplus_adapter",
        lambda: False,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "evalplus", evalplus_package)
    monkeypatch.setitem(sys.modules, "evalplus.eval", eval_module)
    monkeypatch.setitem(sys.modules, "evalplus.data", data_module)
    monkeypatch.setitem(sys.modules, "evalplus.evaluate", evaluate_module)

    samples_path = write_samples_jsonl(
        tmp_path / "samples.jsonl",
        [EvalPlusSample(task_id, "def answer():\n    return 42\n")],
    )

    results = evaluate_samples("humaneval", samples_path, (task_id,))

    assert results[task_id].passed is True
    assert calls == [
        (
            "humaneval",
            0,
            problem,
            "def answer():\n    return 42\n",
            expected,
            False,
            True,
            task_id,
        )
    ]
    normalized = tmp_path / "evalplus-normalized-results.json"
    assert normalized.exists()

    normalized.unlink()
    results = evaluate_samples(
        "humaneval", samples_path, (task_id,), write_results=False
    )

    assert results[task_id].passed is True
    assert not normalized.exists()


def test_evaluate_samples_uses_windows_compatible_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("evalplus")
    task_id = "HumanEval/0"
    problem = {
        "task_id": task_id,
        "entry_point": "increment",
        "base_input": [[1], [4]],
        "plus_input": [[-1], [10]],
        "atol": 0,
    }
    expected = {
        "base": [2, 5],
        "base_time": [0.001, 0.001],
        "plus": [0, 11],
        "plus_time": [0.001, 0.001],
    }
    monkeypatch.setattr(
        evalplus_adapter,
        "_requires_windows_evalplus_adapter",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        evalplus_adapter,
        "_evalplus_problem_set",
        lambda dataset: ({task_id: problem}, {task_id: expected}),
    )
    samples_path = write_samples_jsonl(
        tmp_path / "samples.jsonl",
        [EvalPlusSample(task_id, "def increment(value):\n    return value + 1\n")],
    )

    results = evaluate_samples("humaneval", samples_path, (task_id,))

    assert results[task_id].passed is True


def test_windows_execution_enforces_each_input_time_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("evalplus")
    task_id = "HumanEval/0"
    problem = {
        "task_id": task_id,
        "entry_point": "increment",
        "base_input": [[1], [4]],
        "plus_input": [[-1], [10]],
        "atol": 0,
    }
    expected = {
        "base": [2, 5],
        "base_time": [0.001, 0.001],
        "plus": [0, 11],
        "plus_time": [0.001, 0.001],
    }
    monkeypatch.setattr(
        evalplus_adapter,
        "_requires_windows_evalplus_adapter",
        lambda: True,
    )
    monkeypatch.setattr(
        evalplus_adapter,
        "_evalplus_problem_set",
        lambda dataset: ({task_id: problem}, {task_id: expected}),
    )
    samples_path = write_samples_jsonl(
        tmp_path / "samples.jsonl",
        [
            EvalPlusSample(
                task_id,
                "import time\n"
                "def increment(value):\n"
                "    if value == 1:\n"
                "        time.sleep(1.25)\n"
                "    return value + 1\n",
            )
        ],
    )

    results = evaluate_samples("humaneval", samples_path, (task_id,))

    assert results[task_id].passed is False
    assert results[task_id].failure_reason.startswith(
        "EvalPlus failed: base (status="
    )
    assert results[task_id].failed_test_count >= 1


def test_windows_monitor_rejects_end_after_child_deadline() -> None:
    now = evalplus_adapter.time.perf_counter()

    class FakeProcess:
        def __init__(self) -> None:
            self.checks = 0

        def is_alive(self) -> bool:
            self.checks += 1
            return self.checks == 1

    class FakeSupervision:
        def __init__(self) -> None:
            self.events = [
                ("start", now - 0.1),
                ("end", now),
            ]

        def poll(self, timeout: float) -> bool:
            return bool(self.events)

        def recv(self) -> tuple[str, float]:
            return self.events.pop(0)

    result = evalplus_adapter._monitor_windows_evalplus_worker(
        FakeProcess(),
        FakeSupervision(),
        1.0,
    )

    assert result == (True, False)


def test_windows_monitor_rejects_broken_pipe_while_worker_is_alive() -> None:
    class FakeProcess:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            pass

    class BrokenSupervision:
        def poll(self, timeout: float) -> bool:
            raise OSError("pipe failed")

    with pytest.raises(OSError, match="worker was still running"):
        evalplus_adapter._monitor_windows_evalplus_worker(
            FakeProcess(),
            BrokenSupervision(),
            1.0,
        )


def test_windows_worker_start_failure_closes_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("evalplus")

    class FakeEndpoint:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def start(self) -> None:
            raise OSError("spawn failed")

    parent_endpoint = FakeEndpoint()
    worker_endpoint = FakeEndpoint()

    class FakeContext:
        def Value(self, kind: str, value: int) -> object:
            return types.SimpleNamespace(value=value)

        def Array(self, kind: str, values: list[bool]) -> list[bool]:
            return values

        def Pipe(self, *, duplex: bool) -> tuple[FakeEndpoint, FakeEndpoint]:
            return parent_endpoint, worker_endpoint

        def Process(self, *, target: object, args: object) -> FakeProcess:
            return FakeProcess()

    monkeypatch.setattr(
        evalplus_adapter.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )

    with pytest.raises(
        evalplus_adapter.EvalPlusExecutionError,
        match="failed to start",
    ):
        evalplus_adapter._windows_untrusted_check(
            "humaneval",
            "def answer():\n    return 42\n",
            [],
            "answer",
            [],
            0,
            [],
            fast_check=True,
        )

    assert parent_endpoint.closed is True
    assert worker_endpoint.closed is True


def test_evaluate_samples_propagates_execution_infrastructure_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id = "HumanEval/0"
    problem = {"prompt": "def answer():\n", "entry_point": "answer"}
    expected = {"base": [42], "plus": [42]}
    evalplus_package = types.ModuleType("evalplus")
    eval_module = types.ModuleType("evalplus.eval")
    eval_module.PASS = "pass"
    evaluate_module = types.ModuleType("evalplus.evaluate")
    evaluate_module.check_correctness = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "evalplus", evalplus_package)
    monkeypatch.setitem(sys.modules, "evalplus.eval", eval_module)
    monkeypatch.setitem(sys.modules, "evalplus.evaluate", evaluate_module)
    monkeypatch.setattr(
        evalplus_adapter,
        "_evalplus_problem_set",
        lambda dataset: ({task_id: problem}, {task_id: expected}),
    )

    class InfrastructureError(RuntimeError):
        pass

    def crash(*args: object, **kwargs: object) -> dict[str, object]:
        raise InfrastructureError("worker crashed")

    monkeypatch.setattr(
        evalplus_adapter,
        "EvalPlusExecutionError",
        InfrastructureError,
        raising=False,
    )
    monkeypatch.setattr(evalplus_adapter, "_check_evalplus_correctness", crash)
    samples_path = write_samples_jsonl(
        tmp_path / "samples.jsonl",
        [EvalPlusSample(task_id, "def answer():\n    return 42\n")],
    )

    with pytest.raises(InfrastructureError, match="worker crashed"):
        evaluate_samples("humaneval", samples_path, (task_id,))


def test_evalplus_outcome_parser_accepts_official_result_tuple() -> None:
    assert _evalplus_outcome_passed(
        {"base": ("pass", [True]), "plus": ("pass", [True])},
        "pass",
    ) is True


def test_evaluate_samples_uses_mbpp_output_not_none_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id = "Mbpp/1"
    problem = {"prompt": "def answer():\n", "entry_point": "answer"}
    expected = {"base": [1], "plus": [1]}
    special_tasks = {"answer"}
    captured: dict[str, object] = {}

    evalplus_package = types.ModuleType("evalplus")
    eval_module = types.ModuleType("evalplus.eval")
    eval_module.PASS = "pass"
    data_module = types.ModuleType("evalplus.data")
    data_module.get_human_eval_plus = lambda: {}
    data_module.get_human_eval_plus_hash = lambda: "humaneval-hash"
    data_module.get_mbpp_plus = lambda: {task_id: problem}
    data_module.get_mbpp_plus_hash = lambda: "mbpp-hash"
    evaluate_module = types.ModuleType("evalplus.evaluate")
    special_module = types.ModuleType("evalplus.eval._special_oracle")
    special_module.MBPP_OUTPUT_NOT_NONE_TASKS = special_tasks

    def fake_get_groundtruth(problems: object, task_hash: str, tasks_only_output_not_none: object) -> dict[str, object]:
        captured["tasks_only_output_not_none"] = tasks_only_output_not_none
        assert problems == {task_id: problem}
        assert task_hash == "mbpp-hash"
        return {task_id: expected}

    def fake_check_correctness(
        dataset: str,
        completion_id: int,
        received_problem: object,
        solution: str,
        received_expected: object,
        *,
        base_only: bool,
        fast_check: bool,
        identifier: str,
    ) -> dict[str, object]:
        return {"base": ("pass", [True]), "plus": ("pass", [True])}

    evaluate_module.get_groundtruth = fake_get_groundtruth
    evaluate_module.check_correctness = fake_check_correctness
    monkeypatch.setattr(
        evalplus_adapter,
        "_requires_windows_evalplus_adapter",
        lambda: False,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "evalplus", evalplus_package)
    monkeypatch.setitem(sys.modules, "evalplus.eval", eval_module)
    monkeypatch.setitem(sys.modules, "evalplus.data", data_module)
    monkeypatch.setitem(sys.modules, "evalplus.evaluate", evaluate_module)
    monkeypatch.setitem(sys.modules, "evalplus.eval._special_oracle", special_module)

    samples_path = write_samples_jsonl(
        tmp_path / "samples.jsonl",
        [EvalPlusSample(task_id, "def answer():\n    return 1\n")],
    )

    results = evaluate_samples("mbpp", samples_path, (task_id,))

    assert results[task_id].passed is True
    assert captured["tasks_only_output_not_none"] is special_tasks

def test_build_and_write_samples_jsonl(tmp_path: Path) -> None:
    task = EvalPlusTask(
        task_id="HumanEval/0",
        prompt="def add(a, b):\n",
        entry_point="add",
        dataset="humaneval",
    )
    sample = build_sample(task, "def add(a, b):\n    return a + b\n")
    path = write_samples_jsonl(tmp_path / "samples.jsonl", [sample])

    lines = path.read_text(encoding="utf-8").splitlines()

    assert sample == EvalPlusSample(
        "HumanEval/0",
        "def add(a, b):\n    return a + b\n",
    )
    assert json.loads(lines[0]) == {
        "task_id": "HumanEval/0",
        "solution": "def add(a, b):\n    return a + b\n",
    }
