from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.tasks.models import TaskRecord, TaskStatus
from litecoder.tasks.planning import MissingTaskDependency, PlanningView, TaskCycleError
from litecoder.tasks.store import TaskStore


def task(task_id: str, dependencies: list[str] | None = None) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        subject=f"Subject {task_id}",
        description=f"Description {task_id}",
        status=TaskStatus.PENDING,
        dependencies=[] if dependencies is None else dependencies,
    )


def test_planning_view_orders_dependencies_first() -> None:
    tasks = [task("implement", ["test"]), task("test")]

    assert [item.id for item in PlanningView.ordered_tasks(tasks)] == [
        "test",
        "implement",
    ]


def test_planning_view_rejects_missing_dependency() -> None:
    with pytest.raises(MissingTaskDependency) as error:
        PlanningView.ordered_tasks([task("implement", ["missing"])])

    assert error.value.task_id == "implement"
    assert error.value.dependency_id == "missing"


def test_planning_view_reports_complete_cycle_path() -> None:
    tasks = [
        task("implement", ["test"]),
        task("test", ["review"]),
        task("review", ["implement"]),
    ]

    with pytest.raises(TaskCycleError) as error:
        PlanningView.ordered_tasks(tasks)

    assert error.value.path == ["implement", "test", "review", "implement"]


def test_task_store_round_trips_one_json_file_per_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")

    store.replace_many([task("implement", ["test"]), task("test")])

    assert sorted(path.name for path in store.root.iterdir()) == [
        "implement.json",
        "test.json",
    ]
    assert [item.id for item in store.read_all()] == ["implement", "test"]
    assert store.read("implement").dependencies == ["test"]


def test_task_store_default_load_rejects_missing_dependency(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    store.replace_many([task("implement", ["missing"])])

    with pytest.raises(MissingTaskDependency) as error:
        store.read_all()

    assert error.value.task_id == "implement"
    assert error.value.dependency_id == "missing"

def test_task_store_startup_validation_rejects_corrupt_graph(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    store.replace_many([task("A"), task("B")])
    (store.root / "A.json").write_text(
        '{"schema_version":1,"id":"A","subject":"A","description":"A",'
        '"status":"pending","owner_agent_id":null,"dependencies":["B"],'
        '"worktree_id":null}',
        encoding="utf-8",
    )
    (store.root / "B.json").write_text(
        '{"schema_version":1,"id":"B","subject":"B","description":"B",'
        '"status":"pending","owner_agent_id":null,"dependencies":["A"],'
        '"worktree_id":null}',
        encoding="utf-8",
    )

    with pytest.raises(TaskCycleError) as error:
        store.read_all(validate_graph=True)

    assert error.value.path == ["A", "B", "A"]