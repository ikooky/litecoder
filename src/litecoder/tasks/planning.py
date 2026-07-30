"""Task planning and cycle errors."""

from __future__ import annotations

from litecoder.tasks.models import TaskRecord


class TaskCycleError(ValueError):
    """Raised when the task cycle error conditions occur."""
    def __init__(self, path: list[str]) -> None:
        super().__init__("task dependency cycle: " + " -> ".join(path))
        self.path = path


class MissingTaskDependency(ValueError):
    """Component responsible for the missing task dependency."""
    def __init__(self, task_id: str, dependency_id: str) -> None:
        super().__init__(f"task {task_id!r} depends on missing task {dependency_id!r}")
        self.task_id = task_id
        self.dependency_id = dependency_id


class PlanningView:
    """Component responsible for the planning view."""
    @staticmethod
    def ordered_tasks(tasks: list[TaskRecord]) -> list[TaskRecord]:
        """Return tasks in dependency order."""
        by_id: dict[str, TaskRecord] = {}
        for record in tasks:
            if record.id in by_id:
                raise ValueError("duplicate task id")
            by_id[record.id] = record

        ordered: list[TaskRecord] = []
        state: dict[str, int] = {}
        path: list[str] = []

        def visit(task_id: str, requester: str | None = None) -> None:
            """Handle the visit operation."""
            if task_id not in by_id:
                if requester is None:
                    raise MissingTaskDependency(task_id, task_id)
                raise MissingTaskDependency(requester, task_id)
            marker = state.get(task_id, 0)
            if marker == 1:
                start = path.index(task_id)
                raise TaskCycleError([*path[start:], task_id])
            if marker == 2:
                return
            state[task_id] = 1
            path.append(task_id)
            for dependency_id in by_id[task_id].dependencies:
                visit(dependency_id, task_id)
            path.pop()
            state[task_id] = 2
            ordered.append(by_id[task_id])

        for record in tasks:
            visit(record.id)
        return ordered