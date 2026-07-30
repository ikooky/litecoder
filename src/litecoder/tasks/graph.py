"""Supporting implementation for graph."""

from __future__ import annotations

from dataclasses import dataclass

from litecoder.tasks.models import TaskRecord, validate_task_id
from litecoder.tasks.planning import TaskCycleError


class MissingDependency(ValueError):
    """Component responsible for the missing dependency."""
    def __init__(self, task_id: str, dependency_id: str) -> None:
        super().__init__(
            f"task {task_id!r} depends on missing task {dependency_id!r}"
        )
        self.task_id = task_id
        self.dependency_id = dependency_id


@dataclass(frozen=True, slots=True)
class TaskGraph:
    """Data model representing the task graph."""
    edges: dict[str, tuple[str, ...]]

    @classmethod
    def from_edges(
        cls,
        edges: dict[str, list[str] | tuple[str, ...]],
    ) -> TaskGraph:
        """Construct a value from edges data."""
        normalized: dict[str, tuple[str, ...]] = {}
        seen: set[str] = set()
        for task_id, dependencies in edges.items():
            validate_task_id(task_id)
            key = task_id.casefold()
            if key in seen:
                raise ValueError("duplicate task id")
            seen.add(key)
            normalized[task_id] = tuple(
                validate_task_id(dependency_id)
                for dependency_id in dependencies
            )
        return cls(normalized)

    @classmethod
    def from_records(cls, records: list[TaskRecord]) -> TaskGraph:
        """Construct a value from records data."""
        return cls.from_edges({
            record.id: tuple(record.dependencies)
            for record in records
        })

    def validate_all(self) -> None:
        """Validate the all."""
        state: dict[str, int] = {}
        path: list[str] = []

        def visit(task_id: str, requester: str | None = None) -> None:
            """Handle the visit operation."""
            if task_id not in self.edges:
                raise MissingDependency(requester or task_id, task_id)
            marker = state.get(task_id, 0)
            if marker == 1:
                start = path.index(task_id)
                raise TaskCycleError([*path[start:], task_id])
            if marker == 2:
                return
            state[task_id] = 1
            path.append(task_id)
            for dependency_id in self.edges[task_id]:
                visit(dependency_id, task_id)
            path.pop()
            state[task_id] = 2

        for task_id in self.edges:
            visit(task_id)

    def validate_edge(self, task_id: str, dependency_id: str) -> None:
        """Validate the edge."""
        validate_task_id(task_id)
        validate_task_id(dependency_id)
        if task_id not in self.edges:
            raise MissingDependency(task_id, task_id)
        if dependency_id not in self.edges:
            raise MissingDependency(task_id, dependency_id)
        if task_id == dependency_id:
            raise TaskCycleError([task_id, task_id])
        path = self.find_path(dependency_id, task_id)
        if path is not None:
            raise TaskCycleError([task_id, *path])

    def find_path(self, start: str, target: str) -> list[str] | None:
        """Find the path."""
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node == target:
                return path
            dependencies = self.edges.get(node)
            if dependencies is None:
                continue
            for child in reversed(dependencies):
                if child not in path:
                    stack.append((child, [*path, child]))
        return None