"""Data models for the surrounding subsystem."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class TaskStatus(StrEnum):
    """Enumeration of the task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS = {
    TaskStatus.PENDING: frozenset({
        TaskStatus.IN_PROGRESS,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.IN_PROGRESS: frozenset({
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    }),
    TaskStatus.INTERRUPTED: frozenset({
        TaskStatus.PENDING,
        TaskStatus.IN_PROGRESS,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.FAILED: frozenset({
        TaskStatus.PENDING,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.CANCELLED: frozenset({TaskStatus.PENDING}),
}


def allows_task_transition(
    current: TaskStatus,
    target: TaskStatus,
) -> bool:
    """Handle the allows task transition operation."""
    return target in _ALLOWED_TRANSITIONS[TaskStatus(current)]


@dataclass(slots=True)
class TaskRecord:
    """Data model representing the task record."""
    id: str
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    schema_version: int = 1
    owner_agent_id: str | None = None
    worktree_id: str | None = None

    def __post_init__(self) -> None:
        self.id = validate_task_id(self.id)
        self.subject = _non_empty(self.subject, "subject")
        self.description = _non_empty(self.description, "description")
        self.status = TaskStatus(self.status)
        if self.schema_version != 1:
            raise ValueError("unsupported task schema version")
        self.dependencies = [validate_task_id(value) for value in self.dependencies]
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("duplicate task dependency")
        if self.id in self.dependencies:
            raise ValueError("task cannot depend on itself")
        if self.owner_agent_id is not None:
            self.owner_agent_id = _non_empty(self.owner_agent_id, "owner_agent_id")
        if self.worktree_id is not None:
            self.worktree_id = _non_empty(self.worktree_id, "worktree_id")

    def to_json(self) -> dict[str, Any]:
        """Convert this object to a JSON-compatible value."""
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status.value,
            "owner_agent_id": self.owner_agent_id,
            "dependencies": list(self.dependencies),
            "worktree_id": self.worktree_id,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> TaskRecord:
        """Construct a value from json data."""
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise ValueError("task dependencies are invalid")
        owner = data.get("owner_agent_id")
        worktree = data.get("worktree_id")
        return cls(
            schema_version=_int_field(data.get("schema_version"), "schema_version"),
            id=_str_field(data.get("id"), "id"),
            subject=_str_field(data.get("subject"), "subject"),
            description=_str_field(data.get("description"), "description"),
            status=TaskStatus(_str_field(data.get("status"), "status")),
            owner_agent_id=None if owner is None else _str_field(owner, "owner_agent_id"),
            dependencies=list(dependencies),
            worktree_id=None if worktree is None else _str_field(worktree, "worktree_id"),
        )


@dataclass(frozen=True, slots=True)
class TaskCreate:
    """Data model representing the task create."""
    id: str
    subject: str
    description: str
    dependencies: tuple[str, ...] = ()
    owner_agent_id: str | None = None
    worktree_id: str | None = None

    def __post_init__(self) -> None:
        record = self.to_record()
        object.__setattr__(self, "id", record.id)
        object.__setattr__(self, "subject", record.subject)
        object.__setattr__(self, "description", record.description)
        object.__setattr__(
            self, "dependencies", tuple(record.dependencies)
        )
        object.__setattr__(
            self, "owner_agent_id", record.owner_agent_id
        )
        object.__setattr__(self, "worktree_id", record.worktree_id)

    def to_record(self) -> TaskRecord:
        """Convert this object to a record value."""
        return TaskRecord(
            id=self.id,
            subject=self.subject,
            description=self.description,
            dependencies=list(self.dependencies),
            owner_agent_id=self.owner_agent_id,
            worktree_id=self.worktree_id,
        )


def validate_task_id(value: object) -> str:
    """Validate the task id."""
    if not isinstance(value, str) or not _SAFE_TASK_ID.fullmatch(value):
        raise ValueError("task id is invalid")
    return value


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"task {field_name} is invalid")
    return value


def _str_field(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"task {field_name} is invalid")
    return value


def _int_field(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"task {field_name} is invalid")
    return value