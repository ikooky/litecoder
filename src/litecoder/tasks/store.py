"""Durable storage operations for the surrounding subsystem."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from litecoder.tasks.models import TaskRecord, validate_task_id
from litecoder.tasks.planning import PlanningView


TASK_FILE_MAX_BYTES = 65_536


class TaskStore:
    """Storage interface for the task store."""
    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ValueError("task root must be a Path")
        self.root = root

    def read(self, task_id: str) -> TaskRecord:
        """Read the requested data."""
        validate_task_id(task_id)
        return self._read_file(self._path_for(task_id))

    def read_all(self, *, validate_graph: bool = True) -> list[TaskRecord]:
        """Read the all."""
        if not self.root.exists():
            return []
        records = [
            self._read_file(path)
            for path in sorted(self.root.glob("*.json"), key=lambda item: item.name)
        ]
        if validate_graph:
            PlanningView.ordered_tasks(records)
        return records

    def write(self, record: TaskRecord) -> None:
        """Write the supplied data."""
        self.replace_many([record], remove_unmentioned=False)

    def replace_many(
        self,
        records: Iterable[TaskRecord],
        *,
        remove_unmentioned: bool = False,
    ) -> None:
        """Handle the replace many operation."""
        items = tuple(records)
        seen: set[str] = set()
        for record in items:
            if not isinstance(record, TaskRecord):
                raise ValueError("task record is invalid")
            if record.id in seen:
                raise ValueError("duplicate task id")
            seen.add(record.id)
        self.root.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        try:
            for record in items:
                staged.append((self._write_temp(record), self._path_for(record.id)))
            for source, target in staged:
                os.replace(source, target)
            if remove_unmentioned:
                for path in self.root.glob("*.json"):
                    if path.stem not in seen:
                        path.unlink()
        except OSError as error:
            raise ValueError("Task store is unavailable") from error
        finally:
            for source, _ in staged:
                try:
                    if source.exists():
                        source.unlink()
                except OSError:
                    pass

    def _path_for(self, task_id: str) -> Path:
        validate_task_id(task_id)
        return self.root / f"{task_id}.json"

    def _read_file(self, path: Path) -> TaskRecord:
        """Read the file."""
        try:
            resolved_root = self.root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if not resolved.is_file():
                raise ValueError
            with resolved.open("rb") as handle:
                raw = handle.read(TASK_FILE_MAX_BYTES + 1)
            if len(raw) > TASK_FILE_MAX_BYTES or b"\x00" in raw:
                raise ValueError
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            record = TaskRecord.from_json(data)
            if self._path_for(record.id).name != path.name:
                raise ValueError
            return record
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ValueError("Task store is unavailable") from None

    def _write_temp(self, record: TaskRecord) -> Path:
        """Write the temp."""
        raw = json.dumps(
            record.to_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(raw) > TASK_FILE_MAX_BYTES:
            raise ValueError("task record is invalid")
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".task-", suffix=".tmp", dir=self.root
        )
        path = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
            return path
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise