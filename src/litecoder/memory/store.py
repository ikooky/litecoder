"""Durable storage operations for the surrounding subsystem."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from litecoder.common.locks import NamedFileLock, ResourceLockUnavailable
from litecoder.common.trace import SecretRedactor
from litecoder.memory.models import (
    MemoryEntry,
    MemoryMetadata,
    MemorySnapshot,
    validate_memory_name,
)

_T = TypeVar("_T")
MEMORY_FILE_MAX_BYTES = 65_536
MEMORY_INDEX_MAX_BYTES = 25 * 1024
MEMORY_INDEX_MAX_ENTRIES = 200


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Data model representing the memory write result."""
    paths: tuple[Path, ...]
    total: int


class MemoryConflictError(ValueError):
    """Raised when a compare-and-swap replacement sees a changed store."""


class MemoryStore:
    """Storage interface for the memory store."""
    def __init__(
        self,
        root: Path,
        *,
        file_lock: NamedFileLock | None = None,
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("memory root must be a Path")
        self.root = root
        self.file_lock = file_lock
        self._operation_gate = threading.Lock()
        self._async_operation_lock = asyncio.Lock()

    def _with_file_lock(self, operation: Callable[[], _T]) -> _T:
        with self._operation_gate:
            if self.file_lock is None:
                return operation()
            try:
                with self.file_lock:
                    return operation()
            except ResourceLockUnavailable as error:
                raise ValueError("Memory is unavailable") from error

    def index_exists(self) -> bool:
        """Handle the index exists operation."""
        try:
            return self._with_file_lock(self._index_exists_locked)
        except (OSError, ValueError):
            return False

    def _index_exists_locked(self) -> bool:
        try:
            _reject_link(self.root)
            index = self.root / "MEMORY.md"
            _reject_link(index)
            return self.root.is_dir() and index.is_file()
        except (OSError, ValueError):
            return False

    def scan(self) -> tuple[MemoryMetadata, ...]:
        """Scan the managed memory store."""
        return self._with_file_lock(self._scan_locked)

    def snapshot(self) -> MemorySnapshot:
        """Return an immutable snapshot of the current state."""
        return self._with_file_lock(self._snapshot_locked)

    def read_index(self) -> str:
        """Read the index."""
        return self._with_file_lock(self._read_index_locked)

    def read(self, name: str) -> MemoryEntry:
        """Read the requested data."""
        return self._with_file_lock(lambda: self._read_locked(name))

    def replace_all(
        self,
        entries: Iterable[MemoryEntry],
        *,
        expected: MemorySnapshot | None = None,
    ) -> None:
        """Handle the replace all operation."""
        if expected is not None:
            if not isinstance(expected, MemorySnapshot):
                raise ValueError("expected memory snapshot is invalid")
            if not isinstance(self.file_lock, NamedFileLock):
                raise ValueError("compare-and-swap requires a named lock")
        items = _validate_entry_set(entries)

        def operation() -> None:
            if expected is not None and self._snapshot_locked() != expected:
                raise MemoryConflictError("Memory write conflict")
            self._replace_all_locked(items)

        self._with_file_lock(operation)

    def update(
        self,
        transform: Callable[[tuple[MemoryEntry, ...]], Iterable[MemoryEntry]],
    ) -> None:
        """Update the stored state."""
        def operation() -> None:
            entries = (
                self._entries_from_disk_locked()
                if self._index_exists_locked()
                else ()
            )
            updated = _validate_entry_set(transform(entries))
            self._replace_all_locked(updated)

        self._with_file_lock(operation)

    async def update_async(
        self,
        transform: Callable[[tuple[MemoryEntry, ...]], Iterable[MemoryEntry]],
    ) -> None:
        """Update the async."""
        def operation() -> None:
            entries = (
                self._entries_from_disk_locked()
                if self._index_exists_locked()
                else ()
            )
            updated = _validate_entry_set(transform(entries))
            self._replace_all_locked(updated)

        await self._run_async_file_lock(operation)

    def _scan_locked(self) -> tuple[MemoryMetadata, ...]:
        return tuple(
            entry.metadata()
            for entry in self._entries_from_disk_locked(
                limit=MEMORY_INDEX_MAX_ENTRIES
            )
        )

    def _snapshot_locked(self) -> MemorySnapshot:
        return MemorySnapshot(
            self._read_index_locked(),
            self._entries_from_disk_locked(),
        )

    def _read_index_locked(self) -> str:
        text = self._read_text(self.root / "MEMORY.md", MEMORY_INDEX_MAX_BYTES)
        return text.replace(") \u2014 ", ") - ")

    def _read_locked(self, name: str) -> MemoryEntry:
        try:
            validate_memory_name(name)
            entry = self._read_entry(self.root / f"{name}.md")
            if entry.name != name:
                raise ValueError
            return entry
        except ValueError:
            raise ValueError("Memory is unavailable") from None

    def _entries_from_disk_locked(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[MemoryEntry, ...]:
        try:
            _validate_memory_root(self.root)
            paths = sorted(
                self.root.glob("*.md"),
                key=lambda path: path.name.casefold(),
            )
        except (OSError, ValueError):
            raise ValueError("Memory is unavailable") from None

        entries: list[MemoryEntry] = []
        for path in paths:
            if path.name == "MEMORY.md":
                continue
            try:
                entry = self._read_entry(path)
                if path.name.casefold() != entry.filename.casefold():
                    raise ValueError
            except (OSError, ValueError):
                continue
            entries.append(entry)
            if limit is not None and len(entries) == limit:
                break
        return tuple(entries)

    def _replace_all_locked(self, items: tuple[MemoryEntry, ...]) -> None:
        if not self._index_exists_locked():
            if not items:
                return
            if not self.root.exists():
                self._install_initial_locked(items)
                return
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link(self.root)
        staged_entries: list[tuple[Path, Path]] = []
        staged_index: tuple[Path, Path] | None = None
        backups: dict[Path, Path | None] = {}
        temp_paths: list[Path] = []
        try:
            for entry in items:
                source = self._write_temp(
                    entry.render(),
                    MEMORY_FILE_MAX_BYTES,
                    suffix=".stage",
                )
                target = self.root / entry.filename
                staged_entries.append((source, target))
                temp_paths.append(source)
            index_source = self._write_temp(
                _render_index(items),
                MEMORY_INDEX_MAX_BYTES,
                suffix=".stage",
            )
            index_target = self.root / "MEMORY.md"
            staged_index = (index_source, index_target)
            temp_paths.append(index_source)

            desired_paths = {
                entry.filename.casefold(): self.root / entry.filename
                for entry in items
            }
            obsolete = []
            for entry in self._entries_from_disk_locked():
                current_path = self.root / entry.filename
                desired_path = desired_paths.get(entry.filename.casefold())
                if (
                    desired_path is None
                    or not _paths_share_identity(current_path, desired_path)
                ):
                    obsolete.append(current_path)
            touched = _unique_paths(
                [
                    *(target for _, target in staged_entries),
                    *obsolete,
                    index_target,
                ]
            )
            for target in touched:
                _validate_contained(target, self.root)
                _reject_link(target)
                if _path_exists(target):
                    backup = self._copy_backup(target)
                    backups[target] = backup
                    temp_paths.append(backup)
                else:
                    backups[target] = None

            try:
                for source, target in staged_entries:
                    os.replace(source, target)
                for path in obsolete:
                    path.unlink()
                os.replace(staged_index[0], staged_index[1])
            except OSError as error:
                self._rollback_locked(backups)
                raise ValueError("Memory is unavailable") from error
        except MemoryConflictError:
            raise
        except ValueError:
            raise
        except OSError as error:
            raise ValueError("Memory is unavailable") from error
        finally:
            for path in temp_paths:
                _cleanup_temp(path)

    def _install_initial_locked(self, items: tuple[MemoryEntry, ...]) -> None:
        parent = self.root.parent
        staged_root: Path | None = None
        try:
            _reject_link(parent)
            if not parent.is_dir():
                raise ValueError
            staged_root = Path(tempfile.mkdtemp(
                prefix=".memory-",
                suffix=".stage",
                dir=parent,
            ))
            for entry in items:
                self._write_new_text(
                    staged_root / entry.filename,
                    entry.render(),
                    MEMORY_FILE_MAX_BYTES,
                )
            self._write_new_text(
                staged_root / "MEMORY.md",
                _render_index(items),
                MEMORY_INDEX_MAX_BYTES,
            )
            _reject_link(self.root)
            os.replace(staged_root, self.root)
            staged_root = None
        except ValueError:
            raise
        except OSError as error:
            raise ValueError("Memory is unavailable") from error
        finally:
            if staged_root is not None:
                shutil.rmtree(staged_root, ignore_errors=True)

    def _write_new_text(self, target: Path, text: str, max_bytes: int) -> None:
        """Write the new text."""
        try:
            raw = text.encode("utf-8")
            if len(raw) > max_bytes or b"\x00" in raw:
                raise ValueError("memory entry is invalid")
            with target.open("xb") as handle:
                handle.write(raw)
        except UnicodeEncodeError as error:
            raise ValueError("memory entry is invalid") from error

    def _copy_backup(self, target: Path) -> Path:
        descriptor, filename = tempfile.mkstemp(
            prefix=".memory-",
            suffix=".backup",
            dir=self.root,
        )
        backup = Path(filename)
        try:
            os.close(descriptor)
            shutil.copyfile(target, backup)
            return backup
        except BaseException:
            _cleanup_temp(backup)
            raise

    def _rollback_locked(self, backups: dict[Path, Path | None]) -> None:
        for target, backup in backups.items():
            if backup is not None:
                continue
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        for target, backup in backups.items():
            if backup is None:
                continue
            try:
                os.replace(backup, target)
            except OSError:
                pass

    async def _run_async_file_lock(self, operation: Callable[[], None]) -> None:
        """Run the async file lock."""
        async with self._async_operation_lock:
            gate = asyncio.create_task(asyncio.to_thread(self._operation_gate.acquire))
            acquired = False
            try:
                try:
                    await _await_worker_completion(gate)
                except asyncio.CancelledError:
                    if gate.done() and not gate.cancelled():
                        gate.result()
                        acquired = True
                    raise
                acquired = True
                if self.file_lock is None:
                    worker = asyncio.create_task(asyncio.to_thread(operation))
                    await _await_worker_completion(worker)
                else:
                    try:
                        async with self.file_lock.acquired_async():
                            worker = asyncio.create_task(asyncio.to_thread(operation))
                            await _await_worker_completion(worker)
                    except ResourceLockUnavailable as error:
                        raise ValueError("Memory is unavailable") from error
            finally:
                if acquired:
                    self._operation_gate.release()

    def _read_entry(self, path: Path) -> MemoryEntry:
        return _parse_memory(self._read_text(path, MEMORY_FILE_MAX_BYTES))

    def _read_text(self, path: Path, max_bytes: int) -> str:
        """Read the text."""
        try:
            _reject_link(path)
            _reject_link(path.parent)
            root = self.root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                raise ValueError
            with resolved.open("rb") as handle:
                raw = handle.read(max_bytes + 1)
            if len(raw) > max_bytes or b"\x00" in raw:
                raise ValueError
            return raw.decode("utf-8")
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
            raise ValueError("Memory is unavailable") from None

    def _write_atomic(self, target: Path, text: str, max_bytes: int) -> None:
        source = self._write_temp(text, max_bytes, suffix=".stage")
        try:
            _validate_contained(target, self.root)
            _reject_link(target)
            os.replace(source, target)
        except OSError as error:
            raise ValueError("Memory is unavailable") from error
        finally:
            _cleanup_temp(source)

    def _write_temp(self, text: str, max_bytes: int, *, suffix: str) -> Path:
        """Write the temp."""
        raw = text.encode("utf-8")
        if len(raw) > max_bytes or b"\x00" in raw:
            raise ValueError("memory entry is invalid")
        descriptor, filename = tempfile.mkstemp(
            prefix=".memory-",
            suffix=suffix,
            dir=self.root,
        )
        path = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
            return path
        except BaseException:
            _cleanup_temp(path)
            raise


def write_memory_files(
    store: MemoryStore,
    redactor: SecretRedactor,
    entries: Iterable[MemoryEntry],
) -> MemoryWriteResult:
    """Write the memory files."""
    if not isinstance(store, MemoryStore) or not isinstance(redactor, SecretRedactor):
        raise ValueError("memory rejected")
    try:
        items = _validate_entry_set(entries)
    except ValueError as error:
        raise ValueError("memory rejected") from error

    for entry in items:
        rendered = entry.render()
        if len(rendered.encode("utf-8")) > MEMORY_FILE_MAX_BYTES:
            raise ValueError("memory rejected")
        if redactor.redact_text(rendered) != rendered:
            raise ValueError("memory rejected")
    if not items:
        return MemoryWriteResult((), 0)

    total = 0

    def update(current: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
        nonlocal total
        by_name = {item.name.casefold(): item for item in current}
        for entry in items:
            by_name[entry.name.casefold()] = entry
        updated = tuple(
            sorted(by_name.values(), key=lambda item: item.name.casefold())
        )
        total = len(updated)
        return updated

    store.update(update)
    return MemoryWriteResult(
        tuple(store.root / entry.filename for entry in items),
        total,
    )


def write_memory_file(
    store: MemoryStore,
    redactor: SecretRedactor,
    *,
    name: str,
    memory_type: str,
    description: str,
    body: str,
) -> Path:
    """Write the memory file."""
    try:
        entry = MemoryEntry(name, description, memory_type, body)
    except ValueError as error:
        raise ValueError("memory rejected") from error
    return write_memory_files(store, redactor, (entry,)).paths[0]
def _validate_entry_set(entries: Iterable[MemoryEntry]) -> tuple[MemoryEntry, ...]:
    """Validate the entry set."""
    try:
        items = tuple(entries)
    except TypeError as error:
        raise ValueError("memory entry is invalid") from error
    if len(items) > MEMORY_INDEX_MAX_ENTRIES:
        raise ValueError("memory entry limit exceeded")
    names: set[str] = set()
    for entry in items:
        if not isinstance(entry, MemoryEntry):
            raise ValueError("memory entry is invalid")
        key = entry.name.casefold()
        if key in names:
            raise ValueError("duplicate memory entry")
        names.add(key)
    return tuple(sorted(items, key=lambda entry: entry.name.casefold()))


def _render_index(entries: Iterable[MemoryEntry]) -> str:
    return "".join(
        f"- [{entry.name}]({entry.filename}) - {entry.description}\n"
        for entry in entries
    )


def _parse_memory(text: str) -> MemoryEntry:
    """Parse the memory."""
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        raise ValueError from None
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator or key.strip() in metadata:
            raise ValueError
        metadata[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1:]).lstrip("\n").rstrip("\n")
    return MemoryEntry(
        metadata.get("name", ""),
        metadata.get("description", ""),
        metadata.get("type", ""),
        body,
    )


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return tuple(result)


def _validate_memory_root(root: Path) -> None:
    """Validate the memory root."""
    try:
        _reject_link(root)
        if not root.exists() or not root.is_dir():
            raise ValueError
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Memory is unavailable") from None


def _path_identity(path: Path) -> str:
    return os.path.normcase(str(path))


def _paths_share_identity(first: Path, second: Path) -> bool:
    if _path_identity(first) == _path_identity(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _cleanup_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_contained(path: Path, root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Memory is unavailable") from None


def _reject_link(path: Path) -> None:
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("Memory is unavailable") from error
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag):
        raise ValueError("Memory is unavailable")


async def _await_worker_completion(worker: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(worker)
            break
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
            _clear_current_task_cancellation()
            if worker.done():
                worker.result()
                break
    if cancellation is not None:
        raise cancellation


def _clear_current_task_cancellation() -> None:
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling():
            task.uncancel()
