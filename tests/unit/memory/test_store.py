from __future__ import annotations

import asyncio
import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import litecoder.memory.store as store_module
from litecoder.common.locks import NamedFileLock, ResourceLockUnavailable
from litecoder.common.trace import SecretRedactor
from litecoder.memory.models import MemoryEntry
from litecoder.memory.store import (
    MEMORY_FILE_MAX_BYTES,
    MemoryStore,
    MemoryWriteResult,
    write_memory_file,
    write_memory_files,
)


def test_empty_replacement_does_not_create_memory_paths(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")

    store.replace_all(())

    assert not store.root.exists()
    assert store.index_exists() is False


def test_empty_update_does_not_create_memory_paths(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")

    store.update(lambda entries: entries)

    assert not store.root.exists()


@pytest.mark.asyncio
async def test_empty_async_update_does_not_create_memory_paths(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")

    await store.update_async(lambda entries: entries)

    assert not store.root.exists()

def test_empty_write_memory_files_returns_zero_without_paths(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")

    result = write_memory_files(store, SecretRedactor.with_values(()), ())

    assert result == MemoryWriteResult((), 0)
    assert not store.root.exists()


def test_write_memory_files_reports_paths_and_resulting_total(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    redactor = SecretRedactor.with_values(())

    first = write_memory_files(
        store,
        redactor,
        (MemoryEntry("first", "First memory", "project", "First body."),),
    )
    second = write_memory_files(
        store,
        redactor,
        (
            MemoryEntry("first", "Updated first", "project", "Updated body."),
            MemoryEntry("second", "Second memory", "user", "Second body."),
        ),
    )

    assert first == MemoryWriteResult((store.root / "first.md",), 1)
    assert second == MemoryWriteResult(
        (store.root / "first.md", store.root / "second.md"),
        2,
    )


def test_first_successful_write_installs_entry_and_index_together(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")

    path = write_memory_file(
        store,
        SecretRedactor.with_values(()),
        name="user-preference-tabs",
        memory_type="user",
        description="User prefers tabs for indentation",
        body="Use tabs, not spaces.",
    )

    assert path == store.root / "user-preference-tabs.md"
    assert store.index_exists() is True
    assert sorted(item.name for item in store.root.iterdir()) == [
        "MEMORY.md",
        "user-preference-tabs.md",
    ]
    assert store.read("user-preference-tabs").body == "Use tabs, not spaces."


def test_failed_first_directory_install_leaves_memory_root_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")

    real_replace = store_module.os.replace

    def dispatch_replace(source: object, target: object) -> None:
        if Path(target) == store.root:
            raise OSError("captured initial install failure")
        real_replace(source, target)

    monkeypatch.setattr(store_module.os, "replace", dispatch_replace)

    with pytest.raises(ValueError, match="Memory is unavailable"):
        write_memory_file(
            store,
            SecretRedactor.with_values(()),
            name="first",
            memory_type="project",
            description="First memory",
            body="Must not become partially visible.",
        )

    assert not store.root.exists()
    assert not list(tmp_path.glob(".memory-*.stage"))


def test_index_exists_requires_memory_index_file(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.root.mkdir()
    (store.root / "orphan.md").write_text("orphan", encoding="utf-8")

    assert store.index_exists() is False


def test_write_memory_file_rebuilds_link_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")

    path = write_memory_file(
        store,
        SecretRedactor.with_values(()),
        name="user-preference-tabs",
        memory_type="user",
        description="User prefers tabs for indentation",
        body="Use tabs, not spaces.",
    )

    assert path == store.root / "user-preference-tabs.md"
    assert store.read("user-preference-tabs").body == "Use tabs, not spaces."
    assert store.read_index() == (
        "- [user-preference-tabs](user-preference-tabs.md) "
        "- User prefers tabs for indentation\n"
    )


@pytest.mark.parametrize("memory_type", ["user", "feedback", "project", "reference"])
def test_all_reference_memory_types_are_supported(
    tmp_path: Path, memory_type: str
) -> None:
    store = MemoryStore(tmp_path / ".memory")

    write_memory_file(
        store,
        SecretRedactor.with_values(()),
        name=f"entry-{memory_type}",
        memory_type=memory_type,
        description=f"A {memory_type} memory",
        body="Durable body.",
    )

    assert store.read(f"entry-{memory_type}").type == memory_type


def test_update_exposes_snapshot_and_rebuilds_all_managed_files(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("old", "Old memory", "project", "obsolete")])
    observed: list[tuple[MemoryEntry, ...]] = []

    def transform(current: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
        observed.append(current)
        return (MemoryEntry("new", "New memory", "project", "durable"),)

    store.update(transform)

    assert observed == [(MemoryEntry("old", "Old memory", "project", "obsolete"),)]
    assert store.snapshot().entries == (
        MemoryEntry("new", "New memory", "project", "durable"),
    )
    assert store.snapshot().index == "- [new](new.md) - New memory\n"
    assert not (store.root / "old.md").exists()


def test_snapshot_reads_index_and_entries_under_the_file_lock(tmp_path: Path) -> None:
    lock = RecordingMemoryLock()
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("entry", "Entry", "user", "body")])
    store.file_lock = lock  # type: ignore[assignment]

    snapshot = store.snapshot()

    assert lock.acquired == 1
    assert snapshot == store.snapshot().__class__(
        "- [entry](entry.md) - Entry\n",
        (MemoryEntry("entry", "Entry", "user", "body"),),
    )


def test_read_index_normalizes_legacy_unicode_separator(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.root.mkdir()
    store.root.joinpath("MEMORY.md").write_bytes(
        "- [entry](entry.md) \u2014 Legacy entry\n".encode("utf-8")
    )

    assert store.read_index() == "- [entry](entry.md) - Legacy entry\n"


def test_read_uses_a_bounded_read_for_oversized_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.root.mkdir()
    target = store.root / "entry.md"
    raw = b"x" * (MEMORY_FILE_MAX_BYTES + 1)
    target.write_bytes(raw)
    sizes: list[int] = []

    class TrackingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            sizes.append(size)
            return super().read(size)

    def bounded_open(path: Path, mode: str = "r", **_: object) -> TrackingReader:
        assert path == target
        assert mode == "rb"
        return TrackingReader(raw)

    monkeypatch.setattr(Path, "open", bounded_open)

    with pytest.raises(ValueError, match="Memory is unavailable"):
        store.read("entry")

    assert sizes == [MEMORY_FILE_MAX_BYTES + 1]


def test_invalid_replacement_does_not_remove_existing_memories(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("kept", "Kept memory", "user", "body")])

    with pytest.raises(ValueError, match="duplicate"):
        store.replace_all(
            [
                MemoryEntry("duplicate", "First", "user", "one"),
                MemoryEntry("duplicate", "Second", "user", "two"),
            ]
        )

    assert store.read("kept").body == "body"


@pytest.mark.parametrize("reserved_name", ["memory", "Memory", "MEMORY"])
def test_reserved_index_name_write_attempts_preserve_snapshot(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("kept", "Kept memory", "project", "body")])
    before = store.snapshot()
    before_files = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
    }

    def replacement_entries():
        yield MemoryEntry("replacement", "Replacement", "project", "new")
        yield MemoryEntry(reserved_name, "Reserved index", "project", "bad")

    with pytest.raises(ValueError, match="memory name is invalid"):
        store.replace_all(replacement_entries())

    assert store.snapshot() == before
    assert {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
    } == before_files

    with pytest.raises(ValueError, match="memory rejected"):
        write_memory_file(
            store,
            SecretRedactor.with_values(()),
            name=reserved_name,
            memory_type="project",
            description="Reserved index",
            body="bad",
        )

    assert store.snapshot() == before
    assert {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
    } == before_files
    assert "(memory.md)" not in store.read_index().casefold()


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_replace_all_rolls_back_every_forward_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_call: int,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all(
        [
            MemoryEntry("a", "Old A", "project", "old a"),
            MemoryEntry("b", "Old B", "project", "old b"),
        ]
    )
    before = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
        if path.is_file()
    }
    real_replace = store_module.os.replace
    calls = 0

    def fail_once(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("captured forward failure")
        real_replace(source, target)

    monkeypatch.setattr(store_module.os, "replace", fail_once)

    with pytest.raises(ValueError, match="Memory is unavailable"):
        store.replace_all(
            [
                MemoryEntry("b", "New B", "project", "new b"),
                MemoryEntry("c", "New C", "project", "new c"),
            ]
        )

    after = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not any(path.name.startswith(".memory-") for path in store.root.iterdir())

def test_replace_all_rolls_back_obsolete_unlink_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all(
        [
            MemoryEntry("a", "Old A", "project", "old a"),
            MemoryEntry("b", "Old B", "project", "old b"),
        ]
    )
    before = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
        if path.is_file()
    }
    real_unlink = Path.unlink
    failed = False

    def fail_once(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed
        if path.name == "a.md" and not failed:
            failed = True
            raise OSError("captured obsolete unlink failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_once)

    with pytest.raises(ValueError, match="Memory is unavailable"):
        store.replace_all(
            [
                MemoryEntry("b", "New B", "project", "new b"),
                MemoryEntry("c", "New C", "project", "new c"),
            ]
        )

    after = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not any(path.name.startswith(".memory-") for path in store.root.iterdir())


@pytest.mark.parametrize("failure_call", [2, 4], ids=["staging", "backup"])
def test_replace_all_failure_before_visible_changes_preserves_old_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_call: int,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all(
        [
            MemoryEntry("a", "Old A", "project", "old a"),
            MemoryEntry("b", "Old B", "project", "old b"),
        ]
    )
    before = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
        if path.is_file()
    }
    real_mkstemp = store_module.tempfile.mkstemp
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("captured pre-visible failure")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(store_module.tempfile, "mkstemp", fail_once)

    with pytest.raises(ValueError, match="Memory is unavailable"):
        store.replace_all(
            [
                MemoryEntry("b", "New B", "project", "new b"),
                MemoryEntry("c", "New C", "project", "new c"),
            ]
        )

    after = {
        path.name: path.read_bytes()
        for path in store.root.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not any(path.name.startswith(".memory-") for path in store.root.iterdir())


@pytest.mark.parametrize("operation", ["scan", "read", "read_index", "snapshot"])
def test_public_reads_acquire_exactly_one_named_lock(
    tmp_path: Path,
    operation: str,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("entry", "Entry", "user", "body")])
    lock = RecordingMemoryLock()
    store.file_lock = lock  # type: ignore[assignment]

    if operation == "read":
        store.read("entry")
    else:
        getattr(store, operation)()

    assert lock.acquired == 1


def test_scan_skips_bad_files_but_direct_bad_read_still_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("good", "Good", "project", "usable")])
    (store.root / "malformed.md").write_bytes(b"\xff")
    (store.root / "oversized.md").write_bytes(b"x" * (MEMORY_FILE_MAX_BYTES + 1))
    (store.root / "wrong.md").write_text(
        MemoryEntry("other", "Other", "project", "wrong filename").render(),
        encoding="utf-8",
    )
    (store.root / "linked.md").write_text(
        MemoryEntry("linked", "Linked", "project", "linked body").render(),
        encoding="utf-8",
    )
    disappearing = store.root / "disappearing.md"
    disappearing.write_text(
        MemoryEntry("disappearing", "Disappearing", "project", "gone").render(),
        encoding="utf-8",
    )
    real_reject_link = store_module._reject_link
    real_read_entry = store._read_entry

    def reject_selected_link(path: Path) -> None:
        if path.name == "linked.md":
            raise ValueError("simulated symlink or reparse point")
        real_reject_link(path)

    def remove_before_read(path: Path) -> MemoryEntry:
        if path.name == "disappearing.md":
            path.unlink()
        return real_read_entry(path)

    monkeypatch.setattr(store_module, "_reject_link", reject_selected_link)
    monkeypatch.setattr(store, "_read_entry", remove_before_read)

    assert [item.name for item in store.scan()] == ["good"]
    with pytest.raises(ValueError, match="Memory is unavailable"):
        store.read("wrong")


def test_replace_all_expected_snapshot_conflict_performs_no_write(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "locks"
    first = MemoryStore(
        tmp_path / ".memory",
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    first.replace_all([MemoryEntry("a", "Old A", "project", "old a")])
    expected = first.snapshot()
    second = MemoryStore(
        first.root,
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    second.replace_all(
        [
            MemoryEntry("a", "Old A", "project", "old a"),
            MemoryEntry("concurrent", "Concurrent", "project", "preserve me"),
        ]
    )
    before = second.snapshot()

    with pytest.raises(ValueError, match="conflict") as error:
        first.replace_all(
            [MemoryEntry("replacement", "Replacement", "project", "new")],
            expected=expected,
        )

    assert type(error.value).__name__ == "MemoryConflictError"
    assert second.snapshot() == before
    assert not any(path.name.startswith(".memory-") for path in first.root.iterdir())


def test_compare_and_swap_requires_named_lock(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all([MemoryEntry("a", "Old A", "project", "old a")])
    expected = store.snapshot()

    with pytest.raises(ValueError, match="named lock"):
        store.replace_all(
            [MemoryEntry("replacement", "Replacement", "project", "new")],
            expected=expected,
        )

    assert store.snapshot() == expected


def test_named_lock_serializes_cas_compare_and_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / ".memory"
    lock_root = tmp_path / "locks"
    first = MemoryStore(
        root,
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    second = MemoryStore(
        root,
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    first.replace_all([MemoryEntry("a", "Old A", "project", "old a")])
    expected = first.snapshot()
    compared = threading.Event()
    release_cas = threading.Event()
    concurrent_done = threading.Event()
    failures: list[BaseException] = []
    real_snapshot = first._snapshot_locked

    def pause_after_compare():
        snapshot = real_snapshot()
        compared.set()
        release_cas.wait(timeout=2)
        return snapshot

    monkeypatch.setattr(first, "_snapshot_locked", pause_after_compare)

    def cas_write() -> None:
        try:
            first.replace_all(
                [MemoryEntry("replacement", "Replacement", "project", "new")],
                expected=expected,
            )
        except BaseException as error:
            failures.append(error)

    def concurrent_write() -> None:
        try:
            second.update(
                lambda entries: (
                    *entries,
                    MemoryEntry(
                        "concurrent",
                        "Concurrent",
                        "project",
                        "preserve me",
                    ),
                )
            )
        except BaseException as error:
            failures.append(error)
        finally:
            concurrent_done.set()

    cas_thread = threading.Thread(target=cas_write)
    concurrent_thread = threading.Thread(target=concurrent_write)
    cas_thread.start()
    try:
        assert compared.wait(timeout=1)
        concurrent_thread.start()
        assert not concurrent_done.wait(timeout=0.1)
    finally:
        release_cas.set()
        cas_thread.join(timeout=2)
        concurrent_thread.join(timeout=2)

    assert failures == []
    assert [entry.name for entry in first.snapshot().entries] == [
        "concurrent",
        "replacement",
    ]


def test_full_snapshot_and_replacement_include_entries_beyond_scan_cap(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "locks"
    store = MemoryStore(
        tmp_path / ".memory",
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    store.replace_all((
        MemoryEntry("memory-000", "Memory 0", "project", "Durable fact 0."),
    ))
    for index in range(1, 201):
        entry = MemoryEntry(
            f"memory-{index:03d}",
            f"Memory {index}",
            "project",
            f"Durable fact {index}.",
        )
        (store.root / entry.filename).write_text(entry.render(), encoding="utf-8")

    assert len(store.scan()) == 200
    expected = store.snapshot()
    assert len(expected.entries) == 201

    changed = MemoryEntry(
        "memory-200",
        "Memory 200",
        "project",
        "Changed durable fact.",
    )
    (store.root / changed.filename).write_text(changed.render(), encoding="utf-8")
    with pytest.raises(ValueError, match="conflict"):
        store.replace_all((), expected=expected)

    current = store.snapshot()
    assert len(current.entries) == 201
    store.replace_all((), expected=current)
    assert store.snapshot().entries == ()
    assert list(store.root.glob("memory-*.md")) == []

def test_writer_holds_named_lock_until_reader_can_see_complete_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / ".memory"
    lock_root = tmp_path / "locks"
    writer = MemoryStore(
        root,
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    reader = MemoryStore(
        root,
        file_lock=NamedFileLock.memory("project", lock_root),
    )
    writer.replace_all(
        [
            MemoryEntry("a", "Old A", "project", "old a"),
            MemoryEntry("b", "Old B", "project", "old b"),
        ]
    )
    first_visible_replace = threading.Event()
    release_writer = threading.Event()
    reader_done = threading.Event()
    writer_errors: list[BaseException] = []
    observed: list[tuple[str, ...]] = []
    real_replace = store_module.os.replace
    paused = False

    def pause_after_first_visible_replace(source: object, target: object) -> None:
        nonlocal paused
        real_replace(source, target)
        target_path = Path(target)
        if target_path.parent == root and not paused:
            paused = True
            first_visible_replace.set()
            release_writer.wait(timeout=2)

    monkeypatch.setattr(store_module.os, "replace", pause_after_first_visible_replace)

    def write() -> None:
        try:
            writer.replace_all(
                [
                    MemoryEntry("b", "New B", "project", "new b"),
                    MemoryEntry("c", "New C", "project", "new c"),
                ]
            )
        except BaseException as error:
            writer_errors.append(error)

    def read() -> None:
        observed.append(tuple(item.name for item in reader.scan()))
        reader_done.set()

    writer_thread = threading.Thread(target=write)
    reader_thread = threading.Thread(target=read)
    writer_thread.start()
    try:
        assert first_visible_replace.wait(timeout=1)
        reader_thread.start()
        assert not reader_done.wait(timeout=0.1)
    finally:
        release_writer.set()
        writer_thread.join(timeout=2)
        reader_thread.join(timeout=2)

    assert writer_errors == []
    assert observed == [("b", "c")]

def test_write_memory_file_rejects_redacted_content(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")

    with pytest.raises(ValueError, match="memory rejected"):
        write_memory_file(
            store,
            SecretRedactor.with_values(("very-secret",)),
            name="unsafe",
            memory_type="user",
            description="Contains a secret",
            body="very-secret",
        )

    assert not store.root.exists()


class RecordingMemoryLock:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.acquired = 0

    def __enter__(self) -> object:
        self.acquired += 1
        if self.fail:
            raise ResourceLockUnavailable("memory", Path("memory.lock"))
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_mutations_use_memory_file_lock(tmp_path: Path) -> None:
    lock = RecordingMemoryLock()
    store = MemoryStore(tmp_path / ".memory", file_lock=lock)  # type: ignore[arg-type]

    store.replace_all([MemoryEntry("entry", "Entry", "user", "body")])

    assert lock.acquired == 1


@pytest.mark.asyncio
async def test_cancelled_async_gate_wait_releases_acquired_gate(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store._operation_gate.acquire()
    task = asyncio.create_task(store.update_async(lambda entries: entries))
    while not store._async_operation_lock.locked():
        await asyncio.sleep(0)

    task.cancel()
    store._operation_gate.release()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    reacquired = store._operation_gate.acquire(timeout=0.5)
    try:
        assert reacquired is True
    finally:
        if reacquired:
            store._operation_gate.release()
        else:
            store._operation_gate.release()

    await asyncio.wait_for(store.update_async(lambda entries: entries), timeout=1)
    assert not store.root.exists()

async def test_update_async_keeps_async_lock_until_cancelled_worker_finishes(
    tmp_path: Path,
) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_completed = threading.Event()

    class InspectingLock:
        released = False
        worker_completed_at_release = False

        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                self.released = True
                self.worker_completed_at_release = worker_completed.is_set()

    lock = InspectingLock()
    store = MemoryStore(tmp_path / ".memory")
    store.file_lock = lock  # type: ignore[assignment]

    def transform(entries: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
        worker_started.set()
        release_worker.wait(timeout=1)
        worker_completed.set()
        return entries

    updating = asyncio.create_task(store.update_async(transform))
    await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=1)
    updating.cancel()
    try:
        await asyncio.sleep(0)
        assert not updating.done()
        assert not lock.released

        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await updating

        assert lock.worker_completed_at_release
    finally:
        release_worker.set()


def test_case_only_memory_update_preserves_one_canonical_identity(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    store.replace_all((
        MemoryEntry("Foo", "Original", "project", "old body"),
    ))

    store.replace_all((
        MemoryEntry("foo", "Updated", "project", "new body"),
    ))

    memory_files = [
        path for path in store.root.glob("*.md") if path.name != "MEMORY.md"
    ]
    assert len(memory_files) == 1
    assert {path.name.casefold() for path in memory_files} == {"foo.md"}
    assert store.read("foo") == MemoryEntry(
        "foo", "Updated", "project", "new body"
    )
    assert [entry.name.casefold() for entry in store.snapshot().entries] == ["foo"]


@pytest.mark.parametrize("operation", ["scan", "snapshot"])
@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_reads_reject_missing_or_non_directory_memory_root(
    tmp_path: Path,
    root_kind: str,
    operation: str,
) -> None:
    root = tmp_path / ".memory"
    if root_kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    store = MemoryStore(root)

    with pytest.raises(ValueError, match="Memory is unavailable"):
        getattr(store, operation)()
