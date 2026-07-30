from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from litecoder.context.todos import TodoItem, TodoService, TodoStatus
from litecoder.context.session.models import SessionRecord
from litecoder.context.session.store import SQLiteSessionStore


def _session(tmp_path: Path) -> SessionRecord:
    return SessionRecord.new(
        "session-1",
        "project-1",
        "workspace-1",
        "fake",
        "model",
        workspace_path=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_replace_todos_replaces_only_todos_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    record = _session(tmp_path)
    record.metadata = {
        "feature": {"enabled": True},
        "todos": [{"content": "old", "active_form": "Old", "status": "pending"}],
    }
    await store.create_session(record)

    replacement = [{"content": "new", "active_form": "Writing", "status": "in_progress"}]
    await store.replace_todos("session-1", replacement)

    restored = await store.load_context("session-1")
    assert restored.session.metadata == {
        "feature": {"enabled": True},
        "todos": replacement,
    }
    assert await store.list_todos("session-1") == replacement
    await store.close()


@pytest.mark.asyncio
async def test_todo_service_clears_completed_items_from_sqlite_active_state(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(_session(tmp_path))
    service = TodoService(store)
    completed = TodoItem("Plan", "Planning", TodoStatus.COMPLETED)

    try:
        assert await service.replace("session-1", [completed]) == (completed,)
        assert await service.list("session-1") == ()
        restored = await store.load_context("session-1")
    finally:
        await store.close()

    assert restored.session.metadata["todos"] == []


@pytest.mark.asyncio
async def test_replace_todos_is_atomic_when_metadata_update_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    record = _session(tmp_path)
    record.metadata = {"flag": "preserve", "todos": [{"content": "before"}]}
    await store.create_session(record)
