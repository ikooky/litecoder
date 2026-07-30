from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

import litecoder.context.session.store as session_store
from litecoder.context.session.models import (
    MessageRecord,
    SessionRecord,
    SessionStatus,
)
from litecoder.context.session.store import SQLiteSessionStore


def _session(
    session_id: str,
    *,
    parent_session_id: str | None = None,
    workspace_path: str | None = None,
) -> SessionRecord:
    return SessionRecord.new(
        session_id,
        "project-1",
        "workspace-1",
        "anthropic",
        "model-1",
        parent_session_id=parent_session_id,
        workspace_path=workspace_path or f"/workspaces/{session_id}",
    )


@pytest.mark.asyncio
async def test_session_store_restores_all_session_fields_and_ordered_content_blocks(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path)
    await store.open()
    created_at = datetime(2026, 7, 12, 9, 30, tzinfo=timezone(timedelta(hours=8)))
    updated_at = datetime(2026, 7, 12, 10, 45, tzinfo=timezone(timedelta(hours=8)))
    session = SessionRecord.new(
        "session-1",
        "project-1",
        "workspace-1",
        "anthropic",
        "model-1",
        session_type="derived",
        title="Unicode 会话",
        workspace_path="C:/work/项目",
        status=SessionStatus.IDLE,
        parent_session_id=None,
        metadata={"nested": {"label": "保留"}},
        created_at=created_at,
        updated_at=updated_at,
    )
    await store.create_session(session)
    content = [
        {"type": "thinking", "thinking": "step", "unknown": {"rank": 1}},
        {"type": "text", "text": "hello"},
        {
            "type": "tool_use",
            "id": "tool-1",
            "input": {"path": "文件.txt", "flags": [True, None, 3.5]},
        },
    ]
    message_time = datetime(
        2026, 7, 12, 11, 0, tzinfo=timezone(timedelta(hours=-4))
    )
    await store.append_message(
        MessageRecord(
            session_id="session-1",
            role="assistant",
            content=content,
            token_count=17,
            created_at=message_time,
        )
    )
    await store.append_message(
        MessageRecord(
            session_id="session-1",
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_call_id": "tool-1",
                    "content": {"status": "ok", "extra": [1, 2]},
                }
            ],
        )
    )

    restored = await store.load_context("session-1")

    assert restored.session.id == "session-1"
    assert restored.session.project_id == "project-1"
    assert restored.session.workspace_id == "workspace-1"
    assert restored.session.session_type == "derived"
    assert restored.session.title == "Unicode 会话"
    assert restored.session.provider == "anthropic"
    assert restored.session.model == "model-1"
    assert restored.session.status is SessionStatus.IDLE
    assert restored.session.workspace_path == "C:/work/项目"
    assert restored.session.metadata == {"nested": {"label": "保留"}}
    assert restored.session.created_at == created_at.astimezone(UTC)
    assert restored.session.updated_at == updated_at.astimezone(UTC)
    assert restored.session.created_at.tzinfo is UTC
    assert restored.session.updated_at.tzinfo is UTC
    assert [message.sequence for message in restored.messages] == [1, 2]
    assert restored.messages[0].content == content
    assert restored.messages[0].created_at == message_time.astimezone(UTC)
    assert restored.messages[0].created_at.tzinfo is UTC
    assert restored.messages[1].content[0]["tool_call_id"] == "tool-1"
    await store.close()


def test_session_new_keeps_positional_compatibility_and_validates_new_fields() -> None:
    session = SessionRecord.new("s1", "p1", "w1", "anthropic", "model")

    assert session.session_type == "root"
    assert session.title is None
    assert session.workspace_path == "unresolved-workspace:w1"
    assert not Path(session.workspace_path).is_absolute()
    assert session.created_at.tzinfo is UTC
    assert session.updated_at.tzinfo is UTC
    assert [status.value for status in SessionStatus] == [
        "active",
        "idle",
        "incomplete",
        "failed",
        "cancelled",
    ]
    with pytest.raises(ValueError, match="session_type"):
        SessionRecord.new(
            "s2", "p1", "w1", "anthropic", "model", session_type="  "
        )
    with pytest.raises(ValueError, match="workspace_path"):
        SessionRecord.new(
            "s3", "p1", "w1", "anthropic", "model", workspace_path=""
        )


@pytest.mark.asyncio
async def test_schema_has_exactly_two_application_tables_and_version_one(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path)
    await store.open()
    await store.close()

    with closing(sqlite3.connect(db_path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        parent_fk = connection.execute(
            "PRAGMA foreign_key_list(sessions)"
        ).fetchall()

    assert tables == {"sessions", "messages"}
    assert version == 1
    assert session_columns == {
        "id",
        "project_id",
        "workspace_id",
        "parent_session_id",
        "session_type",
        "title",
        "provider",
        "model",
        "status",
        "workspace_path",
        "metadata_json",
        "created_at",
        "updated_at",
    }
    assert any(
        row[2] == "sessions"
        and row[3] == "parent_session_id"
        and row[4] == "id"
        and row[6].upper() == "CASCADE"
        for row in parent_fk
    )


@pytest.mark.asyncio
async def test_deleting_root_cascades_nested_sessions_and_messages_only(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path)
    await store.open()
    await store.create_session(_session("root"))
    await store.create_session(_session("child", parent_session_id="root"))
    await store.create_session(_session("grandchild", parent_session_id="child"))
    await store.create_session(_session("unrelated"))
    for session_id in ("root", "child", "grandchild", "unrelated"):
        await store.append_message(
            MessageRecord(
                session_id=session_id,
                role="user",
                content=[{"type": "text", "text": session_id}],
            )
        )
    await store.close()

    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM sessions WHERE id = ?", ("root",))
        connection.commit()
        sessions = connection.execute(
            "SELECT id FROM sessions ORDER BY id"
        ).fetchall()
        messages = connection.execute(
            "SELECT session_id FROM messages ORDER BY session_id"
        ).fetchall()

    assert sessions == [("unrelated",)]
    assert messages == [("unrelated",)]


@pytest.mark.asyncio
async def test_store_lifecycle_is_explicit_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")

    with pytest.raises(RuntimeError, match="not open"):
        await store.load_context("missing")
    invalid = _session("invalid-before-open")
    invalid.metadata = {"bad": object()}
    with pytest.raises(RuntimeError, match="not open"):
        await store.create_session(invalid)
    with pytest.raises(RuntimeError, match="not open"):
        await store.mark_status("missing", "not-a-status")  # type: ignore[arg-type]
    await store.open()
    with pytest.raises(RuntimeError, match="already open"):
        await store.open()
    await store.close()
    await store.close()
    with pytest.raises(RuntimeError, match="not open"):
        await store.create_session(_session("after-close"))
    with pytest.raises(RuntimeError, match="not open"):
        await store.mark_status("missing", "not-a-status")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mark_status_updates_timestamp_and_missing_session_fails(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    initial = _session("session-1")
    initial.updated_at = datetime(2000, 1, 1, tzinfo=UTC)
    await store.create_session(initial)

    await store.mark_status("session-1", SessionStatus.FAILED)
    restored = await store.load_context("session-1")

    assert restored.session.status is SessionStatus.FAILED
    assert restored.session.updated_at > initial.updated_at
    with pytest.raises(KeyError, match="missing"):
        await store.mark_status("missing", SessionStatus.IDLE)
    await store.close()


@pytest.mark.asyncio
async def test_unsupported_schema_version_fails_without_leaking_connection(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("PRAGMA user_version = 2")

    store = SQLiteSessionStore(db_path)
    with pytest.raises(RuntimeError, match="Unsupported sessions schema version: 2"):
        await store.open()
    with pytest.raises(RuntimeError, match="not open"):
        await store.load_context("session-1")
    await store.close()
    db_path.unlink()


@pytest.mark.asyncio
async def test_migration_rechecks_version_after_another_store_migrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "sessions.db"
    original_run_write_transaction = session_store._run_write_transaction
    competing_store_opened = False

    async def open_competing_store_before_migration(
        connection: aiosqlite.Connection,
        operation: Callable[[aiosqlite.Connection], Awaitable[object]],
    ) -> object:
        nonlocal competing_store_opened
        if not competing_store_opened:
            competing_store_opened = True
            competing_store = SQLiteSessionStore(db_path)
            await competing_store.open()
            await competing_store.close()
        return await original_run_write_transaction(connection, operation)

    monkeypatch.setattr(
        session_store,
        "_run_write_transaction",
        open_competing_store_before_migration,
    )

    store = SQLiteSessionStore(db_path)
    await store.open()
    assert competing_store_opened
    await store.close()

    with closing(sqlite3.connect(db_path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == 1


@pytest.mark.asyncio
async def test_failed_version_zero_migration_does_not_advance_version(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 0")

    store = SQLiteSessionStore(db_path)
    with pytest.raises(sqlite3.OperationalError):
        await store.open()

    with closing(sqlite3.connect(db_path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert version == 0
    assert tables == {"sessions"}


@pytest.mark.asyncio
async def test_non_json_message_content_is_rejected_without_consuming_sequence(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(_session("session-1"))

    with pytest.raises(TypeError, match="JSON"):
        await store.append_message(
            MessageRecord(
                session_id="session-1",
                role="assistant",
                content=[{"type": "extension", "value": {"not", "json"}}],
            )
        )
    await store.append_message(
        MessageRecord(
            session_id="session-1",
            role="assistant",
            content=[{"type": "text", "text": "valid"}],
        )
    )

    restored = await store.load_context("session-1")
    assert [message.sequence for message in restored.messages] == [1]
    await store.close()


@pytest.mark.asyncio
async def test_non_json_session_metadata_is_rejected_without_partial_insert(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    invalid = _session("invalid")
    invalid.metadata = {"value": object()}

    with pytest.raises(TypeError, match="JSON"):
        await store.create_session(invalid)
    with pytest.raises(KeyError, match="invalid"):
        await store.load_context("invalid")
    await store.close()


@pytest.mark.asyncio
async def test_failed_append_rolls_back_and_connection_remains_usable(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(_session("session-1"))

    with pytest.raises(sqlite3.IntegrityError):
        await store.append_message(
            MessageRecord(
                session_id="missing",
                role="user",
                content=[{"type": "text", "text": "fails foreign key"}],
            )
        )
    await store.append_message(
        MessageRecord(
            session_id="session-1",
            role="user",
            content=[{"type": "text", "text": "succeeds"}],
        )
    )

    restored = await store.load_context("session-1")
    assert [message.sequence for message in restored.messages] == [1]
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_appends_across_store_connections_allocate_unique_sequences(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    first = SQLiteSessionStore(db_path)
    second = SQLiteSessionStore(db_path)
    await first.open()
    await first.create_session(_session("session-1"))
    await second.open()

    async def append(store: SQLiteSessionStore, source: str, index: int) -> None:
        await store.append_message(
            MessageRecord(
                session_id="session-1",
                role="assistant",
                content=[{"type": "text", "text": f"{source}-{index}"}],
            )
        )

    await asyncio.gather(
        *(append(first, "first", index) for index in range(15)),
        *(append(second, "second", index) for index in range(15)),
    )
    restored = await first.load_context("session-1")

    assert [message.sequence for message in restored.messages] == list(range(1, 31))
    assert {
        message.content[0]["text"] for message in restored.messages
    } == {f"first-{index}" for index in range(15)} | {
        f"second-{index}" for index in range(15)
    }
    await second.close()
    await first.close()

@pytest.mark.asyncio
async def test_cancellation_after_begin_rolls_back_and_reuses_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    original_execute = aiosqlite.Connection.execute
    begin_completed = asyncio.Event()
    release_begin = asyncio.Event()

    async def controlled_execute(
        connection: aiosqlite.Connection,
        sql: str,
        parameters: object = None,
    ) -> aiosqlite.Cursor:
        cursor = await original_execute(connection, sql, parameters)
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            begin_completed.set()
            await release_begin.wait()
        return cursor

    monkeypatch.setattr(aiosqlite.Connection, "execute", controlled_execute)
    write = asyncio.create_task(store.create_session(_session("cancelled")))
    await begin_completed.wait()
    write.cancel()
    asyncio.get_running_loop().call_soon(release_begin.set)

    with pytest.raises(asyncio.CancelledError):
        await write

    monkeypatch.setattr(aiosqlite.Connection, "execute", original_execute)
    assert store.connection is not None
    assert not store.connection.in_transaction
    with pytest.raises(KeyError, match="cancelled"):
        await store.load_context("cancelled")
    await store.create_session(_session("subsequent"))
    assert (await store.load_context("subsequent")).session.id == "subsequent"
    await store.close()


@pytest.mark.asyncio
async def test_cancellation_while_commit_is_in_flight_never_reports_cancelled_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(db_path)
    await store.open()
    original_commit = aiosqlite.Connection.commit
    commit_queued = asyncio.Event()

    async def controlled_commit(connection: aiosqlite.Connection) -> None:
        commit_queued.set()
        await original_commit(connection)

    monkeypatch.setattr(aiosqlite.Connection, "commit", controlled_commit)
    reader = sqlite3.connect(db_path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM sessions").fetchall()
        write = asyncio.create_task(store.create_session(_session("commit-race")))
        await commit_queued.wait()
        write.cancel()
        reader.rollback()
        await write
    finally:
        reader.close()
        monkeypatch.setattr(aiosqlite.Connection, "commit", original_commit)

    assert store.connection is not None
    assert not store.connection.in_transaction
    assert (await store.load_context("commit-race")).session.id == "commit-race"
    await store.create_session(_session("subsequent"))
    assert (await store.load_context("subsequent")).session.id == "subsequent"
    await store.close()


@pytest.mark.asyncio
async def test_commit_failure_wins_over_commit_phase_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    original_commit = aiosqlite.Connection.commit
    commit_entered = asyncio.Event()
    release_commit = asyncio.Event()

    async def failing_commit(connection: aiosqlite.Connection) -> None:
        commit_entered.set()
        await release_commit.wait()
        raise sqlite3.OperationalError("forced commit failure")

    monkeypatch.setattr(aiosqlite.Connection, "commit", failing_commit)
    write = asyncio.create_task(store.create_session(_session("commit-failure")))
    await commit_entered.wait()
    write.cancel()
    asyncio.get_running_loop().call_soon(release_commit.set)

    with pytest.raises(sqlite3.OperationalError, match="forced commit failure"):
        await write

    monkeypatch.setattr(aiosqlite.Connection, "commit", original_commit)
    assert store.connection is not None
    assert not store.connection.in_transaction
    with pytest.raises(KeyError, match="commit-failure"):
        await store.load_context("commit-failure")
    await store.create_session(_session("subsequent-after-failure"))
    assert (
        await store.load_context("subsequent-after-failure")
    ).session.id == "subsequent-after-failure"
    await store.close()


@pytest.mark.asyncio
async def test_session_timestamps_are_revalidated_and_normalized_on_write(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    naive = _session("naive")
    naive.created_at = datetime(2026, 7, 12, 10, 30)
    with pytest.raises(ValueError, match="created_at"):
        await store.create_session(naive)
    invalid = _session("invalid")
    invalid.updated_at = "not-a-datetime"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="updated_at"):
        await store.create_session(invalid)

    offset = timezone(timedelta(hours=8))
    created_at = datetime(2026, 7, 12, 10, 30, tzinfo=offset)
    updated_at = datetime(2026, 7, 12, 11, 45, tzinfo=offset)
    normalized = _session("normalized")
    normalized.created_at = created_at
    normalized.updated_at = updated_at
    await store.create_session(normalized)
    assert store.connection is not None
    async with store.connection.execute(
        "SELECT created_at, updated_at FROM sessions WHERE id = ?", ("normalized",)
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    stored_created_at, stored_updated_at = row
    assert stored_created_at == created_at.astimezone(UTC).isoformat()
    assert stored_updated_at == updated_at.astimezone(UTC).isoformat()
    await store.close()


@pytest.mark.asyncio
async def test_message_timestamp_is_revalidated_and_normalized_on_write(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(_session("session-1"))
    naive = MessageRecord(
        session_id="session-1",
        role="user",
        content=[{"type": "text", "text": "naive"}],
    )
    naive.created_at = datetime(2026, 7, 12, 10, 30)
    with pytest.raises(ValueError, match="created_at"):
        await store.append_message(naive)
    invalid = MessageRecord(
        session_id="session-1",
        role="user",
        content=[{"type": "text", "text": "invalid"}],
    )
    invalid.created_at = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="created_at"):
        await store.append_message(invalid)

    offset = timezone(timedelta(hours=-4))
    created_at = datetime(2026, 7, 12, 10, 30, tzinfo=offset)
    normalized = MessageRecord(
        session_id="session-1",
        role="user",
        content=[{"type": "text", "text": "normalized"}],
    )
    normalized.created_at = created_at
    await store.append_message(normalized)
    assert store.connection is not None
    async with store.connection.execute(
        "SELECT created_at FROM messages WHERE session_id = ?", ("session-1",)
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == created_at.astimezone(UTC).isoformat()
    await store.close()


@pytest.mark.asyncio
async def test_root_session_id_rejects_parent_cycles_promptly(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(_session("root"))
    await store.create_session(_session("child", parent_session_id="root"))
    connection = store.connection
    assert connection is not None
    await connection.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("child", "root"),
    )
    await connection.commit()

    try:
        with pytest.raises(RuntimeError, match="session parent cycle detected"):
            await asyncio.wait_for(
                store.root_session_id("root"),
                timeout=0.5,
            )
    finally:
        await store.close()
