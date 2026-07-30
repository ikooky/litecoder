"""Durable storage operations for the surrounding subsystem."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar, cast

import aiosqlite

from .migrations import MIGRATION_1_STATEMENTS
from .models import MessageRecord, SessionContext, SessionRecord, SessionStatus


_T = TypeVar("_T")


@dataclass(slots=True)
class _TransactionOutcome(Generic[_T]):
    """Data model representing the transaction outcome."""
    committed: bool = False
    cancelled: bool = False
    value: _T | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class DeleteSessionTreeResult:
    """Data model representing the delete session tree result."""
    root_session_id: str
    deleted_session_ids: tuple[str, ...]
    artifact_paths: tuple[Path, ...]


class SQLiteSessionStore:
    """Async SQLite store for sessions, messages, TODOs, and recovery state."""
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.connection: aiosqlite.Connection | None = None
        self._operation_lock = asyncio.Lock()

    async def open(self) -> None:
        """Open the database connection and apply pending migrations."""
        async with self._operation_lock:
            if self.connection is not None:
                raise RuntimeError("Session store is already open")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(self.path)
            try:
                await connection.execute("PRAGMA foreign_keys = ON")
                await connection.execute("PRAGMA busy_timeout = 10000")
                await self._migrate(connection)
            except BaseException:
                await connection.close()
                raise
            self.connection = connection

    async def create_session(self, record: SessionRecord) -> None:
        """Persist a new session record."""
        async with self._operation_lock:
            connection = self._require_connection()
            metadata_json = _serialize_json(record.metadata, "session metadata")
            if not record.session_type.strip():
                raise ValueError("session_type must be a non-empty string")
            if not record.workspace_path.strip():
                raise ValueError("workspace_path must be a non-empty string")
            status = SessionStatus(record.status)
            created_at = _timestamp_for_storage(record.created_at, "created_at")
            updated_at = _timestamp_for_storage(record.updated_at, "updated_at")

            async def insert_session(db: aiosqlite.Connection) -> None:
                """Handle the insert session operation."""
                await db.execute(
                    """
                    INSERT INTO sessions (
                        id, project_id, workspace_id, parent_session_id,
                        session_type, title, provider, model, status,
                        workspace_path, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.project_id,
                        record.workspace_id,
                        record.parent_session_id,
                        record.session_type,
                        record.title,
                        record.provider,
                        record.model,
                        status.value,
                        record.workspace_path,
                        metadata_json,
                        created_at,
                        updated_at,
                    ),
                )

            await _run_write_transaction(connection, insert_session)

    async def append_message(self, message: MessageRecord) -> None:
        """Append one message and assign its next session sequence number."""
        async with self._operation_lock:
            connection = self._require_connection()
            content_json = _serialize_json(message.content, "message content")
            created_at = _timestamp_for_storage(message.created_at, "created_at")

            async def insert_message(db: aiosqlite.Connection) -> None:
                """Handle the insert message operation."""
                async with db.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM messages
                    WHERE session_id = ?
                    """,
                    (message.session_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("Failed to allocate a message sequence")
                sequence = int(row[0])
                await db.execute(
                    """
                    INSERT INTO messages (
                        session_id, sequence, role, content_json,
                        token_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.session_id,
                        sequence,
                        message.role,
                        content_json,
                        message.token_count,
                        created_at,
                    ),
                )

            await _run_write_transaction(connection, insert_message)

    async def append_messages(
        self, messages: Collection[MessageRecord]
    ) -> None:
        """Handle the append messages operation."""
        records = tuple(messages)
        if not records:
            return
        session_id = records[0].session_id
        if any(message.session_id != session_id for message in records):
            raise ValueError("messages must belong to one session")
        serialized = [
            (
                _serialize_json(message.content, "message content"),
                _timestamp_for_storage(message.created_at, "created_at"),
            )
            for message in records
        ]

        async with self._operation_lock:
            connection = self._require_connection()

            async def insert_messages(db: aiosqlite.Connection) -> None:
                """Handle the insert messages operation."""
                async with db.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM messages
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("Failed to allocate a message sequence")
                sequence = int(row[0])
                for message, (content_json, created_at) in zip(records, serialized):
                    await db.execute(
                        """
                        INSERT INTO messages (
                            session_id, sequence, role, content_json,
                            token_count, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            sequence,
                            message.role,
                            content_json,
                            message.token_count,
                            created_at,
                        ),
                    )
                    sequence += 1

            await _run_write_transaction(connection, insert_messages)

    async def list_todos(self, session_id: str) -> list[dict[str, object]]:
        """Handle the list todos operation."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        async with self._operation_lock:
            connection = self._require_connection()
            async with connection.execute(
                "SELECT metadata_json FROM sessions WHERE id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            raise KeyError(session_id)
        metadata = json.loads(row[0])
        todos = metadata.get("todos", []) if isinstance(metadata, dict) else []
        if not isinstance(todos, list) or any(
            not isinstance(item, dict) for item in todos
        ):
            raise ValueError("session todos are invalid")
        return json.loads(_serialize_json(todos, "session todos"))

    async def replace_todos(
        self,
        session_id: str,
        todos: list[dict[str, str]],
    ) -> None:
        """Handle the replace todos operation."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if not isinstance(todos, list) or any(
            not isinstance(item, dict) for item in todos
        ):
            raise ValueError("session todos are invalid")
        normalized = json.loads(_serialize_json(todos, "session todos"))
        updated_at = datetime.now(UTC).isoformat()
        async with self._operation_lock:
            connection = self._require_connection()

            async def update_todos(db: aiosqlite.Connection) -> None:
                """Update the todos."""
                async with db.execute(
                    "SELECT metadata_json FROM sessions WHERE id = ?",
                    (session_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise KeyError(session_id)
                metadata = json.loads(row[0])
                if not isinstance(metadata, dict):
                    raise ValueError("session metadata is invalid")
                metadata["todos"] = normalized
                await db.execute(
                    """
                    UPDATE sessions
                    SET metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        _serialize_json(metadata, "session metadata"),
                        updated_at,
                        session_id,
                    ),
                )

            await _run_write_transaction(connection, update_todos)

    async def load_context(self, session_id: str) -> SessionContext:
        """Load a session record together with its ordered messages."""
        async with self._operation_lock:
            connection = self._require_connection()
            async with connection.execute(
                """
                SELECT
                    id, project_id, workspace_id, provider, model,
                    workspace_path, session_type, title, status,
                    parent_session_id, metadata_json, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ) as cursor:
                session_row = await cursor.fetchone()
            if session_row is None:
                raise KeyError(session_id)
            async with connection.execute(
                """
                SELECT sequence, role, content_json, token_count, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ) as cursor:
                message_rows = await cursor.fetchall()

        session = SessionRecord(
            id=session_row[0],
            project_id=session_row[1],
            workspace_id=session_row[2],
            provider=session_row[3],
            model=session_row[4],
            workspace_path=session_row[5],
            session_type=session_row[6],
            title=session_row[7],
            status=SessionStatus(session_row[8]),
            parent_session_id=session_row[9],
            metadata=json.loads(session_row[10]),
            created_at=_parse_utc(session_row[11], "session created_at"),
            updated_at=_parse_utc(session_row[12], "session updated_at"),
        )
        messages = [
            MessageRecord(
                session_id=session_id,
                sequence=row[0],
                role=row[1],
                content=json.loads(row[2]),
                token_count=row[3],
                created_at=_parse_utc(row[4], "message created_at"),
            )
            for row in message_rows
        ]
        return SessionContext(session=session, messages=messages)

    async def list_sessions(
        self, *, project_id: str | None = None
    ) -> list[SessionRecord]:
        """Handle the list sessions operation."""
        if project_id is not None and not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        async with self._operation_lock:
            connection = self._require_connection()
            if project_id is None:
                statement = """
                    SELECT
                        id, project_id, workspace_id, provider, model,
                        workspace_path, session_type, title, status,
                        parent_session_id, metadata_json, created_at, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC, created_at DESC, id
                    """
                arguments: tuple[object, ...] = ()
            else:
                statement = """
                    SELECT
                        id, project_id, workspace_id, provider, model,
                        workspace_path, session_type, title, status,
                        parent_session_id, metadata_json, created_at, updated_at
                    FROM sessions
                    WHERE project_id = ?
                    ORDER BY updated_at DESC, created_at DESC, id
                    """
                arguments = (project_id,)
            async with connection.execute(statement, arguments) as cursor:
                rows = await cursor.fetchall()
        return [_session_from_row(row) for row in rows]

    async def root_session_id(self, session_id: str) -> str:
        """Handle the root session id operation."""
        async with self._operation_lock:
            connection = self._require_connection()
            return await _root_session_id(connection, session_id)

    async def delete_session_tree(self, session_id: str) -> DeleteSessionTreeResult:
        """Delete the session tree."""
        async with self._operation_lock:
            connection = self._require_connection()

            async def delete_tree(
                db: aiosqlite.Connection,
            ) -> DeleteSessionTreeResult:
                """Delete the tree."""
                root_session_id = await _root_session_id(db, session_id)
                async with db.execute(
                    """
                    WITH RECURSIVE tree(id) AS (
                        SELECT id FROM sessions WHERE id = ?
                        UNION ALL
                        SELECT sessions.id
                        FROM sessions
                        JOIN tree ON sessions.parent_session_id = tree.id
                    )
                    SELECT id FROM tree
                    """,
                    (session_id,),
                ) as cursor:
                    id_rows = await cursor.fetchall()
                deleted_ids = tuple(str(row[0]) for row in id_rows)
                if not deleted_ids:
                    raise KeyError(session_id)

                placeholders = ",".join("?" for _ in deleted_ids)
                async with db.execute(
                    f"""
                    SELECT content_json
                    FROM messages
                    WHERE session_id IN ({placeholders})
                    """,
                    deleted_ids,
                ) as cursor:
                    message_rows = await cursor.fetchall()
                candidate_paths = tuple(dict.fromkeys(
                    path
                    for row in message_rows
                    for path in _artifact_paths_from_content(json.loads(row[0]))
                ))
                async with db.execute(
                    f"""
                    SELECT content_json
                    FROM messages
                    WHERE session_id NOT IN ({placeholders})
                    """,
                    deleted_ids,
                ) as cursor:
                    remaining_rows = await cursor.fetchall()
                remaining_paths = {
                    path
                    for row in remaining_rows
                    for path in _artifact_paths_from_content(json.loads(row[0]))
                }
                artifact_paths = tuple(
                    path for path in candidate_paths if path not in remaining_paths
                )

                cursor = await db.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,),
                )
                if cursor.rowcount != 1:
                    raise KeyError(session_id)
                return DeleteSessionTreeResult(
                    root_session_id,
                    deleted_ids,
                    artifact_paths,
                )

            return await _run_write_transaction(connection, delete_tree)

    async def recover_active_sessions(
        self,
        project_id: str,
        exclude_session_ids: Collection[str] = (),
        target_session_ids: Collection[str] | None = None,
    ) -> list[str]:
        """Handle the recover active sessions operation."""
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        excluded_ids = tuple(dict.fromkeys(exclude_session_ids))
        if any(
            not isinstance(session_id, str) or not session_id.strip()
            for session_id in excluded_ids
        ):
            raise ValueError("exclude_session_ids must contain non-empty strings")
        target_ids = (
            None
            if target_session_ids is None
            else tuple(dict.fromkeys(target_session_ids))
        )
        if target_ids is not None and any(
            not isinstance(session_id, str) or not session_id.strip()
            for session_id in target_ids
        ):
            raise ValueError("target_session_ids must contain non-empty strings")
        async with self._operation_lock:
            connection = self._require_connection()

            async def recover(db: aiosqlite.Connection) -> list[str]:
                """Recover durable state after an interrupted operation."""
                if target_ids == ():
                    return []
                arguments: list[object] = [
                    project_id,
                    SessionStatus.ACTIVE.value,
                ]
                filters = ""
                if target_ids is not None:
                    placeholders = ",".join("?" for _ in target_ids)
                    filters += f" AND id IN ({placeholders})"
                    arguments.extend(target_ids)
                if excluded_ids:
                    placeholders = ",".join("?" for _ in excluded_ids)
                    filters += f" AND id NOT IN ({placeholders})"
                    arguments.extend(excluded_ids)
                async with db.execute(
                    (
                        "SELECT id FROM sessions "
                        "WHERE project_id = ? AND status = ?"
                        f"{filters}"
                    ),
                    arguments,
                ) as cursor:
                    rows = await cursor.fetchall()
                session_ids = [str(row[0]) for row in rows]
                if session_ids:
                    placeholders = ",".join("?" for _ in session_ids)
                    await db.execute(
                        f"""
                        UPDATE sessions
                        SET status = ?, updated_at = ?
                        WHERE project_id = ? AND status = ?
                            AND id IN ({placeholders})
                        """,
                        (
                            SessionStatus.INCOMPLETE.value,
                            datetime.now(UTC).isoformat(),
                            project_id,
                            SessionStatus.ACTIVE.value,
                            *session_ids,
                        ),
                    )
                return session_ids

            return await _run_write_transaction(connection, recover)

    async def mark_status(self, session_id: str, status: SessionStatus) -> None:
        """Handle the mark status operation."""
        async with self._operation_lock:
            connection = self._require_connection()
            normalized_status = SessionStatus(status)

            async def update_status(db: aiosqlite.Connection) -> None:
                """Update the status."""
                cursor = await db.execute(
                    """
                    UPDATE sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        normalized_status.value,
                        datetime.now(UTC).isoformat(),
                        session_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(session_id)

            await _run_write_transaction(connection, update_status)

    async def close(self) -> None:
        """Close the database connection and release the operation lock."""
        async with self._operation_lock:
            connection = self.connection
            if connection is None:
                return
            try:
                await connection.close()
            finally:
                self.connection = None

    def _require_connection(self) -> aiosqlite.Connection:
        if self.connection is None:
            raise RuntimeError("Session store is not open")
        return self.connection

    @staticmethod
    async def _migrate(connection: aiosqlite.Connection) -> None:
        """Migrate the requested operation."""
        async with connection.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        version = 0 if row is None else int(row[0])
        if version == 1:
            return
        if version != 0:
            raise RuntimeError(f"Unsupported sessions schema version: {version}")

        async def migrate(db: aiosqlite.Connection) -> None:
            """Migrate the requested operation."""
            async with db.execute("PRAGMA user_version") as cursor:
                row = await cursor.fetchone()
            version = 0 if row is None else int(row[0])
            if version == 1:
                return
            if version != 0:
                raise RuntimeError(f"Unsupported sessions schema version: {version}")
            for statement in MIGRATION_1_STATEMENTS:
                await db.execute(statement)
            await db.execute("PRAGMA user_version = 1")

        await _run_write_transaction(connection, migrate)


async def _run_write_transaction(
    connection: aiosqlite.Connection,
    operation: Callable[[aiosqlite.Connection], Awaitable[_T]],
) -> _T:
    """Run one write transaction without ambiguous cancellation outcomes.

    The worker is shielded because aiosqlite public operations queue SQLite work
    before their await completes. Cancellation requests are therefore communicated
    through an event. Before the commit phase the worker rolls back and cancellation
    is re-raised. Once the commit phase starts, the worker finishes it: a durable
    commit returns success, while commit or cleanup errors outrank cancellation.
    """

    cancel_requested = asyncio.Event()
    worker = asyncio.create_task(
        _transaction_worker(connection, operation, cancel_requested)
    )
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            outcome = await asyncio.shield(worker)
            break
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
            cancel_requested.set()

    if outcome.committed:
        # A cancellation arriving after commit is reported only after durable
        # state is returned, so callers cannot observe an ambiguous outcome.
        if cancellation is not None:
            _clear_current_task_cancellation()
        return cast(_T, outcome.value)
    if outcome.error is not None:
        if cancellation is not None:
            _clear_current_task_cancellation()
        raise outcome.error
    if cancellation is not None:
        raise cancellation
    if outcome.cancelled:
        raise RuntimeError("Transaction rolled back without a cancellation request")
    raise RuntimeError("Transaction finished without an outcome")


async def _transaction_worker(
    connection: aiosqlite.Connection,
    operation: Callable[[aiosqlite.Connection], Awaitable[_T]],
    cancel_requested: asyncio.Event,
) -> _TransactionOutcome[_T]:
    try:
        await connection.execute("BEGIN IMMEDIATE")
        if cancel_requested.is_set():
            await connection.rollback()
            return _TransactionOutcome(cancelled=True)

        value = await operation(connection)
        if cancel_requested.is_set():
            await connection.rollback()
            return _TransactionOutcome(cancelled=True)

        # There is no event-loop suspension between the final cancellation check
        # and entering commit. From this point the commit phase owns the outcome.
        await connection.commit()
        return _TransactionOutcome(committed=True, value=value)
    except BaseException as error:
        if connection.in_transaction:
            try:
                await connection.rollback()
            except BaseException as rollback_error:
                error.add_note(f"Rollback also failed: {rollback_error!r}")
        return _TransactionOutcome(error=error)


def _clear_current_task_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


def _session_from_row(row: object) -> SessionRecord:
    return SessionRecord(
        id=row[0],
        project_id=row[1],
        workspace_id=row[2],
        provider=row[3],
        model=row[4],
        workspace_path=row[5],
        session_type=row[6],
        title=row[7],
        status=SessionStatus(row[8]),
        parent_session_id=row[9],
        metadata=json.loads(row[10]),
        created_at=_parse_utc(row[11], "session created_at"),
        updated_at=_parse_utc(row[12], "session updated_at"),
    )


async def _root_session_id(
    connection: aiosqlite.Connection,
    session_id: str,
) -> str:
    current_session_id = session_id
    visited: set[str] = set()
    while True:
        if current_session_id in visited:
            raise RuntimeError("session parent cycle detected")
        visited.add(current_session_id)
        async with connection.execute(
            "SELECT parent_session_id FROM sessions WHERE id = ?",
            (current_session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise KeyError(session_id)
        parent_session_id = row[0]
        if parent_session_id is None:
            return current_session_id
        current_session_id = str(parent_session_id)


def _artifact_paths_from_content(content: object) -> list[Path]:
    if not isinstance(content, list):
        return []
    paths: list[Path] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        metadata = block.get("metadata")
        if not isinstance(metadata, dict):
            continue
        artifact = metadata.get("artifact")
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        if isinstance(path, str) and path:
            paths.append(Path(path))
    return paths


def _timestamp_for_storage(value: object, field_name: str) -> str:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _serialize_json(value: object, field_name: str) -> str:
    _validate_json(value, field_name, "$")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _validate_json(value: object, field_name: str, path: str) -> None:
    """Validate the json."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_name} contains a non-JSON number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, field_name, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} contains a non-JSON key at {path}")
            _validate_json(item, field_name, f"{path}.{key}")
        return
    raise TypeError(
        f"{field_name} contains a non-JSON value at {path}: "
        f"{type(value).__name__}"
    )


def _parse_utc(value: str, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"Stored {field_name} is not timezone-aware")
    return parsed.astimezone(UTC)
