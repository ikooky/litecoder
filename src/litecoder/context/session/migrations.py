"""SQLite session-schema migrations."""

from __future__ import annotations


MIGRATION_1_STATEMENTS = (
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        parent_session_id TEXT,
        session_type TEXT NOT NULL CHECK (length(trim(session_type)) > 0),
        title TEXT,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('active', 'idle', 'incomplete', 'failed', 'cancelled')
        ),
        workspace_path TEXT NOT NULL CHECK (length(trim(workspace_path)) > 0),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (parent_session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE messages (
        session_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        role TEXT NOT NULL,
        content_json TEXT NOT NULL,
        token_count INTEGER,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, sequence),
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
)

MIGRATION_1_SQL = ";\n".join(
    statement.strip() for statement in MIGRATION_1_STATEMENTS
) + ";"
