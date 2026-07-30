"""Data models for the surrounding subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


UNRESOLVED_WORKSPACE_PREFIX = "unresolved-workspace:"


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class SessionStatus(StrEnum):
    """Enumeration of the session status values."""
    ACTIVE = "active"
    IDLE = "idle"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class SessionRecord:
    """Data model representing the session record."""
    id: str
    project_id: str
    workspace_id: str
    provider: str
    model: str
    workspace_path: str
    session_type: str = "root"
    title: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    parent_session_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.session_type = _non_empty(self.session_type, "session_type")
        self.workspace_path = _non_empty(self.workspace_path, "workspace_path")
        self.status = SessionStatus(self.status)
        self.created_at = _utc_datetime(self.created_at, "created_at")
        self.updated_at = _utc_datetime(self.updated_at, "updated_at")

    @classmethod
    def new(
        cls,
        session_id: str,
        project_id: str,
        workspace_id: str,
        provider: str,
        model: str,
        *,
        session_type: str = "root",
        title: str | None = None,
        workspace_path: str | None = None,
        status: SessionStatus = SessionStatus.ACTIVE,
        parent_session_id: str | None = None,
        metadata: dict[str, object] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> SessionRecord:
        """Create a session while preserving the original five positional arguments.

        The unresolved workspace marker is compatibility-only and deliberately is not
        an absolute path. Runtime callers that know the workspace must pass its
        canonical path explicitly.
        """

        now = datetime.now(UTC)
        return cls(
            id=session_id,
            project_id=project_id,
            workspace_id=workspace_id,
            provider=provider,
            model=model,
            workspace_path=(
                f"{UNRESOLVED_WORKSPACE_PREFIX}{workspace_id}"
                if workspace_path is None
                else workspace_path
            ),
            session_type=session_type,
            title=title,
            status=status,
            parent_session_id=parent_session_id,
            metadata={} if metadata is None else metadata,
            created_at=now if created_at is None else created_at,
            updated_at=now if updated_at is None else updated_at,
        )


@dataclass(slots=True)
class MessageRecord:
    """Data model representing the message record."""
    session_id: str
    role: str
    content: list[dict[str, object]]
    sequence: int | None = None
    token_count: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.created_at = _utc_datetime(self.created_at, "created_at")


@dataclass(slots=True)
class SessionContext:
    """Data model representing the session context."""
    session: SessionRecord
    messages: list[MessageRecord]
