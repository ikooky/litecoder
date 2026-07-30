"""Trace event data models and event names."""

from __future__ import annotations

from typing import NotRequired, Required, TypedDict


class TraceRecord(TypedDict, total=False):
    """Component responsible for the trace record."""
    sequence: Required[int]
    event: Required[str]
    timestamp: NotRequired[str]
    trace_id: NotRequired[str]
    span_id: NotRequired[str]
    parent_span_id: NotRequired[str | None]
    root_session_id: NotRequired[str]
    session_id: NotRequired[str]
    session_id_after: NotRequired[str | None]
    project_id: NotRequired[str]
    workspace_id: NotRequired[str]
    agent_id: NotRequired[str]
    intent: NotRequired[str | None]
    reason: NotRequired[str | None]
    command_id: NotRequired[str]
    command: NotRequired[str]
    status: NotRequired[str]
    duration_ms: NotRequired[int]
    attributes: NotRequired[dict[str, object]]
