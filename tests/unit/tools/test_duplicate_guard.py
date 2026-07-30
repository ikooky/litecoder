from __future__ import annotations

import pytest

from litecoder.tools import ToolCall, ToolSpec
from litecoder.tools.duplicate_guard import (
    DUPLICATE_CALL_WINDOW_ROUNDS,
    DuplicateGuard,
    fingerprint,
)


def call(call_id: str = "call-1", **arguments: object) -> ToolCall:
    return ToolCall(call_id, "read", arguments)


@pytest.mark.asyncio
async def test_matching_success_is_blocked_only_inside_exact_five_round_window() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    original = call("old", path="a.py")
    await guard.record_success(
        "agent", "workspace", 2, round_number=1, call=original, preview={"line": 1}
    )

    duplicate = await guard.check(
        "agent", "workspace", 2, round_number=DUPLICATE_CALL_WINDOW_ROUNDS, call=call("new", path="a.py")
    )
    expired = await guard.check(
        "agent", "workspace", 2, round_number=DUPLICATE_CALL_WINDOW_ROUNDS + 1, call=call("later", path="a.py")
    )

    assert duplicate is not None
    assert duplicate.tool_call_id == "new"
    assert duplicate.status == "duplicate_blocked"
    assert duplicate.metadata["preview"] == {"line": 1}
    assert expired is None


@pytest.mark.asyncio
async def test_cache_is_isolated_by_agent_workspace_and_version() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    read = call(path="a.py")
    await guard.record_success(
        "agent-a", "workspace-a", 0, round_number=1, call=read, preview="A"
    )

    assert await guard.check("agent-b", "workspace-a", 0, round_number=1, call=read) is None
    assert await guard.check("agent-a", "workspace-b", 0, round_number=1, call=read) is None
    assert await guard.check("agent-a", "workspace-a", 1, round_number=1, call=read) is None


@pytest.mark.asyncio
async def test_mutation_success_records_pre_and_post_versions() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    write = ToolCall("write-1", "write", {"path": "a.py", "text": "x"})
    await guard.record_success(
        "agent", "workspace", 4, post_workspace_version=5,
        round_number=3, call=write, preview="written"
    )

    assert await guard.check("agent", "workspace", 4, round_number=3, call=write) is not None
    assert await guard.check("agent", "workspace", 5, round_number=3, call=write) is not None


@pytest.mark.asyncio
async def test_dedupe_none_bypasses_record_and_check() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    volatile = ToolSpec("status", "Status", {}, False, dedupe_policy="none")
    status = ToolCall("status-1", "status", {})
    await guard.record_success(
        "agent", "workspace", 0, round_number=1, call=status, preview="ok", spec=volatile
    )
    assert await guard.check(
        "agent", "workspace", 0, round_number=1, call=status, spec=volatile
    ) is None


@pytest.mark.asyncio
async def test_new_user_message_clear_is_scoped_to_one_agent() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    read = call(path="a.py")
    for agent in ("agent-a", "agent-b"):
        await guard.record_success(
            agent, "workspace", 0, round_number=1, call=read, preview=agent
        )

    await guard.clear_for_new_user_message("agent-a")

    assert await guard.check("agent-a", "workspace", 0, round_number=1, call=read) is None
    assert await guard.check("agent-b", "workspace", 0, round_number=1, call=read) is not None


def test_fingerprint_uses_canonical_json_name_and_workspace() -> None:
    first = ToolCall("one", "read", {"b": 2, "a": {"z": 1}})
    reordered = ToolCall("two", "read", {"a": {"z": 1}, "b": 2})
    assert fingerprint(first, "workspace") == fingerprint(reordered, "workspace")
    assert fingerprint(first, "other") != fingerprint(reordered, "workspace")
    assert fingerprint(first, "workspace") != fingerprint(ToolCall("three", "write", first.arguments), "workspace")


@pytest.mark.asyncio
async def test_duplicate_annotation_is_injected() -> None:
    annotations: list[dict[str, object]] = []

    async def annotate(**payload: object) -> None:
        annotations.append(payload)

    guarded = DuplicateGuard(annotation=annotate)
    read = call(path="a.py")
    await guarded.record_success("agent", "workspace", 0, round_number=1, call=read, preview="A")
    assert await guarded.check("agent", "workspace", 0, round_number=1, call=read) is not None
    assert annotations and annotations[0]["reason"] == "duplicate-tool-call"
