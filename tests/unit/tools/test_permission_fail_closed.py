from __future__ import annotations

import pytest

from litecoder.tools import ToolCall, ToolContext, ToolSpec
from litecoder.tools.permission import PermissionService


@pytest.mark.asyncio
async def test_guard_and_prompt_exceptions_fail_closed_without_raw_text(tmp_path) -> None:
    context = ToolContext("agent", "workspace", tmp_path)

    async def broken_guard(*_):
        raise RuntimeError("guard-secret")

    guarded = await PermissionService(hard_guard=broken_guard).decide(
        ToolSpec("read", "read", {}, False), ToolCall("one", "read", {}), context
    )
    assert guarded.action == "deny"
    assert "guard-secret" not in guarded.reason

    async def broken_prompt(_):
        raise RuntimeError("prompt-secret")

    prompted = await PermissionService(prompt=broken_prompt).decide(
        ToolSpec("deploy", "deploy", {}, True, permission_risk="high"),
        ToolCall("two", "deploy", {}), context,
    )
    assert prompted.action == "deny"
    assert "prompt-secret" not in prompted.reason
