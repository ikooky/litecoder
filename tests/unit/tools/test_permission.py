from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.tools import PermissionBroker, ToolCall, ToolContext, ToolSpec
from litecoder.tools.permission import PermissionMode, PermissionService, PromptChoice


def context(tmp_path: Path, **metadata: object) -> ToolContext:
    return ToolContext("child", "workspace", tmp_path, metadata=metadata)


@pytest.mark.parametrize(
    ("mode", "read_action", "write_action", "external_action"),
    [
        (PermissionMode.ASK, "allow", "prompt", "prompt"),
        (PermissionMode.READ_ONLY, "allow", "deny", "deny"),
        (PermissionMode.BYPASS, "allow", "prompt", "prompt"),
    ],
)
def test_permission_modes_are_exact(mode, read_action, write_action, external_action) -> None:
    service = PermissionService()
    read = ToolSpec("read", "read", {}, False)
    write = ToolSpec("write", "write", {}, True)
    external = ToolSpec("fetch", "fetch", {}, False, permission_risk="external")
    assert service.classify(mode, read).action == read_action
    assert service.classify(mode, write).action == write_action
    assert service.classify(mode, external).action == external_action


@pytest.mark.asyncio
async def test_missing_permission_mode_defaults_to_ask(tmp_path: Path) -> None:
    decision = await PermissionService().decide(
        ToolSpec("write", "write", {}, True),
        ToolCall("call", "write", {"path": "file.txt"}),
        context(tmp_path),
    )

    assert (decision.allowed, decision.action, decision.reason) == (
        False,
        "deny",
        "Permission confirmation unavailable",
    )

def test_bypass_requires_explicit_context_authority(tmp_path: Path) -> None:
    service = PermissionService()
    write = ToolSpec("write", "write", {}, True)
    assert service.classify(PermissionMode.BYPASS, write, context(tmp_path)).action == "prompt"
    assert service.classify(PermissionMode.BYPASS, write, context(tmp_path, bypass_authorized=True)).action == "allow"


@pytest.mark.asyncio
async def test_hard_guard_runs_before_prompt_and_always_denies(tmp_path: Path) -> None:
    prompts = []
    async def prompt(request):
        prompts.append(request)
        return PromptChoice.ALLOW_ONCE
    async def guard(*_):
        return False
    service = PermissionService(prompt=prompt, hard_guard=guard)
    decision = await service.decide(
        ToolSpec("deploy", "deploy", {}, True, permission_risk="high"),
        ToolCall("call", "deploy", {"token": "raw-secret"}),
        context(tmp_path, permission_mode="bypass", bypass_authorized=True),
    )
    assert (decision.allowed, decision.action, prompts) == (False, "deny", [])
    assert "raw-secret" not in decision.reason


@pytest.mark.asyncio
async def test_root_session_approval_is_scoped_and_contains_no_secrets(tmp_path: Path) -> None:
    prompts = []
    async def prompt(request):
        prompts.append(request)
        return PromptChoice.ALLOW_FOR_ROOT_SESSION
    service = PermissionService(prompt=prompt)
    spec = ToolSpec("publish", "publish", {}, True, permission_risk="external")
    call = ToolCall("one", "publish", {"target": "origin", "password": "secret"})
    first = context(tmp_path, permission_mode="ask", root_session_id="root-a")
    same = ToolContext("other", "workspace", tmp_path, metadata={"permission_mode": "ask", "root_session_id": "root-a"})
    other = context(tmp_path, permission_mode="ask", root_session_id="root-b")
    assert (await service.decide(spec, call, first)).allowed
    assert (await service.decide(spec, ToolCall("two", "publish", call.arguments), same)).allowed
    assert len(prompts) == 1
    assert (await service.decide(spec, ToolCall("three", "publish", call.arguments), other)).allowed
    assert len(prompts) == 2
    assert "secret" not in repr(service._session_approvals)
    service.clear_root_session("root-a")
    await service.decide(spec, ToolCall("four", "publish", call.arguments), first)
    assert len(prompts) == 3


@pytest.mark.asyncio
async def test_child_broker_uses_root_session_approval_cache(tmp_path: Path) -> None:
    root_prompts = []
    child_prompts = []

    async def root_prompt(request):
        root_prompts.append(request)
        return PromptChoice.ALLOW_FOR_ROOT_SESSION

    async def child_prompt(request):
        child_prompts.append(request)
        return PromptChoice.DENY

    root_service = PermissionService(prompt=root_prompt)
    child_service = PermissionService(prompt=child_prompt)
    broker = PermissionBroker(root_service)
    spec = ToolSpec("publish", "publish", {}, True, permission_risk="external")
    call = ToolCall("one", "publish", {"target": "origin"})
    child_context = ToolContext(
        "child-a",
        "workspace",
        tmp_path,
        metadata={"permission_mode": "ask", "root_session_id": "root-a"},
        parent_permission_broker=broker,
    )

    first = await child_service.decide(spec, call, child_context)
    second = await child_service.decide(
        spec,
        ToolCall("two", "publish", {"target": "origin"}),
        child_context,
    )

    assert first.allowed
    assert second.allowed
    assert len(root_prompts) == 1
    assert child_prompts == []


@pytest.mark.asyncio
async def test_prompt_denial_is_safe(tmp_path: Path) -> None:
    async def deny(_):
        return PromptChoice.DENY
    decision = await PermissionService(prompt=deny).decide(
        ToolSpec("install", "install", {}, True, permission_risk="high"),
        ToolCall("call", "install", {"api_key": "secret"}),
        context(tmp_path, permission_mode="ask"),
    )
    assert (decision.allowed, decision.action) == (False, "deny")
    assert "secret" not in decision.reason


@pytest.mark.asyncio
async def test_prompt_receives_redacted_tool_arguments(tmp_path: Path) -> None:
    prompts = []

    async def prompt(request):
        prompts.append(request)
        return PromptChoice.DENY

    service = PermissionService(prompt=prompt)
    await service.decide(
        ToolSpec("run_shell", "run", {}, True, permission_risk="high"),
        ToolCall(
            "call",
            "run_shell",
            {"argv": ["curl", "-H", "Bearer secret-token"], "cwd": "."},
        ),
        ToolContext(
            "session",
            "workspace",
            tmp_path,
            metadata={"permission_mode": "ask"},
            secret_values=("secret-token",),
        ),
    )

    assert len(prompts) == 1
    assert prompts[0].arguments == {
        "argv": ["curl", "-H", "[REDACTED]"],
        "cwd": ".",
    }
    assert prompts[0].workspace_root == str(tmp_path.resolve())
    assert prompts[0].tool_call_id == "call"
    assert "secret-token" not in repr(prompts[0])


@pytest.mark.asyncio
async def test_root_approval_is_least_privilege_by_argument_scope(tmp_path: Path) -> None:
    prompts = []

    async def prompt(request):
        prompts.append(request)
        return PromptChoice.ALLOW_FOR_ROOT_SESSION

    service = PermissionService(prompt=prompt)
    spec = ToolSpec("publish", "publish", {}, True, permission_risk="external")
    scope = context(tmp_path, permission_mode="ask", root_session_id="root")
    await service.decide(spec, ToolCall("one", "publish", {"target": "origin"}), scope)
    await service.decide(spec, ToolCall("two", "publish", {"target": "mirror"}), scope)
    assert len(prompts) == 2
