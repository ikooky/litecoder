from __future__ import annotations

import sys

import pytest

from litecoder.common.trace.redaction import SecretRedactor, bind_secret_redactor
from litecoder.hooks import CommandHook, HookEnvelope, HookPoint, discover_command_hooks
from litecoder.settings import HookCommandSettings, Settings
from litecoder.tools import ToolCall


def _config(*, timeout_seconds: float = 1.0, args: tuple[str, ...]) -> HookCommandSettings:
    return HookCommandSettings(
        name="external-policy",
        point="PreToolUse",
        command=sys.executable,
        args=args,
        timeout_seconds=timeout_seconds,
    )


def _envelope(phase: str = "pre") -> HookEnvelope:
    return HookEnvelope(
        point=HookPoint.PRE_TOOL_USE,
        payload={"arguments": {"token": "hook-secret", "path": "old.py"}},
        hook_id="external-policy",
        dispatch_id="dispatch-1",
        phase=phase,
    )


@pytest.mark.asyncio
async def test_command_hook_uses_redacted_json_stdin_and_json_stdout(tmp_path) -> None:
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['payload']['arguments']['token'] == '[REDACTED]'\n"
        "print(json.dumps({'payload': {'arguments': {'path': 'rewritten.py'}}}))\n",
        encoding="utf-8",
    )
    hook = CommandHook(_config(args=(str(script),)))

    with bind_secret_redactor(SecretRedactor.with_values(("hook-secret",))):
        outcome = await hook(_envelope())

    assert outcome.blocked is False
    assert outcome.payload == {"arguments": {"path": "rewritten.py"}}
    assert outcome.diagnostics == ()


@pytest.mark.asyncio
async def test_command_hook_serializes_runtime_tool_call_payload(tmp_path) -> None:
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "call = request['payload']['call']\n"
        "assert call == {'id': 'call-1', 'name': 'read_file', "
        "'arguments': {'path': 'old.py'}}\n"
        "call['arguments']['path'] = 'rewritten.py'\n"
        "print(json.dumps({'payload': {'call': call}}))\n",
        encoding="utf-8",
    )
    hook = CommandHook(_config(args=(str(script),)))

    outcome = await hook(
        HookEnvelope(
            point=HookPoint.PRE_TOOL_USE,
            payload={"call": ToolCall("call-1", "read_file", {"path": "old.py"})},
            hook_id="external-policy",
            dispatch_id="dispatch-runtime",
            phase="pre",
        )
    )

    assert outcome.blocked is False
    assert outcome.payload["call"]["arguments"] == {"path": "rewritten.py"}
    assert outcome.diagnostics == ()


@pytest.mark.asyncio
async def test_pre_command_failure_fails_closed() -> None:
    hook = CommandHook(_config(args=("-c", "print('not-json')")))

    outcome = await hook(_envelope())

    assert outcome.blocked is True
    assert outcome.diagnostics[0].code == "command_invalid_output"
    assert "not-json" not in outcome.diagnostics[0].message


@pytest.mark.asyncio
async def test_post_command_failure_is_reported_without_blocking() -> None:
    hook = CommandHook(_config(args=("-c", "raise SystemExit(1)")))

    outcome = await hook(_envelope("post"))

    assert outcome.blocked is False
    assert outcome.diagnostics[0].code == "command_execution_failed"


@pytest.mark.asyncio
async def test_command_timeout_fails_closed() -> None:
    hook = CommandHook(
        _config(
            timeout_seconds=0.05,
            args=("-c", "import time; time.sleep(10)"),
        )
    )

    outcome = await hook(_envelope())

    assert outcome.blocked is True
    assert outcome.diagnostics[0].code == "command_timed_out"


@pytest.mark.asyncio
async def test_command_output_is_bounded() -> None:
    from litecoder.hooks.command import _MAX_COMMAND_OUTPUT_BYTES

    assert _MAX_COMMAND_OUTPUT_BYTES == 64 * 1024
    hook = CommandHook(
        _config(args=("-c", f"print('x' * {_MAX_COMMAND_OUTPUT_BYTES + 1})"))
    )

    outcome = await hook(_envelope())

    assert outcome.blocked is True
    assert outcome.diagnostics[0].code == "command_output_too_large"


@pytest.mark.asyncio
async def test_command_hook_accepts_input_larger_than_64_kib() -> None:
    from litecoder.hooks.command import _MAX_COMMAND_INPUT_BYTES

    hook = CommandHook(
        _config(
            args=(
                "-c",
                "import json, sys; sys.stdin.read(); print(json.dumps({'payload': {}}))",
            )
        )
    )
    envelope = HookEnvelope(
        point=HookPoint.PRE_TOOL_USE,
        payload={"content": "x" * (64 * 1024)},
        hook_id="external-policy",
        dispatch_id="dispatch-large-input",
        phase="pre",
    )

    outcome = await hook(envelope)

    assert _MAX_COMMAND_INPUT_BYTES == 1024 * 1024
    assert outcome.blocked is False
    assert outcome.diagnostics == ()


@pytest.mark.asyncio
async def test_command_hook_rejects_input_larger_than_1_mib() -> None:
    from litecoder.hooks.command import _MAX_COMMAND_INPUT_BYTES

    hook = CommandHook(_config(args=("-c", "raise AssertionError('not run')")))
    envelope = HookEnvelope(
        point=HookPoint.PRE_TOOL_USE,
        payload={"content": "x" * _MAX_COMMAND_INPUT_BYTES},
        hook_id="external-policy",
        dispatch_id="dispatch-oversized-input",
        phase="pre",
    )

    outcome = await hook(envelope)

    assert outcome.blocked is True
    assert outcome.diagnostics[0].code == "command_input_too_large"


def test_discovery_builds_explicit_command_hook_registrations() -> None:
    settings = Settings(
        hooks=(
            HookCommandSettings(
                name="external-policy",
                enabled=True,
                point="PreToolUse",
                command="policy-hook",
            ),
        )
    )

    discovered = discover_command_hooks(settings)

    assert [(item.name, item.point) for item in discovered] == [
        ("external-policy", HookPoint.PRE_TOOL_USE)
    ]
    assert isinstance(discovered[0].hook, CommandHook)


def test_hook_command_settings_reject_unknown_fields_and_duplicate_names() -> None:
    with pytest.raises(ValueError):
        HookCommandSettings.model_validate(
            {
                "name": "policy",
                "point": "PreToolUse",
                "command": "policy-hook",
                "shell": True,
            }
        )
    with pytest.raises(ValueError, match="unique"):
        Settings(
            hooks=(
                HookCommandSettings(
                    name="policy", point="PreToolUse", command="policy-hook"
                ),
                HookCommandSettings(
                    name="policy", point="PostToolUse", command="policy-hook"
                ),
            )
        )

@pytest.mark.asyncio
async def test_post_command_hook_cannot_mutate_payload() -> None:
    hook = CommandHook(
        _config(
            args=(
                "-c",
                "import json; print(json.dumps({'payload': {'authority': 'elevated'}}))",
            )
        )
    )
    envelope = HookEnvelope(
        point=HookPoint.POST_TOOL_USE,
        payload={"authority": "unchanged"},
        hook_id="external-policy",
        dispatch_id="dispatch-post",
        phase="post",
    )

    outcome = await hook(envelope)

    assert outcome.payload == {"authority": "unchanged"}


@pytest.mark.asyncio
async def test_subagent_start_command_hook_can_block_but_cannot_mutate_payload() -> None:
    hook = CommandHook(
        _config(
            args=(
                "-c",
                "import json; print(json.dumps({'blocked': True, 'payload': {'authority': 'elevated'}}))",
            )
        )
    )
    envelope = HookEnvelope(
        point=HookPoint.SUBAGENT_START,
        payload={"authority": "parent", "workspace": "root", "budget": 3},
        hook_id="external-policy",
        dispatch_id="dispatch-subagent",
        phase="pre",
    )

    outcome = await hook(envelope)

    assert outcome.blocked is True
    assert outcome.payload == {"authority": "parent", "workspace": "root", "budget": 3}


def test_discovery_ignores_disabled_command_hooks() -> None:
    settings = Settings(
        hooks=(
            HookCommandSettings(
                name="disabled", point="PreToolUse", command="policy-hook"
            ),
            HookCommandSettings(
                name="enabled",
                enabled=True,
                point="PostToolUse",
                command="policy-hook",
            ),
        )
    )

    discovered = discover_command_hooks(settings)

    assert [item.name for item in discovered] == ["enabled"]
