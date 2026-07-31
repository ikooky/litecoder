from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from rich.console import Console

import litecoder.ui.permissions as permissions_module
from litecoder.tools.permission import PermissionPrompt, PromptChoice
from litecoder.ui.permissions import select_permission_choice


def test_permission_prompt_selects_allow_with_down_arrow() -> None:
    keys = iter(("down", "enter"))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)

    choice = select_permission_choice(
        PermissionPrompt("run_shell", "external", "external:abc"),
        console=console,
        read_key=lambda: next(keys),
    )

    output = console.file.getvalue()
    assert choice is PromptChoice.ALLOW_ONCE
    assert "run_shell" in output
    assert "external:abc" in output
    assert "allow" in output
    assert "always" in output
    assert "deny" in output
    assert "Allow once" not in output
    assert "Allow for root session" not in output
    assert "Deny" not in output


def test_permission_prompt_shows_shell_command_and_absolute_cwd() -> None:
    keys = iter(("enter",))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)

    choice = select_permission_choice(
        PermissionPrompt(
            "run_shell",
            "high",
            "external:abc",
            {"argv": ["cmd", "/c", "dir"], "cwd": "."},
        ),
        console=console,
        read_key=lambda: next(keys),
    )

    output = console.file.getvalue()
    assert choice is PromptChoice.DENY
    assert "Permission Bash(cmd /c dir)" in output
    assert f"cwd: {Path.cwd().resolve()}" in output
    assert "Permission run_shell" not in output


def test_permission_prompt_resolves_cwd_from_workspace_root() -> None:
    keys = iter(("enter",))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    workspace = Path("C:/workspace/root")

    select_permission_choice(
        PermissionPrompt(
            "run_shell",
            "high",
            "external:abc",
            {"argv": ["python", "-m", "pytest"], "cwd": "tests"},
            workspace_root=str(workspace),
        ),
        console=console,
        read_key=lambda: next(keys),
    )

    output = console.file.getvalue()
    assert f"cwd: {(workspace / 'tests').resolve()}" in output


def test_permission_prompt_fails_closed_on_eof() -> None:
    choice = select_permission_choice(
        PermissionPrompt("write_file", "workspace", "workspace:def"),
        console=Console(file=StringIO(), force_terminal=True, color_system=None),
        read_key=lambda: (_ for _ in ()).throw(EOFError()),
    )

    assert choice is PromptChoice.DENY


def test_permission_prompt_times_out_to_deny(monkeypatch) -> None:
    observed: list[float] = []

    def timeout_reader(timeout_seconds: float) -> str:
        observed.append(timeout_seconds)
        raise TimeoutError

    monkeypatch.setattr(permissions_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(permissions_module, "_read_permission_key", timeout_reader)
    console = Console(file=StringIO(), force_terminal=True, color_system=None)

    choice = select_permission_choice(
        PermissionPrompt("write_file", "workspace", "workspace:def"),
        console=console,
        timeout_seconds=0.25,
    )

    assert choice is PromptChoice.DENY
    assert len(observed) == 1
    assert 0 < observed[0] <= 0.25
    assert "No response within 0.25 seconds defaults to deny." in console.file.getvalue()


@pytest.mark.parametrize(
    "timeout_seconds", [True, 0, -1, float("nan"), float("inf")]
)
def test_permission_prompt_rejects_invalid_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        select_permission_choice(
            PermissionPrompt("write_file", "workspace", "workspace:def"),
            console=Console(file=StringIO(), force_terminal=True, color_system=None),
            read_key=lambda: "enter",
            timeout_seconds=timeout_seconds,
        )


def test_permission_prompt_escape_interrupts_turn() -> None:
    from litecoder.ui.permissions import PermissionInputInterrupt

    with pytest.raises(PermissionInputInterrupt) as error:
        select_permission_choice(
            PermissionPrompt("run_shell", "external", "external:def"),
            console=Console(file=StringIO(), force_terminal=True, color_system=None),
            read_key=lambda: "escape",
        )

    assert error.value.source == "escape"


def test_permission_prompt_ctrl_c_interrupts_turn() -> None:
    from litecoder.ui.permissions import PermissionInputInterrupt

    def interrupt() -> str:
        raise KeyboardInterrupt

    with pytest.raises(PermissionInputInterrupt) as error:
        select_permission_choice(
            PermissionPrompt("run_shell", "external", "external:def"),
            console=Console(file=StringIO(), force_terminal=True, color_system=None),
            read_key=interrupt,
        )

    assert error.value.source == "ctrl_c"
