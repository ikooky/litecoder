from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import sys

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


def test_permission_prompt_pauses_stdout_patch_for_live_refresh() -> None:
    keys = iter(("down", "enter"))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    events: list[str] = []

    class StdoutPatch:
        def suspend(self):  # type: ignore[no-untyped-def]
            class Suspended:
                def __enter__(self) -> None:
                    events.append("pause-enter")

                def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
                    events.append("pause-exit")

            return Suspended()

    setattr(console, permissions_module.input_module._STDOUT_PATCH_ATTRIBUTE, StdoutPatch())

    choice = select_permission_choice(
        PermissionPrompt("run_shell", "external", "external:abc"),
        console=console,
        read_key=lambda: next(keys),
    )

    output = console.file.getvalue()
    assert choice is PromptChoice.ALLOW_ONCE
    assert events == ["pause-enter", "pause-exit"]
    assert "> deny" in output
    assert "> allow" in output


def test_permission_prompt_bypasses_stdout_proxy_for_visible_updates(monkeypatch) -> None:
    keys = iter(("down", "enter"))
    original_stdout = StringIO()
    original_stderr = StringIO()
    proxy_writes: list[str] = []

    class Proxy:
        def __init__(self, *, raw: bool) -> None:
            self.raw = raw

        def write(self, value: str) -> int:
            proxy_writes.append(value)
            return len(value)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

        def isatty(self) -> bool:
            return True

        @property
        def encoding(self) -> str:
            return "utf-8"

    monkeypatch.setattr("prompt_toolkit.patch_stdout.StdoutProxy", Proxy)
    monkeypatch.setattr(sys, "stdout", original_stdout)
    monkeypatch.setattr(sys, "stderr", original_stderr)

    console = Console(force_terminal=True, color_system=None, width=80)
    with permissions_module.input_module._patched_stdout_context(console):
        choice = select_permission_choice(
            PermissionPrompt("run_shell", "external", "external:abc"),
            console=console,
            read_key=lambda: next(keys),
        )

    output = original_stdout.getvalue()
    assert choice is PromptChoice.ALLOW_ONCE
    assert "> deny" in output
    assert "> allow" in output
    assert "Permission" not in "".join(proxy_writes)


def test_permission_prompt_suspends_waiting_status(monkeypatch) -> None:
    keys = iter(("enter",))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    events: list[str] = []

    class Suspended:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
            events.append("exit")

    monkeypatch.setattr(
        permissions_module.input_module,
        "suspend_waiting_status",
        lambda selected_console: events.append(str(selected_console is console)) or Suspended(),
    )

    choice = select_permission_choice(
        PermissionPrompt("write_file", "workspace", "workspace:def"),
        console=console,
        read_key=lambda: next(keys),
    )

    assert choice is PromptChoice.DENY
    assert events == ["True", "enter", "exit"]


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
    from litecoder.ui.input import InputInterrupt

    with pytest.raises(InputInterrupt) as error:
        select_permission_choice(
            PermissionPrompt("run_shell", "external", "external:def"),
            console=Console(file=StringIO(), force_terminal=True, color_system=None),
            read_key=lambda: "escape",
        )

    assert error.value.source == "escape"


def test_permission_prompt_ctrl_c_interrupts_turn() -> None:
    from litecoder.ui.input import InputInterrupt

    def interrupt() -> str:
        raise KeyboardInterrupt

    with pytest.raises(InputInterrupt) as error:
        select_permission_choice(
            PermissionPrompt("run_shell", "external", "external:def"),
            console=Console(file=StringIO(), force_terminal=True, color_system=None),
            read_key=interrupt,
        )

    assert error.value.source == "ctrl_c"
