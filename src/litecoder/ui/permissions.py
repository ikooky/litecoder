"""Interactive permission prompt rendering."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from litecoder.tools.permission import (
    PERMISSION_CONFIRMATION_TIMEOUT_SECONDS,
    PermissionPrompt,
    PromptChoice,
)

PermissionKeyReader = Callable[[], str]


class PermissionInputInterrupt(KeyboardInterrupt):
    """Signal that an interactive permission prompt was interrupted."""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.source = source


_CHOICES: tuple[PromptChoice, ...] = (
    PromptChoice.DENY,
    PromptChoice.ALLOW_ONCE,
    PromptChoice.ALLOW_FOR_ROOT_SESSION,
)

def select_permission_choice(
    prompt: PermissionPrompt,
    *,
    console: Console | None = None,
    read_key: PermissionKeyReader | None = None,
    timeout_seconds: float = PERMISSION_CONFIRMATION_TIMEOUT_SECONDS,
) -> PromptChoice:
    """Select the permission choice."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    output = console or Console()
    if read_key is None and not _stdin_is_interactive():
        return PromptChoice.DENY
    selected = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        with Live(
            _permission_menu(prompt, selected, timeout_seconds),
            console=output,
            auto_refresh=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            live.refresh()
            _flush_console(output)
            while True:
                if read_key is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return PromptChoice.DENY
                    key = _read_permission_key(remaining)
                else:
                    key = read_key()
                if key in {"up", "k"}:
                    selected = (selected - 1) % len(_CHOICES)
                    live.update(
                        _permission_menu(prompt, selected, timeout_seconds),
                        refresh=True,
                    )
                    _flush_console(output)
                    continue
                if key in {"down", "j", "tab"}:
                    selected = (selected + 1) % len(_CHOICES)
                    live.update(
                        _permission_menu(prompt, selected, timeout_seconds),
                        refresh=True,
                    )
                    _flush_console(output)
                    continue
                if key in {"enter", "\r", "\n"}:
                    return _CHOICES[selected]
                if key == "escape":
                    raise PermissionInputInterrupt("escape")
                if key == "q":
                    return PromptChoice.DENY
                direct = _direct_choice(key)
                if direct is not None:
                    return direct
    except PermissionInputInterrupt:
        raise
    except KeyboardInterrupt as error:
        raise PermissionInputInterrupt("ctrl_c") from error
    except (EOFError, StopIteration, TimeoutError):
        return PromptChoice.DENY


def _flush_console(console: Console) -> None:
    flush = getattr(console.file, "flush", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            return


def _permission_menu(
    prompt: PermissionPrompt, selected: int, timeout_seconds: float
) -> Group:
    lines: list[Text] = [
        Text.assemble(
            ("Permission ", "yellow"),
            (_permission_title(prompt), "bold yellow"),
        ),
        Text(f"Risk: {prompt.risk}  Scope: {prompt.scope}", style="dim"),
    ]
    for detail in _permission_detail_lines(prompt):
        lines.append(Text(detail, style="dim"))
    lines.append(
        Text(
            f"No response within {timeout_seconds:g} seconds defaults to deny.",
            style="dim italic",
        )
    )
    for index, choice in enumerate(_CHOICES):
        marker = ">" if index == selected else " "
        line = Text(f"{marker} {_choice_label(choice)}")
        if index == selected:
            line.stylize("reverse bold")
        lines.append(line)
    return Group(*lines)


def _permission_title(prompt: PermissionPrompt) -> str:
    arguments = prompt.arguments
    if prompt.tool_name == "run_shell":
        command = _shell_command(arguments)
        return f"Bash({command})" if command else prompt.tool_name
    path = _optional_string(arguments.get("path"))
    if prompt.tool_name == "read_file" and path:
        return f"Read({_absolute_path(path, prompt.workspace_root)})"
    if prompt.tool_name == "write_file" and path:
        return f"Write({_absolute_path(path, prompt.workspace_root)})"
    if prompt.tool_name == "edit_file" and path:
        return f"Edit({_absolute_path(path, prompt.workspace_root)})"
    pattern = _optional_string(arguments.get("pattern"))
    if prompt.tool_name == "glob_files" and pattern:
        return f"Glob({_absolute_path(pattern, prompt.workspace_root)})"
    return prompt.tool_name


def _permission_detail_lines(prompt: PermissionPrompt) -> list[str]:
    arguments = prompt.arguments
    lines: list[str] = []
    if prompt.tool_name == "run_shell":
        cwd = _optional_string(arguments.get("cwd"))
        if cwd:
            lines.append(f"cwd: {_absolute_path(cwd, prompt.workspace_root)}")
        return lines
    path = _optional_string(arguments.get("path"))
    if path:
        lines.append(f"path: {_absolute_path(path, prompt.workspace_root)}")
    return lines


def _shell_command(arguments: Mapping[str, object]) -> str:
    command = _optional_string(arguments.get("command"))
    if command:
        return command
    argv = arguments.get("argv")
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        values = [item for item in argv if isinstance(item, str)]
        if values:
            return subprocess.list2cmdline(values)
    return ""


def _optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _absolute_path(value: str, workspace_root: str = "") -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return str((root / path).resolve())


def _choice_label(choice: PromptChoice) -> str:
    return {
        PromptChoice.DENY: "deny",
        PromptChoice.ALLOW_ONCE: "allow",
        PromptChoice.ALLOW_FOR_ROOT_SESSION: "always",
    }[choice]


def _direct_choice(key: str) -> PromptChoice | None:
    normalized = key.casefold()
    if normalized == "o":
        return PromptChoice.ALLOW_ONCE
    if normalized == "a":
        return PromptChoice.ALLOW_FOR_ROOT_SESSION
    if normalized == "d":
        return PromptChoice.DENY
    return None


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _read_permission_key(timeout_seconds: float) -> str:
    if os.name == "nt":
        return _read_windows_key(timeout_seconds)
    return _read_posix_key(timeout_seconds)


def _read_windows_key(timeout_seconds: float) -> str:
    """Read the windows key."""
    import msvcrt

    deadline = time.monotonic() + timeout_seconds
    while not msvcrt.kbhit():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        time.sleep(min(0.05, remaining))
    character = msvcrt.getwch()
    if character in {"\x00", "\xe0"}:
        code = msvcrt.getwch()
        return {"H": "up", "P": "down"}.get(code, "")
    if character in {"\r", "\n"}:
        return "enter"
    if character == "\x03":
        raise PermissionInputInterrupt("ctrl_c")
    if character == "\x1b":
        return "escape"
    return character.casefold()


def _read_posix_key(timeout_seconds: float) -> str:
    """Read the posix key."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
        if not ready:
            raise TimeoutError
        character = sys.stdin.read(1)
        if character == "":
            raise EOFError
        if character in {"\r", "\n"}:
            return "enter"
        if character == "\x03":
            raise PermissionInputInterrupt("ctrl_c")
        if character == "\x1b":
            sequence = ""
            for _ in range(2):
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    break
                sequence += sys.stdin.read(1)
            return {"[A": "up", "[B": "down"}.get(sequence, "escape")
        return character.casefold()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
