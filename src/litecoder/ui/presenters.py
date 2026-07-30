"""UI event presenters."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from litecoder.ui.events import RuntimeUIEvent
from litecoder.tools.permission import PermissionPrompt

_WINDOWS_DRIVE_RE = re.compile(r"\b([a-z]):(?=[\\/])")


def normalize_windows_drive_letters(value: str) -> str:
    """Normalize the windows drive letters."""
    return _WINDOWS_DRIVE_RE.sub(lambda match: f"{match.group(1).upper()}:", value)


def compact_number(value: int) -> str:
    """Handle the compact number operation."""
    if value < 1_000:
        return str(value)
    units = (
        (1_000_000_000, "b"),
        (1_000_000, "m"),
        (1_000, "k"),
    )
    for threshold, suffix in units:
        if value < threshold:
            continue
        scaled = value / threshold
        rendered = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{rendered}{suffix}"
    return str(value)


def optional_text(value: object) -> str:
    """Handle the optional text operation."""
    return value.strip() if isinstance(value, str) else ""


def event_tool_key(event: RuntimeUIEvent) -> str:
    """Handle the event tool key operation."""
    return event.tool_call_id or f"{event.sequence}:{event.tool_name or 'tool'}"


def event_tool_arguments(event: RuntimeUIEvent) -> Mapping[str, object]:
    """Handle the event tool arguments operation."""
    value = event.payload.get("arguments")
    return value if isinstance(value, Mapping) else {}


def tool_title(
    tool_name: str,
    arguments: Mapping[str, object],
    *,
    workspace_root: str = "",
) -> str:
    """Handle the tool title operation."""
    if tool_name == "run_shell":
        command = shell_command(arguments)
        return f"Bash({command})" if command else "Bash"
    labels = {
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "glob_files": "Glob",
        "search_text": "Search",
    }
    if tool_name in labels:
        value = optional_text(
            arguments.get("pattern" if tool_name == "glob_files" else "path")
        )
        if tool_name == "search_text":
            value = optional_text(arguments.get("pattern"))
        if value:
            return f"{labels[tool_name]}({absolute_path(value, workspace_root)})"
        return labels[tool_name]
    if tool_name == "git_status":
        return "Git status"
    if tool_name == "git_diff":
        return "Git diff"
    if tool_name == "todo_write":
        return "Update Todos"
    summary = argument_summary(arguments)
    return f"{tool_name}({summary})" if summary else tool_name


def shell_command(arguments: Mapping[str, object]) -> str:
    """Handle the shell command operation."""
    command = optional_text(arguments.get("command"))
    if command:
        return normalize_windows_drive_letters(command)
    argv = arguments.get("argv")
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        values = [item for item in argv if isinstance(item, str)]
        if values:
            return normalize_windows_drive_letters(subprocess.list2cmdline(values))
    return ""


def argument_summary(arguments: Mapping[str, object]) -> str:
    """Handle the argument summary operation."""
    if not arguments:
        return ""
    parts: list[str] = []
    for key, value in list(arguments.items())[:2]:
        rendered = optional_text(value)
        if not rendered:
            try:
                rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                rendered = str(value)
        if len(rendered) > 56:
            rendered = rendered[:53] + "..."
        parts.append(f"{key}={rendered}")
    if len(arguments) > 2:
        parts.append("...")
    return ", ".join(parts)


def success_detail_lines(event: RuntimeUIEvent) -> tuple[str, ...]:
    """Handle the success detail lines operation."""
    preview = event.payload.get("preview")
    lines = preview_lines(preview)
    if lines:
        return lines
    metadata = event.payload.get("metadata")
    if isinstance(metadata, Mapping):
        preview = metadata.get("preview")
        lines = preview_lines(preview)
        if lines:
            return lines
        changed = metadata.get("changed_workspace")
        if changed is True:
            return ("Updated workspace",)
    if event.payload.get("changed_workspace") is True:
        return ("Updated workspace",)
    return ("Done",)


def failure_detail_lines(
    event: RuntimeUIEvent,
    *,
    denied: bool = False,
) -> tuple[str, ...]:
    """Handle the failure detail lines operation."""
    message = optional_text(event.payload.get("message"))
    if not message:
        message = optional_text(event.payload.get("reason"))
    if not message:
        message = "Permission denied" if denied else "Tool execution failed"
    return tuple(message.splitlines()) or (message,)


def preview_lines(value: object, *, limit: int = 8) -> tuple[str, ...]:
    """Handle the preview lines operation."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        raw = value.splitlines() or [value]
        return tuple(raw[:limit])
    if isinstance(value, Mapping):
        try:
            return (json.dumps(value, ensure_ascii=False, sort_keys=True),)
        except (TypeError, ValueError):
            return (str(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered: list[str] = []
        for item in value[:limit]:
            if isinstance(item, str):
                rendered.append(item)
            else:
                try:
                    rendered.append(json.dumps(item, ensure_ascii=False))
                except (TypeError, ValueError):
                    rendered.append(str(item))
        return tuple(rendered)
    return (str(value),)


def absolute_path(value: str, workspace_root: str = "") -> str:
    """Handle the absolute path operation."""
    path = Path(value)
    if path.is_absolute():
        return normalize_windows_drive_letters(str(path))
    root = Path(workspace_root) if workspace_root else Path.cwd()
    try:
        return normalize_windows_drive_letters(str((root / path).resolve()))
    except OSError:
        return normalize_windows_drive_letters(str(root / path))


def memory_diagnostic_text(payload: Mapping[str, object]) -> str:
    """Handle the memory diagnostic text operation."""
    memory = payload.get("memory")
    if not isinstance(memory, Mapping):
        return ""
    operation = optional_text(memory.get("operation")) or "memory"
    status = optional_text(memory.get("status")) or "updated"
    return f"Memory {operation}: {status}"


def memory_count_from_payload(payload: Mapping[str, object]) -> int | None:
    """Handle the memory count from payload operation."""
    for key in ("memory_count", "memory_files", "memories"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
    return None


def turn_finished_text(
    event: RuntimeUIEvent,
    tool_invocations: int = 0,
    memory_count: int = 0,
) -> str:
    """Handle the turn finished text operation."""
    status = optional_text(event.payload.get("status")) or "completed"
    elapsed = event.payload.get("elapsed_seconds")
    parts: list[str] = []
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        parts.append(f"Elapsed {float(elapsed):.1f}s")
    total_tokens = event.payload.get("total_tokens")
    if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
        parts.append(f"Tokens: {compact_number(total_tokens)}")
    if tool_invocations:
        parts.append(f"Tools called: {tool_invocations}")
    if memory_count:
        parts.append(f"Memory: {memory_count}")
    label = {
        "completed": "Completed",
        "cancelled": "Cancelled",
        "failed": "Failed",
        "incomplete": "Incomplete",
    }.get(status, status.replace("_", " ").title())
    return f"{label} · " + ", ".join(parts) if parts else label


def permission_title(prompt: PermissionPrompt) -> str:
    """Handle the permission title operation."""
    if prompt.tool_name == "run_shell":
        command = shell_command(prompt.arguments)
        return f"Bash({command})" if command else "Bash"
    path = optional_text(prompt.arguments.get("path"))
    labels = {
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
    }
    if prompt.tool_name in labels and path:
        return (
            f"{labels[prompt.tool_name]}({absolute_path(path, prompt.workspace_root)})"
        )
    pattern = optional_text(prompt.arguments.get("pattern"))
    if prompt.tool_name == "glob_files" and pattern:
        return f"Glob({absolute_path(pattern, prompt.workspace_root)})"
    return tool_title(
        prompt.tool_name, prompt.arguments, workspace_root=prompt.workspace_root
    )


def permission_detail_lines(prompt: PermissionPrompt) -> tuple[str, ...]:
    """Handle the permission detail lines operation."""
    lines = [f"Risk: {prompt.risk}  Scope: {prompt.scope}"]
    cwd = optional_text(prompt.arguments.get("cwd"))
    if cwd:
        lines.append(f"cwd: {absolute_path(cwd, prompt.workspace_root)}")
    return tuple(lines)
