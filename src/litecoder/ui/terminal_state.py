"""Shared terminal renderer state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from rich.console import Console

_WAIT_STATUS_ATTRIBUTE = "_litecoder_waiting_status"
_TODO_PROGRESS_ATTRIBUTE = "_litecoder_todo_progress"
_FINISHING_MESSAGE = "Finishing response..."
_TODO_VISIBLE_ITEMS = 6
_WAITING_MESSAGE = "Waiting for response..."


@dataclass(frozen=True, slots=True)
class TodoProgressItem:
    """One item in the terminal Todo progress display."""

    content: str
    active_form: str
    status: str


def replace_todo_progress(console: Console, value: object) -> bool:
    """Replace the terminal Todo progress when the value is valid."""

    items = _parse_todo_progress(value)
    if items is None:
        return False
    setattr(console, _TODO_PROGRESS_ATTRIBUTE, items)
    _refresh_waiting_status(console)
    return True


def clear_todo_progress(console: Console) -> None:
    """Clear the current terminal Todo progress."""

    if getattr(console, _TODO_PROGRESS_ATTRIBUTE, ()):
        setattr(console, _TODO_PROGRESS_ATTRIBUTE, ())
        _refresh_waiting_status(console)


def todo_progress_items(console: Console) -> tuple[TodoProgressItem, ...]:
    """Return the validated Todo progress stored on a console."""

    value = getattr(console, _TODO_PROGRESS_ATTRIBUTE, ())
    if not isinstance(value, tuple) or any(
        not isinstance(item, TodoProgressItem) for item in value
    ):
        return ()
    return value


def has_live_waiting_surface(console: Console) -> bool:
    """Return whether a Rich waiting status is currently active."""

    return getattr(console, _WAIT_STATUS_ATTRIBUTE, None) is not None


def stop_waiting_status(console: Console) -> None:
    """Stop the active Rich waiting status, if any."""

    status = getattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    if status is None:
        return
    if getattr(console, _WAIT_STATUS_ATTRIBUTE, None) is status:
        setattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    stop = getattr(status, "stop", None)
    if callable(stop):
        stop()


@contextmanager
def suspend_waiting_status(console: Console):  # type: ignore[no-untyped-def]
    """Temporarily pause the active Rich waiting status."""

    status = getattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    start = None
    if status is not None:
        stop = getattr(status, "stop", None)
        start = getattr(status, "start", None)
        if callable(stop):
            stop()
    try:
        yield
    finally:
        if (
            status is not None
            and getattr(console, _WAIT_STATUS_ATTRIBUTE, None) is status
            and callable(start)
        ):
            start()


def _waiting_status_message(console: Console) -> str:
    items = todo_progress_items(console)
    if not items:
        return _WAITING_MESSAGE
    lines = [_todo_progress_heading(items)]
    for index, (_, text) in enumerate(_todo_progress_lines(items)):
        prefix = "└ " if index == 0 else "  "
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


def _todo_progress_heading(items: tuple[TodoProgressItem, ...]) -> str:
    active = next((item for item in items if item.status == "in_progress"), None)
    if active is not None:
        return _with_ellipsis(active.active_form)
    pending = next((item for item in items if item.status == "pending"), None)
    if pending is not None:
        return _with_ellipsis(pending.active_form)
    return _FINISHING_MESSAGE


def _with_ellipsis(value: str) -> str:
    stripped = value.rstrip()
    return stripped if stripped.endswith(("...", "…")) else f"{stripped}..."


def _todo_progress_lines(
    items: tuple[TodoProgressItem, ...],
) -> list[tuple[str, str]]:
    if not items:
        return []
    start, end = _todo_visible_range(items)
    lines: list[tuple[str, str]] = []
    if start:
        hidden = items[:start]
        completed = sum(item.status == "completed" for item in hidden)
        label = "completed" if completed == len(hidden) else "earlier"
        lines.append(("class:todo.summary", f"… +{len(hidden)} {label}"))
    for item in items[start:end]:
        symbol = "✓" if item.status == "completed" else "□"
        style = {
            "completed": "class:todo.completed",
            "in_progress": "class:todo.active",
            "pending": "class:todo.pending",
        }[item.status]
        lines.append((style, f"{symbol} {item.content}"))
    if end < len(items):
        hidden = items[end:]
        pending = sum(item.status == "pending" for item in hidden)
        label = "pending" if pending == len(hidden) else "remaining"
        lines.append(("class:todo.summary", f"… +{len(hidden)} {label}"))
    return lines


def _todo_visible_range(items: tuple[TodoProgressItem, ...]) -> tuple[int, int]:
    if len(items) <= _TODO_VISIBLE_ITEMS:
        return 0, len(items)
    focus = next(
        (index for index, item in enumerate(items) if item.status == "in_progress"),
        next(
            (index for index, item in enumerate(items) if item.status == "pending"),
            len(items) - 1,
        ),
    )
    start = 0 if focus < _TODO_VISIBLE_ITEMS else max(0, focus - 2)
    end = min(len(items), start + _TODO_VISIBLE_ITEMS)
    return max(0, end - _TODO_VISIBLE_ITEMS), end


def _parse_todo_progress(
    value: object,
) -> tuple[TodoProgressItem, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    items: list[TodoProgressItem] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            return None
        content = raw.get("content")
        active_form = raw.get("active_form")
        status = raw.get("status")
        if (
            not isinstance(content, str)
            or not content.strip()
            or not isinstance(active_form, str)
            or not active_form.strip()
            or status not in {"pending", "in_progress", "completed"}
        ):
            return None
        items.append(TodoProgressItem(content, active_form, status))
    return tuple(items)


def _refresh_waiting_status(console: Console) -> None:
    status = getattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    update = getattr(status, "update", None)
    if callable(update):
        with suppress(Exception):
            update(status=_waiting_status_message(console))
