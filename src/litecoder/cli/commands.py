"""Supporting implementation for commands."""

from __future__ import annotations

from pathlib import Path

from litecoder.context.session.store import (
    DeleteSessionTreeResult,
    SQLiteSessionStore,
)
from litecoder.paths import AppPaths


def current_paths() -> AppPaths:
    """Handle the current paths operation."""
    return AppPaths.discover(Path.cwd())


async def open_session_store(paths: AppPaths) -> SQLiteSessionStore:
    """Open the session store."""
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    return store


def trace_path(paths: AppPaths, root_session_id: str) -> Path:
    """Handle the trace path operation."""
    return paths.trace_path(root_session_id)


def remove_session_files(
    paths: AppPaths,
    deletion: DeleteSessionTreeResult,
) -> None:
    """Remove the session files."""
    trace_path(paths, deletion.root_session_id).unlink(missing_ok=True)
    for artifact_path in _safe_artifact_paths(paths, deletion.artifact_paths):
        artifact_path.unlink(missing_ok=True)


def _safe_artifact_paths(
    paths: AppPaths,
    artifact_paths: tuple[Path, ...],
) -> list[Path]:
    root = (paths.project_dir / "outputs").resolve()
    safe: list[Path] = []
    for artifact_path in artifact_paths:
        try:
            candidate = artifact_path.expanduser().resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        safe.append(candidate)
    return safe