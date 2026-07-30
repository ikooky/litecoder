"""Filesystem identity, project paths, and trace locations."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class TracePathUnavailableError(ValueError):
    """Raised when a safe trace or audit path cannot be constructed."""
    pass


def canonical_path(path: Path) -> Path:
    """Return a normalized, absolute representation of *path*."""
    resolved = path.expanduser().resolve()
    return Path(os.path.normcase(str(resolved)))


def stable_path_id(path: Path) -> str:
    """Return a stable identifier derived from a canonical filesystem path."""
    value = canonical_path(path).as_posix().encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def _git_path(cwd: Path, argument: str) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", argument],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        },
    )
    if result.returncode != 0:
        return None
    candidate = Path(result.stdout.strip())
    return canonical_path(candidate if candidate.is_absolute() else cwd / candidate)


def resolve_git_common_directory(cwd: Path) -> Path | None:
    """Resolve the Git common directory for the working tree, when available."""
    return _git_path(cwd, "--git-common-dir")


def resolve_project_identity(cwd: Path) -> tuple[str, Path]:
    """Resolve the project identifier and canonical storage root."""
    root = resolve_git_common_directory(cwd) or canonical_path(cwd)
    return stable_path_id(root), root


def resolve_workspace_identity(cwd: Path) -> tuple[str, Path]:
    """Resolve the workspace identifier and canonical checkout root."""
    root = _git_path(cwd, "--show-toplevel") or canonical_path(cwd)
    return stable_path_id(root), root


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Resolved user, project, workspace, and runtime storage locations."""
    user_dir: Path
    sessions_db: Path
    project_id: str
    project_dir: Path
    workspace_id: str
    workspace_root: Path

    @property
    def lock_dir(self) -> Path:
        """Return the directory used for process locks."""
        return self.user_dir / "locks"

    def trace_path(self, root_session_id: str) -> Path:
        """Return the validated JSONL trace path for a root session."""
        try:
            project_root = self.project_dir.expanduser().resolve()
            trace_root = (project_root / "traces").resolve()
            trace_root.relative_to(project_root)
            candidate = (trace_root / f"{root_session_id}.jsonl").resolve()
            candidate.relative_to(trace_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise TracePathUnavailableError("Trace is unavailable") from error
        return candidate

    @property
    def command_audit_path(self) -> Path:
        """Return the validated path used for command audit records."""
        try:
            project_root = self.project_dir.expanduser().resolve()
            audit_root = (project_root / "audit").resolve()
            audit_root.relative_to(project_root)
            candidate = (audit_root / "commands.jsonl").resolve()
            candidate.relative_to(audit_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise TracePathUnavailableError("Command audit is unavailable") from error
        return candidate

    @classmethod
    def discover(
        cls,
        cwd: Path,
        home: Path | None = None,
        *,
        isolated: bool = False,
    ) -> "AppPaths":
        """Discover application paths from a working directory and optional home."""
        user_dir = (home or Path.home()) / ".litecoder"
        if isolated:
            workspace_root = canonical_path(cwd)
            project_id = stable_path_id(workspace_root)
            workspace_id = project_id
        else:
            project_id, _ = resolve_project_identity(cwd)
            workspace_id, workspace_root = resolve_workspace_identity(cwd)
        return cls(
            user_dir=user_dir,
            sessions_db=user_dir / "sessions.db",
            project_id=project_id,
            project_dir=user_dir / "projects" / project_id,
            workspace_id=workspace_id,
            workspace_root=workspace_root,
        )
