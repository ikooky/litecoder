"""Git worktree lifecycle management."""

from __future__ import annotations

import asyncio
import configparser
import hashlib
import os
import secrets
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from litecoder.paths import (
    canonical_path,
    resolve_project_identity,
    resolve_workspace_identity,
    stable_path_id,
)

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_BINDING_ID = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NONCE = re.compile(r"^[0-9a-f]{32}$")
_GIT_TIMEOUT_SECONDS = 60.0
_INVALID_BRANCH_CHARS = frozenset(" ~^:?*[\\")
_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class WorktreeError(RuntimeError):
    """Raised when the worktree error conditions occur."""
    pass


@dataclass(frozen=True, slots=True)
class GitResult:
    """Data model representing the git result."""
    returncode: int
    stdout: str
    stderr: str


class ProjectGitLock:
    """Component responsible for the project git lock."""
    _locks: dict[str, asyncio.Lock] = {}
    _registry_lock = Lock()

    def __init__(self, project_id: str) -> None:
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project id is invalid")
        self.project_id = project_id
        with self._registry_lock:
            self._lock = self._locks.setdefault(project_id, asyncio.Lock())

    async def __aenter__(self) -> ProjectGitLock:
        await self._lock.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._lock.release()


@dataclass(frozen=True, slots=True)
class WorktreeBinding:
    """Data model representing the worktree binding."""
    task_id: str
    branch: str
    workspace_id: str
    workspace_root: Path
    project_id: str
    head: str
    nonce: str = ""

    @property
    def id(self) -> str:
        """Handle the id operation."""
        identity = "\0".join(
            (
                self.task_id,
                self.branch,
                self.workspace_id,
                self.project_id,
                self.head,
                self.nonce,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @property
    def path(self) -> Path:
        """Handle the path operation."""
        return self.workspace_root


@dataclass(frozen=True, slots=True)
class _GitWorktree:
    """Data model representing the git worktree."""
    path: Path
    head: str
    branch: str
    prunable: bool
    nonce: str = ""
    task_id: str = ""


class WorktreeManager:
    """Manager coordinating the worktree manager."""
    def __init__(self, project_root: Path, worktree_root: Path) -> None:
        if not isinstance(project_root, Path) or not isinstance(worktree_root, Path):
            raise ValueError("worktree paths must be Path instances")
        project_id, _ = resolve_project_identity(project_root)
        _, workspace_root = resolve_workspace_identity(project_root)
        self.project_id = project_id
        self.project_root = workspace_root
        self.worktree_root = canonical_path(worktree_root)
        self.project_git_lock = ProjectGitLock(project_id)
        self._run_git = run_git

    async def create(self, task_id: str, branch: str) -> WorktreeBinding:
        """Create the requested object."""
        task_id = validate_task_id(task_id)
        branch = validate_branch(branch)
        async with self.project_git_lock:
            self.worktree_root.mkdir(parents=True, exist_ok=True)
            path = self._task_path(task_id)
            if path.exists():
                raise WorktreeError("worktree path already exists")
            existing = await self._git_worktrees_locked()
            if any(item.path == path for item in existing):
                raise WorktreeError("worktree binding already exists")
            try:
                result = await self._run_git(
                    self.project_root, "worktree", "add", "-b", branch, "--", str(path)
                )
                if result.returncode != 0:
                    raise WorktreeError("git could not create the worktree")
                nonce = await self._write_binding_metadata_locked(path, task_id)
                reconciled = await self._git_worktrees_locked()
                item = next((entry for entry in reconciled if entry.path == path), None)
                if item is None or item.branch != branch or item.nonce != nonce:
                    raise WorktreeError("created worktree does not match Git truth")
                return self._binding(task_id, item)
            except BaseException:
                await self._cleanup_created_worktree_locked(path)
                raise

    async def _cleanup_created_worktree_locked(self, path: Path) -> None:
        """Best-effort cleanup for a worktree this operation just created."""
        try:
            records = await self._git_worktrees_locked()
            if not any(item.path == path for item in records):
                return
            await self._run_git(
                self.project_root, "worktree", "remove", "--force", "--", str(path)
            )
            await self._run_git(
                self.project_root, "worktree", "prune", "--expire", "now"
            )
        except asyncio.CancelledError:
            # The original error is authoritative. A later reconciliation can
            # safely discover any leftover Git binding.
            return
        except Exception:
            return

    async def list(self) -> tuple[WorktreeBinding, ...]:
        """Return the available entries."""
        async with self.project_git_lock:
            records = await self._git_worktrees_locked()
            return tuple(
                sorted(
                    self._managed_bindings(records),
                    key=lambda binding: binding.task_id,
                )
            )

    async def remove(
        self, binding: str | WorktreeBinding, *, discard: bool = False
    ) -> WorktreeBinding:
        """Remove the requested operation."""
        binding_id = validate_binding_id(
            binding.id if isinstance(binding, WorktreeBinding) else binding
        )
        if not isinstance(discard, bool):
            raise ValueError("discard must be a bool")
        async with self.project_git_lock:
            records = await self._git_worktrees_locked()
            managed = {
                current.id: (current, item)
                for current, item in self._managed_binding_items(records)
            }
            match = managed.get(binding_id)
            if match is None:
                raise WorktreeError("worktree binding is not present in Git truth")
            current, item = match
            if isinstance(binding, WorktreeBinding) and binding != current:
                raise WorktreeError("worktree binding no longer matches Git truth")

            if not discard:
                await self._ensure_removal_safe_locked(item)
            arguments = ["worktree", "remove"]
            if discard:
                arguments.append("--force")
            arguments.extend(("--", str(item.path)))
            result = await self._run_git(
                self.project_root,
                *arguments,
            )
            if result.returncode != 0 and not item.prunable:
                raise WorktreeError("git could not remove the worktree")

            remaining = await self._git_worktrees_locked()
            target = next((entry for entry in remaining if entry.path == item.path), None)
            if target is None:
                return current
            if not target.prunable:
                raise WorktreeError("removed worktree is still present in Git truth")
            prunable = tuple(entry for entry in remaining if entry.prunable)
            if len(prunable) != 1 or prunable[0].path != item.path:
                raise WorktreeError("cannot safely prune unrelated worktree metadata")
            pruned = await self._run_git(
                self.project_root, "worktree", "prune", "--expire", "now"
            )
            if pruned.returncode != 0:
                raise WorktreeError("git could not prune the worktree")
            remaining = await self._git_worktrees_locked()
            if any(entry.path == item.path for entry in remaining):
                raise WorktreeError("removed worktree is still present in Git truth")
            return current

    async def _ensure_removal_safe_locked(self, item: _GitWorktree) -> None:
        denied = "worktree safety cannot be verified; pass discard=True to force removal"
        if item.prunable or not item.path.is_dir():
            raise WorktreeError(denied)
        status = await self._run_git(
            item.path, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if status.returncode != 0:
            raise WorktreeError(denied)
        if status.stdout.strip():
            raise WorktreeError(
                "worktree has uncommitted changes; pass discard=True to force removal"
            )
        remotes = await self._run_git(self.project_root, "remote")
        if remotes.returncode != 0:
            raise WorktreeError(denied)
        if not remotes.stdout.strip():
            base = await self._run_git(self.project_root, "rev-parse", "HEAD")
            base_head = base.stdout.strip()
            if base.returncode != 0 or not base_head:
                raise WorktreeError(denied)
            local_only = await self._run_git(
                item.path, "rev-list", "--count", "HEAD", "--not", base_head
            )
            local_count = local_only.stdout.strip()
            if local_only.returncode != 0 or not local_count.isdecimal():
                raise WorktreeError(denied)
            if int(local_count) > 0:
                raise WorktreeError(
                    "worktree has local-only commits; pass discard=True to force removal"
                )
            return
        upstream = await self._run_git(
            item.path,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        upstream_name = upstream.stdout.strip()
        if upstream.returncode != 0 or not upstream_name:
            raise WorktreeError(denied)
        divergence = await self._run_git(
            item.path,
            "rev-list",
            "--left-right",
            "--count",
            f"{upstream_name}...HEAD",
        )
        counts = divergence.stdout.split()
        if (
            divergence.returncode != 0
            or len(counts) != 2
            or not all(count.isdecimal() for count in counts)
        ):
            raise WorktreeError(denied)
        if int(counts[1]) > 0:
            raise WorktreeError(
                "worktree has unpushed commits; pass discard=True to force removal"
            )

    def _task_path(self, task_id: str) -> Path:
        path = canonical_path(self.worktree_root / task_id)
        if path == self.worktree_root or not _is_within(self.worktree_root, path):
            raise ValueError("task id is invalid")
        return path

    def _binding(self, task_id: str, item: _GitWorktree) -> WorktreeBinding:
        workspace_root = item.path
        return WorktreeBinding(
            task_id,
            item.branch,
            stable_path_id(workspace_root),
            workspace_root,
            self.project_id,
            item.head,
            item.nonce,
        )

    def _managed_bindings(
        self, records: tuple[_GitWorktree, ...]
    ) -> tuple[WorktreeBinding, ...]:
        return tuple(binding for binding, _ in self._managed_binding_items(records))

    def _managed_binding_items(
        self, records: tuple[_GitWorktree, ...]
    ) -> tuple[tuple[WorktreeBinding, _GitWorktree], ...]:
        main = records[0].path if records else None
        managed: list[tuple[WorktreeBinding, _GitWorktree]] = []
        for item in records:
            if item.path == main or not _is_within(self.worktree_root, item.path):
                continue
            try:
                task_id = validate_task_id(
                    item.task_id
                    or item.path.relative_to(self.worktree_root).as_posix()
                )
            except (ValueError, OSError):
                continue
            managed.append((self._binding(task_id, item), item))
        return tuple(managed)

    async def _git_worktrees_locked(self) -> tuple[_GitWorktree, ...]:
        result = await self._run_git(
            self.project_root, "worktree", "list", "--porcelain", "-z"
        )
        if result.returncode != 0:
            raise WorktreeError("git could not list worktrees")
        records = _parse_porcelain(result.stdout)
        common_dir = await self._git_common_dir_locked()
        metadata = _worktree_metadata(common_dir)
        return tuple(
            _GitWorktree(
                item.path,
                item.head,
                item.branch,
                item.prunable,
                metadata.get(item.path, ("", ""))[0],
                metadata.get(item.path, ("", ""))[1],
            )
            for item in records
        )

    async def _git_common_dir_locked(self) -> Path:
        result = await self._run_git(
            self.project_root,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
        value = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if result.returncode != 0 or not value:
            raise WorktreeError("git common directory is unavailable")
        return canonical_path(Path(value))

    async def _write_binding_metadata_locked(
        self, path: Path, task_id: str
    ) -> str:
        """Write the binding metadata locked."""
        nonce = secrets.token_hex(16)
        enabled = await self._run_git(
            self.project_root, "config", "extensions.worktreeConfig", "true"
        )
        if enabled.returncode != 0:
            raise WorktreeError("git could not enable worktree config")
        written = await self._run_git(
            path, "config", "--worktree", "litecoder.bindingNonce", nonce
        )
        if written.returncode != 0:
            raise WorktreeError("git could not record worktree identity")
        task_written = await self._run_git(
            path, "config", "--worktree", "litecoder.taskId", task_id
        )
        if task_written.returncode != 0:
            raise WorktreeError("git could not record worktree task identity")
        return nonce


def validate_task_id(value: object) -> str:
    """Validate the task id."""
    if (
        not isinstance(value, str)
        or not _SAFE_TASK_ID.fullmatch(value)
        or value in {".", ".."}
        or value.endswith(".")
        or value.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES
    ):
        raise ValueError("task id is invalid")
    return value


def validate_binding_id(value: object) -> str:
    """Validate the binding id."""
    if not isinstance(value, str) or not _SAFE_BINDING_ID.fullmatch(value):
        raise ValueError("worktree binding id is invalid")
    return value


def validate_branch(value: object) -> str:
    """Validate the branch."""
    parts = value.split("/") if isinstance(value, str) else ()
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or value.startswith(("-", ".", "/"))
        or value.endswith(("/", ".", ".lock"))
        or ".." in value
        or "//" in value
        or "@{" in value
        or value == "@"
        or any(part.startswith(".") or part.endswith((".", ".lock")) for part in parts)
        or any(character in _INVALID_BRANCH_CHARS for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("branch is invalid")
    return value


async def run_git(cwd: Path, *arguments: str) -> GitResult:
    """Run the git."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_PAGER": "cat",
            "GIT_EDITOR": "true",
        }
    )
    process: asyncio.subprocess.Process | None = None
    try:
        safe_directory = cwd.resolve().as_posix()
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            f"safe.directory={safe_directory}",
            *arguments,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_GIT_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        if process is not None:
            await _stop_git_process(process)
        raise WorktreeError("git operation timed out") from None
    except asyncio.CancelledError:
        if process is not None:
            await _stop_git_process(process)
        raise
    except (OSError, ValueError):
        if process is not None:
            await _stop_git_process(process)
        raise WorktreeError("git is unavailable") from None
    assert process is not None
    return GitResult(
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def _stop_git_process(process: asyncio.subprocess.Process) -> None:
    """Stop the git process."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except asyncio.TimeoutError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        return
    await process.wait()

def _parse_porcelain(output: str) -> tuple[_GitWorktree, ...]:
    """Parse the porcelain."""
    records: list[_GitWorktree] = []
    fields: list[str] = []
    for field in output.split("\0"):
        if field:
            fields.append(field)
            continue
        if not fields:
            continue
        path: Path | None = None
        head = ""
        branch = "(detached)"
        prunable = False
        for value in fields:
            if value.startswith("worktree "):
                path = canonical_path(Path(value.removeprefix("worktree ")))
            elif value.startswith("HEAD "):
                head = value.removeprefix("HEAD ")
            elif value.startswith("branch refs/heads/"):
                branch = value.removeprefix("branch refs/heads/")
            elif value == "prunable" or value.startswith("prunable "):
                prunable = True
        if path is not None:
            records.append(_GitWorktree(path, head, branch, prunable, ""))
        fields = []
    return tuple(records)


def _worktree_metadata(common_dir: Path) -> dict[Path, tuple[str, str]]:
    root = common_dir / "worktrees"
    try:
        admins = tuple(root.iterdir())
    except OSError:
        return {}
    values: dict[Path, tuple[str, str]] = {}
    for admin in admins:
        if not admin.is_dir():
            continue
        worktree = _admin_worktree_path(admin)
        if worktree is None:
            continue
        values[worktree] = _admin_binding_metadata(admin)
    return values


def _admin_worktree_path(admin: Path) -> Path | None:
    try:
        value = (admin / "gitdir").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value:
        return None
    gitdir = Path(value)
    if not gitdir.is_absolute():
        gitdir = admin / gitdir
    path = gitdir.parent if gitdir.name == ".git" else gitdir
    return canonical_path(path)


def _admin_binding_metadata(admin: Path) -> tuple[str, str]:
    parser = configparser.ConfigParser()
    try:
        with (admin / "config.worktree").open(encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeError, configparser.Error):
        return "", ""
    nonce = parser.get("litecoder", "bindingNonce", fallback="").strip()
    task_id = parser.get("litecoder", "taskId", fallback="").strip()
    return (
        nonce if _SAFE_NONCE.fullmatch(nonce) else "",
        task_id if _SAFE_TASK_ID.fullmatch(task_id) else "",
    )


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        common = os.path.commonpath(
            (os.path.normcase(str(root)), os.path.normcase(str(candidate)))
        )
    except ValueError:
        return False
    return common == os.path.normcase(str(root)) and candidate != root
