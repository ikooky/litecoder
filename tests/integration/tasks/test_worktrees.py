from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from litecoder.hooks import HookManager
from litecoder.paths import stable_path_id
from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskCreate
from litecoder.tasks.store import TaskStore
from litecoder.tasks.worktrees import (
    GitResult,
    WorktreeBinding,
    WorktreeError,
    WorktreeManager,
    _parse_porcelain,
    run_git,
)
from litecoder.tools import (
    DuplicateGuard,
    PermissionMode,
    PermissionService,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    WorkspaceStateRegistry,
)
from litecoder.tools.builtin.worktree import WorktreeRemoveTool, register_worktree_tools


class _Trace:
    async def record(self, _fact: object) -> None:
        return None


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "LiteCoder Tests")
    _git(repository, "config", "user.email", "litecoder@example.invalid")
    (repository / "README.md").write_text("initial\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-qm", "initial")
    return repository


def _context(root: Path, *, permission_mode: str = "ask") -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": permission_mode},
    )


def _executor(tool: object, *, permission_mode: str = "ask") -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(tool)  # type: ignore[arg-type]
    return ToolExecutor(
        registry,
        HookManager(trace_hook=_Trace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"),
        WorkspaceStateRegistry(),
    )


@pytest.mark.asyncio
async def test_run_git_scopes_safe_directory_to_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def spawn(*argv: str, **kwargs: object) -> Process:
        captured.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    result = await run_git(tmp_path, "status", "--short")

    assert result.returncode == 0
    assert captured[0][0] == (
        "git",
        "-c",
        f"safe.directory={tmp_path.resolve().as_posix()}",
        "status",
        "--short",
    )


@pytest.mark.asyncio
async def test_linked_worktrees_have_same_project_and_different_workspace_ids(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")

    first = await manager.create("task-a", "branch-a")
    second = await manager.create("task-b", "branch-b")

    assert first.project_id == second.project_id
    assert first.workspace_id != second.workspace_id
    assert first.path == first.workspace_root
    assert first.id != first.task_id
    assert first.id.isascii() and first.id.isalnum()

    listed = await manager.list()
    assert {binding.task_id for binding in listed} == {"task-a", "task-b"}
    assert not list((tmp_path / "worktrees").glob("*.json"))


@pytest.mark.asyncio
async def test_worktree_preserves_case_sensitive_task_id_on_windows(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")

    created = await manager.create("HumanEval-4-impl", "task/humaneval-4")
    listed = await manager.list()

    assert created.task_id == "HumanEval-4-impl"
    assert listed[0].task_id == "HumanEval-4-impl"
    assert listed[0].id == created.id


@pytest.mark.asyncio
async def test_list_reconciles_git_truth_and_never_returns_main_worktree(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    worktree_root = tmp_path / "worktrees"
    manager = WorktreeManager(repository, worktree_root)
    worktree_root.mkdir()
    external = worktree_root / "external-task"
    _git(repository, "worktree", "add", "-b", "external-branch", str(external))

    listed = await manager.list()
    assert listed == (
        WorktreeBinding(
            "external-task",
            "external-branch",
            listed[0].workspace_id,
            external.resolve(),
            listed[0].project_id,
            listed[0].head,
        ),
    )
    assert all(binding.path != repository.resolve() for binding in listed)


@pytest.mark.asyncio
async def test_project_lock_is_shared_by_managers_from_linked_worktrees(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    worktree_root = tmp_path / "worktrees"
    first_manager = WorktreeManager(repository, worktree_root)
    first = await first_manager.create("task-a", "branch-a")
    linked_manager = WorktreeManager(first.workspace_root, worktree_root)

    active = 0
    maximum_active = 0

    def instrument(manager: WorktreeManager) -> None:
        original = manager._run_git

        async def wrapped(cwd: Path, *args: str):
            nonlocal active, maximum_active
            if args[:4] == ("worktree", "list", "--porcelain", "-z"):
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0.02)
                try:
                    return await original(cwd, *args)
                finally:
                    active -= 1
            return await original(cwd, *args)

        manager._run_git = wrapped  # type: ignore[method-assign]

    instrument(first_manager)
    instrument(linked_manager)
    await asyncio.gather(first_manager.list(), linked_manager.list())

    assert first_manager.project_id == linked_manager.project_id == first.project_id
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_git_environment_cannot_redirect_project_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    manager = WorktreeManager(repository, tmp_path / "worktrees")

    binding = await manager.create("task-a", "branch-a")
    monkeypatch.delenv("GIT_DIR")

    assert binding.path.exists()
    assert "branch-a" in _git(repository, "branch", "--format=%(refname:short)").stdout
    assert "branch-a" not in _git(other, "branch", "--format=%(refname:short)").stdout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_id", "branch"),
    [
        ("../outside", "safe"),
        ("task/child", "safe"),
        ("task.", "safe"),
        ("task", "../evil"),
        ("task", "-bad"),
        ("task", "feature/.hidden"),
    ],
)
async def test_create_rejects_traversal_and_option_like_values(
    tmp_path: Path, task_id: str, branch: str
) -> None:
    manager = WorktreeManager(_repository(tmp_path), tmp_path / "worktrees")
    with pytest.raises(ValueError):
        await manager.create(task_id, branch)


@pytest.mark.asyncio
async def test_remove_requires_confirmed_permission_before_git_mutation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    binding = await manager.create("task-a", "branch-a")
    tool = WorktreeRemoveTool(manager)
    executor = _executor(tool)

    result = await executor.execute(
        ToolCall("remove", "worktree_remove", {"id": binding.id}),
        _context(binding.path, permission_mode=PermissionMode.READ_ONLY),
    )

    assert result.status == "denied"
    assert binding.path.exists()
    assert any(item.path == binding.path for item in await manager.list())


@pytest.mark.asyncio
async def test_confirmed_remove_reconciles_and_removes_git_worktree(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    binding = await manager.create("task-a", "branch-a")
    executor = _executor(WorktreeRemoveTool(manager))

    result = await executor.execute(
        ToolCall("remove", "worktree_remove", {"id": binding.id}),
        _context(binding.path),
    )

    assert result.status == "success"
    assert not binding.path.exists()
    assert await manager.list() == ()


@pytest.mark.asyncio
async def test_remove_rejects_binding_that_no_longer_matches_git_truth(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    binding = await manager.create("task-a", "branch-a")
    _git(repository, "worktree", "remove", str(binding.path))

    with pytest.raises(Exception, match="binding"):
        await manager.remove(binding.id)


def test_porcelain_z_parser_preserves_unquoted_paths_head_branch_and_prunable(
    tmp_path: Path,
) -> None:
    special = tmp_path / '工作 trees "quoted" \\ segment'
    output = (
        f"worktree {special}\0"
        "HEAD 0123456789abcdef0123456789abcdef01234567\0"
        "branch refs/heads/feature/safe\0"
        "prunable gitdir file points to non-existent location\0\0"
    )

    parsed = _parse_porcelain(output)

    assert len(parsed) == 1
    assert parsed[0].path == special.resolve()
    assert parsed[0].head == "0123456789abcdef0123456789abcdef01234567"
    assert parsed[0].branch == "feature/safe"
    assert parsed[0].prunable is True


@pytest.mark.asyncio
async def test_missing_prunable_worktree_lists_without_using_missing_path_as_cwd(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees 空 格")
    binding = await manager.create("task-a", "branch-a")
    shutil.rmtree(binding.path)

    listed = await manager.list()

    assert len(listed) == 1
    assert listed[0].task_id == "task-a"
    assert listed[0].path == binding.path
    assert listed[0].workspace_id == binding.workspace_id


@pytest.mark.asyncio
async def test_remove_prunable_binding_skips_prune_when_git_removes_metadata(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    binding = await manager.create("task-a", "branch-a")
    shutil.rmtree(binding.path)
    calls: list[tuple[str, ...]] = []
    original = manager._run_git

    async def recording(cwd: Path, *args: str):
        calls.append(args)
        return await original(cwd, *args)

    manager._run_git = recording  # type: ignore[method-assign]

    removed = await manager.remove(binding.id, discard=True)

    assert removed.id == binding.id
    assert ("worktree", "remove", "--force", "--", str(binding.path)) in calls
    assert ("worktree", "prune", "--expire", "now") not in calls
    assert await manager.list() == ()


@pytest.mark.asyncio
async def test_remove_prunable_binding_prunes_when_target_is_only_stale_record(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    binding = await manager.create("task-a", "branch-a")
    shutil.rmtree(binding.path)
    calls: list[tuple[str, ...]] = []
    original = manager._run_git

    async def recording(cwd: Path, *args: str):
        calls.append(args)
        if args == ("worktree", "remove", "--force", "--", str(binding.path)):
            return GitResult(0, "", "")
        return await original(cwd, *args)

    manager._run_git = recording  # type: ignore[method-assign]

    removed = await manager.remove(binding.id, discard=True)

    assert removed.id == binding.id
    assert ("worktree", "prune", "--expire", "now") in calls
    assert await manager.list() == ()


@pytest.mark.asyncio
async def test_remove_prunable_binding_refuses_to_prune_unrelated_metadata(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    binding = await manager.create("task-a", "branch-a")
    external = tmp_path / "external"
    _git(repository, "worktree", "add", "-b", "external-branch", str(external))
    shutil.rmtree(binding.path)
    shutil.rmtree(external)
    calls: list[tuple[str, ...]] = []
    original = manager._run_git

    async def recording(cwd: Path, *args: str):
        calls.append(args)
        return await original(cwd, *args)

    manager._run_git = recording  # type: ignore[method-assign]

    removed = await manager.remove(binding.id, discard=True)

    assert removed.id == binding.id
    assert ("worktree", "prune", "--expire", "now") not in calls
    records = _parse_porcelain(
        _git(repository, "worktree", "list", "--porcelain", "-z").stdout
    )
    assert not any(item.path == binding.path for item in records)
    assert any(item.path == external.resolve() and item.prunable for item in records)


@pytest.mark.asyncio
async def test_old_binding_id_cannot_remove_recreated_task_path(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    old = await manager.create("task-a", "branch-a")
    await manager.remove(old.id)
    replacement = await manager.create("task-a", "branch-b")

    with pytest.raises(WorktreeError, match="binding"):
        await manager.remove(old.id)

    assert replacement.path.exists()
    assert await manager.list() == (replacement,)


@pytest.mark.asyncio
async def test_old_binding_id_cannot_remove_exact_state_recreated_task_path(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    old = await manager.create("task-a", "branch-a")
    await manager.remove(old.id)
    _git(repository, "branch", "-D", "branch-a")
    replacement = await manager.create("task-a", "branch-a")

    assert replacement.id != old.id
    with pytest.raises(WorktreeError, match="binding"):
        await manager.remove(old.id)

    assert replacement.path.exists()
    assert await manager.list() == (replacement,)


def test_binding_id_changes_for_every_git_identity_component(tmp_path: Path) -> None:
    base = WorktreeBinding(
        "task-a",
        "branch-a",
        "workspace-a",
        tmp_path / "task-a",
        "project-a",
        "a" * 40,
    )
    variants = (
        WorktreeBinding(
            "task-b",
            base.branch,
            base.workspace_id,
            base.path,
            base.project_id,
            base.head,
        ),
        WorktreeBinding(
            base.task_id,
            "branch-b",
            base.workspace_id,
            base.path,
            base.project_id,
            base.head,
        ),
        WorktreeBinding(
            base.task_id,
            base.branch,
            "workspace-b",
            base.path,
            base.project_id,
            base.head,
        ),
        WorktreeBinding(
            base.task_id,
            base.branch,
            base.workspace_id,
            base.path,
            "project-b",
            base.head,
        ),
        WorktreeBinding(
            base.task_id,
            base.branch,
            base.workspace_id,
            base.path,
            base.project_id,
            "b" * 40,
        ),
        WorktreeBinding(
            base.task_id,
            base.branch,
            base.workspace_id,
            base.path,
            base.project_id,
            base.head,
            "1" * 32,
        ),
    )

    assert len({base.id, *(variant.id for variant in variants)}) == 7


@pytest.mark.asyncio
async def test_remove_token_never_targets_main_or_out_of_root_worktrees(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "managed")
    external = tmp_path / "external"
    _git(repository, "worktree", "add", "-b", "external-branch", str(external))
    head = _git(external, "rev-parse", "HEAD").stdout.strip()
    protected = (
        WorktreeBinding(
            "main",
            "master",
            stable_path_id(repository),
            repository.resolve(),
            manager.project_id,
            head,
        ),
        WorktreeBinding(
            "external",
            "external-branch",
            stable_path_id(external),
            external.resolve(),
            manager.project_id,
            head,
        ),
    )

    for binding in protected:
        with pytest.raises(WorktreeError, match="binding"):
            await manager.remove(binding.id)

    assert repository.exists()
    assert external.exists()


def test_remove_hard_guard_accepts_only_opaque_binding_tokens(tmp_path: Path) -> None:
    manager = WorktreeManager(_repository(tmp_path), tmp_path / "worktrees")
    tool = WorktreeRemoveTool(manager)

    assert tool.hard_guard(
        ToolCall("remove", "worktree_remove", {"id": "task-a"}), _context(tmp_path)
    ) is not None
    assert tool.hard_guard(
        ToolCall("remove", "worktree_remove", {"id": "a" * 64}),
        _context(tmp_path),
    ) is None


@pytest.mark.asyncio
async def test_worktree_tool_registration_and_json_output(tmp_path: Path) -> None:
    manager = WorktreeManager(_repository(tmp_path), tmp_path / "worktrees")
    registry = ToolRegistry()
    task_manager = TaskManager(TaskStore(tmp_path / "tasks"))
    await task_manager.create(TaskCreate("task-a", "Task A", "Task A"))
    register_worktree_tools(registry, manager, task_manager=task_manager)
    assert {tool.spec.name for tool in registry.list()} == {
        "worktree_create",
        "worktree_list",
        "worktree_remove",
    }
    executor = _executor(registry.require("worktree_create"))
    created = await executor.execute(
        ToolCall(
            "create", "worktree_create", {"task_id": "task-a", "branch": "branch-a"}
        ),
        _context(tmp_path),
    )
    assert created.status == "success"
    payload = json.loads(created.content)
    assert payload["task_id"] == "task-a"
    assert (await task_manager.get("task-a")).worktree_id == payload["id"]

    removed = await _executor(registry.require("worktree_remove")).execute(
        ToolCall("remove", "worktree_remove", {"id": payload["id"]}),
        _context(tmp_path),
    )
    assert removed.status == "success"
    assert (await task_manager.get("task-a")).worktree_id is None


@pytest.mark.asyncio
async def test_worktree_list_reconciles_stale_task_binding(tmp_path: Path) -> None:
    manager = WorktreeManager(_repository(tmp_path), tmp_path / "worktrees")
    task_manager = TaskManager(TaskStore(tmp_path / "tasks"))
    await task_manager.create(TaskCreate("task-a", "Task A", "Task A"))
    await task_manager.bind_worktree("task-a", "a" * 64)
    registry = ToolRegistry()
    register_worktree_tools(registry, manager, task_manager=task_manager)

    result = await _executor(registry.require("worktree_list")).execute(
        ToolCall("list", "worktree_list", {}),
        _context(tmp_path),
    )

    assert result.status == "success"
    assert json.loads(result.content) == []
    assert (await task_manager.get("task-a")).worktree_id is None
