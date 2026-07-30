from pathlib import Path
import subprocess

import litecoder.paths as paths_module

from litecoder.paths import (
    AppPaths,
    canonical_path,
    resolve_project_identity,
    resolve_workspace_identity,
    stable_path_id,
)


def test_canonical_path_resolves_relative_segments(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    assert canonical_path(target / ".") == canonical_path(target)


def test_stable_path_id_normalizes_resolved_path(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    assert stable_path_id(target) == stable_path_id(target / ".")


def test_project_identity_falls_back_to_canonical_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths_module, "resolve_git_common_directory", lambda cwd: None)

    project_id, root = resolve_project_identity(tmp_path)
    assert root == canonical_path(tmp_path)
    assert project_id == stable_path_id(root)


def test_workspace_identity_falls_back_to_canonical_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths_module, "_git_path", lambda cwd, argument: None)

    workspace_id, root = resolve_workspace_identity(tmp_path)
    assert root == canonical_path(tmp_path)
    assert workspace_id == stable_path_id(root)


def test_linked_worktree_shares_project_id_but_not_workspace_id(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    linked_worktree = tmp_path / "linked-worktree"
    repository.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        )

    git("init")
    git("config", "user.name", "LiteCoder Tests")
    git("config", "user.email", "litecoder-tests@example.invalid")
    (repository / "README.md").write_text("# Temporary repository\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "--no-gpg-sign", "-m", "Initial commit")
    git("worktree", "add", "--detach", str(linked_worktree), "HEAD")

    repository_paths = AppPaths.discover(cwd=repository, home=tmp_path / "home")
    worktree_paths = AppPaths.discover(cwd=linked_worktree, home=tmp_path / "home")

    assert repository_paths.project_id == worktree_paths.project_id
    assert repository_paths.workspace_id != worktree_paths.workspace_id


def test_app_paths_separates_project_and_workspace(tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = AppPaths.discover(cwd=workspace, home=home)
    assert paths.user_dir == home / ".litecoder"
    assert paths.sessions_db == home / ".litecoder" / "sessions.db"
    assert paths.project_dir.parent == home / ".litecoder" / "projects"


def test_isolated_app_paths_do_not_climb_to_git_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    case_workspace = repository / "eval-runs" / "run-1" / "cases" / "case-1"
    case_workspace.mkdir(parents=True)

    paths = AppPaths.discover(
        cwd=case_workspace,
        home=tmp_path / "home",
        isolated=True,
    )

    expected_root = canonical_path(case_workspace)
    assert paths.workspace_root == expected_root
    assert paths.workspace_id == stable_path_id(expected_root)
    assert paths.project_id == stable_path_id(expected_root)
    assert paths.workspace_root != canonical_path(repository)

def test_app_paths_exposes_lock_directory_without_creating_it(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    paths = AppPaths.discover(cwd=workspace, home=tmp_path / "home")

    assert paths.lock_dir == paths.user_dir / "locks"
    assert not paths.lock_dir.exists()
