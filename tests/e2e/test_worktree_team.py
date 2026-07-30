from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e


def test_tasks_list_smoke_in_git_workspace_without_model(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)

    result = _run_module(["litecoder", "tasks", "list"], repo, tmp_path / "home")

    assert result.returncode == 0, result.stderr + result.stdout
    assert "No tasks." in result.stdout


def _run_module(
    module_args: list[str], cwd: Path, home: Path
) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", *module_args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
