from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e


def test_config_set_key_subprocess_does_not_echo_secret(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config_path = home / ".litecoder" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8"
    )
    secret = "release-secret-value"

    result = _run_module(
        ["litecoder", "config", "set-key", "anthropic", "--key", secret],
        tmp_path,
        home,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Stored key for anthropic" in result.stdout
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert secret in config_path.read_text(encoding="utf-8")
    _remove_lock_file(config_path.parent / "locks" / "config.toml.lock")


def _remove_lock_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while path.exists():
        try:
            path.unlink()
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


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
