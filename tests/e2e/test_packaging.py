from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

FORBIDDEN_PARTS = (
    ".pytest",
    "__pycache__",
    ".pyc",
    "/dist/",
    "/build/",
    ".egg-info",
)


def test_distributions_exclude_generated_paths(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        check=True,
    )

    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = archive.getnames()

    for name in [*wheel_names, *sdist_names]:
        normalized = f"/{name.replace(chr(92), '/')}"
        assert not any(part in normalized for part in FORBIDDEN_PARTS)

    assert any(name.startswith("litecoder/") for name in wheel_names)
    assert any("/src/litecoder/" in f"/{name}" for name in sdist_names)
    assert any(name.endswith("/README.md") for name in sdist_names)
    assert any(name.endswith("/config.example.toml") for name in sdist_names)
    assert any(name.endswith("/pyproject.toml") for name in sdist_names)
