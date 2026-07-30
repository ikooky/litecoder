from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e


def test_readme_documents_only_supported_interactive_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    section_match = re.search(
        r"^### Interactive Commands\s*$\n.*?(?=^## |^### |\Z)",
        readme,
        re.MULTILINE | re.DOTALL,
    )
    assert section_match is not None
    commands = set(re.findall(r"`(/[^`]+)`", section_match.group(0)))
    assert commands == {
        "/clear",
        "/compact",
        "/context",
        "/exit",
        "/help",
        "/memory [name]",
        "/model [provider] [model]",
        "/tasks [task-id]",
        "/trace",
    }
    assert "/cost" not in readme
    assert "/team" not in readme
    assert "not available in this milestone" not in readme


def test_readme_separates_local_and_development_installation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## Quick Start" in readme
    assert 'python -m pip install ".[providers,mcp]"' in readme
    assert "## Development" in readme
    assert 'python -m pip install -e ".[providers,mcp,test]"' in readme


def test_sdist_uses_a_narrow_source_whitelist() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"] == [
        "src/litecoder",
        "README.md",
        "README.zh-CN.md",
        "LICENSE",
        "assets",
        "config.example.toml",
    ]


def test_release_docs_and_ci_use_deterministic_verification_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "os: [ubuntu-latest, macos-latest, windows-latest]" in ci
    assert 'python: ["3.11", "3.13"]' in ci
    assert 'python -m pytest -m "not real_model" -q' in ci
    assert "python -m build --no-isolation" in readme
    assert "python -m build --no-isolation" in ci
    assert "python -m build" not in readme.replace("python -m build --no-isolation", "")
    assert "python -m build" not in ci.replace("python -m build --no-isolation", "")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    test_extra = pyproject["project"]["optional-dependencies"]["test"]
    assert any(dependency.startswith("hatchling") for dependency in test_extra)


def test_readme_documents_evalplus_only_eval_command() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## Evaluation" in readme
    assert "EvalPlus-based evaluation CLI" in readme
    assert 'python -m pip install -e ".[eval,providers,mcp,test]"' in readme
    assert "litecoder-eval run agent-benchmark --dataset humaneval --limit 15" in readme
    assert "litecoder-eval report <run.json>" in readme
    assert "python -m litecoder.eval.suite" in readme
    assert "Evaluation executes generated code." in readme
    assert "process deadlines on Windows are not a complete security sandbox" in readme
