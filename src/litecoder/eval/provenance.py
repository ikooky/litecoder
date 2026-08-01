"""Evaluation provenance and environment metadata."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from tomllib import TOMLDecodeError, loads
from pathlib import Path


def collect_provenance() -> dict[str, object]:
    """Handle the collect provenance operation."""
    return {
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "litecoder_version": _package_version("litecoder"),
            "evalplus_version": _package_version("evalplus"),
        },
        "command": list(sys.argv),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        module_spec = importlib.util.find_spec(name)
        module = __import__(name)
        version = getattr(module, "__version__", None)
        if isinstance(version, str) and version:
            return version
        if module_spec is not None and module_spec.origin:
            source_root = Path(module_spec.origin).resolve().parents[1]
            project_file = source_root / "pyproject.toml"
            if project_file.exists():
                data = loads(project_file.read_text(encoding="utf-8"))
                project = data.get("project")
                if isinstance(project, dict):
                    project_name = project.get("name")
                    project_version = project.get("version")
                    if project_name == name and isinstance(project_version, str):
                        return project_version
    except (ImportError, OSError, TOMLDecodeError, AttributeError):
        pass
    return "unavailable"
