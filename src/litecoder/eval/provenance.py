"""Evaluation provenance and environment metadata."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


def collect_provenance() -> dict[str, object]:
    """Handle the collect provenance operation."""
    return {
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "litecoder_version": _package_version("litecoder"),
            "evalplus_version": _package_version("evalplus"),
        },
        "command": list(sys.argv),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
