"""Shared helpers for built-in tools."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator

from litecoder.tools.models import ToolDenied, ToolFailure


MAX_FILE_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
PROCESS_READ_CHUNK_BYTES = 16_384
MAX_SEARCH_LINE_BYTES = 4_096
MAX_DIRECTORY_ENTRIES = 10_000
MAX_TRAVERSAL_ENTRIES = 100_000
MAX_REGEX_PATTERN_CHARS = 512
DEFAULT_RESULT_LIMIT = 1000
MAX_RESULT_LIMIT = 10_000
_PATH_DENIED = "Denied by workspace safety policy"
_RESERVED_MEMORY_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_-])\.memory(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)


def canonical_workspace_root(root: Path) -> Path:
    """Handle the canonical workspace root operation."""
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        raise ToolDenied(_PATH_DENIED) from None
    if not resolved.is_dir():
        raise ToolDenied(_PATH_DENIED)
    return resolved


def resolve_workspace_path(
    root: Path,
    value: object,
    *,
    require_leaf: bool = False,
) -> Path:
    """Resolve the workspace path."""
    canonical_root = canonical_workspace_root(root)
    relative = normalize_relative_path(value)
    candidate = (canonical_root / Path(*relative.parts)).resolve(strict=False)
    if not _is_within(canonical_root, candidate):
        raise ToolDenied(_PATH_DENIED)
    if require_leaf and candidate == canonical_root:
        raise ToolDenied(_PATH_DENIED)
    return candidate


def has_reserved_memory_reference(value: object) -> bool:
    """Return whether the reserved memory reference condition holds."""
    return isinstance(value, str) and bool(_RESERVED_MEMORY_REFERENCE.search(value))


def normalize_relative_path(value: object) -> PurePosixPath:
    """Normalize the relative path."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or has_reserved_memory_reference(value)
    ):
        raise ToolDenied(_PATH_DENIED)
    normalized = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(normalized)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or value.startswith(("\\\\", "//"))
    ):
        raise ToolDenied(_PATH_DENIED)
    if any(part == ".." for part in posix.parts):
        raise ToolDenied(_PATH_DENIED)
    cleaned = PurePosixPath(*(part for part in posix.parts if part not in {"", "."}))
    if not cleaned.parts:
        return PurePosixPath(".")
    return cleaned


def validate_glob_pattern(value: object) -> str:
    """Validate the glob pattern."""
    relative = normalize_relative_path(value)
    rendered = relative.as_posix()
    if rendered == ".":
        raise ToolDenied(_PATH_DENIED)
    return rendered


def workspace_relative(root: Path, path: Path) -> str:
    """Handle the workspace relative operation."""
    canonical_root = canonical_workspace_root(root)
    resolved = path.resolve(strict=False)
    if not _is_within(canonical_root, resolved) or resolved == canonical_root:
        raise ToolDenied(_PATH_DENIED)
    return resolved.relative_to(canonical_root).as_posix()


def require_existing_file(path: Path) -> None:
    """Handle the require existing file operation."""
    try:
        valid = path.is_file()
    except OSError:
        valid = False
    if not valid:
        raise ToolFailure("Workspace file is unavailable", metadata={"changed_workspace": False})


def require_existing_directory(path: Path) -> None:
    """Handle the require existing directory operation."""
    try:
        valid = path.is_dir()
    except OSError:
        valid = False
    if not valid:
        raise ToolFailure("Workspace directory is unavailable", metadata={"changed_workspace": False})


def require_string(arguments: dict[str, object], name: str, *, allow_empty: bool = False) -> str:
    """Handle the require string operation."""
    value = arguments.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ToolFailure("Invalid tool arguments", metadata={"field": name})
    return value


def optional_bool(arguments: dict[str, object], name: str, default: bool) -> bool:
    """Handle the optional bool operation."""
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolFailure("Invalid tool arguments", metadata={"field": name})
    return value


def optional_int(
    arguments: dict[str, object],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Handle the optional int operation."""
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ToolFailure("Invalid tool arguments", metadata={"field": name})
    if maximum is not None and value > maximum:
        raise ToolFailure("Invalid tool arguments", metadata={"field": name})
    return value


def optional_number(
    arguments: dict[str, object], name: str, default: float, *, minimum: float
) -> float:
    """Handle the optional number operation."""
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolFailure("Invalid tool arguments", metadata={"field": name})
    result = float(value)
    if result < minimum or result != result or result in {float("inf"), float("-inf")}:
        raise ToolFailure("Invalid tool arguments", metadata={"field": name})
    return result


def result_limit(arguments: dict[str, object]) -> int:
    """Handle the result limit operation."""
    return optional_int(
        arguments,
        "limit",
        DEFAULT_RESULT_LIMIT,
        minimum=1,
        maximum=MAX_RESULT_LIMIT,
    )


def iter_workspace_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Handle the iter workspace files operation."""
    canonical_root = canonical_workspace_root(root)
    for current, directories, files in os.walk(canonical_root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if not _is_link_or_reparse(current_path / name)
        )
        for name in sorted(files):
            path = current_path / name
            if _is_link_or_reparse(path):
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_file() or not _is_within(canonical_root, resolved):
                continue
            yield resolved.relative_to(canonical_root).as_posix(), resolved


def matches_glob(relative_path: str, pattern: str) -> bool:
    """Handle the matches glob operation."""
    if fnmatch.fnmatchcase(relative_path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatchcase(relative_path, pattern[3:])


def decode_utf8_text(data: bytes, *, safe_message: str) -> str:
    """Handle the decode utf8 text operation."""
    if b"\x00" in data:
        raise ToolFailure(safe_message, metadata={"changed_workspace": False})
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolFailure(safe_message, metadata={"changed_workspace": False}) from None


def truncate_utf8(value: str, limit: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Handle the truncate utf8 operation."""
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        common = os.path.commonpath((os.path.normcase(str(root)), os.path.normcase(str(candidate))))
    except ValueError:
        return False
    return common == os.path.normcase(str(root))


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)
