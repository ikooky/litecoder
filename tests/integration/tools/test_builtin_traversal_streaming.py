from __future__ import annotations

import os
from pathlib import Path

import pytest

from litecoder.tools import ToolCall, ToolContext
from litecoder.tools.builtin import GlobFilesTool, SearchTextTool
from litecoder.tools.builtin._common import PROCESS_READ_CHUNK_BYTES


def context(root: Path) -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": "ask"},
    )


@pytest.mark.asyncio
async def test_search_reads_secure_file_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    (tmp_path / "large.txt").write_text(
        "needle\n" + ("x" * (PROCESS_READ_CHUNK_BYTES * 4)),
        encoding="utf-8",
    )
    original_fdopen = secure_path.os.fdopen
    read_sizes: list[int] = []

    class ReadProxy:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._stream.read(size)

        def __getattr__(self, name: str) -> object:
            return getattr(self._stream, name)

    def tracking_fdopen(*args: object, **kwargs: object):
        stream = original_fdopen(*args, **kwargs)
        mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
        return ReadProxy(stream) if mode == "rb" else stream

    monkeypatch.setattr(secure_path.os, "fdopen", tracking_fdopen)
    result = await SearchTextTool().execute(
        ToolCall("stream-search", "search_text", {"query": "needle"}),
        context(tmp_path),
    )

    assert result.status == "success"
    assert result.metadata["count"] == 1
    assert read_sizes
    assert all(0 < size <= PROCESS_READ_CHUNK_BYTES for size in read_sizes)
    assert len(read_sizes) > 1


@pytest.mark.asyncio
async def test_directory_entry_cap_stops_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    monkeypatch.setattr(secure_path, "MAX_DIRECTORY_ENTRIES", 3)
    for name in ("z.txt", "a.txt", "m.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    result = await GlobFilesTool().execute(
        ToolCall("dir-cap", "glob_files", {"pattern": "*.txt"}),
        context(tmp_path),
    )

    assert result.status == "success"
    assert result.content == ""
    assert result.metadata["traversal_truncated"] is True
    assert result.metadata["directory_entries_truncated"] is True
    assert result.metadata["traversed_entries"] == 4


@pytest.mark.asyncio
async def test_total_traversal_cap_stops_and_reports_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    monkeypatch.setattr(secure_path, "MAX_DIRECTORY_ENTRIES", 100)
    monkeypatch.setattr(secure_path, "MAX_TRAVERSAL_ENTRIES", 4)
    for directory in ("a", "b", "c"):
        child = tmp_path / directory
        child.mkdir()
        (child / "one.txt").write_text("one", encoding="utf-8")
        (child / "two.txt").write_text("two", encoding="utf-8")

    result = await GlobFilesTool().execute(
        ToolCall("total-cap", "glob_files", {"pattern": "**/*.txt"}),
        context(tmp_path),
    )

    assert result.status == "success"
    assert result.metadata["traversal_truncated"] is True
    assert result.metadata["traversed_entries"] <= 5
    assert result.content.splitlines() == sorted(result.content.splitlines())