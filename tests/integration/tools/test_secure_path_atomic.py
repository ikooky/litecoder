from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.tools import ToolCall, ToolContext, ToolFailure
from litecoder.tools.builtin import WriteFileTool


@pytest.mark.asyncio
async def test_pre_replace_fsync_failure_cleans_temp_and_leaves_target_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("pre-replace-failure")

    monkeypatch.setattr(secure_path.os, "fsync", fail_fsync)
    with pytest.raises(ToolFailure):
        await WriteFileTool().execute(
            ToolCall(
                "pre-replace",
                "write_file",
                {"path": "target.txt", "content": "new"},
            ),
            ToolContext("agent", "workspace", tmp_path),
        )

    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.txt.litecoder-*.tmp"))
