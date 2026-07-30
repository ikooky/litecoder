from __future__ import annotations

import re
from pathlib import Path

import pytest

from litecoder.tools import ToolCall, ToolContext, ToolFailure
from litecoder.tools.builtin import SearchTextTool
from litecoder.tools.builtin._common import MAX_REGEX_PATTERN_CHARS


def context(root: Path) -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": "ask"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    (
        "a" * (MAX_REGEX_PATTERN_CHARS + 1),
        "界" * (MAX_REGEX_PATTERN_CHARS + 1),
    ),
)
async def test_over_limit_literal_is_rejected_before_escape_compile_or_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    import litecoder.tools.builtin.search as search

    class ForbiddenRe:
        IGNORECASE = re.IGNORECASE

        @staticmethod
        def escape(_: str) -> str:
            raise AssertionError("literal query must be bounded before escape")

        @staticmethod
        def compile(*_: object, **__: object):
            raise AssertionError("literal query must be bounded before compile")

    def forbidden_traversal(*_: object, **__: object):
        raise AssertionError("literal query must be bounded before traversal")

    monkeypatch.setattr(search, "re", ForbiddenRe)
    monkeypatch.setattr(search, "secure_iter_files", forbidden_traversal)

    with pytest.raises(ToolFailure, match="Invalid search pattern"):
        await SearchTextTool().execute(
            ToolCall("over-limit", "search_text", {"query": query}),
            context(tmp_path),
        )


@pytest.mark.asyncio
async def test_max_literal_and_case_insensitive_multibyte_queries_remain_useful(
    tmp_path: Path,
) -> None:
    maximum = "界" * MAX_REGEX_PATTERN_CHARS
    (tmp_path / "maximum.txt").write_text(maximum, encoding="utf-8")
    (tmp_path / "case.txt").write_text("ÄPFEL 界", encoding="utf-8")

    maximum_result = await SearchTextTool().execute(
        ToolCall(
            "maximum",
            "search_text",
            {"query": maximum, "glob": "maximum.txt"},
        ),
        context(tmp_path),
    )
    case_result = await SearchTextTool().execute(
        ToolCall(
            "case",
            "search_text",
            {
                "query": "äpfel 界",
                "case_sensitive": False,
                "glob": "case.txt",
            },
        ),
        context(tmp_path),
    )

    assert maximum_result.metadata["matches"][0]["column"] == 1
    assert case_result.metadata["matches"][0]["column"] == 1


def test_max_literal_uses_bounded_kmp_state(tmp_path: Path) -> None:
    from litecoder.tools.builtin.search import _search_chunks

    query = "a" * MAX_REGEX_PATTERN_CHARS

    class ForbiddenPattern:
        def finditer(self, _: str):
            raise AssertionError("literal matching must not build regex windows")

    matches, oversized, has_more = _search_chunks(
        (b"x" + query.encode("utf-8"),),
        relative="bounded.txt",
        compiled=ForbiddenPattern(),  # type: ignore[arg-type]
        context=context(tmp_path),
        remaining_limit=10,
        literal_query=query,
    )

    assert oversized is False
    assert has_more is False
    assert matches[0]["column"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "text"),
    (
        ("i", "İ"),
        ("İ", "i"),
        ("i", "ı"),
        ("ı", "I"),
        ("s", "ſ"),
        ("ſ", "S"),
        ("k", "K"),
        ("K", "K"),
    ),
)
async def test_case_insensitive_literal_preserves_python_special_equivalence(
    tmp_path: Path, query: str, text: str
) -> None:
    (tmp_path / "unicode.txt").write_text(f"x{text}y", encoding="utf-8")

    result = await SearchTextTool().execute(
        ToolCall(
            "unicode-equivalence",
            "search_text",
            {
                "query": query,
                "case_sensitive": False,
                "glob": "unicode.txt",
            },
        ),
        context(tmp_path),
    )

    assert result.metadata["matches"][0]["column"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "text"), (("ß", "ss"), ("ss", "ß")))
async def test_case_insensitive_literal_does_not_expand_sharp_s(
    tmp_path: Path, query: str, text: str
) -> None:
    (tmp_path / "sharp-s.txt").write_text(text, encoding="utf-8")

    result = await SearchTextTool().execute(
        ToolCall(
            "sharp-s",
            "search_text",
            {
                "query": query,
                "case_sensitive": False,
                "glob": "sharp-s.txt",
            },
        ),
        context(tmp_path),
    )

    assert result.metadata["matches"] == []