"""Markdown rendering helpers for terminal output."""

from __future__ import annotations

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown, TableElement
from rich.table import Table


class FoldingTableElement(TableElement):
    """Render Markdown table cells by folding instead of truncating."""

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        for renderable in super().__rich_console__(console, options):
            if isinstance(renderable, Table):
                for column in renderable.columns:
                    column.overflow = "fold"
                    column.no_wrap = False
            yield renderable


class WrappingMarkdown(Markdown):
    """Component responsible for the wrapping markdown."""
    elements = {**Markdown.elements, "table_open": FoldingTableElement}
