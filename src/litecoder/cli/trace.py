"""Supporting implementation for trace."""

from __future__ import annotations

import asyncio

import typer

from litecoder.cli.commands import current_paths, open_session_store, trace_path
from litecoder.paths import TracePathUnavailableError


def trace_command(session_id: str) -> None:
    """Handle the trace command operation."""
    try:
        asyncio.run(_trace(session_id))
    except KeyError:
        raise typer.BadParameter(f"Unknown session {session_id!r}") from None
    except TracePathUnavailableError as error:
        raise typer.BadParameter(str(error)) from None


async def _trace(session_id: str) -> None:
    paths = current_paths()
    store = await open_session_store(paths)
    try:
        root_session_id = await store.root_session_id(session_id)
    finally:
        await store.close()
    path = trace_path(paths, root_session_id)
    if not path.exists():
        typer.echo(f"No trace found for root session {root_session_id}.", err=True)
        raise typer.Exit(1)
    typer.echo(path.read_text(encoding="utf-8"), nl=False)