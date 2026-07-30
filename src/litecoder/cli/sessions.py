"""Supporting implementation for sessions."""

from __future__ import annotations

import asyncio

import typer

from litecoder.cli.commands import (
    current_paths,
    open_session_store,
    remove_session_files,
    trace_path,
)
from litecoder.context.session.models import SessionRecord
from litecoder.paths import TracePathUnavailableError


app = typer.Typer(help="Manage LiteCoder sessions.")


@app.command("list")
def list_sessions() -> None:
    """Handle the list sessions operation."""
    try:
        asyncio.run(_list_sessions())
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None


@app.command("show")
def show_session(session_id: str) -> None:
    """Handle the show session operation."""
    try:
        asyncio.run(_show_session(session_id))
    except KeyError:
        raise typer.BadParameter(f"Unknown session {session_id!r}") from None


@app.command("delete")
def delete_session(session_id: str) -> None:
    """Delete the session."""
    if not typer.confirm(f"Delete session {session_id} and child sessions?"):
        raise typer.Exit(1)
    try:
        asyncio.run(_delete_session(session_id))
    except KeyError:
        raise typer.BadParameter(f"Unknown session {session_id!r}") from None
    except TracePathUnavailableError as error:
        raise typer.BadParameter(str(error)) from None


async def _list_sessions() -> None:
    paths = current_paths()
    store = await open_session_store(paths)
    try:
        sessions = await store.list_sessions(project_id=paths.project_id)
    finally:
        await store.close()
    if not sessions:
        typer.echo("No sessions.")
        return
    for session in sessions:
        typer.echo(_session_row(session))


async def _show_session(session_id: str) -> None:
    paths = current_paths()
    store = await open_session_store(paths)
    try:
        context = await store.load_context(session_id)
    finally:
        await store.close()
    session = context.session
    typer.echo(f"id: {session.id}")
    typer.echo(f"type: {session.session_type}")
    typer.echo(f"status: {session.status.value}")
    typer.echo(f"provider: {session.provider}")
    typer.echo(f"model: {session.model}")
    typer.echo(f"title: {session.title or ''}")
    typer.echo(f"parent: {session.parent_session_id or ''}")
    typer.echo(f"workspace: {session.workspace_path}")
    typer.echo(f"messages: {len(context.messages)}")


async def _delete_session(session_id: str) -> None:
    paths = current_paths()
    store = await open_session_store(paths)
    try:
        root_session_id = await store.root_session_id(session_id)
        trace_path(paths, root_session_id)
        deletion = await store.delete_session_tree(session_id)
    finally:
        await store.close()
    remove_session_files(paths, deletion)
    typer.echo(
        f"Deleted {len(deletion.deleted_session_ids)} session(s) from "
        f"root {deletion.root_session_id}."
    )


def _session_row(session: SessionRecord) -> str:
    title = session.title or ""
    parent = session.parent_session_id or ""
    return (
        f"{session.id}\t{session.status.value}\t{session.session_type}\t"
        f"{session.provider}/{session.model}\tparent={parent}\t{title}"
    )