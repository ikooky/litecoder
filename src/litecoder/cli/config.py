"""Supporting implementation for config."""

from pathlib import Path
import tomllib

import typer

from litecoder.settings import set_provider_key


app = typer.Typer()


@app.command("set-key")
def set_key(
    provider: str,
    key: str = typer.Option(..., prompt=True, hide_input=True),
) -> None:
    """Set the key."""
    try:
        set_provider_key(Path.home() / ".litecoder" / "config.toml", provider, key)
    except tomllib.TOMLDecodeError:
        raise typer.BadParameter("Configuration file is invalid") from None
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(f"Stored key for {provider}")
