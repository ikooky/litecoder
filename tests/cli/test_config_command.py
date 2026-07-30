from pathlib import Path
import time

import pytest
from typer.testing import CliRunner

from litecoder.cli import config as config_command
from litecoder.cli.app import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def _cleanup_config_lock(tmp_path: Path):
    yield
    _remove_lock_file(tmp_path / "home" / ".litecoder" / "locks" / "config.toml.lock")


def _remove_lock_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while path.exists():
        try:
            path.unlink()
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _write_provider(home: Path) -> Path:
    config_path = home / ".litecoder" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8"
    )
    return config_path


def test_config_set_key_stores_key_for_existing_provider(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    config_path = _write_provider(home)
    monkeypatch.setattr(config_command.Path, "home", lambda: home)

    result = runner.invoke(
        app, ["config", "set-key", "anthropic", "--key", "secret-value"]
    )

    assert result.exit_code == 0, result.output
    assert "Stored key for anthropic" in result.output
    assert 'api_key = "secret-value"' in config_path.read_text(encoding="utf-8")


def test_config_set_key_prompt_hides_key(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _write_provider(home)
    monkeypatch.setattr(config_command.Path, "home", lambda: home)

    result = runner.invoke(
        app,
        ["config", "set-key", "anthropic"],
        input="secret-value\n",
    )

    assert result.exit_code == 0, result.output
    assert "secret-value" not in result.output

def test_config_set_key_reports_undefined_provider_without_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    _write_provider(home)
    monkeypatch.setattr(config_command.Path, "home", lambda: home)

    result = runner.invoke(
        app, ["config", "set-key", "missing", "--key", "secret-value"]
    )

    assert result.exit_code == 2
    assert "must be defined" in result.output
    assert "Traceback" not in result.output
    assert "secret-value" not in result.output


def test_config_set_key_reports_invalid_toml_without_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    config_path = _write_provider(home)
    config_path.write_text("invalid = [", encoding="utf-8")
    monkeypatch.setattr(config_command.Path, "home", lambda: home)

    result = runner.invoke(
        app, ["config", "set-key", "anthropic", "--key", "secret-value"]
    )

    assert result.exit_code == 2
    assert "Configuration file is invalid" in result.output
    assert "Traceback" not in result.output
    assert "secret-value" not in result.output


def test_config_set_key_reports_invalid_schema_without_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    config_path = _write_provider(home)
    config_path.write_text(
        '[providers]\nanthropic = "invalid"\n', encoding="utf-8"
    )
    monkeypatch.setattr(config_command.Path, "home", lambda: home)

    result = runner.invoke(
        app, ["config", "set-key", "anthropic", "--key", "secret-value"]
    )

    assert result.exit_code == 2
    assert "Configuration file is invalid" in result.output
    assert "Traceback" not in result.output
    assert "secret-value" not in result.output
