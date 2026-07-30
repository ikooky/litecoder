from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import tomllib

import pytest
from pydantic import ValidationError

from litecoder.paths import AppPaths
import litecoder.settings as settings_module
from litecoder.settings import Settings, ensure_user_config, set_provider_key


def _paths(tmp_path: Path) -> AppPaths:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AppPaths.discover(workspace, tmp_path / "home")


def test_ensure_user_config_creates_default_openai_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    chmod_calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)
    monkeypatch.setattr(
        settings_module.os,
        "chmod",
        lambda target, mode: chmod_calls.append((Path(target), mode)),
    )

    created = ensure_user_config(paths)

    config_path = paths.user_dir / "config.toml"
    assert created == config_path
    assert config_path.exists()
    assert chmod_calls == [(config_path, 0o600)]
    loaded = Settings.load(paths)
    assert loaded.default_provider == "openai"
    provider = loaded.providers["openai"]
    assert provider.type == "openai-responses"
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.model == "gpt-5.6-sol"
    assert provider.api_key_env == "OPENAI_API_KEY"
    assert provider.api_key is None


def test_ensure_user_config_preserves_existing_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    config_path = paths.user_dir / "config.toml"
    original = "existing = true\n"
    config_path.write_text(original, encoding="utf-8")

    created = ensure_user_config(paths)

    assert created is None
    assert config_path.read_text(encoding="utf-8") == original


def test_ensure_user_config_is_safe_for_concurrent_startups(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)
    start = threading.Barrier(2)

    def initialize() -> Path | None:
        start.wait()
        return ensure_user_config(paths)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results.count(paths.user_dir / "config.toml") == 1
    assert results.count(None) == 1
    assert Settings.load(paths).providers["openai"].model == "gpt-5.6-sol"


def test_environment_key_overrides_file_key(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\napi_key = "file-key"\n',
        encoding="utf-8",
    )

    loaded = Settings.load(paths, {"ANTHROPIC_API_KEY": "env-key"})

    assert (
        loaded.resolve_api_key(
            "anthropic", {"ANTHROPIC_API_KEY": "env-key"}
        ).get_secret_value()
        == "env-key"
    )


def test_empty_selected_environment_key_does_not_fall_back_to_file(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\napi_key = "file-key"\n',
        encoding="utf-8",
    )
    loaded = Settings.load(paths, {})

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY.*empty"):
        loaded.resolve_api_key("anthropic", {"ANTHROPIC_API_KEY": ""})


def test_openai_protocol_uses_only_its_configured_environment_variable(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        "\n".join(
            [
                '[providers.local]',
                'type = "openai-chat-completions"',
                'api_key = "file-key"',
                'api_key_env = "LOCAL_API_KEY"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    loaded = Settings.load(paths, {})

    resolved = loaded.resolve_api_key(
        "local", {"OPENAI_API_KEY": "wrong", "LOCAL_API_KEY": "right"}
    )

    assert resolved.get_secret_value() == "right"


def test_openai_protocol_defaults_to_openai_api_key(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[providers.local]\ntype = "openai-chat-completions"\napi_key = "file-key"\n',
        encoding="utf-8",
    )
    loaded = Settings.load(paths, {})

    resolved = loaded.resolve_api_key("local", {"OPENAI_API_KEY": "env-key"})

    assert resolved.get_secret_value() == "env-key"


def test_load_reads_only_user_configuration(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.workspace_root.joinpath("config.toml").write_text(
        'default_provider = "project-provider"\n', encoding="utf-8"
    )

    loaded = Settings.load(paths, {})

    assert loaded.default_provider is None
    assert loaded.providers == {}


def test_load_rejects_unknown_provider_type(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[providers.invalid]\ntype = "other"\n', encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        Settings.load(paths, {})


def test_missing_api_key_is_explicit(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8"
    )
    loaded = Settings.load(paths, {})

    with pytest.raises(ValueError, match="No API key configured for anthropic"):
        loaded.resolve_api_key("anthropic", {})


def test_empty_file_api_key_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\napi_key = ""\n',
        encoding="utf-8",
    )
    loaded = Settings.load(paths, {})

    with pytest.raises(ValueError, match="API key.*empty"):
        loaded.resolve_api_key("anthropic", {})


def test_example_configuration_matches_settings_schema() -> None:
    example = Path(__file__).resolve().parents[2] / "config.example.toml"
    loaded = Settings.model_validate(
        tomllib.loads(example.read_text(encoding="utf-8"))
    )

    assert loaded.default_provider == "anthropic"
    assert loaded.providers["anthropic"].type == "anthropic-messages"
    assert loaded.providers["anthropic"].api_key is not None
    assert (
        loaded.providers["anthropic"].api_key.get_secret_value()
        == "replace-with-your-anthropic-api-key"
    )
    assert loaded.providers["openai"].api_key is not None
    assert (
        loaded.providers["openai"].api_key.get_secret_value()
        == "replace-with-your-openai-api-key"
    )

def test_set_provider_key_uses_dedicated_lock_directory(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[providers.anthropic]\ntype = "anthropic-messages"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)

    set_provider_key(path, "anthropic", "secret-value")

    assert (tmp_path / "locks" / "config.toml.lock").exists()
    assert not (tmp_path / "config.toml.lock").exists()

def test_set_provider_key_replaces_atomically(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8")
    replacements: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(settings_module.os, "replace", record_replace)
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)

    set_provider_key(path, "anthropic", "secret-value")

    assert 'api_key = "secret-value"' in path.read_text(encoding="utf-8")
    assert len(replacements) == 1
    assert replacements[0][1] == path
    assert replacements[0][0].parent == path.parent
    assert replacements[0][0].suffix == ".tmp"
    assert not list(tmp_path.glob("*.tmp"))


def test_set_provider_key_uses_unique_restrictive_temporary_files(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8")
    calls: list[dict[str, object]] = []
    replacements: list[Path] = []
    original_mkstemp = tempfile.mkstemp
    original_replace = os.replace

    def record_mkstemp(**kwargs):
        calls.append(kwargs)
        return original_mkstemp(**kwargs)

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append(Path(source))
        original_replace(source, destination)

    monkeypatch.setattr(
        settings_module,
        "tempfile",
        SimpleNamespace(mkstemp=record_mkstemp),
        raising=False,
    )
    monkeypatch.setattr(settings_module.os, "replace", record_replace)
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)

    set_provider_key(path, "anthropic", "first")
    set_provider_key(path, "anthropic", "second")

    assert calls == [
        {"prefix": "config.toml.", "suffix": ".tmp", "dir": path.parent},
        {"prefix": "config.toml.", "suffix": ".tmp", "dir": path.parent},
    ]
    assert len(set(replacements)) == 2


def test_set_provider_key_cleans_up_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8")
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)
    monkeypatch.setattr(
        settings_module.os,
        "replace",
        lambda source, destination: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        set_provider_key(path, "anthropic", "secret-value")

    assert not list(tmp_path.glob("*.tmp"))
    assert "secret-value" not in path.read_text(encoding="utf-8")

def test_set_provider_key_requires_existing_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="must be defined"):
        set_provider_key(path, "anthropic", "secret-value")


def test_set_provider_key_rejects_empty_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = '[providers.anthropic]\ntype = "anthropic-messages"\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="API key.*empty"):
        set_provider_key(path, "anthropic", "")

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_key_writes_preserve_both_provider_updates(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                '[providers.anthropic]',
                'type = "anthropic-messages"',
                '',
                '[providers.local]',
                'type = "openai-chat-completions"',
                '',
            ]
        ),
        encoding="utf-8",
    )
    start = threading.Barrier(2)
    original_loads = settings_module.tomllib.loads

    def slow_loads(content: str):
        time.sleep(0.2)
        return original_loads(content)

    def store(provider: str, key: str) -> None:
        start.wait()
        set_provider_key(path, provider, key)

    monkeypatch.setattr(settings_module.tomllib, "loads", slow_loads)
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(store, "anthropic", "anthropic-key"),
            executor.submit(store, "local", "local-key"),
        ]
        for future in futures:
            future.result()

    content = path.read_text(encoding="utf-8")
    assert 'api_key = "anthropic-key"' in content
    assert 'api_key = "local-key"' in content


def test_set_provider_key_rejects_schema_invalid_provider_shape(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = '[providers]\nanthropic = "invalid"\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration file is invalid"):
        set_provider_key(path, "anthropic", "secret-value")

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))

def test_posix_key_storage_applies_owner_only_mode(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8")
    chmod_calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(settings_module, "IS_WINDOWS", False)
    monkeypatch.setattr(
        settings_module.os,
        "chmod",
        lambda target, mode: chmod_calls.append((Path(target), mode)),
    )

    set_provider_key(path, "anthropic", "secret-value")

    assert chmod_calls == [(path, 0o600)]


def test_windows_key_storage_warns_without_exposing_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.anthropic]\ntype = "anthropic-messages"\n', encoding="utf-8")
    monkeypatch.setattr(settings_module, "IS_WINDOWS", True)
    monkeypatch.setattr(settings_module, "_windows_acl_is_broad", lambda _: True)

    set_provider_key(path, "anthropic", "secret-value")

    captured = capsys.readouterr()
    assert "broad" in captured.err.lower()
    assert "secret-value" not in captured.err


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("S-1-5-21-1000\nS-1-5-32-545\n", True),
        ("S-1-5-21-1000\n", False),
    ],
)
def test_windows_acl_check_isolated_behind_subprocess(
    tmp_path: Path, monkeypatch, output: str, expected: bool
) -> None:
    monkeypatch.setattr(
        settings_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    assert settings_module._windows_acl_is_broad(tmp_path / "config.toml") is expected

def test_windows_acl_check_passes_literal_path_through_environment(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def record_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(settings_module.subprocess, "run", record_run)

    settings_module._windows_acl_is_broad(path)

    command, kwargs = calls[0]
    assert "$env:LITECODER_CONFIG_PATH" in command[4]
    assert kwargs["env"]["LITECODER_CONFIG_PATH"] == str(path)

def test_windows_acl_inspection_failure_emits_key_free_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    path = tmp_path / "config.toml"
    monkeypatch.setattr(
        settings_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    settings_module.warn_if_broad_windows_acl(path)

    captured = capsys.readouterr()
    assert "could not be verified" in captured.err.lower()
    assert "secret-value" not in captured.err


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("powershell missing"),
        PermissionError("powershell denied"),
        OSError("powershell failed"),
    ],
)
def test_windows_acl_invocation_exception_emits_key_free_warning(
    tmp_path: Path, monkeypatch, capsys, error: OSError
) -> None:
    path = tmp_path / "config.toml"

    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(settings_module.subprocess, "run", raise_error)

    settings_module.warn_if_broad_windows_acl(path)

    captured = capsys.readouterr()
    assert "could not be verified" in captured.err.lower()
    assert "secret-value" not in captured.err


def test_mcp_server_settings_validate_stdio_and_http(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        "\n".join(
            [
                '[mcp_servers.docs]',
                'transport = "stdio"',
                'command = "python"',
                'args = ["server.py"]',
                '',
                '[mcp_servers.remote]',
                'transport = "streamable-http"',
                'url = "https://example.invalid/mcp"',
                '',
            ]
        ),
        encoding="utf-8",
    )

    loaded = Settings.load(paths, {})

    assert loaded.mcp_servers["docs"].command == "python"
    assert loaded.mcp_servers["docs"].args == ("server.py",)
    assert loaded.mcp_servers["remote"].url == "https://example.invalid/mcp"


def test_stdio_mcp_server_requires_command(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        '[mcp_servers.docs]\ntransport = "stdio"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings.load(paths, {})
