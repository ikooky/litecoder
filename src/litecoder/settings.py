"""Application configuration models and validation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import threading
import tomllib
from typing import Literal, Mapping

import portalocker
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator
from rich.console import Console
import tomli_w

from litecoder.paths import AppPaths


IS_WINDOWS = os.name == "nt"
_CONFIG_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_CONFIG_THREAD_LOCKS_GUARD = threading.Lock()
_DEFAULT_CONFIG = """# LiteCoder configuration.
# Set OPENAI_API_KEY in the environment before starting LiteCoder.

default_provider = "openai"

[providers.openai]
type = "openai-responses"
base_url = "https://api.openai.com/v1"
model = "gpt-5.6-sol"
api_key_env = "OPENAI_API_KEY"
"""


class ProviderSettings(BaseModel):
    """Component responsible for the provider settings."""
    type: Literal[
        "anthropic-messages",
        "openai-chat-completions",
        "openai-responses",
    ]
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_key_env: str | None = None


class MCPServerSettings(BaseModel):
    """Component responsible for the mcp server settings."""
    transport: Literal["stdio", "streamable-http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerSettings":
        """Validate the transport fields."""
        if self.transport == "stdio":
            if self.command is None or not self.command.strip():
                raise ValueError("stdio MCP servers require command")
        elif self.url is None or not self.url.strip():
            raise ValueError("streamable-http MCP servers require url")
        return self


class HookCommandSettings(BaseModel):
    """An explicitly configured external hook command."""

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = False
    point: Literal[
        "UserPromptSubmit",
        "PreModelCall",
        "PostModelCall",
        "PreToolUse",
        "PostToolUse",
        "ToolError",
        "AgentStop",
        "SubagentStart",
        "SubagentStop",
    ]
    command: str
    args: tuple[str, ...] = ()
    timeout_seconds: float = 5.0

    @field_validator("enabled", mode="before")
    @classmethod
    def validate_enabled(cls, value: object) -> bool:
        """Validate the enabled."""
        if type(value) is not bool:
            raise TypeError("hook command enabled must be a bool")
        return value

    @field_validator("name", "command", mode="before")
    @classmethod
    def validate_text_fields(cls, value: object) -> str:
        """Validate the text fields."""
        if not isinstance(value, str):
            raise TypeError("hook command text fields must be strings")
        return value

    @field_validator("args", mode="before")
    @classmethod
    def validate_args(cls, value: object) -> tuple[str, ...]:
        """Validate the args."""
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) for item in value
        ):
            raise TypeError("hook command args must be an array of strings")
        return tuple(value)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_type(cls, value: object) -> float | int:
        """Validate the timeout type."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("hook command timeout_seconds must be a number")
        return value

    @model_validator(mode="after")
    def validate_command(self) -> "HookCommandSettings":
        """Validate the command."""
        if not self.name.strip():
            raise ValueError("hook command name must not be empty")
        if not self.command.strip() or "\x00" in self.command:
            raise ValueError("hook command must not be empty or contain NUL")
        if any("\x00" in item for item in self.args):
            raise ValueError("hook command args must not contain NUL")
        if not 0 < self.timeout_seconds <= 30:
            raise ValueError("hook command timeout_seconds must be between 0 and 30")
        return self


class Settings(BaseModel):
    """Component responsible for the settings."""
    default_provider: str | None = None
    default_model: str | None = None
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    mcp_servers: dict[str, MCPServerSettings] = Field(default_factory=dict)
    hooks: tuple[HookCommandSettings, ...] = ()

    @model_validator(mode="after")
    def validate_hook_names(self) -> "Settings":
        """Validate the hook names."""
        names = [hook.name for hook in self.hooks]
        if len(names) != len(set(names)):
            raise ValueError("hook command names must be unique")
        return self

    @classmethod
    def load(
        cls,
        paths: AppPaths,
        environ: Mapping[str, str] = os.environ,
    ) -> Settings:
        """Load the requested records."""
        del environ
        config_path = paths.user_dir / "config.toml"
        raw = (
            tomllib.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        return cls.model_validate(raw)

    def resolve_api_key(
        self,
        provider_name: str,
        environ: Mapping[str, str] = os.environ,
    ) -> SecretStr:
        """Resolve the api key."""
        provider = self.providers[provider_name]
        variable = provider.api_key_env
        if variable is None:
            variable = (
                "ANTHROPIC_API_KEY"
                if provider.type == "anthropic-messages"
                else "OPENAI_API_KEY"
            )
        if variable is not None and variable in environ:
            value = environ[variable]
            if not value:
                raise ValueError(f"Environment variable {variable} is empty")
            return SecretStr(value)
        if provider.api_key is None:
            raise ValueError(f"No API key configured for {provider_name}")
        if not provider.api_key.get_secret_value():
            raise ValueError(f"API key configured for {provider_name} is empty")
        return provider.api_key


def _windows_acl_is_broad(config_path: Path) -> bool | None:
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$acl = Get-Acl -LiteralPath $env:LITECODER_CONFIG_PATH; "
        "$acl.Access | ForEach-Object { "
        "$_.IdentityReference.Translate("
        "[System.Security.Principal.SecurityIdentifier]).Value }"
    )
    environment = os.environ.copy()
    environment["LITECODER_CONFIG_PATH"] = str(config_path)
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    broad_sids = {"S-1-1-0", "S-1-5-11", "S-1-5-32-545"}
    identities = {line.strip() for line in result.stdout.splitlines()}
    return not broad_sids.isdisjoint(identities)


def warn_if_broad_windows_acl(config_path: Path) -> None:
    """Handle the warn if broad windows acl operation."""
    is_broad = _windows_acl_is_broad(config_path)
    if is_broad is None:
        Console(stderr=True).print(
            "[yellow]Warning:[/yellow] config file permissions could not be verified."
        )
    elif is_broad:
        Console(stderr=True).print(
            "[yellow]Warning:[/yellow] config file permissions allow broad access."
        )


def _thread_lock_for(config_path: Path) -> threading.RLock:
    canonical = Path(os.path.normcase(str(config_path.absolute())))
    with _CONFIG_THREAD_LOCKS_GUARD:
        return _CONFIG_THREAD_LOCKS.setdefault(canonical, threading.RLock())


def _write_config_atomically(config_path: Path, content: str) -> None:
    """Write a configuration file without exposing partial content."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content.encode("utf-8"))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_user_config(paths: AppPaths) -> Path | None:
    """Create the default user configuration when it does not exist."""
    config_path = paths.user_dir / "config.toml"
    if config_path.exists():
        return None
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = config_path.parent / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{config_path.name}.lock"
    with _thread_lock_for(config_path):
        with portalocker.Lock(str(lock_path), mode="a", timeout=10):
            if config_path.exists():
                return None
            _write_config_atomically(config_path, _DEFAULT_CONFIG)
            if not IS_WINDOWS:
                os.chmod(config_path, 0o600)
            return config_path


def _set_provider_key_locked(config_path: Path, provider: str, key: str) -> None:
    """Set the provider key locked."""
    data = (
        tomllib.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    try:
        validated = Settings.model_validate(data)
    except ValidationError as error:
        raise ValueError("Configuration file is invalid") from error
    if provider not in validated.providers:
        raise ValueError(f"Provider {provider!r} must be defined before storing its key")
    providers = data.setdefault("providers", {})
    providers[provider]["api_key"] = key
    _write_config_atomically(config_path, tomli_w.dumps(data))
    if IS_WINDOWS:
        warn_if_broad_windows_acl(config_path)
    else:
        os.chmod(config_path, 0o600)


def set_provider_key(config_path: Path, provider: str, key: str) -> None:
    """Set the provider key."""
    if not key:
        raise ValueError("API key cannot be empty")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = config_path.parent / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{config_path.name}.lock"
    with _thread_lock_for(config_path):
        with portalocker.Lock(str(lock_path), mode="a", timeout=10):
            _set_provider_key_locked(config_path, provider, key)
