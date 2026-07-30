from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from litecoder.providers.compatible import CompatibleProvider
from litecoder.providers.registry import ProviderRegistry, close_default_async_clients
from litecoder.settings import ProviderSettings


async def completion(**kwargs: object) -> object:
    del kwargs
    return object()


async def responses(**kwargs: object) -> object:
    del kwargs
    return object()


@pytest.mark.parametrize(
    "api_style",
    [
        "anthropic-messages",
        "openai-chat-completions",
        "openai-responses",
    ],
)
def test_registry_builds_protocol_provider(api_style: str) -> None:
    registry = ProviderRegistry(completion=completion, responses=responses)

    created = registry.create(
        "primary",
        ProviderSettings(
            type=api_style,  # type: ignore[arg-type]
            model="model",
            api_key=SecretStr("secret"),
            base_url="https://gateway.invalid/v1",
        ),
    )

    assert isinstance(created, CompatibleProvider)
    assert created.model == "model"
    assert created.api_style == api_style
    assert created.base_url == "https://gateway.invalid/v1"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            ProviderSettings(
                type="anthropic-messages", api_key=SecretStr("secret")
            ),
            "non-empty model",
        ),
        (
            ProviderSettings(type="anthropic-messages", model="model"),
            "API key",
        ),
        (
            ProviderSettings(
                type="anthropic-messages",
                model=" ",
                api_key=SecretStr("secret"),
            ),
            "non-empty model",
        ),
        (
            ProviderSettings(
                type="anthropic-messages",
                model="model",
                api_key=SecretStr(""),
            ),
            "API key",
        ),
        (
            ProviderSettings.model_construct(
                type="unsupported",
                model="model",
                api_key=SecretStr("secret"),
            ),
            "Unsupported provider type",
        ),
    ],
)
def test_registry_rejects_invalid_configuration_without_exposing_keys(
    settings: ProviderSettings, message: str
) -> None:
    registry = ProviderRegistry(completion=completion, responses=responses)

    with pytest.raises(ValueError, match=message) as caught:
        registry.create("provider-name", settings)

    assert "secret" not in str(caught.value)


def test_registry_lazily_imports_optional_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object):
        if name == "litellm":
            raise ModuleNotFoundError("missing optional dependency", name="litellm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    registry = ProviderRegistry()

    with pytest.raises(RuntimeError, match=r"pip install 'litecoder\[providers\]'"):
        registry.create(
            "primary",
            ProviderSettings(
                type="anthropic-messages",
                model="model",
                api_key=SecretStr("secret"),
            ),
        )


def test_provider_package_exports_generic_provider_and_registry() -> None:
    from litecoder.providers import (
        CompatibleProvider as ExportedProvider,
        ProviderRegistry as ExportedRegistry,
    )

    assert ExportedProvider is CompatibleProvider
    assert ExportedRegistry is ProviderRegistry


@pytest.mark.asyncio
async def test_close_default_async_clients_uses_litellm_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(close_litellm_async_clients=close),
    )

    await close_default_async_clients()

    assert closed == [True]
