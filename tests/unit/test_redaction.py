from __future__ import annotations

import asyncio

import pytest

from litecoder.common.trace.redaction import (
    SecretRedactor,
    bind_secret_redactor,
    current_secret_redactor,
)


_EXPOSURE_MESSAGE = "sensitive value was exposed"


def _assert_secret_absent(secret: str, rendered: str) -> None:
    if secret in rendered:
        pytest.fail(_EXPOSURE_MESSAGE, pytrace=False)


def test_secret_absence_guard_uses_a_constant_failure_message() -> None:
    secret = "-".join(("configured", "value"))

    with pytest.raises(pytest.fail.Exception) as captured:
        _assert_secret_absent(secret, secret)

    message = str(captured.value)
    if secret in message:
        pytest.fail("secret absence guard exposed its input", pytrace=False)
    assert message == _EXPOSURE_MESSAGE


def test_redacts_exact_values_and_bearer_tokens() -> None:
    redactor = SecretRedactor.with_values(["configured-value"])

    value = redactor.redact_text(
        "key=configured-value Authorization: bEaReR abc.def.ghi"
    )

    _assert_secret_absent("configured-value", value)
    _assert_secret_absent("abc.def.ghi", value)
    assert value.count("[REDACTED]") == 2


def test_redacts_overlapping_exact_values_longest_first() -> None:
    redactor = SecretRedactor.with_values(["prefix", "prefix-suffix"])

    value = redactor.redact_text("key=prefix-suffix")

    _assert_secret_absent("prefix", value)
    _assert_secret_absent("prefix-suffix", value)
    assert value == "key=[REDACTED]"


def test_redactor_repr_does_not_expose_configured_values() -> None:
    redactor = SecretRedactor.with_values(["configured-value"])

    _assert_secret_absent("configured-value", repr(redactor))


@pytest.mark.asyncio
async def test_scoped_redactor_defaults_empty_restores_and_reaches_child_tasks() -> None:
    secret = "-".join(("scoped", "runtime", "secret"))
    scoped = SecretRedactor.with_values([secret])
    default = current_secret_redactor()

    assert default.redact_text(secret) == secret
    with bind_secret_redactor(scoped):
        assert current_secret_redactor() is scoped
        inherited = await asyncio.create_task(_current_redactor())
        assert inherited is scoped
    assert current_secret_redactor() is default


async def _current_redactor() -> SecretRedactor:
    await asyncio.sleep(0)
    return current_secret_redactor()


def test_redacts_nested_structures_without_mutating_input() -> None:
    source = {
        "headers": {"authorization": "Bearer token-value"},
        "safe": ["ok", ("configured-value", 7)],
    }

    result = SecretRedactor.with_values(["configured-value"]).redact_data(source)

    rendered = repr(result)
    _assert_secret_absent("configured-value", rendered)
    _assert_secret_absent("token-value", rendered)
    assert result == {
        "headers": {"authorization": "[REDACTED]"},
        "safe": ["ok", ("[REDACTED]", 7)],
    }
    if source["headers"]["authorization"] != "Bearer token-value":
        pytest.fail("source authorization value was mutated", pytrace=False)
    if source["safe"][1][0] != "configured-value":
        pytest.fail("source configured value was mutated", pytrace=False)
    if result is source:
        pytest.fail("source mapping was reused", pytrace=False)
    assert result["headers"] is not source["headers"]
    assert result["safe"] is not source["safe"]


def test_sensitive_mapping_keys_are_case_insensitive_and_keys_are_preserved() -> None:
    source = {
        "API_KEY": "configured-value",
        "PassWord": 1234,
        "ordinary": 42,
        7: "safe",
    }

    result = SecretRedactor.with_values([]).redact_data(source)

    rendered = repr(result)
    _assert_secret_absent("configured-value", rendered)
    _assert_secret_absent("1234", rendered)
    assert result == {
        "API_KEY": "[REDACTED]",
        "PassWord": "[REDACTED]",
        "ordinary": 42,
        7: "safe",
    }
    assert set(result) == set(source)

def test_redacts_secret_bearing_mapping_keys_and_preserves_safe_collisions() -> None:
    first_secret = "-".join(("first", "configured", "key"))
    second_secret = "-".join(("second", "configured", "key"))
    bearer_token = ".".join(("runtime", "bearer", "credential"))
    source = {
        first_secret: "first",
        "[REDACTED]": "safe-base",
        "[REDACTED]#2": "safe-suffix",
        second_secret: "second",
        f"Bearer {bearer_token}": "third",
        "ordinary": "unchanged",
    }

    result = SecretRedactor.with_values(
        [first_secret, second_secret]
    ).redact_data(source)

    rendered = repr(result)
    _assert_secret_absent(first_secret, rendered)
    _assert_secret_absent(second_secret, rendered)
    _assert_secret_absent(bearer_token, rendered)
    assert result == {
        "[REDACTED-KEY:1]": "first",
        "[REDACTED]": "safe-base",
        "[REDACTED]#2": "safe-suffix",
        "[REDACTED-KEY:4]": "second",
        "[REDACTED-KEY:5]": "third",
        "ordinary": "unchanged",
    }
