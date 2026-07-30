from __future__ import annotations

from collections.abc import Mapping

import pytest

from litecoder.common.trace import TraceContext, trace_annotation
from litecoder.common.trace.redaction import SecretRedactor, bind_secret_redactor
from litecoder.hooks import TraceHook


def _assert_absent(value: str, rendered: str) -> None:
    if value in rendered:
        pytest.fail("sensitive value was exposed", pytrace=False)


class Sink:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    async def record(self, payload: Mapping[str, object]) -> None:
        self.rows.append(dict(payload))


@pytest.mark.asyncio
async def test_empty_scoped_redactor_still_redacts_before_nonredacting_sink() -> None:
    sink = Sink()
    context = TraceContext.root("trace-1", "session-1", "lead", sink)
    with context.bind(), bind_secret_redactor(SecretRedactor.with_values(())):
        await trace_annotation(
            intent="inspect Bearer annotation-token",
            reason=None,
            attributes={"authorization": "Bearer attribute-token"},
        )
        await TraceHook().record(
            {
                "authorization": "Bearer fact-token",
                "message": "Bearer message-token",
            }
        )

    rendered = repr(sink.rows)
    for secret in (
        "annotation-token",
        "attribute-token",
        "fact-token",
        "message-token",
    ):
        _assert_absent(secret, rendered)


def test_mapping_redaction_is_idempotent_across_collision_suffixes() -> None:
    secret = "configured-key"
    source = {
        secret: "secret-key-value",
        "[REDACTED]": "safe-base",
        "[REDACTED]#2": "safe-suffix",
        "authorization": "Bearer token-value",
    }
    redactor = SecretRedactor.with_values([secret])

    first = redactor.redact_data(source)
    second = redactor.redact_data(first)

    assert second == first
    _assert_absent(secret, repr(second))
    _assert_absent("token-value", repr(second))


def test_generated_redaction_tokens_are_stable_against_marker_shaped_secrets() -> None:
    configured = (
        "credential-secret",
        "REDACTED",
        "KEY",
        "[",
        "]",
        ":",
        "1",
        "#2",
    )
    redactor = SecretRedactor.with_values(configured)
    source = {
        "credential-secret": "prefix credential-secret suffix",
        "authorization": "Bearer runtime-token",
    }

    first = redactor.redact_data(source)
    second = redactor.redact_data(first)
    third = redactor.redact_data(second)

    assert second == first
    assert third == first
    generated = [*first.keys(), *first.values()]
    for value in generated:
        for secret in configured:
            assert secret not in str(value)
    assert "runtime-token" not in repr(first)

    import json

    json.dumps(first)


def test_generated_redaction_tokens_survive_deepcopy() -> None:
    import copy

    redactor = SecretRedactor.with_values(("secret", "REDACTED"))
    redacted = redactor.redact_text("prefix secret suffix")

    copied = copy.deepcopy(redacted)

    assert copied == redacted
    assert redactor.redact_text(copied) == copied
