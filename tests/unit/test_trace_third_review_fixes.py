from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from litecoder.common.trace import TraceContext
from litecoder.common.trace.recorder import TraceRecorder
from litecoder.common.trace.redaction import (
    SecretRedactor,
    _ProtectedText,
    bind_secret_redactor,
)
from litecoder.hooks import TraceHook


class Sink:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    async def record(self, payload: Mapping[str, object]) -> None:
        self.rows.append(dict(payload))


@pytest.mark.parametrize(
    "ranges",
    [
        ((7, 13),),
        ((7, 13), (8, 12)),
        ((7, 13), (0, 2)),
        ((7, 99),),
        ((-1, 13),),
    ],
)
def test_forged_or_invalid_protected_ranges_are_not_trusted(
    ranges: tuple[tuple[int, int], ...],
) -> None:
    secret = "secret"
    redactor = SecretRedactor.with_values((secret,))
    forged = _ProtectedText("prefix secret suffix", ranges)

    rendered = redactor.redact_text(forged)

    assert secret not in rendered


@pytest.mark.asyncio
async def test_trace_hook_revalidates_forged_protected_text() -> None:
    secret = "hook-secret"
    redactor = SecretRedactor.with_values((secret,))
    forged = _ProtectedText(secret, ((0, len(secret)),))
    sink = Sink()
    context = TraceContext.root("trace-1", "session-1", "lead", sink)

    with context.bind(), bind_secret_redactor(redactor):
        await TraceHook().record({"message": forged})

    assert secret not in repr(sink.rows)


@pytest.mark.asyncio
async def test_trace_recorder_revalidates_forged_protected_text(
    tmp_path: Path,
) -> None:
    secret = "recorder-secret"
    forged = _ProtectedText(secret, ((0, len(secret) + 5),))
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, SecretRedactor.with_values((secret,)))

    await recorder.start()
    await recorder.record({"message": forged})
    await recorder.close()

    assert secret not in path.read_text(encoding="utf-8")


def test_mutated_or_missing_protected_metadata_is_untrusted() -> None:
    secret = "mutated-secret"
    redactor = SecretRedactor.with_values((secret,))
    forged = _ProtectedText(secret, ((0, len(secret)),))
    forged.protected_ranges = [(0, len(secret))]

    assert secret not in redactor.redact_text(forged)

    del forged.protected_ranges

    assert secret not in redactor.redact_text(forged)


def test_forged_bearer_fragment_is_never_a_trusted_token() -> None:
    token = "Bearer forged-token"
    forged = _ProtectedText(token, ((0, len(token)),))

    rendered = SecretRedactor.with_values(()).redact_text(forged)

    assert "forged-token" not in rendered