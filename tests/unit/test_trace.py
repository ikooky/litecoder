from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from litecoder.common.trace.context import TraceContext
from litecoder.common.trace.emit import trace_annotation
from litecoder.common.trace.recorder import TraceRecorder
from litecoder.common.trace.redaction import SecretRedactor


_EXPOSURE_MESSAGE = "sensitive value was exposed"


def _assert_secret_absent(secret: str, rendered: str) -> None:
    if secret in rendered:
        pytest.fail(_EXPOSURE_MESSAGE, pytrace=False)


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_recorder_assigns_monotonic_sequence(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))
    await recorder.start()
    context = TraceContext.root("trace-1", "session-1", "lead", recorder)

    with context.bind():
        await trace_annotation(intent="inspect", reason=None, attributes={"path": "a.py"})
        await trace_annotation(intent=None, reason="cache-hit", attributes={})

    await recorder.close()

    rows = _read_rows(path)
    assert [row["sequence"] for row in rows] == [1, 2]
    assert {row["trace_id"] for row in rows} == {"trace-1"}
    assert {row["root_session_id"] for row in rows} == {"session-1"}
    assert [row["event"] for row in rows] == ["trace.annotation"] * 2


@pytest.mark.asyncio
async def test_context_binding_restores_parent_and_async_children_inherit() -> None:
    recorder = object()
    parent = TraceContext.root("trace-1", "session-1", "lead", recorder)
    child = TraceContext(
        trace_id="trace-1",
        span_id="child",
        parent_span_id="root",
        root_session_id="session-1",
        session_id="session-2",
        agent_id="worker",
        recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="No active TraceContext"):
        TraceContext.current()

    with parent.bind():
        assert TraceContext.current() is parent
        assert await asyncio.create_task(_current_context()) is parent
        with child.bind():
            assert TraceContext.current() is child
        assert TraceContext.current() is parent

    with pytest.raises(RuntimeError, match="No active TraceContext"):
        TraceContext.current()


async def _current_context() -> TraceContext:
    await asyncio.sleep(0)
    return TraceContext.current()


@pytest.mark.asyncio
async def test_annotation_requires_context_and_intent_or_reason(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="intent or reason is required"):
        await trace_annotation(intent=None, reason=None, attributes={})

    with pytest.raises(RuntimeError, match="No active TraceContext"):
        await trace_annotation(intent="inspect", reason=None, attributes={})

    assert not (tmp_path / "trace.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["start", "end", "status", "duration", "result", "error"])
async def test_annotation_rejects_lifecycle_fact_attributes(key: str) -> None:
    with pytest.raises(ValueError, match="lifecycle facts are not allowed"):
        await trace_annotation(intent="inspect", reason=None, attributes={key: "value"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attributes",
    [
        {"value": object()},
        {"value": {1: "non-string-key"}},
        {"value": float("nan")},
    ],
)
async def test_annotation_rejects_non_json_attributes(attributes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="attributes must be JSON-compatible"):
        await trace_annotation(intent="inspect", reason=None, attributes=attributes)


@pytest.mark.asyncio
async def test_annotation_rejects_text_that_cannot_be_encoded_as_utf8() -> None:
    invalid_text = chr(0xD800)

    with pytest.raises(ValueError, match="attributes must be JSON-compatible"):
        await trace_annotation(
            intent="inspect",
            reason=None,
            attributes={"nested": {"value": invalid_text}},
        )


@pytest.mark.asyncio
async def test_annotation_accepts_nested_json_attributes(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))
    await recorder.start()
    context = TraceContext.root("trace-1", "session-1", "lead", recorder)
    attributes = {"nested": [None, True, 4, 1.5, "value", {"key": "value"}]}

    with context.bind():
        await trace_annotation(intent="inspect", reason="needed", attributes=attributes)

    await recorder.close()

    assert _read_rows(path)[0]["attributes"] == attributes


@pytest.mark.asyncio
async def test_recorder_redacts_before_persisting(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    secret = "-".join(("runtime", "configured", "value"))
    recorder = TraceRecorder(path, SecretRedactor.with_values([secret]))
    await recorder.start()
    context = TraceContext.root("trace-1", "session-1", "lead", recorder)

    with context.bind():
        await trace_annotation(
            intent="inspect",
            reason=None,
            attributes={"note": secret, "authorization": "Bearer abc.def.ghi"},
        )

    await recorder.close()

    rendered = path.read_text(encoding="utf-8")
    _assert_secret_absent(secret, rendered)
    _assert_secret_absent("abc.def.ghi", rendered)
    assert rendered.count("[REDACTED]") == 2


@pytest.mark.asyncio
async def test_recorder_redacts_secret_bearing_attribute_keys(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    configured_key = "-".join(("runtime", "configured", "key"))
    bearer_token = ".".join(("runtime", "bearer", "credential"))
    recorder = TraceRecorder(path, SecretRedactor.with_values([configured_key]))
    await recorder.start()
    context = TraceContext.root("trace-1", "session-1", "lead", recorder)

    with context.bind():
        await trace_annotation(
            intent="inspect",
            reason=None,
            attributes={
                configured_key: "first",
                "[REDACTED]": "safe-base",
                "[REDACTED]#2": "safe-suffix",
                f"Bearer {bearer_token}": "second",
                "ordinary": "unchanged",
            },
        )

    await recorder.close()

    rendered = path.read_text(encoding="utf-8")
    _assert_secret_absent(configured_key, rendered)
    _assert_secret_absent(bearer_token, rendered)
    assert _read_rows(path)[0]["attributes"] == {
        "[REDACTED-KEY:1]": "first",
        "[REDACTED]": "safe-base",
        "[REDACTED]#2": "safe-suffix",
        "[REDACTED-KEY:4]": "second",
        "ordinary": "unchanged",
    }


@pytest.mark.asyncio
async def test_recorder_resumes_from_last_valid_sequence(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"sequence": 3, "event": "existing"}) + "\n",
        encoding="utf-8",
    )
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))

    await recorder.start()
    await recorder.record({"event": "new", "sequence": 999})
    await recorder.close()

    rows = _read_rows(path)
    assert [row["sequence"] for row in rows] == [3, 4]
    assert rows[-1]["event"] == "new"


@pytest.mark.asyncio
async def test_recorder_appends_after_valid_record_without_final_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"sequence": 3, "event": "existing"}),
        encoding="utf-8",
    )
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))

    await recorder.start()
    await recorder.record({"event": "new"})
    await recorder.close()

    rows = _read_rows(path)
    assert [row["sequence"] for row in rows] == [3, 4]


@pytest.mark.asyncio
async def test_recorder_snapshots_nested_payload_when_it_enters_queue(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))
    payload = {"event": "snapshot", "nested": {"value": "before"}}

    await recorder.start()
    await recorder.record(payload)
    payload["nested"]["value"] = "after"
    await recorder.close()

    assert _read_rows(path)[0]["nested"] == {"value": "before"}


@pytest.mark.asyncio
async def test_recorder_recovers_incomplete_tail_once_and_redacts_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    secret = "-".join(("discarded", "configured", "value"))
    valid = json.dumps({"sequence": 7, "event": "existing"}).encode()
    incomplete = b'{"sequence": 8, "note": "' + secret.encode() + b'"'
    path.write_bytes(valid + b"\n" + incomplete)
    recorder = TraceRecorder(path, SecretRedactor.with_values([secret]))

    await recorder.start()
    await recorder.record({"event": "new"})
    await recorder.close()

    rendered = path.read_text(encoding="utf-8")
    _assert_secret_absent(secret, rendered)
    rows = _read_rows(path)
    assert [row["sequence"] for row in rows] == [7, 8, 9]
    assert [row["event"] for row in rows] == [
        "existing",
        "trace.recovery",
        "new",
    ]
    assert rows[1]["reason"] == "incomplete_trailing_line"
    assert "[REDACTED]" in rows[1]["discarded_tail"]
    assert sum(row["event"] == "trace.recovery" for row in rows) == 1


@pytest.mark.asyncio
async def test_recorder_quarantines_multiple_corrupt_tail_lines_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    valid = json.dumps({"sequence": 5, "event": "existing"}).encode()
    corrupt_tail = b"not-json\n" + b'{"sequence": 6'
    path.write_bytes(valid + b"\n" + corrupt_tail)
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))

    await recorder.start()
    await recorder.record({"event": "new"})
    await recorder.close()

    rows = _read_rows(path)
    assert [row["sequence"] for row in rows] == [5, 6, 7]
    assert [row["event"] for row in rows] == [
        "existing",
        "trace.recovery",
        "new",
    ]
    assert rows[1]["discarded_tail"] == corrupt_tail.decode()
    assert sum(row["event"] == "trace.recovery" for row in rows) == 1


@pytest.mark.asyncio
async def test_recorder_recovers_entirely_corrupt_file_as_one_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    secret = "-".join(("corrupt", "configured", "value"))
    corrupt = b"not-json\n" + secret.encode() + b"\n"
    path.write_bytes(corrupt)
    recorder = TraceRecorder(path, SecretRedactor.with_values([secret]))

    await recorder.start()
    await recorder.close()

    rendered = path.read_text(encoding="utf-8")
    _assert_secret_absent(secret, rendered)
    rows = _read_rows(path)
    assert len(rows) == 1
    assert rows[0]["sequence"] == 1
    assert rows[0]["event"] == "trace.recovery"
    assert rows[0]["reason"] == "incomplete_trailing_line"
    assert "[REDACTED]" in rows[0]["discarded_tail"]
    assert rendered.endswith("\n")


@pytest.mark.asyncio
async def test_recorder_reports_worker_failure_to_later_producers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "removed" / "trace.jsonl"
    secret = "-".join(("queued", "configured", "value"))
    recorder = TraceRecorder(path, SecretRedactor.with_values([secret]))
    await recorder.start()
    path.parent.rmdir()
    await asyncio.sleep(0)

    try:
        with pytest.raises(RuntimeError, match="TraceRecorder worker failed") as captured:
            await recorder.record({"event": "after-failure", "note": secret})
    finally:
        for _ in range(2):
            with pytest.raises(FileNotFoundError):
                await recorder.close()

    _assert_secret_absent(secret, str(captured.value))
    _assert_secret_absent(secret, repr(captured.value.__cause__))
    assert isinstance(captured.value.__cause__, FileNotFoundError)


@pytest.mark.asyncio
async def test_recorder_enforces_lifecycle(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        tmp_path / "trace.jsonl", SecretRedactor.with_values([])
    )

    with pytest.raises(RuntimeError, match="not started"):
        await recorder.record({"event": "too-early"})

    await recorder.start()
    with pytest.raises(RuntimeError, match="already started"):
        await recorder.start()
    await recorder.close()
    await recorder.close()

    with pytest.raises(RuntimeError, match="closed"):
        await recorder.record({"event": "too-late"})


@pytest.mark.asyncio
async def test_concurrent_records_keep_file_order_equal_to_sequence(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, SecretRedactor.with_values([]))
    await recorder.start()

    await asyncio.gather(
        *(recorder.record({"event": "concurrent", "value": value}) for value in range(20))
    )
    await recorder.close()

    rows = _read_rows(path)
    assert [row["sequence"] for row in rows] == list(range(1, 21))
    assert {row["value"] for row in rows} == set(range(20))
