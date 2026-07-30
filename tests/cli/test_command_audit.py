from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from litecoder.cli.command_audit import CommandAuditRecorder
from litecoder.cli.local_commands import LocalCommandRouter
from litecoder.common.trace import SecretRedactor, TraceRecorder
from litecoder.context.manual_compaction import ManualCompactionReport
from litecoder.paths import AppPaths


def make_paths(tmp_path: Path) -> AppPaths:
    return AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-1",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class Runtime:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.store = object()
        self.provider_name = "fake"
        self.model = "model-a"
        self.provider_models = {"fake": "model-a"}


async def test_local_command_records_successful_start_and_end(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    result = await LocalCommandRouter(Runtime(paths)).dispatch(
        "/help",
        session_id=None,
    )  # type: ignore[arg-type]

    assert result.audit_status == "success"
    rows = read_rows(paths.command_audit_path)
    assert [row["sequence"] for row in rows] == [1, 2]
    assert [row["event"] for row in rows] == [
        "local.command.start",
        "local.command.end",
    ]
    assert rows[0]["command_id"] == rows[1]["command_id"]
    assert rows[0]["command"] == rows[1]["command"] == "/help"
    assert rows[1]["status"] == "success"
    assert rows[1]["attributes"]["outcome"] == "shown"
    assert "message" not in rows[1]["attributes"]
    assert isinstance(rows[0]["timestamp"], str)
    assert isinstance(rows[1]["duration_ms"], int)


class DurableRuntime(Runtime):
    def __init__(self, paths: AppPaths) -> None:
        super().__init__(paths)
        self.start_rows: list[dict[str, object]] = []

    async def compact_session(
        self,
        session_id: str,
    ) -> ManualCompactionReport:
        self.start_rows = read_rows(self.paths.command_audit_path)
        return ManualCompactionReport(90, 30, True)


async def test_start_is_durable_before_command_side_effects(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    runtime = DurableRuntime(paths)

    await LocalCommandRouter(runtime).dispatch(
        "/compact",
        session_id="root",
    )  # type: ignore[arg-type]

    assert len(runtime.start_rows) == 1
    assert runtime.start_rows[0]["event"] == "local.command.start"
    assert runtime.start_rows[0]["session_id"] == "root"
    assert runtime.start_rows[0]["root_session_id"] == "root"


async def test_unknown_command_is_rejected_without_persisting_raw_arguments(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    secret_argument = "unconfigured-secret-value"

    result = await LocalCommandRouter(Runtime(paths)).dispatch(
        f"/unknown {secret_argument}",
        session_id=None,
    )  # type: ignore[arg-type]

    assert result.audit_status == "rejected"
    rendered = paths.command_audit_path.read_text(encoding="utf-8")
    assert secret_argument not in rendered
    rows = read_rows(paths.command_audit_path)
    assert rows[0]["attributes"] == {"argument_count": 1}
    assert rows[1]["status"] == "rejected"
    assert rows[1]["attributes"]["code"] == "unknown_command"
    assert rows[1]["attributes"]["message"] == "Unknown local command: /unknown"


class RuntimeErrorRuntime(Runtime):
    def __init__(self, paths: AppPaths, secret: str) -> None:
        super().__init__(paths)
        self.secret = secret
        self.trace_redactor = SecretRedactor.with_values((secret,))

    async def compact_session(
        self,
        session_id: str,
    ) -> ManualCompactionReport:
        raise RuntimeError(f"storage {self.secret} " + ("x" * 2_000))


async def test_expected_command_failure_records_bounded_redacted_message(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    secret = "configured-command-secret"

    result = await LocalCommandRouter(
        RuntimeErrorRuntime(paths, secret)
    ).dispatch("/compact", session_id="root")  # type: ignore[arg-type]

    assert result.audit_status == "failed"
    rows = read_rows(paths.command_audit_path)
    attributes = rows[-1]["attributes"]
    assert attributes["code"] == "runtime_error"
    assert "[REDACTED]" in attributes["message"]
    assert len(attributes["message"].encode("utf-8")) <= 1_000
    assert secret not in paths.command_audit_path.read_text(encoding="utf-8")


class FailingRuntime(Runtime):
    async def compact_session(
        self,
        session_id: str,
    ) -> ManualCompactionReport:
        raise OSError("storage failed")


async def test_unexpected_command_failure_is_recorded_and_reraised(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)

    with pytest.raises(OSError, match="storage failed"):
        await LocalCommandRouter(FailingRuntime(paths)).dispatch(
            "/compact",
            session_id="root",
        )  # type: ignore[arg-type]

    rows = read_rows(paths.command_audit_path)
    assert rows[-1]["event"] == "local.command.end"
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["attributes"] == {
        "code": "unexpected_error",
        "error_type": "OSError",
    }


class BlockingRuntime(Runtime):
    def __init__(self, paths: AppPaths) -> None:
        super().__init__(paths)
        self.entered = asyncio.Event()

    async def compact_session(
        self,
        session_id: str,
    ) -> ManualCompactionReport:
        self.entered.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


async def test_cancelled_command_records_cancelled_end(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    runtime = BlockingRuntime(paths)
    task = asyncio.create_task(
        LocalCommandRouter(runtime).dispatch(
            "/compact",
            session_id="root",
        )  # type: ignore[arg-type]
    )
    await runtime.entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    rows = read_rows(paths.command_audit_path)
    assert [row["event"] for row in rows] == [
        "local.command.start",
        "local.command.end",
    ]
    assert rows[-1]["status"] == "cancelled"
    assert rows[-1]["attributes"] == {"code": "cancelled"}


class CountingRuntime(Runtime):
    def __init__(self, paths: AppPaths) -> None:
        super().__init__(paths)
        self.compact_calls = 0

    async def compact_session(
        self,
        session_id: str,
    ) -> ManualCompactionReport:
        self.compact_calls += 1
        return ManualCompactionReport(90, 30, True)


async def test_command_does_not_execute_when_start_audit_fails(
    tmp_path: Path,
) -> None:
    runtime = CountingRuntime(make_paths(tmp_path))
    router = LocalCommandRouter(runtime)  # type: ignore[arg-type]

    async def unavailable(payload: object) -> None:
        raise OSError("audit unavailable")

    router.audit.record = unavailable  # type: ignore[method-assign]
    result = await router.dispatch("/compact", session_id="root")

    assert runtime.compact_calls == 0
    assert result.audit_status == "failed"
    assert result.audit_code == "audit_unavailable"
    assert "was not executed" in result.message


async def test_concurrent_audit_writers_keep_monotonic_sequence(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    redactor = SecretRedactor.with_values(())
    first = CommandAuditRecorder(paths, redactor)
    second = CommandAuditRecorder(paths, redactor)

    await asyncio.gather(
        first.record({"event": "first"}),
        second.record({"event": "second"}),
    )

    rows = read_rows(paths.command_audit_path)
    assert [row["sequence"] for row in rows] == [1, 2]
    assert {row["event"] for row in rows} == {"first", "second"}


async def test_record_and_flush_is_visible_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, SecretRedactor.with_values(()))
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        "litecoder.common.trace.recorder.os.fsync",
        fsync_calls.append,
    )

    await recorder.start()
    await recorder.record_and_flush({"event": "durable"})

    assert read_rows(path)[0]["event"] == "durable"
    assert len(fsync_calls) == 1
    await recorder.close()
