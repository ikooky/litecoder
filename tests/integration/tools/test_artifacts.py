from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import SecretStr

import litecoder.cli.app as app_module
from litecoder.agent.loop import AgentLoop, RuntimeBudgets
import litecoder.tools.artifacts as artifacts_module
from litecoder.common.trace import SecretRedactor, TraceContext, TraceRecorder
from litecoder.context.manager import ContextManager
from litecoder.context.session.models import MessageRecord, SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.hooks import HookManager, HookOutcome, HookPoint, TraceHook
from litecoder.paths import AppPaths
from litecoder.providers.models import ProviderEvent, StopReason, ToolCallBlock
from litecoder.settings import ProviderSettings, Settings
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
    WorkspaceStateRegistry,
)
from litecoder.tools.artifacts import (
    TOOL_RESULT_INLINE_BYTES,
    ArtifactStore,
    ProjectArtifactStores,
)
from tests.fakes.provider import FakeProvider


class _NullTrace:
    async def record(self, fact: Mapping[str, object]) -> None:
        del fact


class _RecordingTrace:
    def __init__(self) -> None:
        self.facts: list[dict[str, object]] = []

    async def record(self, fact: Mapping[str, object]) -> None:
        self.facts.append(dict(fact))


class _ContentTool:
    def __init__(
        self,
        content: str,
        *,
        mutates: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.spec = ToolSpec("content", "return content", {}, mutates)
        self.content = content
        self.metadata = metadata or {}
        self.calls = 0

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        del call, context
        self.calls += 1
        return ToolExecution.success(
            self.content,
            metadata=self.metadata,
            changed_workspace=self.spec.mutates_workspace,
            preview=self.content[:20],
        )


def _context(
    root: Path, *, secrets: tuple[str, ...] = (), session_id: str = "agent"
) -> ToolContext:
    return ToolContext(
        session_id,
        "workspace",
        root,
        metadata={
            "round_number": 1,
            "permission_mode": "ask",
            "root_session_id": "root",
        },
        secret_values=secrets,
    )


def _executor(
    tool: _ContentTool,
    *,
    store: ArtifactStore | object | None,
    hooks: HookManager | None = None,
) -> tuple[ToolExecutor, WorkspaceStateRegistry]:
    registry = ToolRegistry()
    registry.register(tool)
    workspaces = WorkspaceStateRegistry()
    executor = ToolExecutor(
        registry,
        hooks or HookManager(trace_hook=_NullTrace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=lambda _: "Allow once"),
        workspaces,
        artifact_store=store,
    )
    return executor, workspaces


@pytest.mark.asyncio
async def test_artifact_store_redacts_before_atomic_write_and_returns_metadata(
    tmp_path: Path,
) -> None:
    secret = "split-" + "private-value"
    root = tmp_path / "outputs"
    store = ArtifactStore(root, SecretRedactor.with_values((secret,)))

    reference = await store.persist(
        "../../CON\\Bearer secret-call-id",
        f"configured={secret}\nAuthorization: Bearer bearer-value\n" + "x" * 100,
    )

    persisted = reference.path.read_text(encoding="utf-8")
    assert reference.path.parent == root.resolve()
    assert reference.path.name.startswith("artifact-")
    assert reference.path.suffix == ".txt"
    assert "CON" not in reference.path.name
    assert secret not in persisted
    assert "bearer-value" not in persisted
    assert persisted.count("[REDACTED]") == 2
    assert reference.preview in persisted
    assert reference.bytes == len(persisted.encode("utf-8"))
    assert len(reference.tool_call_id_sha256) == 64
    assert not list(root.glob("*.tmp"))


@pytest.mark.asyncio
async def test_artifact_store_uses_short_temp_name_under_deep_windows_path(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows legacy path regression")

    tool_call_id = "deep-path-call"
    digest = hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()
    final_name = f"artifact-{digest}.txt"
    old_temp_name = f".{final_name}.12345678.tmp"
    root = tmp_path / "outputs"
    old_temp_length = len(str(root / old_temp_name))
    if old_temp_length < 264:
        root = root / ("d" * (264 - old_temp_length - 1))

    if len(str(root / final_name)) >= 260:
        pytest.skip("pytest temporary root is too deep for the final artifact")

    assert len(str(root / old_temp_name)) >= 260
    assert len(str(root / ".artifact-12345678.tmp")) < 260
    store = ArtifactStore(root, SecretRedactor.with_values(()))

    reference = await store.persist(tool_call_id, "persisted")

    assert reference.path.name == final_name
    assert reference.path.read_text(encoding="utf-8") == "persisted"
    assert not list(root.glob("*.tmp"))


@pytest.mark.asyncio
async def test_untrusted_call_ids_have_distinct_safe_paths(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs", SecretRedactor.with_values(()))
    identifiers = (
        "../same",
        "..\\same",
        "/absolute/same",
        "C:\\absolute\\same",
        "CON",
        "con.txt",
        "NUL",
        "a/b",
        "a\\b",
    )

    references = [await store.persist(identifier, identifier) for identifier in identifiers]

    assert len({reference.path for reference in references}) == len(identifiers)
    assert all(reference.path.parent == store.root for reference in references)
    assert all("/" not in reference.path.name and "\\" not in reference.path.name for reference in references)
    assert all(identifier not in reference.path.name for identifier, reference in zip(identifiers, references))


@pytest.mark.asyncio
async def test_atomic_replace_failure_preserves_existing_artifact_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "outputs"
    store = ArtifactStore(root, SecretRedactor.with_values(()))
    original = await store.persist("call-1", "old")

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("replace-secret")

    monkeypatch.setattr(artifacts_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace-secret"):
        await store.persist("call-1", "new")

    assert original.path.read_text(encoding="utf-8") == "old"
    assert not list(root.glob("*.tmp"))


@pytest.mark.asyncio
async def test_executor_uses_redacted_utf8_bytes_and_offloads_only_above_threshold(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    store = ArtifactStore(root, SecretRedactor.with_values(()))
    exact_tool = _ContentTool("x" * TOOL_RESULT_INLINE_BYTES)
    exact_executor, _ = _executor(exact_tool, store=store)

    exact = await exact_executor.execute(
        ToolCall("exact", "content", {}), _context(tmp_path)
    )

    assert exact.content == "x" * TOOL_RESULT_INLINE_BYTES
    assert "artifact" not in exact.metadata
    assert not root.exists()

    multibyte = "眉" * (TOOL_RESULT_INLINE_BYTES // 3 + 1)
    large_tool = _ContentTool(multibyte)
    large_executor, _ = _executor(large_tool, store=store)
    large = await large_executor.execute(
        ToolCall("large", "content", {}), _context(tmp_path)
    )

    assert large.status == "success"
    assert multibyte not in large.content
    assert len(large.content.encode("utf-8")) < TOOL_RESULT_INLINE_BYTES
    artifact = large.metadata["artifact"]
    assert isinstance(artifact, dict)
    artifact_path = Path(artifact["path"])
    assert artifact_path.read_text(encoding="utf-8") == multibyte
    json.dumps(large.metadata)


@pytest.mark.asyncio
async def test_post_hook_observes_execution_before_bounded_redacted_offload(
    tmp_path: Path,
) -> None:
    secret = "configured-output-secret"
    bearer = "Bearer bearer-output-secret"
    content = f"{secret}\n{bearer}\n" + "z" * TOOL_RESULT_INLINE_BYTES
    metadata = {
        "detail": secret,
        "nested": {"authorization": bearer},
    }
    tool = _ContentTool(content, metadata=metadata)
    trace = _RecordingTrace()
    hooks = HookManager(trace_hook=trace)
    observations: list[tuple[str, dict[str, object]]] = []

    async def observe(envelope):
        execution = envelope.payload["execution"]
        observations.append((execution.content, dict(execution.metadata)))
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.POST_TOOL_USE, observe, name="observe")
    store = ArtifactStore(
        tmp_path / "outputs", SecretRedactor.with_values((secret,))
    )
    executor, _ = _executor(tool, store=store, hooks=hooks)

    result = await executor.execute(
        ToolCall("large-secret", "content", {}),
        _context(tmp_path, secrets=(secret,)),
    )

    assert observations == [(content, metadata)]
    artifact = result.metadata["artifact"]
    persisted = Path(artifact["path"]).read_text(encoding="utf-8")
    rendered = repr(result)
    for unsafe in (secret, "bearer-output-secret"):
        assert unsafe not in persisted
        assert unsafe not in rendered
    assert "z" * 2_000 not in result.content
    assert result.metadata["detail"] == "[REDACTED]"
    assert result.metadata["nested"] == {"authorization": "[REDACTED]"}
    stages = [
        fact["stage"]
        for fact in trace.facts
        if fact.get("event") == "tool.runtime"
    ]
    assert stages[-3:] == ["post", "artifact", "final"]
    assert trace.facts[-2]["status"] == "stored"


class _FailingStore:
    async def persist(self, tool_call_id: str, content: str):
        del tool_call_id, content
        raise OSError("artifact-persistence-secret")


@pytest.mark.asyncio
async def test_artifact_failure_is_bounded_and_preserves_version_and_dedupe(
    tmp_path: Path,
) -> None:
    secret = "unreturned-output-secret"
    content = secret + "x" * TOOL_RESULT_INLINE_BYTES
    tool = _ContentTool(content, mutates=True, metadata={"detail": secret})
    executor, workspaces = _executor(tool, store=_FailingStore())
    tool_context = _context(tmp_path, secrets=(secret,))

    first = await executor.execute(
        ToolCall("first", "content", {"path": "same"}), tool_context
    )
    second = await executor.execute(
        ToolCall("second", "content", {"path": "same"}), tool_context
    )

    assert first.status == "tool_error"
    assert first.metadata["artifact_error"] is True
    assert first.metadata["automatic_retry"] is False
    assert first.metadata["changed_workspace"] is True
    assert first.metadata["workspace_version"] == 1
    assert secret not in repr(first)
    assert "artifact-persistence-secret" not in repr(first)
    assert len(first.content.encode("utf-8")) < 2_000
    assert second.status == "duplicate_blocked"
    assert tool.calls == 1
    assert workspaces.get("workspace").version == 1


@pytest.mark.asyncio
async def test_build_runtime_injects_project_output_store_with_runtime_redactor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-id",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-id",
        workspace_id="workspace-id",
        workspace_root=tmp_path,
    )
    settings = Settings(
        default_provider="test",
        default_model="model",
        providers={
            "test": ProviderSettings(
                type="anthropic-messages",
                model="model",
                api_key=SecretStr("runtime-secret"),
            )
        },
    )
    captured: list[object | None] = []
    real_executor = app_module.ToolExecutor

    class _CapturingExecutor(real_executor):
        def __init__(
            self,
            *args,
            artifact_store=None,
            artifact_store_resolver=None,
            **kwargs,
        ):
            assert artifact_store is None
            captured.append(artifact_store_resolver)
            super().__init__(
                *args,
                artifact_store=artifact_store,
                artifact_store_resolver=artifact_store_resolver,
                **kwargs,
            )

    monkeypatch.setattr(app_module.AppPaths, "discover", lambda _cwd: paths)
    monkeypatch.setattr(app_module.Settings, "load", lambda _paths: settings)
    monkeypatch.setattr(app_module, "ToolExecutor", _CapturingExecutor)

    runtime = await app_module.build_runtime(tmp_path)
    await runtime.close()

    assert len(captured) == 1
    resolver = captured[0]
    assert isinstance(resolver, ProjectArtifactStores)
    other_context = _context(tmp_path)
    other_context.metadata["project_id"] = "persisted-other-project"
    resolved = resolver(other_context)
    assert resolved.root == (
        paths.user_dir / "projects" / "persisted-other-project" / "outputs"
    ).resolve()
    assert resolved.redactor.values == ("runtime-secret",)


@pytest.mark.asyncio
async def test_final_metadata_remains_bounded_and_keeps_artifact_reference(
    tmp_path: Path,
) -> None:
    metadata = {f"field_{index}": "m" * 1_000 for index in range(16)}
    tool = _ContentTool("x" * (TOOL_RESULT_INLINE_BYTES + 1), metadata=metadata)
    executor, _ = _executor(
        tool,
        store=ArtifactStore(tmp_path / "outputs", SecretRedactor.with_values(())),
    )

    result = await executor.execute(
        ToolCall("near-limit", "content", {}), _context(tmp_path)
    )

    assert "artifact" in result.metadata
    assert len(
        json.dumps(result.metadata, ensure_ascii=False).encode("utf-8")
    ) <= 16_384
    assert Path(result.metadata["artifact"]["path"]).exists()


@pytest.mark.asyncio
async def test_metadata_truncation_marker_stays_within_the_size_limit(
    tmp_path: Path,
) -> None:
    metadata = {f"k{index}": "" for index in range(2_000)}
    tool = _ContentTool("small", metadata=metadata)
    executor, _ = _executor(tool, store=None)

    result = await executor.execute(
        ToolCall("many-fields", "content", {}), _context(tmp_path)
    )

    assert result.metadata["metadata_truncated"] is True
    assert len(
        json.dumps(
            result.metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ) <= 16_384


@pytest.mark.asyncio
async def test_agent_loop_routes_artifact_to_persisted_session_project(
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / ".litecoder"
    store = SQLiteSessionStore(user_dir / "sessions.db")
    await store.open()
    await store.create_session(
        SessionRecord.new(
            "resumed",
            "persisted-project",
            "workspace",
            "fake",
            "model",
            workspace_path=str(tmp_path),
        )
    )
    content = "r" * (TOOL_RESULT_INLINE_BYTES + 1)
    tool = _ContentTool(content)
    registry = ToolRegistry()
    registry.register(tool)
    resolver = ProjectArtifactStores(
        user_dir, SecretRedactor.with_values(())
    )
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=_NullTrace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(),
        WorkspaceStateRegistry(),
        artifact_store_resolver=resolver,
    )
    tool_call = ToolCallBlock("large-call", "content", {})
    provider = FakeProvider(
        [
            [
                ProviderEvent.tool_call_completed(0, tool_call),
                ProviderEvent.content_block_completed(
                    0,
                    {
                        "type": "tool_call",
                        "call_id": "large-call",
                        "name": "content",
                        "input": {},
                    },
                ),
                ProviderEvent.response_completed(
                    StopReason.TOOL_USE, "tool_use"
                ),
            ],
            [
                ProviderEvent.content_block_completed(
                    0, {"type": "text", "text": "done"}
                ),
                ProviderEvent.response_completed(
                    StopReason.END_TURN, "end_turn"
                ),
            ],
        ]
    )
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=ContextManager(store, model="model"),
        tools=registry,
        executor=executor,
        duplicates=DuplicateGuard(annotation=lambda **_: None),
        budgets=RuntimeBudgets(max_rounds=4, max_tokens=100),
    )

    result = await loop.run_turn("resumed", "produce large output")
    restored = await store.load_context("resumed")
    await store.close()

    assert result.status == "completed"
    artifact_metadata = restored.messages[2].content[0]["metadata"]["artifact"]
    artifact_path = Path(artifact_metadata["path"])
    assert artifact_path.parent == (
        user_dir / "projects" / "persisted-project" / "outputs"
    ).resolve()
    assert artifact_path.read_text(encoding="utf-8") == content
    assert not (user_dir / "projects" / "current-project" / "outputs").exists()


@pytest.mark.asyncio
async def test_project_artifact_resolver_scopes_equal_call_ids_by_session(
    tmp_path: Path,
) -> None:
    resolver = ProjectArtifactStores(
        tmp_path / ".litecoder", SecretRedactor.with_values(())
    )
    first_context = _context(tmp_path, session_id="session-a")
    second_context = _context(tmp_path, session_id="session-b")
    first_context.metadata["project_id"] = "project"
    second_context.metadata["project_id"] = "project"
    first = await resolver(first_context).persist(
        "same-call", "first"
    )
    second = await resolver(second_context).persist(
        "same-call", "second"
    )

    assert first.path != second.path
    assert first.path.read_text(encoding="utf-8") == "first"
    assert second.path.read_text(encoding="utf-8") == "second"


@pytest.mark.asyncio
async def test_session_deletion_preserves_shared_artifact_reference(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    shared_path = str(tmp_path / "shared-artifact.txt")
    content = [{"metadata": {"artifact": {"path": shared_path}}}]
    try:
        await store.create_session(SessionRecord.new("keep", "project", "workspace", "fake", "model", workspace_path=str(tmp_path)))
        await store.create_session(SessionRecord.new("drop", "project", "workspace", "fake", "model", workspace_path=str(tmp_path)))
        await store.append_message(MessageRecord("keep", "tool", content))
        await store.append_message(MessageRecord("drop", "tool", content))

        deletion = await store.delete_session_tree("drop")
        assert deletion.artifact_paths == ()
    finally:
        await store.close()

def test_project_artifact_resolver_rejects_unsafe_project_segments(
    tmp_path: Path,
) -> None:
    resolver = ProjectArtifactStores(
        tmp_path / ".litecoder", SecretRedactor.with_values(())
    )
    for unsafe in ("../escape", "..\\escape", "/absolute", "C:\\absolute", "CON"):
        context = _context(tmp_path)
        context.metadata["project_id"] = unsafe
        with pytest.raises(ValueError, match="project_id"):
            resolver(context)

@pytest.mark.asyncio
async def test_trace_bounds_post_payload_before_persistence(
    tmp_path: Path,
) -> None:
    content = "trace-output-" + "t" * TOOL_RESULT_INLINE_BYTES
    trace_path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    hooks = HookManager()
    observed: list[str] = []

    async def observe(envelope):
        observed.append(envelope.payload["execution"].content)
        return HookOutcome(envelope.payload)

    hooks.register(HookPoint.POST_TOOL_USE, observe, name="observe")
    executor, _ = _executor(
        _ContentTool(content),
        store=ArtifactStore(
            tmp_path / "outputs", SecretRedactor.with_values(())
        ),
        hooks=hooks,
    )
    trace_context = TraceContext.root(
        "trace-id", "root-session", "agent", recorder
    )

    with trace_context.bind():
        result = await executor.execute(
            ToolCall("trace-large", "content", {}), _context(tmp_path)
        )
    await recorder.close()

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    rendered = trace_path.read_text(encoding="utf-8")
    assert observed == [content]
    assert "t" * 2_000 not in rendered
    assert any(
        row.get("stage") == "artifact"
        and row.get("status") == "stored"
        and row["artifact"]["path"] == result.metadata["artifact"]["path"]
        for row in rows
    )
    assert any(
        isinstance(row.get("payload"), dict)
        and isinstance(row["payload"].get("execution"), dict)
        and isinstance(row["payload"]["execution"].get("content"), dict)
        and row["payload"]["execution"]["content"].get("truncated") is True
        for row in rows
    )

@pytest.mark.asyncio
async def test_artifact_reference_is_redacted_in_returned_result(
    tmp_path: Path,
) -> None:
    secret = "path-runtime-secret"
    root = tmp_path / secret / "outputs"
    executor, _ = _executor(
        _ContentTool("x" * (TOOL_RESULT_INLINE_BYTES + 1)),
        store=ArtifactStore(root, SecretRedactor.with_values((secret,))),
    )

    result = await executor.execute(
        ToolCall("path-secret", "content", {}),
        _context(tmp_path, secrets=(secret,)),
    )

    assert secret not in result.content
    assert secret not in repr(result.metadata)
    assert len(list(root.glob("artifact-*.txt"))) == 1

class _GuardedLargeMapping(Mapping[str, int]):
    def __init__(self) -> None:
        self.iterated = 0

    def __len__(self) -> int:
        return 100

    def __iter__(self):
        for index in range(100):
            self.iterated += 1
            if self.iterated > 50:
                raise AssertionError("trace mapping preview consumed all entries")
            yield f"key-{index}"

    def __getitem__(self, key: str) -> int:
        return int(key.removeprefix("key-"))


@pytest.mark.asyncio
async def test_trace_projection_is_lazy_and_preserves_redacted_key_collisions(
    tmp_path: Path,
) -> None:
    first_secret = "first-trace-key-secret"
    second_secret = "second-trace-key-secret"
    redactor = SecretRedactor.with_values((first_secret, second_secret))
    trace_path = tmp_path / "projection.jsonl"
    recorder = TraceRecorder(trace_path, redactor)
    await recorder.start()
    trace_context = TraceContext.root(
        "trace-projection", "root", "agent", recorder
    )
    guarded = _GuardedLargeMapping()

    with trace_context.bind():
        from litecoder.common.trace import bind_secret_redactor

        with bind_secret_redactor(redactor):
            await TraceHook().record(
                {
                    "large": guarded,
                    "collisions": {
                        first_secret: 1,
                        second_secret: 2,
                    },
                }
            )
    await recorder.close()

    row = json.loads(trace_path.read_text(encoding="utf-8"))
    assert guarded.iterated == 50
    assert row["large"]["size"] == 100
    assert row["large"]["truncated"] is True
    assert set(row["collisions"].values()) == {1, 2}
    assert first_secret not in repr(row)
    assert second_secret not in repr(row)

@pytest.mark.asyncio
async def test_trace_projection_has_a_shared_total_budget(tmp_path: Path) -> None:
    trace_path = tmp_path / "bounded-projection.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    trace_context = TraceContext.root(
        "trace-bounded", "root", "agent", recorder
    )
    nested = {
        f"group-{index}": ["x" * 1_000 for _ in range(50)]
        for index in range(50)
    }

    with trace_context.bind():
        await TraceHook().record({"nested": nested})
    await recorder.close()

    rendered = trace_path.read_bytes()
    assert len(rendered) <= 65_536
    assert rendered.count(b"x" * 1_000) < 50

@pytest.mark.asyncio
async def test_trace_budget_preserves_payload_first_runtime_identity(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "runtime-identity.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    trace_context = TraceContext.root(
        "trace-runtime", "root", "agent", recorder
    )
    bulky = [["x" * 1_000 for _ in range(50)] for _ in range(50)]

    with trace_context.bind():
        await TraceHook().record(
            {
                "payload": bulky,
                "event": "tool.runtime",
                "status": "stored",
                "tool_call_id": "call-identity",
                "tool_name": "content",
            }
        )
    await recorder.close()

    row = json.loads(trace_path.read_text(encoding="utf-8"))
    assert row["event"] == "tool.runtime"
    assert row["status"] == "stored"
    assert row["tool_call_id"] == "call-identity"
    assert row["tool_name"] == "content"


@pytest.mark.asyncio
async def test_real_hook_dispatch_fallback_remains_attributable(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "hook-identity.jsonl"
    recorder = TraceRecorder(trace_path, SecretRedactor.with_values(()))
    await recorder.start()
    trace_context = TraceContext.root(
        "trace-hook", "root", "agent", recorder
    )
    hooks = HookManager()
    bulky = [["x" * 1_000 for _ in range(50)] for _ in range(50)]

    with trace_context.bind():
        await hooks.dispatch_pre(HookPoint.PRE_TOOL_USE, {"payload": bulky})
    await recorder.close()

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    start = next(row for row in rows if row["event"] == "hook.dispatch.start")
    assert start["point"] == "PreToolUse"
    assert start["phase"] == "pre"
    assert start["status"] == "started"
    assert isinstance(start["dispatch_id"], str)
    assert start["dispatch_id"]
