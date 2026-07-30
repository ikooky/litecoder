from __future__ import annotations

import math
from pathlib import Path

import pytest

from litecoder.tools import (
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolPartialFailure,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class ExampleTool:
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            description="example",
            input_schema={"type": "object"},
            mutates_workspace=False,
        )

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        return ToolExecution.success("ok")


def test_tool_models_take_deep_json_snapshots(tmp_path: Path) -> None:
    schema = {"properties": {"paths": ["a.py"]}}
    arguments = {"paths": ["a.py"]}
    metadata = {"nested": [1]}
    spec = ToolSpec("read", "Read", schema, False)
    call = ToolCall("call-1", "read", arguments)
    context = ToolContext("agent-1", "workspace-1", tmp_path, metadata=metadata)
    execution = ToolExecution("success", "ok", metadata, False, {"line": 1})

    schema["properties"]["paths"].append("changed.py")
    arguments["paths"].append("changed.py")
    metadata["nested"].append(2)

    assert spec.input_schema == {"properties": {"paths": ["a.py"]}}
    assert call.arguments == {"paths": ["a.py"]}
    assert context.metadata == {"nested": [1]}
    assert execution.metadata == {"nested": [1]}
    assert execution.preview == {"line": 1}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ToolSpec("", "Read", {}, False),
        lambda: ToolCall("", "read", {}),
        lambda: ToolCall("call", "", {}),
        lambda: ToolContext("", "workspace", Path(".")),
        lambda: ToolContext("agent", "", Path(".")),
        lambda: ToolResult("", "success", "ok"),
        lambda: ToolExecution("", "ok"),
    ],
)
def test_tool_models_reject_empty_stable_identifiers(factory: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    [
        lambda secret: ToolCall("call", "read", {"value": secret}),
        lambda secret: ToolSpec("read", "Read", {"value": secret}, False),
        lambda secret: ToolResult("call", "success", "ok", {"value": secret}),
        lambda secret: ToolExecution("success", "ok", {"value": math.inf}),
    ],
)
def test_json_validation_errors_never_include_raw_values(factory: object) -> None:
    secret = object()
    with pytest.raises(ValueError) as captured:
        factory(secret)  # type: ignore[operator]
    assert repr(secret) not in str(captured.value)


def test_tool_spec_validates_control_metadata() -> None:
    spec = ToolSpec(
        "write",
        "Write",
        {},
        True,
        concurrency="exclusive",
        permission_risk="workspace",
        dedupe_policy="none",
    )
    assert spec.concurrency == "exclusive"
    assert spec.permission_risk == "workspace"
    assert spec.dedupe_policy == "none"

    with pytest.raises(ValueError, match="dedupe_policy"):
        ToolSpec("bad", "Bad", {}, False, dedupe_policy="sometimes")  # type: ignore[arg-type]


def test_execution_to_result_keeps_call_id_and_independent_snapshots() -> None:
    execution = ToolExecution.success(
        "complete", metadata={"count": 1}, preview={"summary": "done"}
    )
    result = execution.to_result("call-7")
    execution.metadata["count"] = 2
    execution.preview["summary"] = "changed"  # type: ignore[index]

    assert result == ToolResult(
        "call-7", "success", "complete", {"count": 1, "preview": {"summary": "done"}}
    )


def test_partial_failure_exposes_only_safe_failure_fields() -> None:
    raw_secret = "raw-secret-value"
    error = ToolPartialFailure(
        "Operation partly applied",
        changed_workspace=True,
        metadata={"phase": "rename"},
    )

    assert error.safe_message == "Operation partly applied"
    assert error.changed_workspace is True
    assert error.metadata == {"phase": "rename"}
    assert raw_secret not in repr(error)


def test_registry_is_deterministic_and_rejects_duplicates() -> None:
    registry = ToolRegistry()
    second = ExampleTool("zeta")
    first = ExampleTool("alpha")
    registry.register(second)
    registry.register(first)

    listed = registry.list()
    assert listed == (first, second)
    assert registry.require("alpha") is first
    assert isinstance(listed, tuple)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ExampleTool("alpha"))
    with pytest.raises(KeyError, match="not registered"):
        registry.require("missing")
