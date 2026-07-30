from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from litecoder.tools.models import ToolCall, ToolContext, ToolFailure
from litecoder.tools.skills import LoadSkillTool, SkillCatalog, SKILL_MAX_BYTES


def _skill(root: Path, name: str, body: str, description: str = "desc") -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8")
    return path


def test_project_precedence_and_deterministic_casefold_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user = tmp_path / "user"
    bundled = tmp_path / "bundled"
    _skill(project / ".litecoder" / "skills", "Review", "project")
    _skill(user / "skills", "review", "user")
    _skill(bundled, "review", "bundled")
    _skill(bundled, "zeta", "bundled")
    _skill(bundled, "Alpha", "bundled")

    catalog = SkillCatalog.discover(project, user, bundled)

    assert catalog.resolve("review").source == "project"
    assert catalog.resolve("REVIEW").name == "Review"
    assert [item.name for item in catalog.list()] == ["Alpha", "Review", "zeta"]
    metadata = catalog.resolve("review")
    assert not hasattr(metadata, "path")
    assert not hasattr(metadata, "source_root")
    assert len(metadata.path_identity) == 64
    with pytest.raises(FrozenInstanceError):
        metadata.name = "changed"  # type: ignore[misc]
    assert catalog.prompt_metadata()[1] == {
        "name": "Review",
        "source": "project",
        "description": "desc",
    }


def test_same_source_case_collision_is_rejected_where_supported(tmp_path: Path) -> None:
    root = tmp_path / "bundled"
    _skill(root, "Review", "one")
    collision_dir = root / "review"
    try:
        collision_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pytest.skip("case-sensitive directories are unavailable")
    (collision_dir / "SKILL.md").write_text("two", encoding="utf-8")

    with pytest.raises(ValueError, match="collision"):
        SkillCatalog.discover(tmp_path / "project", tmp_path / "user", root)


def test_skill_prompt_catalog_has_a_total_budget_and_stable_name_fallback(
    tmp_path: Path,
) -> None:
    bundled = tmp_path / "bundled"
    for name in ("Alpha", "Bravo", "Charlie"):
        _skill(bundled, name, "body", description=name * 40)
    catalog = SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)

    metadata = catalog.prompt_metadata(max_chars=260)

    assert len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))) <= 260
    assert [item["name"] for item in metadata] == ["Alpha", "Bravo", "Charlie"]
    assert all(item["source"] == "bundled" for item in metadata)
    assert any(item["description"].endswith("…") for item in metadata)
    assert metadata == catalog.prompt_metadata(max_chars=260)
    with pytest.raises(ValueError, match="max_chars"):
        catalog.prompt_metadata(max_chars=0)


def test_skill_prompt_catalog_omits_deterministically_when_names_exceed_budget(
    tmp_path: Path,
) -> None:
    bundled = tmp_path / "bundled"
    for name in ("Alpha", "Bravo", "Charlie", "Delta"):
        _skill(bundled, name, "body")
    catalog = SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)

    metadata = catalog.prompt_metadata(max_chars=120)

    assert len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))) <= 120
    assert metadata[0]["name"] == "Alpha"
    assert metadata == catalog.prompt_metadata(max_chars=120)


def test_unsafe_names_are_rejected(tmp_path: Path) -> None:
    unsafe_root = tmp_path / "unsafe-bundled"
    unsafe = unsafe_root / "bad name" / "SKILL.md"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("unsafe", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        SkillCatalog.discover(tmp_path / "other-project", tmp_path / "other-user", unsafe_root)


@pytest.mark.asyncio
async def test_load_skill_is_safe_read_only_redacted_and_revalidates(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    skill_file = _skill(bundled, "review", "token=sk-private")
    catalog = SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)
    tool = LoadSkillTool(catalog)
    context = ToolContext("agent", "workspace", tmp_path, secret_values=("sk-private",))

    result = await tool.execute(ToolCall("call-1", "load_skill", {"name": "review"}), context)
    assert "sk-private" not in result.content
    assert "[REDACTED]" in result.content
    assert tool.spec.permission_risk == "safe"
    assert tool.spec.concurrency == "shared"
    assert tool.spec.mutates_workspace is False

    skill_file.unlink()
    with pytest.raises(ToolFailure, match="unavailable"):
        await tool.execute(ToolCall("call-2", "load_skill", {"name": "review"}), context)


@pytest.mark.asyncio
async def test_unknown_and_unsafe_skill_names_fail_without_paths(tmp_path: Path) -> None:
    catalog = SkillCatalog.discover(tmp_path / "project", tmp_path / "user", tmp_path / "bundled")
    tool = LoadSkillTool(catalog)
    context = ToolContext("agent", "workspace", tmp_path)
    for name in ("missing", "../escape"):
        with pytest.raises(ToolFailure) as captured:
            await tool.execute(ToolCall("call", "load_skill", {"name": name}), context)
        assert str(tmp_path) not in str(captured.value)


def test_unrelated_directories_are_ignored_and_invalid_skill_files_fail(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    (bundled / "notes with spaces").mkdir(parents=True)
    _skill(bundled, "valid", "ok")
    assert [item.name for item in SkillCatalog.discover(
        tmp_path / "project", tmp_path / "user", bundled
    ).list()] == ["valid"]

    (bundled / "valid" / "SKILL.md").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="invalid"):
        SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)

    (bundled / "valid" / "SKILL.md").write_bytes(b"x" * (SKILL_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="invalid"):
        SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)


@pytest.mark.asyncio
async def test_load_rejects_content_changed_after_discovery(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    skill_file = _skill(bundled, "review", "original")
    catalog = SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)
    skill_file.write_text("changed", encoding="utf-8")

    with pytest.raises(ToolFailure, match="unavailable"):
        await LoadSkillTool(catalog).execute(
            ToolCall("call", "load_skill", {"name": "review"}),
            ToolContext("agent", "workspace", tmp_path),
        )


def test_discovery_rejects_symlinked_source_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    _skill(actual, "review", "body")
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match="unsafe"):
        SkillCatalog.discover(tmp_path / "project", tmp_path / "user", linked)


@pytest.mark.asyncio
async def test_load_rejects_symlink_swap_where_supported(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    skill_file = _skill(bundled, "review", "body")
    catalog = SkillCatalog.discover(tmp_path / "project", tmp_path / "user", bundled)
    skill_file.unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        skill_file.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ToolFailure, match="unavailable"):
        await LoadSkillTool(catalog).execute(
            ToolCall("call", "load_skill", {"name": "review"}),
            ToolContext("agent", "workspace", tmp_path),
        )


@pytest.mark.asyncio
async def test_real_runtime_registers_load_skill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from litecoder.cli.app import build_runtime
    from litecoder.paths import AppPaths
    from litecoder.tools.registry import ToolRegistry

    user_dir = tmp_path / ".litecoder"
    user_dir.mkdir()
    (user_dir / "config.toml").write_text(
        'default_provider = "fake"\ndefault_model = "model"\n'
        '[providers.fake]\ntype = "openai-chat-completions"\nmodel = "model"\napi_key = "key"\n',
        encoding="utf-8",
    )
    paths = AppPaths(
        user_dir=user_dir, sessions_db=user_dir / "sessions.db",
        project_id="project", project_dir=user_dir / "projects" / "project",
        workspace_id="workspace", workspace_root=tmp_path,
    )
    monkeypatch.setattr(
        "litecoder.cli.app.AppPaths.discover", staticmethod(lambda cwd: paths)
    )
    names: list[str] = []
    original = ToolRegistry.register

    def recording_register(self: ToolRegistry, tool: object) -> None:
        names.append(tool.spec.name)  # type: ignore[attr-defined]
        original(self, tool)  # type: ignore[arg-type]

    monkeypatch.setattr(ToolRegistry, "register", recording_register)
    runtime = await build_runtime(tmp_path)
    try:
        assert "load_skill" in names
    finally:
        await runtime.close()

@pytest.mark.asyncio
async def test_load_skill_resolver_uses_tool_context_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other"
    _skill(workspace / ".litecoder" / "skills", "review", "workspace-body")
    _skill(other_workspace / ".litecoder" / "skills", "review", "wrong-body")
    seen: list[Path] = []

    def resolve_catalog(root: Path) -> SkillCatalog:
        seen.append(root)
        return SkillCatalog.discover(root, tmp_path / "user", tmp_path / "bundled")

    tool = LoadSkillTool(catalog_resolver=resolve_catalog)
    result = await tool.execute(
        ToolCall("call", "load_skill", {"name": "review"}),
        ToolContext("agent", "workspace", workspace),
    )

    assert seen == [workspace]
    assert "workspace-body" in result.content
    assert "wrong-body" not in result.content
