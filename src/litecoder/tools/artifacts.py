"""Evaluation artifact collection and storage."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from litecoder.common.trace import SecretRedactor


TOOL_RESULT_INLINE_BYTES = 32_768
ARTIFACT_PREVIEW_BYTES = 1_000


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Data model representing the artifact reference."""
    path: Path
    preview: str
    bytes: int
    tool_call_id_sha256: str

    def as_metadata(self) -> dict[str, object]:
        """Handle the as metadata operation."""
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "tool_call_id_sha256": self.tool_call_id_sha256,
        }


_WINDOWS_DEVICE_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ProjectArtifactStores:
    """Component responsible for the project artifact stores."""
    def __init__(self, user_dir: Path, redactor: SecretRedactor) -> None:
        if not isinstance(user_dir, Path):
            raise ValueError("user_dir must be a Path")
        if not isinstance(redactor, SecretRedactor):
            raise ValueError("redactor must be a SecretRedactor")
        self.user_dir = user_dir.expanduser().resolve()
        self.redactor = redactor
        self._projects_root = (self.user_dir / "projects").resolve()
        self._stores: dict[tuple[str, str], ArtifactStore] = {}

    def __call__(self, context: object) -> ArtifactStore:
        metadata = getattr(context, "metadata", None)
        project_id = metadata.get("project_id") if isinstance(metadata, dict) else None
        project_id = _safe_project_id(project_id)
        session_id = getattr(context, "agent_session_id", "")
        scope = session_id.strip() if isinstance(session_id, str) else ""
        store_key = (project_id, scope)
        store = self._stores.get(store_key)
        if store is None:
            root = (self._projects_root / project_id / "outputs").resolve()
            try:
                root.relative_to(self._projects_root)
            except ValueError as error:
                raise ValueError("project_id must be a safe path segment") from error
            store = ArtifactStore(root, self.redactor, scope=scope)
            self._stores[store_key] = store
        return store


def _safe_project_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("project_id must be a safe path segment")
    if value in {".", ".."} or any(character in value for character in "/\\:"):
        raise ValueError("project_id must be a safe path segment")
    if any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in value):
        raise ValueError("project_id must be a safe path segment")
    if value.rstrip(" .") != value:
        raise ValueError("project_id must be a safe path segment")
    if value.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES:
        raise ValueError("project_id must be a safe path segment")
    return value


class ArtifactStore:
    """Storage interface for the artifact store."""
    def __init__(
        self, root: Path, redactor: SecretRedactor, *, scope: str = ""
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("root must be a Path")
        if not isinstance(redactor, SecretRedactor):
            raise ValueError("redactor must be a SecretRedactor")
        if not isinstance(scope, str):
            raise ValueError("scope must be text")
        self.root = root.expanduser().resolve()
        self.redactor = redactor
        self.scope = scope

    async def persist(
        self, tool_call_id: str, content: str
    ) -> ArtifactReference:
        """Persist the generated artifact."""
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool_call_id must not be empty")
        if not isinstance(content, str):
            raise ValueError("content must be text")
        redacted = self.redactor.redact_text(content)
        identity = tool_call_id if not self.scope else f"{self.scope}\0{tool_call_id}"
        digest = hashlib.sha256(
            identity.encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        path = await asyncio.to_thread(self._write_atomic, digest, redacted)
        return ArtifactReference(
            path=path,
            preview=_truncate_utf8(redacted, ARTIFACT_PREVIEW_BYTES),
            bytes=len(redacted.encode("utf-8")),
            tool_call_id_sha256=digest,
        )

    def _write_atomic(self, digest: str, content: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"artifact-{digest}.txt"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".artifact-", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")
