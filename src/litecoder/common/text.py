"""Low-level text helpers with no dependencies on the rest of the package.

Kept in ``common`` so that leaf modules (``hooks.builtin``, ``tools.executor``,
``tools.artifacts``, ``context.compaction``) can share one implementation of
UTF-8-safe truncation without pulling in ``tools.builtin`` and the circular
import chain it triggers.
"""

from __future__ import annotations


def truncate_utf8(value: str, limit: int) -> tuple[str, bool]:
    """Truncate ``value`` to at most ``limit`` bytes on a UTF-8 boundary.

    Returns the (possibly truncated) text and whether truncation happened. A
    non-positive ``limit`` always yields the empty string.
    """
    if limit <= 0:
        return "", bool(value)
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def truncate_utf8_text(value: str, limit: int) -> str:
    """Truncate ``value`` to at most ``limit`` bytes, returning only the text.

    Convenience wrapper around :func:`truncate_utf8` for callers that do not
    need the truncation flag.
    """
    return truncate_utf8(value, limit)[0]
