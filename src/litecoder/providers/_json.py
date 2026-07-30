"""JSON-compatible value validation and snapshots."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def snapshot_json(value: object, field_name: str) -> JsonValue:
    """Handle the snapshot json operation."""
    return _snapshot_json(value, field_name, set())


def snapshot_mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    """Handle the snapshot mapping operation."""
    snapshot = snapshot_json(value, field_name)
    if not isinstance(snapshot, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return snapshot


def snapshot_object_list(value: object, field_name: str) -> list[dict[str, JsonValue]]:
    """Handle the snapshot object list operation."""
    snapshot = snapshot_json(value, field_name)
    if not isinstance(snapshot, list) or any(not isinstance(item, dict) for item in snapshot):
        raise ValueError(f"{field_name} must be a list of JSON objects")
    return snapshot  # type: ignore[return-value]


def _snapshot_json(value: object, field_name: str, active: set[int]) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise _invalid_json(field_name, "non-finite floats are not allowed")

    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise _invalid_json(field_name, "a cycle was detected")
        active.add(identity)
        try:
            return [_snapshot_json(item, field_name, active) for item in value]
        finally:
            active.remove(identity)

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise _invalid_json(field_name, "a cycle was detected")
        if any(not isinstance(key, str) for key in value):
            raise _invalid_json(field_name, "object keys must be strings")
        active.add(identity)
        try:
            return {
                key: _snapshot_json(item, field_name, active)
                for key, item in value.items()
            }
        finally:
            active.remove(identity)

    raise _invalid_json(field_name, "unsupported value type")


def _invalid_json(field_name: str, reason: str) -> ValueError:
    return ValueError(f"{field_name} must contain only JSON-compatible values: {reason}")
