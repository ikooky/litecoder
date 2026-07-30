from __future__ import annotations

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from litecoder.tools.executor import (
    _safe_validation_message,
    _validation_code,
    _validation_path,
)


def _error(schema: dict[str, object], instance: object) -> ValidationError:
    return next(Draft202012Validator(schema).iter_errors(instance))


def test_missing_required_argument_feedback_is_specific_without_values() -> None:
    error = _error(
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        {"secret": "must-not-appear"},
    )

    message = _safe_validation_message(error)

    assert message == "Invalid tool arguments: required field $.path is missing"
    assert "must-not-appear" not in message
    assert _validation_code(error) == "required"
    assert _validation_path(error) == "$"


def test_nested_type_feedback_exposes_only_safe_path() -> None:
    error = _error(
        {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {"timeout": {"type": "number"}},
                }
            },
        },
        {"options": {"timeout": "sensitive-invalid-value"}},
    )

    message = _safe_validation_message(error)

    assert message == (
        "Invalid tool arguments: $.options.timeout has the wrong type"
    )
    assert "sensitive-invalid-value" not in message
    assert _validation_path(error) == "$.options.timeout"
