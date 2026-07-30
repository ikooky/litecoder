from __future__ import annotations

import pytest

from litecoder.context.token_budget import TokenBudget, estimate_tokens


def test_budget_gives_recent_context_only_the_remaining_capacity() -> None:
    budget = TokenBudget(total=1000, reserve=200)

    allocation = budget.allocate(
        system=200, tools=200, memories=100, recent=400
    )

    assert allocation.system == 200
    assert allocation.tools == 200
    assert allocation.memories == 100
    assert allocation.recent == 300
    assert allocation.reserve == 200
    assert allocation.truncated is True


def test_budget_preserves_recent_context_when_it_fits_exactly() -> None:
    allocation = TokenBudget(total=10, reserve=2).allocate(
        system=2, tools=2, memories=1, recent=3
    )

    assert allocation.recent == 3
    assert allocation.truncated is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"total": True, "reserve": 0}, "total"),
        ({"total": 10.0, "reserve": 0}, "total"),
        ({"total": 10, "reserve": False}, "reserve"),
        ({"total": 10, "reserve": -1}, "reserve"),
        ({"total": 5, "reserve": 6}, "reserve"),
    ],
)
def test_budget_rejects_invalid_capacity_values(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TokenBudget(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["system", "tools", "memories", "recent"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_budget_rejects_invalid_allocation_values(
    field: str, value: object
) -> None:
    values: dict[str, object] = {
        "system": 1,
        "tools": 1,
        "memories": 1,
        "recent": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        TokenBudget(total=10, reserve=1).allocate(**values)  # type: ignore[arg-type]


def test_estimate_tokens_is_deterministic_and_utf8_aware() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("眉眉") == 2
    assert estimate_tokens("abcde") == estimate_tokens("abcde") == 2


@pytest.mark.parametrize("value", [None, b"text", 123])
def test_estimate_tokens_rejects_non_text(value: object) -> None:
    with pytest.raises(ValueError, match="text"):
        estimate_tokens(value)  # type: ignore[arg-type]
