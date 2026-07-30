from __future__ import annotations

import pytest

from litecoder.tasks.graph import MissingDependency, TaskGraph
from litecoder.tasks.planning import TaskCycleError


def test_new_edge_reports_complete_cycle() -> None:
    graph = TaskGraph.from_edges({
        "A": ["B"],
        "B": ["C"],
        "C": [],
    })

    with pytest.raises(TaskCycleError) as error:
        graph.validate_edge("C", "A")

    assert error.value.path == ["C", "A", "B", "C"]


def test_startup_validation_rejects_manually_corrupted_graph() -> None:
    graph = TaskGraph.from_edges({"A": ["B"], "B": ["A"]})

    with pytest.raises(TaskCycleError) as error:
        graph.validate_all()

    assert error.value.path == ["A", "B", "A"]


def test_graph_rejects_missing_dependency() -> None:
    graph = TaskGraph.from_edges({"A": ["missing"]})

    with pytest.raises(MissingDependency) as error:
        graph.validate_all()

    assert error.value.task_id == "A"
    assert error.value.dependency_id == "missing"