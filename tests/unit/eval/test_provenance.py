from __future__ import annotations

from litecoder.eval.provenance import collect_provenance


def test_provenance_omits_repository_metadata() -> None:
    provenance = collect_provenance()

    assert set(provenance) == {"command", "runtime"}
    assert "repository" not in provenance
