"""Tests for the SSAlign two-stage structure-search tool."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment.ssalign import (
    SSAlignHit,
    SSAlignQuery,
    SSAlignQueryResult,
    SSAlignSearchConfig,
    SSAlignSearchInput,
    SSAlignSearchOutput,
    run_ssalign_search,
)
from proto_tools.tools.structure_alignment.ssalign.ssalign_search import example_input
from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.tool_io import MissingAssetError

_DUMMY_DATA = Path(__file__).parent.parent / "dummy_data"
_FIXTURE_PDB = str(
    Path(__file__).parent.parent.parent / "proto_tools" / "tools" / "structure_alignment" / "example_input_fixture.pdb"
)
_TARGET_PDBS = [
    str(_DUMMY_DATA / "pdl1.pdb"),
    str(_DUMMY_DATA / "renin_af3.pdb"),
    str(_DUMMY_DATA / "test_structure_similarity.pdb"),
]


# ── Config validators ───────────────────────────────────────────────────────


def test_config_requires_exactly_one_target():
    """Exactly one of target_structures / ssalign_db must be set — both or neither is rejected."""
    with pytest.raises(ValidationError, match="exactly one"):
        SSAlignSearchConfig(target_structures=[_FIXTURE_PDB], ssalign_db="/some/db")
    with pytest.raises(ValidationError, match="exactly one"):
        SSAlignSearchConfig()


def test_config_max_target_must_not_exceed_prefilter_target():
    """max_target above prefilter_target is incoherent and rejected."""
    with pytest.raises(ValidationError, match="must be <= prefilter_target"):
        SSAlignSearchConfig(target_structures=[_FIXTURE_PDB], max_target=3000, prefilter_target=2000)


def test_input_requires_at_least_one_query():
    """An empty query list fails the after-validator."""
    with pytest.raises(ValidationError, match="at least one query"):
        SSAlignSearchInput(queries=[])


# ── Registration / example ──────────────────────────────────────────────────


def test_registry_registration():
    """The tool is registered with the expected iterable wiring and category."""
    spec = ToolRegistry.get("ssalign-search")
    assert spec.iterable_input_field == "queries"
    assert spec.iterable_output_field == "results"
    assert spec.uses_gpu is True
    assert spec.category == "structure_alignment"


def test_example_input_builds():
    """example_input() validates and yields exactly one query."""
    example = example_input()
    assert isinstance(example, SSAlignSearchInput)
    assert len(example.queries) == 1


# ── Output models / export ──────────────────────────────────────────────────


def test_output_export_roundtrip(tmp_path):
    """JSON export reloads the query bundle; CSV export has a header plus one row per hit."""
    output = SSAlignSearchOutput(
        results=[
            SSAlignQueryResult(
                query_id="q0",
                num_hits=2,
                hits=[
                    SSAlignHit(target_id="target_0", prefilter_score=1.0, ss_score=1.11, rank=1, refined=False),
                    SSAlignHit(
                        target_id="target_1",
                        prefilter_score=0.42,
                        saligner_score=12.3,
                        ss_score=0.791,
                        rank=2,
                        refined=True,
                    ),
                ],
            )
        ]
    )

    output.export("demo", export_path=tmp_path, file_format="json")
    output.export("demo", export_path=tmp_path, file_format="csv")

    json_path = tmp_path / "demo.json"
    csv_path = tmp_path / "demo.csv"
    assert json_path.exists()
    assert csv_path.exists()

    reloaded = json.loads(json_path.read_text())
    assert reloaded["results"][0]["query_id"] == "q0"
    assert len(reloaded["results"][0]["hits"]) == 2
    assert reloaded["results"][0]["hits"][0]["target_id"] == "target_0"

    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["query_id", "rank", "target_id", "prefilter_score", "saligner_score", "ss_score", "refined"]
    assert len(rows) == 3  # header + 2 hits
    assert rows[1][2] == "target_0"
    assert rows[2][2] == "target_1"


def test_mode2_missing_db_raises_missing_asset():
    """Mode 2 fails fast (before dispatch) when the prebuilt SSAlignDB directory is absent."""
    inputs = example_input()
    config = SSAlignSearchConfig(ssalign_db="/nonexistent/ssalign_db", dim=512)
    with pytest.raises(MissingAssetError):
        run_ssalign_search(inputs, config)


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
def test_mode1_build_on_the_fly_self_search():
    """Mode-1 build-on-the-fly: a query that is also a target must rank itself first at cosine ~1."""
    query_pdb = _TARGET_PDBS[0]
    inputs = SSAlignSearchInput(queries=[SSAlignQuery(structure=query_pdb, query_id="self")])
    config = SSAlignSearchConfig(
        target_structures=_TARGET_PDBS,
        device="cpu",
        mode=1,
        prefilter_target=3,
        max_target=3,
    )
    result = run_ssalign_search(inputs, config)

    assert result.success
    assert result.tool_id == "ssalign-search"
    assert len(result.results) == 1

    bundle = result.results[0]
    assert bundle.query_id == "self"
    top = bundle.hits[0]
    assert top.target_id == "target_0"  # the query is target_structures[0]
    assert top.prefilter_score >= 0.99
    assert [h.rank for h in bundle.hits] == list(range(1, len(bundle.hits) + 1))
    for hit in bundle.hits:
        assert hit.ss_score == pytest.approx(0.55 * hit.prefilter_score + 0.56)


@pytest.mark.integration
def test_mode1_refine_path():
    """A high prefilter_threshold sends non-self hits to SAligner refinement; the self-hit stays unrefined."""
    query_pdb = _TARGET_PDBS[0]
    inputs = SSAlignSearchInput(queries=[SSAlignQuery(structure=query_pdb, query_id="self")])
    config = SSAlignSearchConfig(
        target_structures=_TARGET_PDBS,
        device="cpu",
        mode=1,
        prefilter_target=3,
        max_target=3,
        prefilter_threshold=0.99,
    )
    result = run_ssalign_search(inputs, config)

    hits = result.results[0].hits
    refined = [h for h in hits if h.refined]
    assert refined, "expected at least one below-threshold hit to be SAligner-refined"
    assert all(h.saligner_score is not None for h in refined)

    self_hit = next(h for h in hits if h.target_id == "target_0")
    assert self_hit.prefilter_score >= 0.99
    assert self_hit.refined is False
    assert self_hit.saligner_score is None
