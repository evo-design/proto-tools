"""Tests for ccd-lookup tool (pdbeccdutils wrapper)."""

import json

import pytest

from proto_tools.entities.ligands import Fragment, Ligands
from proto_tools.tools.database_retrieval import (
    CcdEnrichment,
    CcdLookupConfig,
    CcdLookupInput,
    CcdLookupOutput,
    run_ccd_lookup,
)
from proto_tools.tools.database_retrieval.ccd_lookup.ccd_lookup import _build_fragment
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

# Share one warm worker across the module — parsing the CCD bundle dominates per-test cost.
_persistent_ccd_lookup = make_persistent_fixture("ccd_lookup", gpu=False)


# ── Unit tests ──────────────────────────────────────────────────────────────


def test_input_normalizes_single_string_to_list():
    """The custom validator lifts a single string identifier to a 1-element list."""
    assert CcdLookupInput(identifiers="ATP").identifiers == ["ATP"]


def test_output_computed_counts_split_resolved_and_unresolved():
    """num_resolved / num_unresolved partition enrichments by ccd_code is None."""
    output = CcdLookupOutput(
        tool_id="ccd-lookup",
        execution_time=0.0,
        success=True,
        ligands=Ligands(fragments=[Fragment(ccd_code="ATP"), Fragment(ccd_code="HEM"), Fragment(smiles="CCO")]),
        enrichments=[
            CcdEnrichment(ccd_code="ATP"),
            CcdEnrichment(ccd_code="HEM"),
            CcdEnrichment(ccd_code=None),
        ],
    )
    assert (output.num_resolved, output.num_unresolved) == (2, 1)


def test_output_rejects_mismatched_parallel_arrays():
    """The parallel-array invariant is enforced: ligands and enrichments lengths must match."""
    with pytest.raises(ValueError, match="parallel arrays"):
        CcdLookupOutput(
            tool_id="ccd-lookup",
            execution_time=0.0,
            success=True,
            ligands=Ligands(fragments=[Fragment(ccd_code="ATP")]),
            enrichments=[CcdEnrichment(ccd_code="ATP"), CcdEnrichment(ccd_code=None)],
        )


def test_enrichment_rejects_partial_resolution():
    """ccd_code=None requires the descriptive fields to also be None."""
    with pytest.raises(ValueError, match="must be fully resolved or fully None"):
        CcdEnrichment(ccd_code=None, formula="C2 H6 O")


def test_build_fragment_placeholder_keeps_ccd_code_none_through_json_roundtrip():
    """Unparseable input falls back to a placeholder whose ccd_code stays None after re-validation."""
    record = {"smiles": None, "ccd_code": None, "name": None}
    frag = _build_fragment(record, original_identifier="NOTACCDORSMILES")
    assert frag.ccd_code is None
    roundtripped = type(frag).model_validate_json(frag.model_dump_json())
    assert roundtripped.ccd_code is None


def test_output_export_json_writes_ligands_and_enrichments(tmp_path):
    """JSON export round-trips both fragments and enrichment records."""
    output = CcdLookupOutput(
        tool_id="ccd-lookup",
        execution_time=0.0,
        success=True,
        ligands=Ligands(fragments=[Fragment(ccd_code="ATP")]),
        enrichments=[CcdEnrichment(ccd_code="ATP", formula="C10 H16 N5 O13 P3", formula_weight=507.181)],
    )
    output.export(name="out", export_path=str(tmp_path), file_format="json")
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["enrichments"][0]["ccd_code"] == "ATP"
    assert payload["enrichments"][0]["formula"] == "C10 H16 N5 O13 P3"
    assert payload["ligands"]["fragments"][0]["ccd_code"] == "ATP"


# ── Integration tests (require pdbeccdutils env + components.cif) ───────────
# Skipped in CI: each loads the ~700 MB CCD bundle (SMILES paths re-parse for
# indexing), which blows past the GH runner wall-clock budget.


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_offline_enrichment_full_pass():
    """One batch covers every offline path: CCD codes, parent codes, SMILES, no-match, order, defaults."""
    paracetamol = "CC(=O)NC1=CC=C(C=C1)O"
    long_alkane = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"  # valid SMILES, no CCD match
    result = run_ccd_lookup(
        CcdLookupInput(identifiers=["ATP", "SEP", "MSE", paracetamol, long_alkane]),
        CcdLookupConfig(),
    )
    validate_output(result)
    atp, sep, mse, tylenol, junk = result.enrichments
    atp_frag, *_ = result.ligands.fragments

    assert atp.ccd_code == "ATP"
    assert atp.formula == "C10 H16 N5 O13 P3"
    assert atp.formula_weight == pytest.approx(507.18, abs=1.0)
    assert atp.inchikey == "ZKHQWZAMYRWXGA-KQYNXXCUSA-N"
    assert atp.released is True
    assert atp.release_status == "REL"
    assert {"logp", "tpsa", "num_h_donors", "num_h_acceptors", "num_heavy_atoms"} <= set(atp.physchem_properties)
    assert atp.physchem_properties["num_heavy_atoms"] == 31
    assert atp_frag.mol.GetNumHeavyAtoms() == 31

    assert sep.ccd_code == "SEP" and sep.parent_ccd_code == "SER"
    assert mse.ccd_code == "MSE" and mse.parent_ccd_code == "MET"

    assert tylenol.ccd_code is not None
    assert (tylenol.formula or "").replace(" ", "") == "C8H9NO2"

    assert junk.ccd_code is None

    assert [e.ccd_code for e in result.enrichments] == ["ATP", "SEP", "MSE", tylenol.ccd_code, None]
    assert len(result.ligands.fragments) == 5

    assert all(e.cross_references is None for e in result.enrichments)
    assert all(e.pdb_structures is None for e in result.enrichments)


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_include_cross_references_unichem():
    """include_cross_references=True maps ATP to the canonical external compound IDs."""
    result = run_ccd_lookup(
        CcdLookupInput(identifiers=["ATP"]),
        CcdLookupConfig(include_cross_references=True),
    )
    validate_output(result)
    xrefs = result.enrichments[0].cross_references
    assert xrefs is not None
    sources = {s.lower() for s in xrefs}
    assert {"chembl", "drugbank", "pubchem", "chebi"} <= sources
    assert "CHEMBL14249" in xrefs["chembl"]
    assert "DB00171" in xrefs["drugbank"]


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_garbage_input_returns_placeholder_fragment():
    """End-to-end: garbage input yields a None-ccd_code placeholder without crashing."""
    result = run_ccd_lookup(CcdLookupInput(identifiers=["NOTACCDORSMILES"]), CcdLookupConfig())
    validate_output(result)
    assert result.enrichments[0].ccd_code is None
    assert result.enrichments[0].errors
    assert result.ligands.fragments[0].ccd_code is None


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_include_pdb_usage_rcsb():
    """include_pdb_usage=True returns 4-character PDB codes; HEM appears in thousands of entries."""
    result = run_ccd_lookup(
        CcdLookupInput(identifiers=["HEM"]),
        CcdLookupConfig(include_pdb_usage=True),
    )
    validate_output(result)
    pdb_ids = result.enrichments[0].pdb_structures
    assert pdb_ids is not None
    assert len(pdb_ids) > 100
    assert all(isinstance(pid, str) and len(pid) == 4 for pid in pdb_ids)
