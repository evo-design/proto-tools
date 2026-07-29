"""Tests for ccd-lookup tool (pdbeccdutils wrapper)."""

import json

import pytest

from proto_tools.entities.ligands import Fragment
from proto_tools.tools.database_retrieval import (
    CcdLookupConfig,
    CcdLookupInput,
    CcdLookupOutput,
    CcdLookupResult,
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


def _unresolvable_fragment() -> Fragment:
    """A fragment that carries no CCD code, via the tool's own placeholder path."""
    return _build_fragment({"smiles": None, "ccd_code": None, "name": None}, "NOTACCDORSMILES")


def test_output_computed_counts_split_resolved_and_unresolved():
    """num_resolved / num_unresolved partition entries by whether the fragment carries a code."""
    output = CcdLookupOutput(
        tool_id="ccd-lookup",
        execution_time=0.0,
        success=True,
        results=[
            CcdLookupResult(fragment=Fragment(ccd_code="ATP")),
            CcdLookupResult(fragment=Fragment(ccd_code="HEM")),
            CcdLookupResult(fragment=_unresolvable_fragment()),
        ],
    )
    assert (output.num_resolved, output.num_unresolved) == (2, 1)


def test_result_rejects_unexplained_metadata_without_a_code():
    """CCD metadata beside a code-less fragment is a leak unless the row declares why."""
    with pytest.raises(ValueError, match="resolved_without_structure"):
        CcdLookupResult(fragment=_unresolvable_fragment(), formula="C2 H6 O")


def test_result_allows_metadata_without_a_code_when_declared():
    """A resolved entry whose structure could not be rebuilt is a legitimate degraded row."""
    result = CcdLookupResult(
        fragment=_unresolvable_fragment(),
        formula="C2 H6 O",
        resolved_without_structure=True,
    )

    assert result.fragment.ccd_code is None
    assert result.formula == "C2 H6 O"


def test_an_unrelated_warning_does_not_excuse_leaked_metadata():
    """Only the declared flag opens the escape hatch, so incidental warnings cannot widen it."""
    with pytest.raises(ValueError, match="resolved_without_structure"):
        CcdLookupResult(
            fragment=_unresolvable_fragment(),
            formula="C2 H6 O",
            warnings=["Physchem properties failed: something unrelated"],
        )


def test_build_fragment_keeps_a_resolved_code_when_the_record_smiles_is_unusable():
    """A valid CCD code survives a record whose SMILES will not parse."""
    fragment = _build_fragment({"ccd_code": "ATP", "smiles": "this-is-not-smiles", "name": "ATP"}, "ATP")

    assert fragment.ccd_code == "ATP"


def test_build_fragment_placeholder_keeps_ccd_code_none_through_json_roundtrip():
    """Unparseable input falls back to a placeholder whose ccd_code stays None after re-validation."""
    record = {"smiles": None, "ccd_code": None, "name": None}
    frag = _build_fragment(record, original_identifier="NOTACCDORSMILES")
    assert frag.ccd_code is None
    roundtripped = type(frag).model_validate_json(frag.model_dump_json())
    assert roundtripped.ccd_code is None


def test_output_export_json_writes_ligands_and_results(tmp_path):
    """JSON export round-trips both fragments and result records."""
    output = CcdLookupOutput(
        tool_id="ccd-lookup",
        execution_time=0.0,
        success=True,
        results=[
            CcdLookupResult(
                fragment=Fragment(ccd_code="ATP"),
                formula="C10 H16 N5 O13 P3",
                formula_weight=507.181,
            )
        ],
    )
    output.export(name="out", export_path=str(tmp_path), file_format="json")
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["results"][0]["fragment"]["ccd_code"] == "ATP"
    assert payload["results"][0]["formula"] == "C10 H16 N5 O13 P3"
    assert payload["ligands"]["fragments"][0]["ccd_code"] == "ATP"


# ── Integration tests (require pdbeccdutils env + components.cif) ───────────
# Skipped in CI: each loads the ~700 MB CCD bundle (SMILES paths re-parse for
# indexing), which blows past the GH runner wall-clock budget.


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_offline_full_pass():
    """One batch covers every offline path: CCD codes, parent codes, SMILES, no-match, order, defaults."""
    paracetamol = "CC(=O)NC1=CC=C(C=C1)O"
    long_alkane = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"  # valid SMILES, no CCD match
    result = run_ccd_lookup(
        CcdLookupInput(identifiers=["ATP", "SEP", "MSE", paracetamol, long_alkane]),
        CcdLookupConfig(),
    )
    validate_output(result)
    atp, sep, mse, tylenol, junk = result.results
    atp_frag, *_ = result.ligands.fragments

    assert atp.fragment.ccd_code == "ATP"
    assert atp.formula == "C10 H16 N5 O13 P3"
    assert atp.formula_weight == pytest.approx(507.18, abs=1.0)
    assert atp.inchikey == "ZKHQWZAMYRWXGA-KQYNXXCUSA-N"
    assert atp.released is True
    assert atp.release_status == "REL"
    assert {"logp", "tpsa", "num_h_donors", "num_h_acceptors", "num_heavy_atoms"} <= set(atp.physchem_properties)
    assert atp.physchem_properties["num_heavy_atoms"] == 31
    assert atp_frag.mol.GetNumHeavyAtoms() == 31

    assert sep.fragment.ccd_code == "SEP" and sep.parent_ccd_code == "SER"
    assert mse.fragment.ccd_code == "MSE" and mse.parent_ccd_code == "MET"

    assert tylenol.fragment.ccd_code is not None
    assert (tylenol.formula or "").replace(" ", "") == "C8H9NO2"

    assert junk.fragment.ccd_code is None

    assert [e.fragment.ccd_code for e in result.results] == [
        "ATP",
        "SEP",
        "MSE",
        tylenol.fragment.ccd_code,
        None,
    ]
    assert len(result.ligands.fragments) == 5

    assert all(e.cross_references is None for e in result.results)
    assert all(e.pdb_structures is None for e in result.results)


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_include_cross_references_unichem():
    """include_cross_references=True maps ATP to the canonical external compound IDs."""
    result = run_ccd_lookup(
        CcdLookupInput(identifiers=["ATP"]),
        CcdLookupConfig(include_cross_references=True),
    )
    validate_output(result)
    xrefs = result.results[0].cross_references
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
    assert result.results[0].fragment.ccd_code is None
    assert result.results[0].errors
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
    pdb_ids = result.results[0].pdb_structures
    assert pdb_ids is not None
    assert len(pdb_ids) > 100
    assert all(isinstance(pid, str) and len(pid) == 4 for pid in pdb_ids)
