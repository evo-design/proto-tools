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
from proto_tools.tools.database_retrieval.ccd_lookup.ccd_lookup import _build_fragment, _status
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
    """num_resolved counts resolved results; num_unresolved counts both other statuses."""
    output = CcdLookupOutput(
        tool_id="ccd-lookup",
        execution_time=0.0,
        success=True,
        results=[
            CcdLookupResult(fragment=Fragment(ccd_code="ATP"), status="resolved"),
            CcdLookupResult(fragment=Fragment(ccd_code="HEM"), status="resolved"),
            CcdLookupResult(fragment=_unresolvable_fragment(), formula="C2 H6 O", status="no_ccd_code"),
            CcdLookupResult(fragment=_unresolvable_fragment(), status="not_found"),
        ],
    )
    assert (output.num_resolved, output.num_unresolved) == (2, 2)


def test_result_rejects_metadata_when_nothing_matched():
    """CCD metadata on a not_found result can only have leaked from another record."""
    with pytest.raises(ValueError, match="not_found"):
        CcdLookupResult(fragment=_unresolvable_fragment(), formula="C2 H6 O", status="not_found")


def test_result_keeps_metadata_for_a_match_without_a_code():
    """A matched entry the fragment can't carry keeps its CCD metadata."""
    result = CcdLookupResult(fragment=_unresolvable_fragment(), formula="C2 H6 O", status="no_ccd_code")

    assert result.fragment.ccd_code is None
    assert result.formula == "C2 H6 O"


def test_an_unrelated_warning_does_not_excuse_leaked_metadata():
    """Only the status allows metadata without a code, so incidental warnings cannot widen it."""
    with pytest.raises(ValueError, match="not_found"):
        CcdLookupResult(
            fragment=_unresolvable_fragment(),
            formula="C2 H6 O",
            status="not_found",
            warnings=["Physchem properties failed: something unrelated"],
        )


@pytest.mark.parametrize(
    ("fragment", "status"),
    [
        (lambda: Fragment(ccd_code="ATP"), "no_ccd_code"),
        (lambda: Fragment(ccd_code="ATP"), "not_found"),
        (_unresolvable_fragment, "resolved"),
    ],
)
def test_status_must_match_whether_the_fragment_carries_a_code(fragment, status: str):
    """Only a resolved result carries a CCD code, and every resolved result does."""
    with pytest.raises(ValueError, match="only a resolved result carries a CCD code"):
        CcdLookupResult(fragment=fragment(), status=status)


_ENTRY = {"formula": "C2 H6 O", "release_status": "REL"}


@pytest.mark.parametrize(
    ("fragment", "record", "status", "reason"),
    [
        (lambda: Fragment(ccd_code="ATP"), {"ccd_code": "ATP", **_ENTRY}, "resolved", None),
        (_unresolvable_fragment, {"ccd_code": None}, "not_found", None),
        # The standalone sets ccd_code for any code-shaped identifier, so a missing entry is still not_found
        (_unresolvable_fragment, {"ccd_code": "ZZZZZ"}, "not_found", None),
        (_unresolvable_fragment, {"ccd_code": "ZZZZZ", **_ENTRY}, "no_ccd_code", "not in proto-tools' CCD table"),
        (_unresolvable_fragment, {"ccd_code": "ATP", **_ENTRY}, "no_ccd_code", "no SMILES RDKit can parse"),
    ],
)
def test_status_classifies_each_outcome_and_explains_a_missing_code(fragment, record, status: str, reason):
    """A matched entry the fragment lacks gets an error naming why; other outcomes add none."""
    errors: list[str] = []
    assert _status(fragment(), record, errors) == status
    if reason is None:
        assert errors == []
    else:
        assert len(errors) == 1
        assert reason in errors[0]


def test_build_fragment_keeps_a_resolved_code_when_the_record_smiles_is_unusable():
    """A valid CCD code survives a record whose SMILES will not parse."""
    fragment = _build_fragment({"ccd_code": "ATP", "smiles": "this-is-not-smiles", "name": "ATP"}, "ATP")

    assert fragment.ccd_code == "ATP"


def test_build_fragment_never_reads_a_matched_code_as_smiles():
    """A matched entry whose structure can't be built gets the placeholder, not its code parsed as SMILES."""
    # FCO is an iron complex; read as a SMILES it would be fluoromethanol
    record = {"ccd_code": "FCO", "smiles": "unparseable", "name": "IRON COMPLEX", "formula": "C3 Fe N2 O"}
    fragment = _build_fragment(record, "FCO")

    assert fragment.smiles == "[He]"
    assert fragment.ccd_code is None


def test_build_fragment_reads_an_unmatched_identifier_as_smiles():
    """With no CCD entry, a code-shaped identifier may still be the SMILES the caller meant."""
    fragment = _build_fragment({"ccd_code": "CCCCO", "smiles": None, "name": None}, "CCCCO")

    assert fragment.smiles == "CCCCO"


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
                status="resolved",
            )
        ],
    )
    output.export(name="out", export_path=str(tmp_path), file_format="json")
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["results"][0]["fragment"]["ccd_code"] == "ATP"
    assert payload["results"][0]["formula"] == "C10 H16 N5 O13 P3"
    assert payload["results"][0]["status"] == "resolved"
    assert payload["ligands"]["fragments"][0]["ccd_code"] == "ATP"


# ── Integration tests (require pdbeccdutils env + components.cif) ───────────
# Skipped in CI: each loads the ~700 MB CCD bundle, which blows past the GH runner wall-clock budget.


@pytest.mark.skip_ci
@pytest.mark.uses_env
def test_ccd_lookup_offline_full_pass():
    """One batch covers the offline CCD-code path: metadata, parent codes, order, and enrichment defaults."""
    result = run_ccd_lookup(CcdLookupInput(identifiers=["ATP", "SEP", "MSE"]), CcdLookupConfig())
    validate_output(result)
    atp, sep, mse = result.results
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

    assert [e.fragment.ccd_code for e in result.results] == ["ATP", "SEP", "MSE"]
    assert len(result.ligands.fragments) == 3

    assert all(e.cross_references is None for e in result.results)
    assert all(e.pdb_structures is None for e in result.results)


@pytest.mark.skip_ci
@pytest.mark.uses_env
def test_ccd_lookup_returns_metadata_for_entries_rdkit_cannot_sanitize():
    """C8L (cucurbit[8]uril) overflows RDKit's CIP labeler, yet still returns its CCD metadata.

    wwPDB revisions can make an entry sanitizable (CZL and HB1 were, in 2026-09), so the test
    first checks that C8L still takes the unsanitized fallback.
    """
    result = run_ccd_lookup(CcdLookupInput(identifiers=["C8L"]), CcdLookupConfig())
    validate_output(result)
    (c8l,) = result.results

    assert any("sanitization failed" in w for w in c8l.warnings), "C8L no longer exercises the fallback"
    assert c8l.formula == "C48 H48 N32 O16"
    assert c8l.formula_weight == pytest.approx(1329.10, abs=1.0)
    assert not c8l.errors


@pytest.mark.skip_ci
@pytest.mark.uses_env
def test_ccd_lookup_computes_descriptors_without_sanitization():
    """sanitize=False still yields formula_weight and RDKit physchem descriptors."""
    result = run_ccd_lookup(CcdLookupInput(identifiers=["ATP"]), CcdLookupConfig(sanitize=False))
    validate_output(result)
    (atp,) = result.results

    assert atp.formula_weight == pytest.approx(507.18, abs=1.0)
    assert {"molecular_weight", "logp", "tpsa", "num_h_donors", "num_h_acceptors"} <= set(atp.physchem_properties)
    assert not any("formula_weight" in w for w in atp.warnings)


@pytest.mark.skip_ci
@pytest.mark.uses_env
def test_ccd_lookup_keeps_a_code_rdkit_cannot_parse_off_a_different_molecule():
    """FCO's SMILES doesn't parse, and its code read as a SMILES is fluoromethanol, so it gets the placeholder."""
    result = run_ccd_lookup(CcdLookupInput(identifiers=["FCO"]), CcdLookupConfig()).results[0]

    assert result.status == "no_ccd_code"
    assert result.formula is not None and "Fe" in result.formula
    assert result.fragment.smiles == "[He]"


@pytest.mark.skip_ci
@pytest.mark.uses_env
def test_ccd_lookup_network_enrichment():
    """UniChem maps ATP to canonical external compound IDs; RCSB lists HEM in thousands of PDB entries."""
    result = run_ccd_lookup(
        CcdLookupInput(identifiers=["ATP", "HEM"]),
        CcdLookupConfig(include_cross_references=True, include_pdb_usage=True),
    )
    validate_output(result)
    atp, hem = result.results

    xrefs = atp.cross_references
    assert xrefs is not None
    sources = {s.lower() for s in xrefs}
    assert {"chembl", "drugbank", "pubchem", "chebi"} <= sources
    assert "CHEMBL14249" in xrefs["chembl"]
    assert "DB00171" in xrefs["drugbank"]

    pdb_ids = hem.pdb_structures
    assert pdb_ids is not None
    assert len(pdb_ids) > 100
    assert all(isinstance(pid, str) and len(pid) == 4 for pid in pdb_ids)
