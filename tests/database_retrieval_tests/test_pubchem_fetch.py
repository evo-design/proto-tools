"""tests/database_retrieval_tests/test_pubchem_fetch.py.

Tests for the PubChem fetch tool.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    PubChemFetchConfig,
    PubChemFetchInput,
    run_pubchem_fetch,
)
from proto_tools.tools.database_retrieval.pubchem.pubchem_fetch import (
    _fetch_properties,
    _fetch_synonyms,
    _resolve_to_cids,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cid": 2244, "name": "aspirin"},
        {"name": "aspirin", "smiles": "CCO", "inchikey": "X-Y-Z"},
    ],
    ids=["zero", "two", "three"],
)
def test_input_rejects_anything_other_than_one_identifier(kwargs):
    with pytest.raises(ValidationError, match="Provide exactly one of"):
        PubChemFetchInput(**kwargs)


def test_resolve_to_cids_skips_http_when_cid_given():
    """Direct CID input must not trigger any HTTP call."""
    session = MagicMock()
    cids = _resolve_to_cids(PubChemFetchInput(cid=42), session)
    assert cids == [42]
    session.get.assert_not_called()


@pytest.mark.parametrize(
    "status_code, payload, expected",
    [
        (200, {"IdentifierList": {"CID": [2244]}}, [2244]),
        (200, {"IdentifierList": {"CID": [2244, 12345]}}, [2244, 12345]),
        (404, None, []),
    ],
    ids=["single", "multiple", "not-found"],
)
def test_resolve_to_cids_parses_response(status_code, payload, expected):
    session = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    session.get.return_value = response
    cids = _resolve_to_cids(PubChemFetchInput(name="anything"), session)
    assert cids == expected


def test_fetch_properties_raises_when_property_table_empty():
    """An empty Properties array indicates a malformed response and must surface clearly."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"PropertyTable": {"Properties": []}}
    session.get.return_value = response
    with pytest.raises(ValueError, match="no Properties entries"):
        _fetch_properties(2244, PubChemFetchConfig(), session)


def test_fetch_synonyms_truncates_to_50():
    """Synonym list is capped at 50 client-side regardless of server response."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "InformationList": {"Information": [{"CID": 2244, "Synonym": [f"syn-{i}" for i in range(100)]}]}
    }
    session.get.return_value = response
    synonyms = _fetch_synonyms(2244, session)
    assert len(synonyms) == 50
    assert synonyms == [f"syn-{i}" for i in range(50)]


def test_fetch_synonyms_404_returns_empty_list():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    session.get.return_value = response
    assert _fetch_synonyms(2244, session) == []


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skip_ci
@pytest.mark.integration
@pytest.mark.parametrize(
    "input_kwargs",
    [
        {"cid": 2244},
        {"name": "aspirin"},
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
    ],
    ids=["cid", "name", "smiles", "inchikey"],
)
def test_pubchem_fetch_resolves_aspirin_from_every_identifier(input_kwargs):
    """All four identifier types must converge on aspirin's CID 2244."""
    output = run_pubchem_fetch(PubChemFetchInput(**input_kwargs), PubChemFetchConfig())
    assert output.success
    assert output.tool_id == "pubchem-fetch"
    assert output.cid == 2244
    assert output.molecular_formula == "C9H8O4"


@pytest.mark.integration
def test_pubchem_fetch_full_property_bundle_aspirin():
    """The default property bundle must populate every typed field for a real compound."""
    output = run_pubchem_fetch(PubChemFetchInput(name="aspirin"), PubChemFetchConfig())
    assert output.success
    assert output.cid == 2244
    assert output.title is not None and "aspirin" in output.title.lower()
    assert output.molecular_formula == "C9H8O4"
    assert output.molecular_weight == pytest.approx(180.16, rel=1e-2)
    assert output.smiles is not None
    assert output.connectivity_smiles is not None
    assert output.inchi is not None and output.inchi.startswith("InChI=1S/C9H8O4")
    assert output.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert "benzoic acid" in output.iupac_name.lower()
    assert output.exact_mass == pytest.approx(180.04, abs=0.1)
    assert output.tpsa == pytest.approx(63.6, abs=1.0)
    assert output.heavy_atom_count == 13
    assert output.hbond_donor_count == 1
    assert output.hbond_acceptor_count == 4
    assert output.rotatable_bond_count == 3
    assert output.charge == 0
    assert output.synonyms == []  # off by default


@pytest.mark.integration
def test_pubchem_fetch_drug_like_compound_lipinski_signals():
    """Ibuprofen is a textbook Lipinski-compliant drug; the descriptor counts must reflect that."""
    output = run_pubchem_fetch(PubChemFetchInput(cid=3672), PubChemFetchConfig())
    assert output.success
    assert output.molecular_formula == "C13H18O2"
    assert output.molecular_weight is not None and output.molecular_weight < 500  # rule of five
    assert output.hbond_donor_count is not None and output.hbond_donor_count <= 5
    assert output.hbond_acceptor_count is not None and output.hbond_acceptor_count <= 10


@pytest.mark.integration
def test_pubchem_fetch_smiles_canonicalization_roundtrip():
    """A non-canonical SMILES must still resolve to the canonical CID."""
    non_canonical = "OC(=O)c1ccccc1OC(C)=O"  # aspirin written backwards
    output = run_pubchem_fetch(PubChemFetchInput(smiles=non_canonical), PubChemFetchConfig())
    assert output.success
    assert output.cid == 2244
    assert output.smiles is not None
    # The wrapper exposes PubChem's canonical SMILES, which differs from the input
    assert output.smiles != non_canonical


@pytest.mark.skip_ci
@pytest.mark.integration
def test_pubchem_fetch_with_synonyms():
    """Enabling synonyms triggers the second HTTP call and returns up to 50 synonyms."""
    output = run_pubchem_fetch(
        PubChemFetchInput(name="aspirin"),
        PubChemFetchConfig(include_synonyms=True),
    )
    assert output.success
    assert 0 < len(output.synonyms) <= 50
    assert "aspirin" in [s.lower() for s in output.synonyms]


@pytest.mark.integration
def test_pubchem_fetch_custom_property_subset_leaves_other_fields_unset():
    """Asking for only two properties must leave all other typed fields as None."""
    output = run_pubchem_fetch(
        PubChemFetchInput(name="aspirin"),
        PubChemFetchConfig(properties=["MolecularFormula", "MolecularWeight"]),
    )
    assert output.success
    assert output.molecular_formula == "C9H8O4"
    assert output.molecular_weight is not None
    assert output.smiles is None
    assert output.inchi is None
    assert output.tpsa is None


@pytest.mark.integration
def test_workflow_name_to_smiles_to_inchikey_roundtrip_converges():
    """Chained workflow: name -> CID, then SMILES -> CID, then InChIKey -> CID.

    Whenever a downstream tool emits a SMILES or InChIKey that a caller hands
    back to this wrapper, the round-trip must converge on the same compound. A
    failure here means the wrapper's identifier resolution is asymmetric -- a
    silent bug that would corrupt downstream design provenance.
    """
    by_name = run_pubchem_fetch(PubChemFetchInput(name="caffeine"), PubChemFetchConfig())
    assert by_name.success
    assert by_name.cid == 2519
    assert by_name.smiles is not None
    assert by_name.inchikey is not None

    by_smiles = run_pubchem_fetch(PubChemFetchInput(smiles=by_name.smiles), PubChemFetchConfig())
    assert by_smiles.success
    assert by_smiles.cid == by_name.cid
    assert by_smiles.molecular_formula == by_name.molecular_formula
    assert by_smiles.inchikey == by_name.inchikey

    by_inchikey = run_pubchem_fetch(PubChemFetchInput(inchikey=by_name.inchikey), PubChemFetchConfig())
    assert by_inchikey.success
    assert by_inchikey.cid == by_name.cid
    # The resolved canonical SMILES must be invariant across all three entry points.
    assert by_inchikey.smiles == by_name.smiles


@pytest.mark.integration
def test_pubchem_fetch_unknown_name_returns_failure():
    """An unresolvable name raises with a descriptive error message identifying the input."""
    with pytest.raises(
        Exception,
        match=r"PubChem returned no CIDs.*name='notacompoundxyz123def'|name='notacompoundxyz123def'.*PubChem returned no CIDs",
    ):
        run_pubchem_fetch(PubChemFetchInput(name="notacompoundxyz123def"), PubChemFetchConfig())


@pytest.mark.integration
def test_pubchem_fetch_include_description():
    """Setting `include_description=True` populates `descriptions` with at least one entry."""
    output = run_pubchem_fetch(
        PubChemFetchInput(cid=2244),
        PubChemFetchConfig(include_description=True),
    )
    assert output.success
    assert len(output.descriptions) > 0
    # At least one description should mention aspirin or salicylate
    combined = " ".join(output.descriptions).lower()
    assert "aspirin" in combined or "salicyl" in combined


@pytest.mark.integration
def test_pubchem_fetch_include_aids_returns_assay_ids():
    """Setting `include_aids=True` returns a list of integer assay IDs."""
    output = run_pubchem_fetch(
        PubChemFetchInput(cid=2244),
        PubChemFetchConfig(include_aids=True),
    )
    assert output.success
    # Aspirin has hundreds of bioassays
    assert len(output.bioassay_ids) > 100
    assert all(isinstance(aid, int) and aid > 0 for aid in output.bioassay_ids)


@pytest.mark.integration
def test_pubchem_fetch_default_excludes_optional_payloads():
    """Default config does NOT make extra HTTP calls for synonyms/descriptions/aids."""
    output = run_pubchem_fetch(PubChemFetchInput(cid=2244), PubChemFetchConfig())
    assert output.success
    assert output.synonyms == []
    assert output.descriptions == []
    assert output.bioassay_ids == []
