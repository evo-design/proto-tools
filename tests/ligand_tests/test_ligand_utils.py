"""tests/ligand_tests/test_ligand_utils.py.

Tests for ligand utility functions.
"""

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from proto_tools.entities.ligands.utils import (
    fetch_pubchem_txt,
    is_mol_valid,
    is_smiles_valid,
    lookup_name_via_pubchem,
    lookup_smiles_via_pubchem,
)


@pytest.mark.parametrize("smiles,expected", [("CCO", True), ("INVALID_SMILES", False)], ids=["valid", "invalid"])
def test_is_smiles_valid(smiles, expected):
    assert is_smiles_valid(smiles) is expected


@pytest.mark.parametrize("mol,expected", [(Chem.MolFromSmiles("C"), True), (None, False)], ids=["valid-mol", "none"])
def test_is_mol_valid(mol, expected):
    assert is_mol_valid(mol) is expected


def test_fetch_pubchem_retry_on_429():
    resp_429 = MagicMock(status_code=429)
    resp_200 = MagicMock(status_code=200, text="CCO\n")
    with patch("proto_tools.entities.ligands.utils.requests.get", side_effect=[resp_429, resp_200]):
        with patch("proto_tools.entities.ligands.utils.time.sleep"):
            assert fetch_pubchem_txt("https://example.com") == "CCO"


def test_fetch_pubchem_success_first_attempt():
    resp = MagicMock(status_code=200, text="CCO\n")
    with patch("proto_tools.entities.ligands.utils.requests.get", return_value=resp):
        assert fetch_pubchem_txt("https://example.com") == "CCO"


def test_fetch_pubchem_200_empty_text():
    resp = MagicMock(status_code=200, text="  \n")
    with patch("proto_tools.entities.ligands.utils.requests.get", return_value=resp):
        assert fetch_pubchem_txt("https://example.com") is None


def test_fetch_pubchem_request_exception():
    import requests as req

    with patch("proto_tools.entities.ligands.utils.requests.get", side_effect=req.RequestException("timeout")):
        with patch("proto_tools.entities.ligands.utils.time.sleep"):
            assert fetch_pubchem_txt("https://example.com") is None


def test_fetch_pubchem_all_429_exhausted():
    resp = MagicMock(status_code=429)
    with patch("proto_tools.entities.ligands.utils.requests.get", return_value=resp):
        with patch("proto_tools.entities.ligands.utils.time.sleep"):
            assert fetch_pubchem_txt("https://example.com") is None


def test_lookup_smiles_via_pubchem_not_found():
    with patch("proto_tools.entities.ligands.utils.fetch_pubchem_txt", return_value=None):
        with pytest.raises(ValueError, match="Could not find SMILES"):
            lookup_smiles_via_pubchem("anything")


@pytest.mark.parametrize(
    "side_effects,expected",
    [
        (["12345\n", "aspirin"], "aspirin"),
        (["12345\n", None], "Unknown"),
    ],
    ids=["happy-path", "name-fetch-fails"],
)
def test_lookup_name_via_pubchem(side_effects, expected):
    with patch("proto_tools.entities.ligands.utils.fetch_pubchem_txt", side_effect=side_effects):
        assert lookup_name_via_pubchem("CC(=O)Oc1ccccc1C(=O)O") == expected


@pytest.mark.skip_ci
@pytest.mark.integration
def test_get_smiles_valid_name():
    smiles = lookup_smiles_via_pubchem("Aspirin")
    assert isinstance(smiles, str)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None


@pytest.mark.skip_ci
@pytest.mark.integration
def test_get_smiles_invalid_name():
    with pytest.raises(ValueError, match="Could not find SMILES for"):
        lookup_smiles_via_pubchem("ThisIsNotARealCompound1234")


@pytest.mark.skip_ci
@pytest.mark.integration
def test_get_name_invalid_smiles():
    name = lookup_name_via_pubchem("C1C1C1C1C1")
    assert name == "Unknown"


@pytest.mark.skip_ci
@pytest.mark.integration
def test_get_name_empty_smiles():
    name = lookup_name_via_pubchem("")
    assert name == "Unknown"
