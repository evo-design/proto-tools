"""tests/ligand_tests/test_ligand_utils.py.

Tests for ligand utility functions.
"""

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from proto_tools.entities.ligands.utils import (
    fetch_pubchem_txt,
    get_name_from_smiles,
    get_smiles_from_name,
    is_mol_valid,
    is_smiles_valid,
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


@pytest.mark.integration
def test_get_smiles_valid_name():
    smiles = get_smiles_from_name("Aspirin")
    assert isinstance(smiles, str)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None


@pytest.mark.integration
def test_get_smiles_invalid_name():
    with pytest.raises(ValueError, match="Could not find SMILES for"):
        get_smiles_from_name("ThisIsNotARealCompound1234")


@pytest.mark.integration
def test_get_name_invalid_smiles():
    name = get_name_from_smiles("C1C1C1C1C1")
    assert name == "Unknown"


@pytest.mark.integration
def test_get_name_empty_smiles():
    name = get_name_from_smiles("")
    assert name == "Unknown"
