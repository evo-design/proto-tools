"""tests/ligand_tests/test_ligand_utils.py

Tests for ligand utility functions."""

import pytest
from rdkit import Chem

from proto_tools.entities.ligands.utils import (
    get_name_from_smiles,
    get_smiles_from_name,
)


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
