# test_utils.py
import pytest
from rdkit import Chem

from bio_programming_tools.entities.ligands.utils import (
    get_name_from_smiles,
    get_smiles_from_name,
)


@pytest.mark.skip_ci
class TestLigandUtils:
    """Tests for ligand utility functions."""

    # ===============================
    # Tests for get_smiles_from_name
    # ===============================

    def test_get_smiles_valid_name(self):
        smiles = get_smiles_from_name("Aspirin")
        assert isinstance(smiles, str)
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None  # should be a valid molecule

    def test_get_smiles_invalid_name(self):
        with pytest.raises(ValueError):
            get_smiles_from_name("ThisIsNotARealCompound1234")

    # ===============================
    # Tests for get_name_from_smiles
    # ===============================

    def test_get_name_invalid_smiles(self):
        # Random invalid SMILES
        invalid_smiles = "C1C1C1C1C1"
        name = get_name_from_smiles(invalid_smiles)
        assert name == "Unknown"

    def test_get_name_empty_smiles(self):
        name = get_name_from_smiles("")
        assert name == "Unknown"
