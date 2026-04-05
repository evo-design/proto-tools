"""tests/ligand_tests/test_fragments.py.

Tests for Fragment entity.
"""

import pytest
from rdkit import Chem

from proto_tools.entities.ligands import Fragment
from tests.ligand_tests.ligand_inputs import LIGAND_TEST_FILES


@pytest.mark.integration
def test_fragment_from_valid_smiles():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles=smiles)
    assert frag.mol is not None
    assert frag.smiles == Chem.MolToSmiles(Chem.RemoveHs(frag.mol), canonical=True)


def test_fragment_from_invalid_smiles():
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        Fragment(smiles="INVALIDSMILES")


@pytest.mark.integration
def test_fragment_from_mol_object():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    frag = Fragment.from_mol(mol)
    assert frag.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True) == frag.smiles


@pytest.mark.integration
def test_generate_conformers():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles=smiles)
    frag.generate_conformers(num_conformers=2)
    assert len(frag.conformers) == 2


def test_round_trip():
    frag = Fragment(smiles="CCO", name="ethanol")
    reconstructed = Fragment.model_validate(frag.model_dump())
    assert reconstructed.smiles == frag.smiles
    assert reconstructed.name == frag.name
    assert reconstructed.mol is not None
