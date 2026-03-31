"""tests/ligand_tests/test_fragments.py

Tests for Fragment entity."""

import pytest
from rdkit import Chem

from proto_tools.entities.ligands import Fragment
from tests.ligand_tests.ligand_inputs import LIGAND_TEST_FILES


@pytest.mark.integration
def test_fragment_from_valid_smiles():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles)
    assert isinstance(frag, Fragment)
    assert frag.mol is not None
    assert frag.smiles == Chem.MolToSmiles(Chem.RemoveHs(frag.mol), canonical=True)


def test_fragment_from_invalid_smiles():
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        Fragment("INVALIDSMILES")


@pytest.mark.integration
def test_fragment_from_mol_object():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    frag = Fragment(mol)
    assert isinstance(frag, Fragment)
    assert mol.GetNumAtoms() == frag.mol.GetNumAtoms()
    assert mol.GetNumBonds() == frag.mol.GetNumBonds()
    assert mol.GetProp("_Name") == frag.mol.GetProp("_Name")
    assert Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True) == frag.smiles


@pytest.mark.integration
def test_generate_conformers():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles)
    frag.generate_conformers(num_conformers=2)
    assert len(frag.conformers) == 2
    for conf in frag.conformers:
        assert conf is not None


@pytest.mark.integration
def test_fragment_name_assignment():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles)
    assert frag.name is not None
    assert isinstance(frag.name, str)
