"""tests/ligand_tests/test_ligand.py

Tests for Ligands entity."""

import pytest

from bio_programming_tools.entities.ligands import Fragment, Ligands
from bio_programming_tools.entities.structures.utils import is_valid_structure
from tests.ligand_tests.ligand_inputs import LIGAND_TEST_FILES


# ── Loading ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_ligands_from_single_fragment_smi():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    ligands = Ligands(smi_path)
    assert len(ligands) == 1
    frag = ligands[0]
    assert isinstance(frag, Fragment)
    assert frag.mol is not None


@pytest.mark.integration
@pytest.mark.parametrize("sdf_key", ["2d_sdf", "3d_sdf"])
def test_ligands_from_single_fragment_sdf(sdf_key):
    sdf_path = LIGAND_TEST_FILES["single_fragment"][sdf_key]
    ligands = Ligands(sdf_path)
    assert len(ligands) == 1
    frag = ligands[0]
    assert isinstance(frag, Fragment)
    assert frag.mol is not None


@pytest.mark.integration
def test_ligands_from_multiple_fragment_smi():
    smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
    ligands = Ligands(smi_path)
    assert len(ligands) > 1
    for frag in ligands:
        assert isinstance(frag, Fragment)
        assert frag.mol is not None


@pytest.mark.integration
@pytest.mark.parametrize("sdf_key", ["2d_sdf", "3d_sdf"])
def test_ligands_from_multiple_fragment_sdf(sdf_key):
    sdf_path = LIGAND_TEST_FILES["multiple_fragment"][sdf_key]
    ligands = Ligands(sdf_path)
    assert len(ligands) > 1
    for frag in ligands:
        assert isinstance(frag, Fragment)
        assert frag.mol is not None


@pytest.mark.integration
def test_generate_conformers_for_all():
    smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
    ligands = Ligands(smi_path)
    ligands.generate_conformers(num_conformers=2, prune_rms_threshold=0)
    for frag in ligands:
        assert len(frag.conformers) == 2


@pytest.mark.integration
def test_get_smiles_list_and_names_list():
    smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
    ligands = Ligands(smi_path)
    smiles_list = ligands.get_smiles_list()
    names_list = ligands.get_names_list()
    assert all(isinstance(s, str) for s in smiles_list)
    assert all(isinstance(n, str) for n in names_list)
    assert ".".join(smiles_list) == ligands.smiles


# ── PDB generation ──────────────────────────────────────────────────────


@pytest.mark.integration
def test_to_pdb_single_fragment():
    ligands = Ligands("CCO")
    pdb_string = ligands.to_pdb()

    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0
    assert is_valid_structure(pdb_string)
    assert "HETATM" in pdb_string or "ATOM" in pdb_string
    assert "END" in pdb_string

    lines = pdb_string.split('\n')
    atom_lines = [line for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
    assert len(atom_lines) > 0
    # Chain ID is at position 21 in PDB format
    assert all(line[21] == 'A' for line in atom_lines)


@pytest.mark.integration
def test_to_pdb_multiple_fragments():
    ligands = Ligands(["CCO", "CO"])
    ligands.generate_conformers(num_conformers=1)
    pdb_string = ligands.to_pdb(spacing=5.0)

    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0
    assert is_valid_structure(pdb_string)

    lines = pdb_string.split('\n')
    atom_lines = [line for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
    chain_ids = set(line[21] for line in atom_lines)
    assert len(chain_ids) == 2
    assert 'A' in chain_ids
    assert 'B' in chain_ids

    ter_lines = [line for line in lines if line.startswith('TER')]
    assert len(ter_lines) == 2

    chain_a_atoms = [line for line in atom_lines if line[21] == 'A']
    chain_b_atoms = [line for line in atom_lines if line[21] == 'B']

    # X coordinates at columns 31-38 in PDB format
    chain_a_x_coords = [float(line[30:38]) for line in chain_a_atoms]
    chain_b_x_coords = [float(line[30:38]) for line in chain_b_atoms]

    # Ensure spatial separation between fragments
    assert max(chain_a_x_coords) < min(chain_b_x_coords)


@pytest.mark.integration
def test_to_pdb_write_file(tmp_path):
    ligands = Ligands("CCO")
    pdb_path = tmp_path / "test.pdb"
    pdb_string = ligands.to_pdb(filepath=pdb_path)

    assert pdb_path.exists()
    assert pdb_path.read_text() == pdb_string
    assert is_valid_structure(pdb_path)


@pytest.mark.integration
def test_to_pdb_empty_ligands():
    ligands = Ligands("CCO")
    ligands.fragments = []

    with pytest.raises(ValueError, match="Cannot generate PDB: no fragments"):
        ligands.to_pdb()


@pytest.mark.integration
def test_to_pdb_spacing_parameter():
    """Larger spacing pushes chain B further along the X-axis."""
    ligands = Ligands(["CCO", "CO"])

    pdb_small_spacing = ligands.to_pdb(spacing=1.0)
    pdb_large_spacing = ligands.to_pdb(spacing=20.0)

    assert is_valid_structure(pdb_small_spacing)
    assert is_valid_structure(pdb_large_spacing)

    def get_chain_b_min_x(pdb_string):
        lines = pdb_string.split('\n')
        chain_b_atoms = [line for line in lines if (line.startswith('HETATM') or line.startswith('ATOM')) and line[21] == 'B']
        return min(float(line[30:38]) for line in chain_b_atoms)

    assert get_chain_b_min_x(pdb_large_spacing) > get_chain_b_min_x(pdb_small_spacing)
