# tests/test_ligands.py
import tempfile
from pathlib import Path

import pytest

from bio_programming_tools.entities.ligands import Fragment, Ligands
from bio_programming_tools.entities.structures.utils import is_valid_structure
from tests.ligand_tests.ligand_inputs import LIGAND_TEST_FILES


@pytest.mark.skip_ci
class TestLigands:
    """Tests for Ligands functionality."""

    def test_ligands_from_single_fragment_smi(self):
        smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
        ligands = Ligands(smi_path)
        assert len(ligands) == 1
        frag = ligands[0]
        assert isinstance(frag, Fragment)
        assert frag.mol is not None

    @pytest.mark.parametrize("sdf_key", ["2d_sdf", "3d_sdf"])
    def test_ligands_from_single_fragment_sdf(self, sdf_key):
        sdf_path = LIGAND_TEST_FILES["single_fragment"][sdf_key]
        ligands = Ligands(sdf_path)
        assert len(ligands) == 1
        frag = ligands[0]
        assert isinstance(frag, Fragment)
        assert frag.mol is not None

    def test_ligands_from_multiple_fragment_smi(self):
        smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
        ligands = Ligands(smi_path)
        assert len(ligands) > 1
        for frag in ligands:
            assert isinstance(frag, Fragment)
            assert frag.mol is not None

    @pytest.mark.parametrize("sdf_key", ["2d_sdf", "3d_sdf"])
    def test_ligands_from_multiple_fragment_sdf(self, sdf_key):
        sdf_path = LIGAND_TEST_FILES["multiple_fragment"][sdf_key]
        ligands = Ligands(sdf_path)
        assert len(ligands) > 1
        for frag in ligands:
            assert isinstance(frag, Fragment)
            assert frag.mol is not None

    def test_generate_conformers_for_all(self):
        smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
        ligands = Ligands(smi_path)
        ligands.generate_conformers(num_conformers=2, prune_rms_threshold=0)
        for frag in ligands:
            assert len(frag.conformers) == 2

    def test_get_smiles_list_and_names_list(self):
        smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
        ligands = Ligands(smi_path)
        smiles_list = ligands.get_smiles_list()
        names_list = ligands.get_names_list()
        assert all(isinstance(s, str) for s in smiles_list)
        assert all(isinstance(n, str) for n in names_list)
        assert ".".join(smiles_list) == ligands.smiles

    def test_to_pdb_single_fragment(self):
        """Test PDB generation for a single fragment."""
        # Create a simple ligand (ethanol)
        ligands = Ligands("CCO")

        # Generate PDB string
        pdb_string = ligands.to_pdb()

        # Verify PDB string is generated
        assert isinstance(pdb_string, str)
        assert len(pdb_string) > 0

        # Verify PDB is valid using the helper function
        assert is_valid_structure(pdb_string), "Generated PDB string is not valid"

        # Verify PDB format
        assert "HETATM" in pdb_string or "ATOM" in pdb_string
        assert "END" in pdb_string

        # Verify chain ID is assigned
        lines = pdb_string.split('\n')
        atom_lines = [line for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
        assert len(atom_lines) > 0
        # Check that chain ID is 'A' (position 21 in PDB format)
        assert all(line[21] == 'A' for line in atom_lines)

    def test_to_pdb_multiple_fragments(self):
        """Test PDB generation for multiple fragments with non-overlapping positions."""
        # Create ligands with multiple fragments (ethanol and methanol)
        ligands = Ligands(["CCO", "CO"])

        # Generate conformers
        ligands.generate_conformers(num_conformers=1)

        # Generate PDB string
        pdb_string = ligands.to_pdb(spacing=5.0)

        # Verify PDB string is generated
        assert isinstance(pdb_string, str)
        assert len(pdb_string) > 0

        # Verify PDB is valid using the helper function
        assert is_valid_structure(pdb_string), "Generated PDB string is not valid"

        # Verify multiple chains
        lines = pdb_string.split('\n')
        atom_lines = [line for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
        chain_ids = set(line[21] for line in atom_lines)
        assert len(chain_ids) == 2  # Two fragments should have two different chain IDs
        assert 'A' in chain_ids
        assert 'B' in chain_ids

        # Verify TER records are present
        ter_lines = [line for line in lines if line.startswith('TER')]
        assert len(ter_lines) == 2  # One TER per fragment

        # Verify no overlap by checking X-coordinates are separated
        chain_a_atoms = [line for line in atom_lines if line[21] == 'A']
        chain_b_atoms = [line for line in atom_lines if line[21] == 'B']

        # Extract X coordinates (columns 31-38 in PDB format)
        chain_a_x_coords = [float(line[30:38]) for line in chain_a_atoms]
        chain_b_x_coords = [float(line[30:38]) for line in chain_b_atoms]

        # Check that max X of chain A < min X of chain B (ensuring separation)
        assert max(chain_a_x_coords) < min(chain_b_x_coords)

    def test_to_pdb_write_file(self):
        """Test writing PDB to file."""
        ligands = Ligands("CCO")

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_string = ligands.to_pdb(filepath=pdb_path)

            # Verify file was created
            assert pdb_path.exists()

            # Verify file content matches returned string
            with open(pdb_path, 'r') as f:
                file_content = f.read()
            assert file_content == pdb_string

            # Verify the written file is a valid PDB structure
            assert is_valid_structure(pdb_path), "Written PDB file is not valid"

    def test_to_pdb_empty_ligands(self):
        """Test that empty Ligands raises an error."""
        # Create an empty Ligands object - need to use a valid input first then clear
        ligands = Ligands("CCO")
        ligands.fragments = []  # Clear fragments

        with pytest.raises(ValueError, match="Cannot generate PDB: no fragments"):
            ligands.to_pdb()

    def test_to_pdb_spacing_parameter(self):
        """Test that spacing parameter affects fragment separation."""
        ligands = Ligands(["CCO", "CO"])

        # Generate with different spacing values
        pdb_small_spacing = ligands.to_pdb(spacing=1.0)
        pdb_large_spacing = ligands.to_pdb(spacing=20.0)

        # Verify both PDB strings are valid
        assert is_valid_structure(pdb_small_spacing), "PDB with small spacing is not valid"
        assert is_valid_structure(pdb_large_spacing), "PDB with large spacing is not valid"

        # Extract X coordinates for chain B in both cases
        def get_chain_b_min_x(pdb_string):
            lines = pdb_string.split('\n')
            chain_b_atoms = [line for line in lines if (line.startswith('HETATM') or line.startswith('ATOM')) and line[21] == 'B']
            x_coords = [float(line[30:38]) for line in chain_b_atoms]
            return min(x_coords)

        min_x_small = get_chain_b_min_x(pdb_small_spacing)
        min_x_large = get_chain_b_min_x(pdb_large_spacing)

        # With larger spacing, chain B should start further along X-axis
        assert min_x_large > min_x_small
