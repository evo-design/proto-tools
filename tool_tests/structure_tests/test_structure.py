"""
test_structure.py

Tests for structure tools.
"""
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from bio_programming.bio_tools.entities.structures import BFactorType, Structure

TEST_PDB_FILE = Path(__file__).parent.parent.parent / "dummy_data" / "renin_af3.pdb"
TEST_CIF_FILE = Path(__file__).parent.parent.parent / "dummy_data" / "renin.cif"


@pytest.fixture(scope="module")
def test_pdb_file_content() -> str:
    with open(TEST_PDB_FILE, "r") as f:
        return f.read()


@pytest.fixture(scope="module")
def test_cif_file_content() -> str:
    with open(TEST_CIF_FILE, "r") as f:
        return f.read()


@pytest.fixture
def protein_from_pdb_file():
    """Create Structure from PDB file path."""
    return Structure(TEST_PDB_FILE)


@pytest.fixture
def protein_from_cif_file():
    """Create Structure from CIF file path."""
    return Structure(TEST_CIF_FILE)


@pytest.fixture
def protein_from_pdb_content(test_pdb_file_content):
    """Create Structure from PDB content string."""
    return Structure(test_pdb_file_content)


@pytest.fixture
def protein_from_cif_content(test_cif_file_content):
    """Create Structure from CIF content string."""
    return Structure(test_cif_file_content)


class TestStructureInitialization:
    """Test Structure initialization with various inputs."""

    def test_init_from_pdb_filepath(self, protein_from_pdb_file):
        """Test initialization from PDB file path."""
        assert protein_from_pdb_file is not None
        assert protein_from_pdb_file.structure_format == "pdb"

    def test_init_from_cif_filepath(self, protein_from_cif_file):
        """Test initialization from CIF file path."""
        assert protein_from_cif_file is not None
        assert protein_from_cif_file.structure_format == "cif"

    def test_init_from_pdb_content_string(self, protein_from_pdb_content):
        """Test initialization from PDB content string."""
        assert protein_from_pdb_content is not None
        assert protein_from_pdb_content.structure_format == "pdb"

    def test_init_from_cif_content_string(self, protein_from_cif_content):
        """Test initialization from CIF content string."""
        assert protein_from_cif_content is not None
        assert protein_from_cif_content.structure_format == "cif"

    def test_init_with_invalid_structure(self):
        """Test initialization with invalid structure content raises error."""
        with pytest.raises(ValueError, match="Structure content is invalid"):
            Structure("invalid structure content")

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Structure(Path("/nonexistent/file.pdb"))


class TestStructureFormatConversion:
    """Test format conversion between PDB and CIF."""

    def test_pdb_file_to_cif_property(self, protein_from_pdb_file):
        """Test converting PDB file to CIF via property."""
        cif_content = protein_from_pdb_file.structure_cif
        assert cif_content is not None
        assert isinstance(cif_content, str)
        assert len(cif_content) > 0
        # CIF should contain data_ block
        assert "data_" in cif_content or "_atom_site" in cif_content

    def test_cif_file_to_pdb_property(self, protein_from_cif_file):
        """Test converting CIF file to PDB via property."""
        pdb_content = protein_from_cif_file.structure_pdb
        assert pdb_content is not None
        assert isinstance(pdb_content, str)
        assert len(pdb_content) > 0
        # PDB should contain ATOM records
        assert "ATOM" in pdb_content

    def test_pdb_content_to_cif_property(self, protein_from_pdb_content):
        """Test converting PDB content to CIF via property."""
        cif_content = protein_from_pdb_content.structure_cif
        assert cif_content is not None
        assert len(cif_content) > 0

    def test_cif_content_to_pdb_property(self, protein_from_cif_content):
        """Test converting CIF content to PDB via property."""
        pdb_content = protein_from_cif_content.structure_pdb
        assert pdb_content is not None
        assert len(pdb_content) > 0

    def test_pdb_to_pdb_returns_original(self, protein_from_pdb_file):
        """Test that PDB to PDB returns original structure."""
        pdb_content = protein_from_pdb_file.structure_pdb
        assert pdb_content == protein_from_pdb_file.structure

    def test_cif_to_cif_returns_original(self, protein_from_cif_file):
        """Test that CIF to CIF returns original structure."""
        cif_content = protein_from_cif_file.structure_cif
        assert cif_content == protein_from_cif_file.structure


class TestStructureGemmiIntegration:
    """Test integration with gemmi Structure objects."""

    def test_gemmi_struct_lazy_loading(self, protein_from_pdb_file):
        """Test that gemmi structure is lazily loaded."""
        # Before access, should be None
        assert protein_from_pdb_file._gemmi_struct is None
        # After access, should be loaded
        gemmi_struct = protein_from_pdb_file.gemmi_struct
        assert gemmi_struct is not None
        assert protein_from_pdb_file._gemmi_struct is not None

    def test_gemmi_struct_from_pdb(self, protein_from_pdb_file):
        """Test gemmi structure loading from PDB."""
        gemmi_struct = protein_from_pdb_file.gemmi_struct
        assert len(gemmi_struct) > 0  # Has at least one model

    def test_gemmi_struct_from_cif(self, protein_from_cif_file):
        """Test gemmi structure loading from CIF."""
        gemmi_struct = protein_from_cif_file.gemmi_struct
        assert len(gemmi_struct) > 0  # Has at least one model

    def test_gemmi_struct_cached(self, protein_from_pdb_file):
        """Test that gemmi structure is cached after first access."""
        struct1 = protein_from_pdb_file.gemmi_struct
        struct2 = protein_from_pdb_file.gemmi_struct
        # Should be the exact same object (cached)
        assert struct1 is struct2


class TestStructureFileIO:
    """Test file I/O operations."""

    def test_write_cif(self, protein_from_pdb_file):
        """Test writing structure to CIF file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            protein_from_pdb_file.write_cif(tmp_path)
            assert tmp_path.exists()

            # Verify content is valid CIF
            content = tmp_path.read_text()
            assert len(content) > 0
            assert "data_" in content or "_atom_site" in content

            # Should be able to load it back
            protein_reloaded = Structure(tmp_path)
            assert protein_reloaded is not None
        finally:
            tmp_path.unlink()

    def test_write_pdb(self, protein_from_cif_file):
        """Test writing structure to PDB file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            protein_from_cif_file.write_pdb(tmp_path)
            assert tmp_path.exists()

            # Verify content is valid PDB
            content = tmp_path.read_text()
            assert len(content) > 0
            assert "ATOM" in content

            # Should be able to load it back
            protein_reloaded = Structure(tmp_path)
            assert protein_reloaded is not None
        finally:
            tmp_path.unlink()

    def test_write_cif_with_string_path(self, protein_from_pdb_file):
        """Test writing CIF with string path instead of Path object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            protein_from_pdb_file.write_cif(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink()

    def test_write_pdb_with_string_path(self, protein_from_cif_file):
        """Test writing PDB with string path instead of Path object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            protein_from_cif_file.write_pdb(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink()


class TestStructureSequenceExtraction:
    """Test sequence extraction methods."""

    def test_get_chain_sequences(self, protein_from_pdb_file):
        """Test extracting all chain sequences."""
        sequences = protein_from_pdb_file.get_chain_sequences()
        assert isinstance(sequences, dict)
        assert len(sequences) > 0
        # All values should be non-empty strings
        for chain_id, sequence in sequences.items():
            assert isinstance(chain_id, str)
            assert isinstance(sequence, str)
            assert len(sequence) > 0

    def test_get_chain_ids(self, protein_from_pdb_file):
        """Test extracting chain IDs."""
        chain_ids = protein_from_pdb_file.get_chain_ids()
        assert isinstance(chain_ids, list)
        assert len(chain_ids) > 0
        # Should match keys from get_chain_sequences
        sequences = protein_from_pdb_file.get_chain_sequences()
        assert set(chain_ids) == set(sequences.keys())

    def test_get_chain_sequence_first_chain(self, protein_from_pdb_file):
        """Test extracting sequence of first chain (no chain_id specified)."""
        sequence = protein_from_pdb_file.get_chain_sequence()
        assert isinstance(sequence, str)
        assert len(sequence) > 0
        # Should match first chain from get_chain_sequences
        sequences = protein_from_pdb_file.get_chain_sequences()
        first_sequence = next(iter(sequences.values()))
        assert sequence == first_sequence

    def test_get_chain_sequence_specific_chain(self, protein_from_pdb_file):
        """Test extracting sequence of specific chain."""
        chain_ids = protein_from_pdb_file.get_chain_ids()
        if len(chain_ids) > 0:
            first_chain_id = chain_ids[0]
            sequence = protein_from_pdb_file.get_chain_sequence(first_chain_id)
            assert isinstance(sequence, str)
            assert len(sequence) > 0

    def test_get_chain_sequence_invalid_chain(self, protein_from_pdb_file):
        """Test that requesting invalid chain raises error."""
        with pytest.raises(ValueError, match="Chain .* not found"):
            protein_from_pdb_file.get_chain_sequence("INVALID_CHAIN_XYZ")

    def test_sequences_from_converted_formats(self, protein_from_pdb_file):
        """Test that sequences are preserved through format conversion."""
        # Get original sequences from PDB
        original_sequences = protein_from_pdb_file.get_chain_sequences()

        # Convert to CIF and back, create new Structure
        cif_content = protein_from_pdb_file.structure_cif
        protein_from_converted_cif = Structure(cif_content)

        converted_sequences = protein_from_converted_cif.get_chain_sequences()

        # Sequences should be preserved
        assert (
            original_sequences == converted_sequences
        ), f"Original sequences: {original_sequences}, Converted sequences: {converted_sequences}"


class TestStructurePydanticSerialization:
    """Test Pydantic serialization and deserialization."""

    def test_serialize_to_dict(self, protein_from_pdb_file):
        """Test serialization to dictionary."""
        serialized = protein_from_pdb_file._serialize_to_dict()
        assert isinstance(serialized, dict)
        assert "structure" in serialized
        assert "structure_format" in serialized
        assert "b_factor_type" in serialized
        assert serialized["structure_format"] == "pdb"
        assert serialized["b_factor_type"] == "UNSPECIFIED"

    def test_validate_from_dict(self, protein_from_pdb_file):
        """Test deserialization from dictionary."""
        serialized = protein_from_pdb_file._serialize_to_dict()
        reconstructed = Structure._validate_from_dict(serialized)

        assert isinstance(reconstructed, Structure)
        assert reconstructed.structure_format == "pdb"
        assert reconstructed.b_factor_type == protein_from_pdb_file.b_factor_type

    def test_pydantic_model_integration(self, protein_from_pdb_file):
        """Test that Structure works in Pydantic models."""

        class TestModel(BaseModel):
            structure: Structure

        # Should be able to create model with Structure
        model = TestModel(structure=protein_from_pdb_file)
        assert model.structure is not None

    def test_pydantic_serialization_round_trip(self, protein_from_pdb_file):
        """Test full serialization/deserialization round trip with Pydantic."""

        class TestModel(BaseModel):
            structure: Structure

        # Create model
        original_model = TestModel(structure=protein_from_pdb_file)

        # Serialize to dict
        model_dict = original_model.model_dump()

        # Deserialize back
        reconstructed_model = TestModel.model_validate(model_dict)

        # Verify structure is intact
        assert reconstructed_model.structure is not None
        assert reconstructed_model.structure.b_factor_type == protein_from_pdb_file.b_factor_type

        # Sequences should be the same
        original_sequences = protein_from_pdb_file.get_chain_sequences()
        reconstructed_sequences = reconstructed_model.structure.get_chain_sequences()
        assert original_sequences == reconstructed_sequences

    def test_validate_from_dict_with_b_factor_type(self):
        """Test deserialization preserves B-factor type."""
        protein = Structure(TEST_PDB_FILE, b_factor_type=BFactorType.PLDDT)
        serialized = protein._serialize_to_dict()
        reconstructed = Structure._validate_from_dict(serialized)

        assert reconstructed.b_factor_type == BFactorType.PLDDT

    def test_validate_from_dict_missing_structure(self):
        """Test deserialization fails with missing structure."""
        with pytest.raises(ValueError, match="Missing 'structure'"):
            Structure._validate_from_dict(
                {"b_factor_type": "UNSPECIFIED", "structure_format": "pdb"}
            )

    def test_validate_from_dict_missing_structure_format(self):
        """Test deserialization fails with missing structure_format."""
        ps = Structure._validate_from_dict(
            {"structure": "ATOM  1", "b_factor_type": "UNSPECIFIED"}
        )
        assert ps.structure_format == "pdb"

    def test_validate_from_dict_with_protein_structure_instance(self, protein_from_pdb_file):
        """Test that passing existing Structure returns it unchanged."""
        result = Structure._validate_from_dict(protein_from_pdb_file)
        assert result is protein_from_pdb_file

    def test_visualize(self, protein_from_pdb_file):
        """Ensure visualize method does not fail"""
        _ = protein_from_pdb_file.visualize(show_legend=False)
