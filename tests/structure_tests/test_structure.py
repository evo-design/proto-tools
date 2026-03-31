"""tests/structure_tests/test_structure.py

Tests for the Structure entity."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from proto_tools.entities.structures import BFactorType, Structure

_TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
_TEST_CIF_FILE = Path(__file__).parent.parent / "dummy_data" / "renin.cif"


@pytest.fixture(scope="module")
def test_pdb_file_content() -> str:
    with open(_TEST_PDB_FILE, "r") as f:
        return f.read()


@pytest.fixture(scope="module")
def test_cif_file_content() -> str:
    with open(_TEST_CIF_FILE, "r") as f:
        return f.read()


@pytest.fixture
def protein_from_pdb_file():
    """Create Structure from PDB file path."""
    return Structure(_TEST_PDB_FILE)


@pytest.fixture
def protein_from_cif_file():
    """Create Structure from CIF file path."""
    return Structure(_TEST_CIF_FILE)


@pytest.fixture
def protein_from_pdb_content(test_pdb_file_content):
    """Create Structure from PDB content string."""
    return Structure(test_pdb_file_content)


@pytest.fixture
def protein_from_cif_content(test_cif_file_content):
    """Create Structure from CIF content string."""
    return Structure(test_cif_file_content)


# ── Initialization ────────────────────────────────────────────────────────────

def test_init_from_pdb_filepath(protein_from_pdb_file):
    assert protein_from_pdb_file is not None
    assert protein_from_pdb_file.structure_format == "pdb"


def test_init_from_cif_filepath(protein_from_cif_file):
    assert protein_from_cif_file is not None
    assert protein_from_cif_file.structure_format == "cif"


def test_init_from_pdb_content_string(protein_from_pdb_content):
    assert protein_from_pdb_content is not None
    assert protein_from_pdb_content.structure_format == "pdb"


def test_init_from_cif_content_string(protein_from_cif_content):
    assert protein_from_cif_content is not None
    assert protein_from_cif_content.structure_format == "cif"


def test_init_with_invalid_structure():
    with pytest.raises(ValueError, match="Structure content is invalid"):
        Structure("invalid structure content")


def test_init_with_nonexistent_file():
    with pytest.raises(FileNotFoundError, match="File not found"):
        Structure(Path("/nonexistent/file.pdb"))


# ── Format conversion ────────────────────────────────────────────────────────

def test_pdb_file_to_cif_property(protein_from_pdb_file):
    cif_content = protein_from_pdb_file.structure_cif
    assert cif_content is not None
    assert isinstance(cif_content, str)
    assert len(cif_content) > 0
    assert "data_" in cif_content or "_atom_site" in cif_content


def test_cif_file_to_pdb_property(protein_from_cif_file):
    pdb_content = protein_from_cif_file.structure_pdb
    assert pdb_content is not None
    assert isinstance(pdb_content, str)
    assert len(pdb_content) > 0
    assert "ATOM" in pdb_content


def test_pdb_content_to_cif_property(protein_from_pdb_content):
    cif_content = protein_from_pdb_content.structure_cif
    assert cif_content is not None
    assert len(cif_content) > 0


def test_cif_content_to_pdb_property(protein_from_cif_content):
    pdb_content = protein_from_cif_content.structure_pdb
    assert pdb_content is not None
    assert len(pdb_content) > 0


def test_pdb_to_pdb_returns_original(protein_from_pdb_file):
    pdb_content = protein_from_pdb_file.structure_pdb
    assert pdb_content == protein_from_pdb_file.structure


def test_cif_to_cif_returns_original(protein_from_cif_file):
    cif_content = protein_from_cif_file.structure_cif
    assert cif_content == protein_from_cif_file.structure


# ── Gemmi integration ────────────────────────────────────────────────────────

def test_gemmi_struct_lazy_loading(protein_from_pdb_file):
    """Verify gemmi structure is lazily loaded on first access."""
    assert protein_from_pdb_file._gemmi_struct is None
    gemmi_struct = protein_from_pdb_file.gemmi_struct
    assert gemmi_struct is not None
    assert protein_from_pdb_file._gemmi_struct is not None


def test_gemmi_struct_from_pdb(protein_from_pdb_file):
    gemmi_struct = protein_from_pdb_file.gemmi_struct
    assert len(gemmi_struct) > 0


def test_gemmi_struct_from_cif(protein_from_cif_file):
    gemmi_struct = protein_from_cif_file.gemmi_struct
    assert len(gemmi_struct) > 0


def test_gemmi_struct_cached(protein_from_pdb_file):
    """Verify gemmi structure is cached after first access."""
    struct1 = protein_from_pdb_file.gemmi_struct
    struct2 = protein_from_pdb_file.gemmi_struct
    assert struct1 is struct2


# ── File I/O ──────────────────────────────────────────────────────────────────

def test_write_cif(protein_from_pdb_file, tmp_path):
    out = tmp_path / "out.cif"
    protein_from_pdb_file.write_cif(out)
    assert out.exists()

    content = out.read_text()
    assert len(content) > 0
    assert "data_" in content or "_atom_site" in content

    # Round-trip: should be loadable
    Structure(out)


def test_write_pdb(protein_from_cif_file, tmp_path):
    out = tmp_path / "out.pdb"
    protein_from_cif_file.write_pdb(out)
    assert out.exists()

    content = out.read_text()
    assert len(content) > 0
    assert "ATOM" in content

    # Round-trip: should be loadable
    Structure(out)


def test_write_cif_with_string_path(protein_from_pdb_file, tmp_path):
    out = str(tmp_path / "out.cif")
    protein_from_pdb_file.write_cif(out)
    assert Path(out).exists()


def test_write_pdb_with_string_path(protein_from_cif_file, tmp_path):
    out = str(tmp_path / "out.pdb")
    protein_from_cif_file.write_pdb(out)
    assert Path(out).exists()


# ── Sequence extraction ───────────────────────────────────────────────────────

def test_get_chain_sequences(protein_from_pdb_file):
    sequences = protein_from_pdb_file.get_chain_sequences()
    assert isinstance(sequences, dict)
    assert len(sequences) > 0
    for chain_id, sequence in sequences.items():
        assert isinstance(chain_id, str)
        assert isinstance(sequence, str)
        assert len(sequence) > 0


def test_get_chain_ids(protein_from_pdb_file):
    chain_ids = protein_from_pdb_file.get_chain_ids()
    assert isinstance(chain_ids, list)
    assert len(chain_ids) > 0
    sequences = protein_from_pdb_file.get_chain_sequences()
    assert set(chain_ids) == set(sequences.keys())


def test_get_chain_sequence_first_chain(protein_from_pdb_file):
    sequence = protein_from_pdb_file.get_chain_sequence()
    assert isinstance(sequence, str)
    assert len(sequence) > 0
    sequences = protein_from_pdb_file.get_chain_sequences()
    first_sequence = next(iter(sequences.values()))
    assert sequence == first_sequence


def test_get_chain_sequence_specific_chain(protein_from_pdb_file):
    chain_ids = protein_from_pdb_file.get_chain_ids()
    first_chain_id = chain_ids[0]
    sequence = protein_from_pdb_file.get_chain_sequence(first_chain_id)
    assert isinstance(sequence, str)
    assert len(sequence) > 0


def test_get_chain_sequence_invalid_chain(protein_from_pdb_file):
    with pytest.raises(ValueError, match="Chain .* not found"):
        protein_from_pdb_file.get_chain_sequence("INVALID_CHAIN_XYZ")


def test_sequences_preserved_through_format_conversion(protein_from_pdb_file):
    original_sequences = protein_from_pdb_file.get_chain_sequences()
    cif_content = protein_from_pdb_file.structure_cif
    protein_from_converted_cif = Structure(cif_content)
    converted_sequences = protein_from_converted_cif.get_chain_sequences()
    assert original_sequences == converted_sequences


# ── Pydantic serialization ────────────────────────────────────────────────────

def test_serialize_to_dict(protein_from_pdb_file):
    serialized = protein_from_pdb_file._serialize_to_dict()
    assert isinstance(serialized, dict)
    assert "structure" in serialized
    assert "structure_format" in serialized
    assert "b_factor_type" in serialized
    assert serialized["structure_format"] == "pdb"
    assert serialized["b_factor_type"] == "unspecified"


def test_validate_from_dict(protein_from_pdb_file):
    serialized = protein_from_pdb_file._serialize_to_dict()
    reconstructed = Structure._validate_from_dict(serialized)
    assert isinstance(reconstructed, Structure)
    assert reconstructed.structure_format == "pdb"
    assert reconstructed.b_factor_type == protein_from_pdb_file.b_factor_type


def test_pydantic_model_integration(protein_from_pdb_file):
    """Test that Structure works in Pydantic models."""

    class _StructureModel(BaseModel):
        structure: Structure

    model = _StructureModel(structure=protein_from_pdb_file)
    assert model.structure is not None


def test_pydantic_serialization_round_trip(protein_from_pdb_file):
    """Test full serialization/deserialization round trip with Pydantic."""

    class _StructureModel(BaseModel):
        structure: Structure

    original_model = _StructureModel(structure=protein_from_pdb_file)
    model_dict = original_model.model_dump()
    reconstructed_model = _StructureModel.model_validate(model_dict)

    assert reconstructed_model.structure is not None
    assert reconstructed_model.structure.b_factor_type == protein_from_pdb_file.b_factor_type

    original_sequences = protein_from_pdb_file.get_chain_sequences()
    reconstructed_sequences = reconstructed_model.structure.get_chain_sequences()
    assert original_sequences == reconstructed_sequences


def test_validate_from_dict_with_b_factor_type():
    protein = Structure(_TEST_PDB_FILE, b_factor_type=BFactorType.PLDDT)
    serialized = protein._serialize_to_dict()
    reconstructed = Structure._validate_from_dict(serialized)
    assert reconstructed.b_factor_type == BFactorType.PLDDT


def test_validate_from_dict_missing_structure():
    with pytest.raises(ValueError, match="Missing 'structure'"):
        Structure._validate_from_dict(
            {"b_factor_type": "unspecified", "structure_format": "pdb"}
        )


def test_validate_from_dict_missing_structure_format():
    ps = Structure._validate_from_dict(
        {"structure": "ATOM  1", "b_factor_type": "unspecified"}
    )
    assert ps.structure_format == "pdb"


def test_validate_from_dict_with_structure_instance(protein_from_pdb_file):
    """Passing existing Structure returns it unchanged."""
    result = Structure._validate_from_dict(protein_from_pdb_file)
    assert result is protein_from_pdb_file


def test_visualize(protein_from_pdb_file):
    """Ensure visualize method does not fail."""
    _ = protein_from_pdb_file.visualize(show_legend=False)


# ── Metrics serialization ────────────────────────────────────────────────────

def test_metrics_survive_serialize_deserialize_round_trip():
    """Metrics dict should survive _serialize_to_dict → _validate_from_dict."""
    protein = Structure(_TEST_PDB_FILE, metrics={"plddt": 85.2, "ptm": 0.9})
    serialized = protein._serialize_to_dict()
    reconstructed = Structure._validate_from_dict(serialized)
    assert reconstructed.metrics == {"plddt": 85.2, "ptm": 0.9}
