"""tests/structure_tests/test_structure.py.

Tests for the Structure entity.
"""

import warnings
from pathlib import Path

import gemmi
import pytest
from pydantic import BaseModel

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.entities.structures.utils import (
    convert_cif_str_to_pdb_str,
    convert_pdb_str_to_cif_str,
    detect_structure_format,
    load_structure_file,
)

_TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
_TEST_CIF_FILE = Path(__file__).parent.parent / "dummy_data" / "renin.cif"


@pytest.fixture(scope="module")
def test_pdb_file_content() -> str:
    with open(_TEST_PDB_FILE) as f:
        return f.read()


@pytest.fixture(scope="module")
def test_cif_file_content() -> str:
    with open(_TEST_CIF_FILE) as f:
        return f.read()


@pytest.fixture
def protein_from_pdb_file():
    return Structure.from_file(_TEST_PDB_FILE)


@pytest.fixture
def protein_from_cif_file():
    return Structure.from_file(_TEST_CIF_FILE)


# ── Initialization ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source,expected_format",
    [
        ("pdb_file", "pdb"),
        ("cif_file", "cif"),
        ("pdb_content", "pdb"),
        ("cif_content", "cif"),
    ],
)
def test_init_detects_format(source, expected_format, test_pdb_file_content, test_cif_file_content):
    sources = {
        "pdb_file": lambda: Structure.from_file(_TEST_PDB_FILE),
        "cif_file": lambda: Structure.from_file(_TEST_CIF_FILE),
        "pdb_content": lambda: Structure(structure=test_pdb_file_content),
        "cif_content": lambda: Structure(structure=test_cif_file_content),
    }
    s = sources[source]()
    assert s.structure_format == expected_format


def test_init_with_invalid_structure():
    with pytest.raises(ValueError, match="Structure content is invalid"):
        Structure(structure="invalid structure content", structure_format="pdb")


def test_init_with_nonexistent_file():
    with pytest.raises(FileNotFoundError, match="File not found"):
        Structure.from_file(Path("/nonexistent/file.pdb"))


@pytest.mark.parametrize("path_type", [str, Path])
def test_init_accepts_path_in_structure_field(path_type):
    """Structure(structure=<path>) loads the file transparently and sets source."""
    s = Structure(structure=path_type(_TEST_PDB_FILE))
    assert s.structure_format == "pdb"
    assert s.source == str(_TEST_PDB_FILE)
    assert "ATOM" in s.structure


def test_init_path_does_not_override_explicit_source():
    s = Structure(structure=str(_TEST_PDB_FILE), source="custom-source")
    assert s.source == "custom-source"


def test_init_content_string_not_treated_as_path(test_pdb_file_content):
    """Multi-line content strings must not be mistaken for filesystem paths."""
    s = Structure(structure=test_pdb_file_content)
    assert s.source is None
    assert s.structure == test_pdb_file_content


# ── Format conversion ────────────────────────────────────────────────────────


def test_pdb_to_cif_conversion(protein_from_pdb_file):
    cif_content = protein_from_pdb_file.structure_cif
    assert isinstance(cif_content, str)
    assert len(cif_content) > 0
    assert "data_" in cif_content or "_atom_site" in cif_content


def test_cif_to_pdb_conversion(protein_from_cif_file):
    pdb_content = protein_from_cif_file.structure_pdb
    assert isinstance(pdb_content, str)
    assert "ATOM" in pdb_content


def test_cif_to_pdb_no_warnings_for_clean_cif(test_cif_file_content):
    """A normal single-chain CIF (renin) should convert without any lossy-data warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert_cif_str_to_pdb_str(test_cif_file_content)
    lossy_warnings = [w for w in caught if "CIF→PDB conversion" in str(w.message)]
    assert lossy_warnings == []


def test_cif_to_pdb_warns_on_long_chain_id(test_cif_file_content):
    """Multi-character chain IDs cannot fit PDB's 1-char column — must warn."""
    # Rename chain A to AA, re-emit CIF, then convert.
    struct = gemmi.make_structure_from_block(gemmi.cif.read_string(test_cif_file_content)[0])
    for model in struct:
        for chain in model:
            chain.name = chain.name + chain.name  # "A" -> "AA"
            break
        break
    modified_cif = struct.make_mmcif_document().as_string()

    with pytest.warns(UserWarning, match="chain ID"):
        convert_cif_str_to_pdb_str(modified_cif)


@pytest.mark.parametrize("fixture_name", ["protein_from_pdb_file", "protein_from_cif_file"])
def test_same_format_returns_original(fixture_name, request):
    """Accessing the same format as stored returns the original string."""
    protein = request.getfixturevalue(fixture_name)
    if protein.structure_format == "pdb":
        assert protein.structure_pdb == protein.structure
    else:
        assert protein.structure_cif == protein.structure


def test_sequences_preserved_through_format_conversion(protein_from_pdb_file):
    original_sequences = protein_from_pdb_file.get_chain_sequences()
    cif_content = protein_from_pdb_file.structure_cif
    converted = Structure(structure=cif_content)
    assert converted.get_chain_sequences() == original_sequences


# ── Gemmi integration ────────────────────────────────────────────────────────


def test_gemmi_struct_lazy_loading_and_caching(protein_from_pdb_file):
    assert protein_from_pdb_file._gemmi_struct is None
    struct1 = protein_from_pdb_file.gemmi_struct
    assert struct1 is not None
    assert len(struct1) > 0
    assert protein_from_pdb_file.gemmi_struct is struct1


@pytest.mark.parametrize("fixture_name", ["protein_from_pdb_file", "protein_from_cif_file"])
def test_gemmi_struct_parses_both_formats(fixture_name, request):
    protein = request.getfixturevalue(fixture_name)
    assert len(protein.gemmi_struct) > 0


# ── File I/O ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "write_method,suffix,content_check",
    [
        ("write_cif", ".cif", lambda c: "data_" in c or "_atom_site" in c),
        ("write_pdb", ".pdb", lambda c: "ATOM" in c),
    ],
)
def test_write_and_round_trip(protein_from_pdb_file, tmp_path, write_method, suffix, content_check):
    out = tmp_path / f"out{suffix}"
    getattr(protein_from_pdb_file, write_method)(out)
    assert out.exists()
    content = out.read_text()
    assert content_check(content)
    Structure.from_file(out)


# ── Sequence extraction ───────────────────────────────────────────────────────


def test_get_chain_sequences_and_ids(protein_from_pdb_file):
    sequences = protein_from_pdb_file.get_chain_sequences()
    assert len(sequences) > 0
    for chain_id, sequence in sequences.items():
        assert isinstance(chain_id, str)
        assert len(sequence) > 0

    chain_ids = protein_from_pdb_file.get_chain_ids()
    assert set(chain_ids) == set(sequences.keys())

    # Default (no chain_id) returns first chain
    assert protein_from_pdb_file.get_chain_sequence() == next(iter(sequences.values()))

    # Specific chain
    assert len(protein_from_pdb_file.get_chain_sequence(chain_ids[0])) > 0


def test_get_chain_sequence_invalid_chain(protein_from_pdb_file):
    with pytest.raises(ValueError, match=r"Chain .* not found"):
        protein_from_pdb_file.get_chain_sequence("INVALID_CHAIN_XYZ")


# ── Pydantic serialization ────────────────────────────────────────────────────


def test_model_dump_and_validate_round_trip(protein_from_pdb_file):
    dumped = protein_from_pdb_file.model_dump()
    assert dumped["structure_format"] == "pdb"
    assert dumped["b_factor_type"] == "unspecified"

    reconstructed = Structure.model_validate(dumped)
    assert reconstructed.structure_format == "pdb"
    assert reconstructed.b_factor_type == protein_from_pdb_file.b_factor_type


def test_nested_pydantic_model_round_trip(protein_from_pdb_file):
    """Structure survives serialization when nested in another Pydantic model."""

    class _StructureModel(BaseModel):
        structure: Structure

    original = _StructureModel(structure=protein_from_pdb_file)
    reconstructed = _StructureModel.model_validate(original.model_dump())

    assert reconstructed.structure.b_factor_type == protein_from_pdb_file.b_factor_type
    assert reconstructed.structure.get_chain_sequences() == protein_from_pdb_file.get_chain_sequences()


def test_model_validate_with_b_factor_type():
    protein = Structure.from_file(_TEST_PDB_FILE, b_factor_type=BFactorType.PLDDT)
    dumped = protein.model_dump()
    assert Structure.model_validate(dumped).b_factor_type == BFactorType.PLDDT


def test_model_validate_missing_structure():
    with pytest.raises(ValueError):
        Structure.model_validate({"b_factor_type": "unspecified", "structure_format": "pdb"})


def test_model_validate_auto_detects_format(test_pdb_file_content):
    s = Structure.model_validate({"structure": test_pdb_file_content, "b_factor_type": "unspecified"})
    assert s.structure_format == "pdb"


def test_visualize(protein_from_pdb_file):
    _ = protein_from_pdb_file.visualize(show_legend=False)


def test_metrics_survive_round_trip():
    protein = Structure.from_file(_TEST_PDB_FILE, metrics={"plddt": 85.2, "ptm": 0.9})
    reconstructed = Structure.model_validate(protein.model_dump())
    assert reconstructed.metrics == {"plddt": 85.2, "ptm": 0.9}


def test_structure_approx_equal_matching():
    a = Structure.from_file(_TEST_PDB_FILE)
    b = Structure.from_file(_TEST_PDB_FILE)
    a.approx_equal(b)


def test_structure_getattr_metrics():
    protein = Structure.from_file(_TEST_PDB_FILE, metrics={"ptm": 0.9})
    assert protein.ptm == 0.9
    with pytest.raises(AttributeError):
        _ = protein.nonexistent_field


def test_detect_structure_format_empty():
    with pytest.raises(ValueError, match="Empty structure content"):
        detect_structure_format("")


def test_load_structure_file_bad_extension(tmp_path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("content")
    with pytest.raises(ValueError, match="Invalid structure file extension"):
        load_structure_file(bad_file)


def test_convert_empty_strings():
    assert convert_pdb_str_to_cif_str("") == ""
    assert convert_cif_str_to_pdb_str("") == ""
