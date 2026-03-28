"""tests/ligand_tests/test_ccd_utils.py

Tests for CCD utilities (SMILES / CCD code mapping)."""

import pytest

from bio_programming_tools.entities.ligands.ccd_utils import (
    COMMON_MODIFICATIONS,
    get_canonical_component,
    get_ccd_description,
    get_modifications_for_component,
    is_valid_ccd_code,
    map_ccd_code_to_smiles,
    map_smiles_to_ccd_code,
)


# ── SMILES / CCD mapping ────────────────────────────────────────────────


def test_map_known_smiles_to_ccd():
    result = map_smiles_to_ccd_code("C([C@@H](C(=O)O)N)OP(=O)(O)O")
    assert result == "SEP"


@pytest.mark.integration
def test_map_unknown_smiles_returns_none():
    result = map_smiles_to_ccd_code("INVALID_SMILES_XYZ123")
    assert result is None


def test_map_empty_smiles_returns_none():
    result = map_smiles_to_ccd_code("")
    assert result is None


def test_map_known_ccd_to_smiles():
    result = map_ccd_code_to_smiles("SEP")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_map_ccd_case_insensitive():
    result_upper = map_ccd_code_to_smiles("SEP")
    result_lower = map_ccd_code_to_smiles("sep")
    result_mixed = map_ccd_code_to_smiles("SeP")
    assert result_upper == result_lower == result_mixed


def test_map_unknown_ccd_returns_none():
    result = map_ccd_code_to_smiles("z_z")
    assert result is None


def test_map_empty_ccd_returns_none():
    result = map_ccd_code_to_smiles("")
    assert result is None


# ── CCD descriptions ────────────────────────────────────────────────────


def test_get_description_for_known_code():
    result = get_ccd_description("SEP")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_description_case_insensitive():
    result_upper = get_ccd_description("SEP")
    result_lower = get_ccd_description("sep")
    assert result_upper == result_lower


def test_get_description_for_unknown_code():
    result = get_ccd_description("z_z")
    assert result is None


# ── CCD validation ──────────────────────────────────────────────────────


def test_valid_ccd_code_returns_true():
    assert is_valid_ccd_code("SEP") is True
    assert is_valid_ccd_code("TPO") is True
    assert is_valid_ccd_code("2MG") is True


def test_invalid_ccd_code_returns_false():
    assert is_valid_ccd_code("z_z") is False
    assert is_valid_ccd_code("INVALID") is False


def test_empty_ccd_code_returns_false():
    assert is_valid_ccd_code("") is False


def test_all_common_modifications_are_valid():
    for mod_code in COMMON_MODIFICATIONS:
        assert (
            map_ccd_code_to_smiles(mod_code) is not None
        ), f"{mod_code.upper()} modification not found in CCD database"


# ── Canonical component mapping ─────────────────────────────────────────


def test_protein_ptm_to_canonical():
    assert get_canonical_component("SEP") == "S"  # Phosphoserine -> Serine
    assert get_canonical_component("TPO") == "T"  # Phosphothreonine -> Threonine
    assert get_canonical_component("PTR") == "Y"  # Phosphotyrosine -> Tyrosine
    assert get_canonical_component("MSE") == "M"  # Selenomethionine -> Methionine
    assert get_canonical_component("HYP") == "P"  # Hydroxyproline -> Proline


def test_rna_modifications_to_canonical():
    assert get_canonical_component("2MG") == "G"
    assert get_canonical_component("5MC") == "C"
    assert get_canonical_component("PSU") == "U"
    assert get_canonical_component("7MG") == "G"
    assert get_canonical_component("H2U") == "U"


def test_dna_modifications_to_canonical():
    assert get_canonical_component("6MA") == "A"
    assert get_canonical_component("8OG") == "G"
    assert get_canonical_component("6OG") == "G"
    assert get_canonical_component("5CM") == "C"


def test_standard_components_return_none():
    """Standard (unmodified) components should return None."""
    assert get_canonical_component("ALA") is None
    assert get_canonical_component("GLY") is None
    assert get_canonical_component("SER") is None
    assert get_canonical_component("THR") is None


def test_canonical_case_insensitive_lookup():
    assert get_canonical_component("SEP") == "S"
    assert get_canonical_component("sep") == "S"
    assert get_canonical_component("SeP") == "S"


def test_canonical_unknown_code_returns_none():
    assert get_canonical_component("XYZ") is None
    assert get_canonical_component("INVALID") is None
    assert get_canonical_component("") is None


def test_all_common_modifications_have_canonical_forms():
    modifications_with_parents = {
        "SEP", "TPO", "PTR", "HYP", "HY3", "MLY", "MLZ", "M3L",
        "ALY", "MEA", "CSD", "CSO", "2MG", "5MC", "PSU", "7MG",
        "H2U", "M2G", "OMC", "OMG", "1MA", "6OG", "6MA", "5CM", "8OG"
    }

    for mod_code in COMMON_MODIFICATIONS:
        if mod_code in modifications_with_parents:
            result = get_canonical_component(mod_code)
            assert result is not None, f"{mod_code} should have a canonical form"
            assert len(result) == 1, f"{mod_code} canonical form should be single letter"


# ── Modifications for component ─────────────────────────────────────────


def test_protein_serine_modifications():
    mods = get_modifications_for_component("protein", "S")
    assert isinstance(mods, list)
    assert len(mods) > 0
    assert "SEP" in mods


def test_protein_threonine_modifications():
    mods = get_modifications_for_component("protein", "T")
    assert isinstance(mods, list)
    assert "TPO" in mods


def test_protein_tyrosine_modifications():
    mods = get_modifications_for_component("protein", "Y")
    assert isinstance(mods, list)
    assert "PTR" in mods


def test_rna_guanosine_modifications():
    mods = get_modifications_for_component("rna", "G")
    assert isinstance(mods, list)
    assert "2MG" in mods
    assert "7MG" in mods
    assert "PSU" not in mods  # Pseudouridine is U, not G


def test_rna_uridine_modifications():
    mods = get_modifications_for_component("rna", "U")
    assert isinstance(mods, list)
    assert "PSU" in mods


def test_dna_adenine_modifications():
    mods = get_modifications_for_component("dna", "A")
    assert isinstance(mods, list)
    assert "1AP" in mods


def test_entity_type_distinction():
    """RNA and DNA should have different modifications for the same base."""
    rna_g_mods = get_modifications_for_component("rna", "G")
    dna_g_mods = get_modifications_for_component("dna", "G")
    assert set(rna_g_mods) != set(dna_g_mods)


def test_empty_result_for_no_modifications():
    mods = get_modifications_for_component("protein", "Z")
    assert isinstance(mods, list)


def test_case_insensitive_canonical_letter():
    mods_upper = get_modifications_for_component("protein", "S")
    mods_lower = get_modifications_for_component("protein", "s")
    assert mods_upper == mods_lower


def test_case_insensitive_entity_type():
    mods_lower = get_modifications_for_component("protein", "S")
    mods_upper = get_modifications_for_component("PROTEIN", "S")
    mods_mixed = get_modifications_for_component("Protein", "S")
    assert mods_lower == mods_upper == mods_mixed


def test_invalid_entity_type_raises_error():
    with pytest.raises(ValueError, match="Invalid entity_type"):
        get_modifications_for_component("invalid", "A")


# ── Round-trip and edge cases ────────────────────────────────────────────


def test_multiple_roundtrips():
    test_codes = ["SEP", "TPO", "2MG", "PSU", "6OG"]
    for original_ccd in test_codes:
        smiles = map_ccd_code_to_smiles(original_ccd)
        assert smiles is not None, f"Failed to get SMILES for {original_ccd}"

        ccd_from_smiles = map_smiles_to_ccd_code(smiles)
        assert ccd_from_smiles == original_ccd, f"Roundtrip failed for {original_ccd}"


def test_whitespace_in_ccd_code():
    result = map_ccd_code_to_smiles("  SEP  ")
    assert result is None


@pytest.mark.integration
def test_case_sensitivity_smiles():
    smiles_sep = map_ccd_code_to_smiles("SEP")
    result1 = map_smiles_to_ccd_code(smiles_sep)
    result2 = map_smiles_to_ccd_code(smiles_sep.upper())
    result3 = map_smiles_to_ccd_code(smiles_sep.lower())
    assert result1 == "SEP"
    assert result2 is None or result2 == "SEP"
    assert result3 is None or result3 == "SEP"
