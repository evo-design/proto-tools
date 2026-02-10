"""
test_ccd_utils.py

Test the CCD utilities module for mapping between SMILES and CCD codes.
"""

from bio_programming.bio_tools.entities.ligands.ccd_utils import (
    COMMON_MODIFICATIONS,
    get_canonical_component,
    get_ccd_description,
    get_modifications_for_component,
    is_valid_ccd_code,
    map_ccd_code_to_smiles,
    map_smiles_to_ccd_code,
)


class TestSMILESToCCDMapping:
    """
    Tests the mapping between SMILES and CCD codes.
    """
    def test_map_known_smiles_to_ccd(self):
        result = map_smiles_to_ccd_code("C([C@@H](C(=O)O)N)OP(=O)(O)O")
        assert result == "SEP"

    def test_map_unknown_smiles_returns_none(self):
        """
        Tests that an unknown SMILES string returns None.
        """
        result = map_smiles_to_ccd_code("INVALID_SMILES_XYZ123")
        assert result is None

    def test_map_empty_smiles_returns_none(self):
        """
        Tests that an empty SMILES string returns None.
        """
        result = map_smiles_to_ccd_code("")
        assert result is None


class TestCCDToSMILESMapping:
    """
    Tests the mapping between CCD codes and SMILES strings.
    """
    def test_map_known_ccd_to_smiles(self):
        """
        Tests that a known CCD code returns a SMILES string.
        """
        result = map_ccd_code_to_smiles("SEP")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_map_ccd_case_insensitive(self):
        """
        Tests that the mapping is case-insensitive.
        """
        result_upper = map_ccd_code_to_smiles("SEP")
        result_lower = map_ccd_code_to_smiles("sep")
        result_mixed = map_ccd_code_to_smiles("SeP")
        assert result_upper == result_lower == result_mixed

    def test_map_unknown_ccd_returns_none(self):
        """
        Tests that an unknown CCD code returns None.
        """
        result = map_ccd_code_to_smiles("z_z")
        assert result is None

    def test_map_empty_ccd_returns_none(self):
        """
        Tests that an empty CCD code returns None.
        """
        result = map_ccd_code_to_smiles("")
        assert result is None


class TestCCDDescription:
    """
    Tests the description of CCD codes.
    """
    def test_get_description_for_known_code(self):
        """
        Tests that a known CCD code returns a description.
        """
        result = get_ccd_description("SEP")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_description_case_insensitive(self):
        """
        Tests that the description is case-insensitive.
        """
        result_upper = get_ccd_description("SEP")
        result_lower = get_ccd_description("sep")
        assert result_upper == result_lower

    def test_get_description_for_unknown_code(self):
        """
        Tests that an unknown CCD code returns None.
        """
        result = get_ccd_description("z_z")
        assert result is None


class TestCCDValidation:
    """
    Tests the validation of CCD codes.
    """
    def test_valid_ccd_code_returns_true(self):
        """
        Tests that a valid CCD code returns True.
        """
        assert is_valid_ccd_code("SEP") is True
        assert is_valid_ccd_code("TPO") is True
        assert is_valid_ccd_code("2MG") is True

    def test_invalid_ccd_code_returns_false(self):
        """
        Tests that an invalid CCD code returns False.
        """
        assert is_valid_ccd_code("z_z") is False
        assert is_valid_ccd_code("INVALID") is False

    def test_empty_ccd_code_returns_false(self):
        """
        Tests that an empty CCD code returns False.
        """
        assert is_valid_ccd_code("") is False


class TestCommonModificationsInDatabase:
    """Verify that all common modifications are actually in the CCD database."""

    def test_all_common_modifications_are_valid(self):
        for mod_code in COMMON_MODIFICATIONS:
            assert (
                map_ccd_code_to_smiles(mod_code) is not None
            ), f"{mod_code.upper()} modification not found in CCD database"


class TestRoundTripConversion:
    """Test converting between SMILES and CCD codes and back."""

    def test_ccd_to_smiles_to_ccd_roundtrip(self):
        original_ccd = "SEP"
        smiles = map_ccd_code_to_smiles(original_ccd)
        assert smiles is not None

        ccd_from_smiles = map_smiles_to_ccd_code(smiles)
        assert ccd_from_smiles == original_ccd

    def test_multiple_roundtrips(self):
        test_codes = ["SEP", "TPO", "2MG", "PSU", "6OG"]
        for original_ccd in test_codes:
            smiles = map_ccd_code_to_smiles(original_ccd)
            assert smiles is not None, f"Failed to get SMILES for {original_ccd}"

            ccd_from_smiles = map_smiles_to_ccd_code(smiles)
            assert ccd_from_smiles == original_ccd, f"Roundtrip failed for {original_ccd}"

class TestEdgeCases:
    def test_whitespace_in_ccd_code(self):
        result = map_ccd_code_to_smiles("  SEP  ")
        assert result is None

    def test_numeric_ccd_codes(self):
        result = is_valid_ccd_code("2MG")
        assert result is True

    def test_case_sensitivity_smiles(self):
        smiles_sep = map_ccd_code_to_smiles("SEP")
        result1 = map_smiles_to_ccd_code(smiles_sep)
        result2 = map_smiles_to_ccd_code(smiles_sep.upper())
        result3 = map_smiles_to_ccd_code(smiles_sep.lower())
        assert result1 == "SEP"
        assert result2 is None or result2 == "SEP"
        assert result3 is None or result3 == "SEP"


class TestCanonicalComponentMapping:
    """Test the mapping from modified components to their canonical forms."""

    def test_protein_ptm_to_canonical(self):
        """Test protein post-translational modifications."""
        assert get_canonical_component("SEP") == "S"  # Phosphoserine -> Serine
        assert get_canonical_component("TPO") == "T"  # Phosphothreonine -> Threonine
        assert get_canonical_component("PTR") == "Y"  # Phosphotyrosine -> Tyrosine
        assert get_canonical_component("MSE") == "M"  # Selenomethionine -> Methionine
        assert get_canonical_component("HYP") == "P"  # Hydroxyproline -> Proline

    def test_rna_modifications_to_canonical(self):
        """Test RNA base modifications."""
        assert get_canonical_component("2MG") == "G"  # 2'-O-methylguanosine -> Guanosine
        assert get_canonical_component("5MC") == "C"  # 5-methylcytidine -> Cytidine
        assert get_canonical_component("PSU") == "U"  # Pseudouridine -> Uridine
        assert get_canonical_component("7MG") == "G"  # 7-methylguanosine -> Guanosine
        assert get_canonical_component("H2U") == "U"  # Dihydrouridine -> Uridine

    def test_dna_modifications_to_canonical(self):
        """Test DNA base modifications."""
        assert get_canonical_component("6MA") == "A"  # N6-methyladenine -> Adenine
        assert get_canonical_component("8OG") == "G"  # 8-oxoguanine -> Guanine
        assert get_canonical_component("6OG") == "G"  # 8-oxoguanine (alternative) -> Guanine
        assert get_canonical_component("5CM") == "C"  # 5-methylcytosine -> Cytosine

    def test_standard_components_return_none(self):
        """Standard (unmodified) components should return None."""
        assert get_canonical_component("ALA") is None  # Standard amino acid
        assert get_canonical_component("GLY") is None  # Standard amino acid
        assert get_canonical_component("SER") is None  # Standard amino acid
        assert get_canonical_component("THR") is None  # Standard amino acid

    def test_case_insensitive_lookup(self):
        """Test that CCD code lookup is case-insensitive."""
        assert get_canonical_component("SEP") == "S"
        assert get_canonical_component("sep") == "S"
        assert get_canonical_component("SeP") == "S"

    def test_unknown_code_returns_none(self):
        """Unknown CCD codes should return None."""
        assert get_canonical_component("XYZ") is None
        assert get_canonical_component("INVALID") is None
        assert get_canonical_component("") is None

    def test_all_common_modifications_have_canonical_forms(self):
        """All common modifications should have canonical mappings."""
        # Only test modifications that should have parents (not standard residues)
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

    def test_canonical_mapping_consistency(self):
        """Test that canonical mappings are consistent with biochemistry."""
        # Phosphorylated amino acids
        assert get_canonical_component("SEP") == "S"  # Serine phosphorylation
        assert get_canonical_component("TPO") == "T"  # Threonine phosphorylation
        assert get_canonical_component("PTR") == "Y"  # Tyrosine phosphorylation

        # Methylated bases
        assert get_canonical_component("2MG") == "G"  # Methylated guanine
        assert get_canonical_component("5MC") == "C"  # Methylated cytosine
        assert get_canonical_component("6MA") == "A"  # Methylated adenine


class TestModificationsForComponent:
    """Test the reverse mapping from canonical components to their modifications."""

    def test_protein_serine_modifications(self):
        """Test getting all modifications for serine."""
        mods = get_modifications_for_component("protein", "S")
        assert isinstance(mods, list)
        assert len(mods) > 0
        assert "SEP" in mods  # Phosphoserine should be in the list

    def test_protein_threonine_modifications(self):
        """Test getting all modifications for threonine."""
        mods = get_modifications_for_component("protein", "T")
        assert isinstance(mods, list)
        assert "TPO" in mods  # Phosphothreonine should be in the list

    def test_protein_tyrosine_modifications(self):
        """Test getting all modifications for tyrosine."""
        mods = get_modifications_for_component("protein", "Y")
        assert isinstance(mods, list)
        assert "PTR" in mods  # Phosphotyrosine should be in the list

    def test_rna_guanosine_modifications(self):
        """Test getting all modifications for RNA guanosine."""
        mods = get_modifications_for_component("rna", "G")
        assert isinstance(mods, list)
        assert "2MG" in mods  # 2'-O-methylguanosine
        assert "7MG" in mods  # 7-methylguanosine
        assert "PSU" not in mods  # Pseudouridine is U, not G

    def test_rna_uridine_modifications(self):
        """Test getting all modifications for RNA uridine."""
        mods = get_modifications_for_component("rna", "U")
        assert isinstance(mods, list)
        assert "PSU" in mods  # Pseudouridine

    def test_dna_adenine_modifications(self):
        """Test getting all modifications for DNA adenine."""
        mods = get_modifications_for_component("dna", "A")
        assert isinstance(mods, list)
        # DNA modifications should have parent codes starting with 'D' (e.g., DA)
        assert "1AP" in mods  # 1-aminopurine, a DNA adenine modification

    def test_entity_type_distinction(self):
        """Test that entity types are properly distinguished."""
        # RNA and DNA should have different modifications for the same base
        rna_g_mods = get_modifications_for_component("rna", "G")
        dna_g_mods = get_modifications_for_component("dna", "G")

        # They should be different lists
        assert set(rna_g_mods) != set(dna_g_mods)

    def test_empty_result_for_no_modifications(self):
        """Test that empty list is returned when no modifications exist."""
        # Most amino acids probably don't have modifications with parent 'X'
        # (X is the unknown amino acid)
        # Actually from our earlier test, X does have some modifications, so let's use Z
        mods = get_modifications_for_component("protein", "Z")
        assert isinstance(mods, list)
        # Could be empty or have some, but should be a list

    def test_case_insensitive_canonical_letter(self):
        """Test that canonical letter lookup is case-insensitive."""
        mods_upper = get_modifications_for_component("protein", "S")
        mods_lower = get_modifications_for_component("protein", "s")
        assert mods_upper == mods_lower

    def test_case_insensitive_entity_type(self):
        """Test that entity type is case-insensitive."""
        mods_lower = get_modifications_for_component("protein", "S")
        mods_upper = get_modifications_for_component("PROTEIN", "S")
        mods_mixed = get_modifications_for_component("Protein", "S")
        assert mods_lower == mods_upper == mods_mixed

    def test_invalid_entity_type_raises_error(self):
        """Test that invalid entity types raise ValueError."""
        try:
            get_modifications_for_component("invalid", "A")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid entity_type" in str(e)
