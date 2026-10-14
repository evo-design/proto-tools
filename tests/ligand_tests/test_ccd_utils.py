"""tests/ligand_tests/test_ccd_utils.py.

Tests for CCD utilities (SMILES / CCD code mapping).
"""

import pytest

from proto_tools.entities.ligands.ccd_utils import (
    COMMON_MODIFICATIONS,
    count_heavy_atoms_for_ccd,
    get_all_ccd_codes,
    get_canonical_component,
    get_ccd_description,
    get_modifications_for_component,
    is_valid_ccd_code,
    map_ccd_code_to_smiles,
    map_smiles_to_ccd_code,
)

# ── SMILES / CCD mapping ────────────────────────────────────────────────


def test_lookups_are_case_insensitive():
    """Every CCD entry point normalizes case, for the code, the canonical letter, and the entity type."""
    assert map_ccd_code_to_smiles("SEP") == map_ccd_code_to_smiles("sep") == map_ccd_code_to_smiles("SeP")
    assert get_ccd_description("SEP") == get_ccd_description("sep")
    assert get_canonical_component("SEP") == get_canonical_component("sep") == get_canonical_component("SeP") == "S"
    mods = get_modifications_for_component("protein", "S")
    assert get_modifications_for_component("protein", "s") == mods
    assert get_modifications_for_component("PROTEIN", "S") == mods
    assert get_modifications_for_component("Protein", "S") == mods


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
        assert map_ccd_code_to_smiles(mod_code) is not None, (
            f"{mod_code.upper()} modification not found in CCD database"
        )


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


def test_canonical_unknown_code_returns_none():
    assert get_canonical_component("XYZ") is None
    assert get_canonical_component("INVALID") is None
    assert get_canonical_component("") is None


# ── Modifications for component ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("entity_type", "canonical_letter", "expected", "absent"),
    [
        ("protein", "S", ["SEP"], []),
        ("protein", "T", ["TPO"], []),
        ("protein", "Y", ["PTR"], []),
        ("rna", "G", ["2MG", "7MG"], ["PSU"]),  # Pseudouridine is U, not G
        ("rna", "U", ["PSU"], []),
        ("dna", "A", ["1AP"], []),
    ],
)
def test_modifications_for_component(entity_type, canonical_letter, expected, absent):
    """Each canonical component maps to the modifications registered for its entity type."""
    mods = get_modifications_for_component(entity_type, canonical_letter)
    assert isinstance(mods, list)
    assert set(expected) <= set(mods)
    assert not set(absent) & set(mods)


def test_entity_type_distinction():
    """RNA and DNA should have different modifications for the same base."""
    rna_g_mods = get_modifications_for_component("rna", "G")
    dna_g_mods = get_modifications_for_component("dna", "G")
    assert set(rna_g_mods) != set(dna_g_mods)


def test_invalid_entity_type_raises_error():
    with pytest.raises(ValueError, match="Invalid entity_type"):
        get_modifications_for_component("invalid", "A")


# ── Round-trip and edge cases ────────────────────────────────────────────


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


def test_inchikey_cache_contains_known_ccd_entries():
    """InChIKey cache is populated for common biology-relevant ligands.

    Validates the cache build correctly emits InChIKey entries (regression guard
    for the InChIKey 2nd-tier lookup). Asserts a minimum number of probed CCDs
    were actually found — without this counter, the test
    would silently pass if all probes happened to be ambiguity-filtered out.
    """
    from rdkit import Chem

    from proto_tools.entities.ligands.ccd_utils import _get_inchikey_cache

    cache = _get_inchikey_cache()
    # All three probes are common biology ligands with known-unambiguous InChIKeys
    # in the wwPDB CCD (verified at PR time). TYR and HEM are NOT used as probes:
    # TYR shares its InChIKey with at least one other CCD entry; HEM produces an
    # empty InChIKey (RDKit/InChI can't process metal-coordinated macrocycles).
    probes = ["ATP", "FAD", "GTP"]
    found = 0
    for ccd_code in probes:
        smiles = map_ccd_code_to_smiles(ccd_code)
        assert smiles is not None
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))  # type: ignore[no-untyped-call]
        # If this InChIKey is in the cache, it must point to the right CCD.
        if inchikey in cache:
            assert cache[inchikey] == ccd_code, f"InChIKey {inchikey} maps to {cache[inchikey]}, expected {ccd_code}"
            found += 1

    # Guard against a future RDKit/InChI update silently filtering all probes —
    # without this counter the per-iteration assert above would silently pass.
    assert found == len(probes), (
        f"Expected all {len(probes)} probes ({probes}) in InChIKey cache, only found {found}. "
        "If this fails after an RDKit/InChI update, verify whether the probe ligands' "
        "InChIKeys changed and pick replacements known to be unambiguous."
    )


# ── Parameterized canonical round-trip tests ───────────────────────────


@pytest.mark.parametrize(
    "ccd_code",
    ["ATP", "SEP", "TPO", "HEM", "FAD", "ADP", "NAD", "GTP"],
    ids=["ATP", "SEP", "TPO", "HEM", "FAD", "ADP", "NAD", "GTP"],
)
def test_valid_ligand_roundtrip(ccd_code):
    """CCD → RDKit SMILES → CCD round-trip works via canonical comparison."""
    smiles = map_ccd_code_to_smiles(ccd_code)
    assert smiles is not None, f"No SMILES for {ccd_code}"
    result = map_smiles_to_ccd_code(smiles)
    assert result == ccd_code, f"Round-trip failed: {ccd_code} → {smiles} → {result}"


@pytest.mark.parametrize(
    "smiles",
    [
        "INVALID_NOT_SMILES",
        "",
        "X" * 100,
        "C1CC1CC1CC1CC1CC1CC1CC1CC1CC1CC1CC1CC1CC1CC1",
    ],
    ids=["garbage", "empty", "long_garbage", "novel_molecule"],
)
def test_invalid_ligand_no_ccd(smiles):
    """Invalid or novel SMILES return None from CCD lookup."""
    result = map_smiles_to_ccd_code(smiles)
    assert result is None


def test_rdkit_canonical_smiles_matches_ccd():
    """RDKit-canonical SMILES for a known molecule matches its CCD entry."""
    from rdkit import Chem

    # Phosphoserine: OE form differs from RDKit form
    oe_smiles = "C([C@@H](C(=O)O)N)OP(=O)(O)O"
    rdkit_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(oe_smiles), canonical=True)
    assert oe_smiles != rdkit_canonical, "OE and RDKit forms should differ"
    assert map_smiles_to_ccd_code(rdkit_canonical) == "SEP"
    assert map_smiles_to_ccd_code(oe_smiles) == "SEP"


# ── Validity is independent of RDKit parseability ──────────────────────


def test_is_valid_ccd_code_accepts_rdkit_unparseable_entry():
    """A CCD code present in the file but with an RDKit-unparseable SMILES is still valid.

    08T contains a beryllium atom with valence 4 that RDKit rejects, but it is
    a real wwPDB CCD entry. ``is_valid_ccd_code`` should return True regardless.
    """
    assert is_valid_ccd_code("08T") is True
    # The canonical mapping is rightfully None — we can't compute one without RDKit.
    assert map_ccd_code_to_smiles("08T") is None


def test_get_all_ccd_codes_includes_rdkit_unparseable_entries():
    """get_all_ccd_codes returns the full raw set, not just the canonicalizable subset."""
    all_codes = get_all_ccd_codes()
    assert "08T" in all_codes  # RDKit-unparseable but file-present


def test_count_heavy_atoms_falls_back_for_unsanitizable_entries():
    """A real component whose SMILES RDKit will not sanitize is still counted, not refused.

    A1IPL is para-boronophenylalanine: its four-bond boron breaks RDKit's valence rules, so it
    never reaches the canonical cache. Counting atoms needs only the molecular graph, and these
    codes are reachable as residue modifications, so refusing them would fail token counting.
    """
    assert map_ccd_code_to_smiles("A1IPL") is None  # not canonicalizable
    assert count_heavy_atoms_for_ccd("A1IPL") == 16
    assert count_heavy_atoms_for_ccd("A1LZ3") == 28


def test_count_heavy_atoms_still_rejects_codes_absent_from_the_database():
    """The fallback does not turn an unknown code into a silent answer."""
    with pytest.raises(ValueError, match="Unknown CCD code"):
        count_heavy_atoms_for_ccd("ZZZZZ")
