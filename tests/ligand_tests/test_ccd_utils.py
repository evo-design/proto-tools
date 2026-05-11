"""tests/ligand_tests/test_ccd_utils.py.

Tests for CCD utilities (SMILES / CCD code mapping).
"""

import pytest

from proto_tools.entities.ligands.ccd_utils import (
    COMMON_MODIFICATIONS,
    get_all_ccd_codes,
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
        "SEP",
        "TPO",
        "PTR",
        "HYP",
        "HY3",
        "MLY",
        "MLZ",
        "M3L",
        "ALY",
        "MEA",
        "CSD",
        "CSO",
        "2MG",
        "5MC",
        "PSU",
        "7MG",
        "H2U",
        "M2G",
        "OMC",
        "OMG",
        "1MA",
        "6OG",
        "6MA",
        "5CM",
        "8OG",
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


def test_map_smiles_novel_returns_none():
    """A SMILES with no canonical or InChIKey match in the CCD returns None."""
    assert map_smiles_to_ccd_code("c1ccc(C(=O)NCCNCCN)cc1") is None


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


def test_inchikey_cache_drops_ambiguous_keys():
    """When multiple CCDs share an InChIKey, none of them are in the cache.

    Validates the safe-by-default collision handling (skip ambiguous matches
    rather than picking arbitrarily).
    """
    from collections import defaultdict
    from pathlib import Path

    from rdkit import Chem, RDLogger

    from proto_tools.entities.ligands.ccd_utils import CCD_DATABASE_PATH, _get_inchikey_cache

    cache = _get_inchikey_cache()

    # Find an InChIKey collision in the bundled CCD file
    RDLogger.DisableLog("rdApp.*")
    inchikey_groups: dict[str, list[str]] = defaultdict(list)
    with open(Path(CCD_DATABASE_PATH)) as f:
        for line in f:
            fields = line.rstrip().split("\t")
            if len(fields) < 2:
                continue
            mol = Chem.MolFromSmiles(fields[0])
            if mol is None:
                continue
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey:
                inchikey_groups[inchikey].append(fields[1])
    RDLogger.EnableLog("rdApp.*")

    ambiguous_keys = [k for k, codes in inchikey_groups.items() if len(codes) > 1]
    assert ambiguous_keys, "Expected at least one InChIKey collision in the CCD"

    # None of the ambiguous keys should appear in the lookup cache.
    leaked = [k for k in ambiguous_keys if k in cache]
    assert not leaked, f"Ambiguous InChIKeys leaked into cache: {leaked[:3]}..."


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
