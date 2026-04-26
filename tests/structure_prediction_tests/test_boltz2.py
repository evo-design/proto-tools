"""tests/structure_prediction_tests/test_boltz2.py.

Tests for Boltz2.
"""

from proto_tools.tools.structure_prediction import Chain

# ── Ligand YAML shape: CCD-prefer dispatch (#502) ───────────────────────────


def _boltz2_ligand_entries(chains):
    """Build a Boltz2 YAML payload from ``chains`` and return its ligand entries (parsed)."""
    import yaml

    from proto_tools.tools.structure_prediction.boltz2.helpers import complex_to_yaml

    parsed = yaml.safe_load(complex_to_yaml(chains))
    return [entry["ligand"] for entry in parsed["sequences"] if "ligand" in entry]


def test_boltz2_ligand_uses_ccd_code_when_available():
    """Fragment with a resolved ccd_code serializes to ``ccd: <code>``, not raw SMILES."""
    from proto_tools.entities.ligands import Fragment

    atp = Fragment(ccd_code="ATP")
    assert atp.ccd_code == "ATP"  # invariant guard

    [ligand_entry] = _boltz2_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), atp])
    assert ligand_entry == {"id": "B", "ccd": "ATP"}


def test_boltz2_ligand_falls_back_to_smiles_when_no_ccd_match():
    """Novel ligand (SMILES with no wwPDB CCD entry) serializes as raw SMILES."""
    from proto_tools.entities.ligands import Fragment

    # Synthetic perfluorinated terphenyl chain — not in the wwPDB CCD database.
    novel_smiles = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    novel = Fragment(smiles=novel_smiles)
    assert novel.ccd_code is None  # invariant guard

    [ligand_entry] = _boltz2_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), novel])
    assert ligand_entry == {"id": "B", "smiles": novel.smiles}
