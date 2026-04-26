"""tests/ligand_tests/test_fragments.py.

Tests for Fragment entity.
"""

import pytest
from rdkit import Chem

from proto_tools.entities.ligands import Fragment
from tests.ligand_tests.ligand_inputs import LIGAND_TEST_FILES


@pytest.mark.integration
def test_fragment_from_valid_smiles():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles=smiles)
    assert frag.mol is not None
    assert frag.smiles == Chem.MolToSmiles(Chem.RemoveHs(frag.mol), canonical=True)


def test_fragment_from_invalid_smiles():
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        Fragment(smiles="INVALIDSMILES")


@pytest.mark.integration
def test_fragment_from_mol_object():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    frag = Fragment.from_mol(mol)
    assert frag.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True) == frag.smiles


@pytest.mark.integration
def test_generate_conformers():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    with open(smi_path) as f:
        smiles = f.read().strip()
    frag = Fragment(smiles=smiles)
    frag.generate_conformers(num_conformers=2)
    assert len(frag.conformers) == 2


def test_round_trip():
    frag = Fragment(smiles="CCO", name="ethanol")
    reconstructed = Fragment.model_validate(frag.model_dump())
    assert reconstructed.smiles == frag.smiles
    assert reconstructed.name == frag.name
    assert reconstructed.mol is not None


# ── CCD code construction ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "ccd_code",
    ["ATP", "ZN", "HEM", "SEP", "NAD"],
    ids=["ATP", "ZN", "HEM", "SEP", "NAD"],
)
def test_fragment_from_ccd_code(ccd_code):
    """Fragment constructed from CCD code resolves SMILES automatically."""
    frag = Fragment(ccd_code=ccd_code)
    assert frag.ccd_code == ccd_code
    assert frag.smiles is not None
    assert len(frag.smiles) > 0
    assert frag.mol is not None


def test_fragment_from_ccd_code_case_insensitive():
    frag = Fragment(ccd_code="atp")
    assert frag.ccd_code == "ATP"


def test_fragment_from_smiles_resolves_ccd():
    """Fragment constructed from SMILES auto-resolves CCD code when possible."""
    from proto_tools.entities.ligands import map_ccd_code_to_smiles

    zn_smiles = map_ccd_code_to_smiles("ZN")
    frag = Fragment(smiles=zn_smiles)
    assert frag.ccd_code == "ZN"


def test_fragment_from_novel_smiles_has_none_ccd():
    frag = Fragment(smiles="c1ccc(C(=O)NCCNCCN)cc1")
    assert frag.smiles is not None
    assert frag.ccd_code is None


def test_fragment_neither_smiles_nor_ccd_raises():
    with pytest.raises(ValueError, match="At least one of"):
        Fragment()


def test_fragment_invalid_smiles_raises():
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        Fragment(smiles="NOT_A_VALID_SMILES")


def test_fragment_invalid_ccd_raises():
    with pytest.raises(ValueError, match="Invalid CCD code"):
        Fragment(ccd_code="ZZZZZZZ")


def test_fragment_mismatched_smiles_and_ccd_raises():
    """Providing both smiles and ccd_code that refer to different molecules raises."""
    with pytest.raises(ValueError, match="different molecules"):
        Fragment(smiles="CCO", ccd_code="ATP")


def test_fragment_matching_smiles_and_ccd():
    """Providing both smiles and ccd_code that agree resolves correctly."""
    from proto_tools.entities.ligands import map_ccd_code_to_smiles

    sep_smiles = map_ccd_code_to_smiles("SEP")
    frag = Fragment(smiles=sep_smiles, ccd_code="SEP")
    assert frag.ccd_code == "SEP"
    assert frag.smiles == sep_smiles


def test_fragment_ccd_serialization_roundtrip():
    """CCD code survives model_dump → model_validate round-trip."""
    frag = Fragment(ccd_code="ATP")
    dumped = frag.model_dump()
    assert dumped["ccd_code"] == "ATP"
    reconstructed = Fragment.model_validate(dumped)
    assert reconstructed.ccd_code == "ATP"
    assert reconstructed.smiles == frag.smiles


def test_fragment_entity_type_default():
    """Fragment self-identifies as a 'ligand' entity for chain-list integration."""
    assert Fragment(smiles="CCO").entity_type == "ligand"
    assert Fragment(ccd_code="ATP").entity_type == "ligand"


def test_fragment_rejects_multi_fragment_smiles():
    """Multi-component SMILES (dot-separated) raises with a pointer to Ligands."""
    with pytest.raises(ValueError, match="Use `Ligands"):
        Fragment(smiles="CCO.O=C=O")
