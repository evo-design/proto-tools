"""tests/ligand_tests/test_ligand.py.

Tests for Ligands entity.
"""

import pytest
from rdkit import Chem

from proto_tools.entities.ligands import Fragment, Ligands
from proto_tools.entities.ligands.ligands import parse_fragments_from_string_or_path
from proto_tools.entities.structures.utils import is_valid_structure
from tests.ligand_tests.ligand_inputs import LIGAND_TEST_FILES

# ── Loading ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_ligands_from_single_fragment_smi():
    smi_path = LIGAND_TEST_FILES["single_fragment"]["smi"]
    ligands = Ligands.from_file(smi_path)
    assert len(ligands) == 1
    frag = ligands[0]
    assert isinstance(frag, Fragment)
    assert frag.mol is not None


@pytest.mark.integration
@pytest.mark.parametrize("sdf_key", ["2d_sdf", "3d_sdf"])
def test_ligands_from_single_fragment_sdf(sdf_key):
    sdf_path = LIGAND_TEST_FILES["single_fragment"][sdf_key]
    ligands = Ligands.from_file(sdf_path)
    assert len(ligands) == 1
    frag = ligands[0]
    assert isinstance(frag, Fragment)
    assert frag.mol is not None


@pytest.mark.integration
def test_ligands_from_multiple_fragment_smi():
    smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
    ligands = Ligands.from_file(smi_path)
    assert len(ligands) > 1
    for frag in ligands:
        assert isinstance(frag, Fragment)
        assert frag.mol is not None


@pytest.mark.integration
@pytest.mark.parametrize("sdf_key", ["2d_sdf", "3d_sdf"])
def test_ligands_from_multiple_fragment_sdf(sdf_key):
    sdf_path = LIGAND_TEST_FILES["multiple_fragment"][sdf_key]
    ligands = Ligands.from_file(sdf_path)
    assert len(ligands) > 1
    for frag in ligands:
        assert isinstance(frag, Fragment)
        assert frag.mol is not None


@pytest.mark.integration
def test_generate_conformers_for_all():
    smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
    ligands = Ligands.from_file(smi_path)
    ligands.generate_conformers(num_conformers=2, prune_rms_threshold=0)
    for frag in ligands:
        assert len(frag.conformers) == 2


@pytest.mark.integration
def test_get_smiles_list_and_names_list():
    smi_path = LIGAND_TEST_FILES["multiple_fragment"]["smi"]
    ligands = Ligands.from_file(smi_path)
    smiles_list = ligands.get_smiles_list()
    assert all(isinstance(s, str) for s in smiles_list)
    assert ".".join(smiles_list) == ligands.smiles


# ── PDB generation ──────────────────────────────────────────────────────


@pytest.mark.integration
def test_to_pdb_single_fragment():
    ligands = Ligands.from_smiles("CCO")
    pdb_string = ligands.to_pdb()

    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0
    assert is_valid_structure(pdb_string)
    assert "HETATM" in pdb_string or "ATOM" in pdb_string
    assert "END" in pdb_string

    lines = pdb_string.split("\n")
    atom_lines = [line for line in lines if line.startswith(("HETATM", "ATOM"))]
    assert len(atom_lines) > 0
    assert all(line[21] == "A" for line in atom_lines)


@pytest.mark.integration
def test_to_pdb_multiple_fragments():
    ligands = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    ligands.generate_conformers(num_conformers=1)
    pdb_string = ligands.to_pdb(spacing=5.0)

    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0
    assert is_valid_structure(pdb_string)

    lines = pdb_string.split("\n")
    atom_lines = [line for line in lines if line.startswith(("HETATM", "ATOM"))]
    chain_ids = {line[21] for line in atom_lines}
    assert len(chain_ids) == 2
    assert "A" in chain_ids
    assert "B" in chain_ids

    ter_lines = [line for line in lines if line.startswith("TER")]
    assert len(ter_lines) == 2

    chain_a_atoms = [line for line in atom_lines if line[21] == "A"]
    chain_b_atoms = [line for line in atom_lines if line[21] == "B"]

    chain_a_x_coords = [float(line[30:38]) for line in chain_a_atoms]
    chain_b_x_coords = [float(line[30:38]) for line in chain_b_atoms]

    assert max(chain_a_x_coords) < min(chain_b_x_coords)


@pytest.mark.integration
def test_to_pdb_write_file(tmp_path):
    ligands = Ligands.from_smiles("CCO")
    pdb_path = tmp_path / "test.pdb"
    pdb_string = ligands.to_pdb(filepath=pdb_path)

    assert pdb_path.exists()
    assert pdb_path.read_text() == pdb_string
    assert is_valid_structure(pdb_path)


@pytest.mark.integration
def test_to_pdb_empty_ligands():
    ligands = Ligands(fragments=[])

    with pytest.raises(ValueError, match="Cannot generate PDB: no fragments"):
        ligands.to_pdb()


@pytest.mark.integration
def test_to_pdb_spacing_parameter():
    """Larger spacing pushes chain B further along the X-axis."""
    ligands = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])

    pdb_small_spacing = ligands.to_pdb(spacing=1.0)
    pdb_large_spacing = ligands.to_pdb(spacing=20.0)

    assert is_valid_structure(pdb_small_spacing)
    assert is_valid_structure(pdb_large_spacing)

    def get_chain_b_min_x(pdb_string):
        lines = pdb_string.split("\n")
        chain_b_atoms = [line for line in lines if (line.startswith(("HETATM", "ATOM"))) and line[21] == "B"]
        return min(float(line[30:38]) for line in chain_b_atoms)

    assert get_chain_b_min_x(pdb_large_spacing) > get_chain_b_min_x(pdb_small_spacing)


def test_ligands_round_trip():
    """Ligands survive model_dump → model_validate."""
    ligands = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    dumped = ligands.model_dump()
    reconstructed = Ligands.model_validate(dumped)
    assert len(reconstructed) == 2
    assert reconstructed[0].smiles == "CCO"
    assert reconstructed[1].smiles == "CO"


def test_fragment_smiles_canonicalization():
    frag = Fragment(smiles="c1ccccc1")
    assert frag.smiles == "c1ccccc1"
    assert frag.mol is not None


def test_to_smi_round_trip(tmp_path):
    ligands = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    smi_path = tmp_path / "test.smi"
    ligands.to_smi(filepath=smi_path)
    reloaded = Ligands.from_file(smi_path)
    assert set(reloaded.get_smiles_list()) == set(ligands.get_smiles_list())


# ── Fragment protocol ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "other,expected",
    [
        (Fragment(smiles="CCO"), True),
        (Fragment(smiles="CO"), False),
        ("not a fragment", NotImplemented),
    ],
    ids=["same", "different", "not-fragment"],
)
def test_fragment_eq(other, expected):
    assert Fragment(smiles="CCO").__eq__(other) is expected


def test_fragment_hash():
    a = Fragment(smiles="CCO")
    b = Fragment(smiles="CCO")
    assert hash(a) == hash(b)
    assert hash(a) == hash("CCO")


def test_fragment_from_mol_multi_fragment_raises():
    mol = Chem.MolFromSmiles("CCO.CO")
    with pytest.raises(ValueError, match=r"must contain exactly one fragment"):
        Fragment.from_mol(mol)


# ── Ligands protocol ────────────────────────────────────────────────────


def test_ligands_from_mols():
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("CO")
    ligands = Ligands.from_mols([mol1, mol2])
    assert len(ligands) == 2
    assert set(ligands.get_smiles_list()) == {"CCO", "CO"}


@pytest.mark.parametrize(
    "other,expected",
    [
        (Ligands(fragments=[Fragment(smiles="CO"), Fragment(smiles="CCO")]), True),
        (Ligands(fragments=[Fragment(smiles="CO")]), False),
        ("not ligands", NotImplemented),
    ],
    ids=["same-reordered", "different", "not-ligands"],
)
def test_ligands_eq(other, expected):
    a = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    assert a.__eq__(other) is expected


def test_ligands_hash():
    a = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    b = Ligands(fragments=[Fragment(smiles="CO"), Fragment(smiles="CCO")])
    assert hash(a) == hash(b)


# ── I/O ──────────────────────────────────────────────────────────────────


def test_to_sdf_round_trip(tmp_path):
    """SDF write+parse roundtrip preserves SMILES across all fragments."""
    ligands = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    sdf_path = tmp_path / "out.sdf"
    ligands.to_sdf(sdf_path)
    reloaded = Ligands.from_file(sdf_path)
    assert set(reloaded.get_smiles_list()) == set(ligands.get_smiles_list())


@pytest.mark.parametrize("ext", [".smi", ".sdf"], ids=["smi", "sdf"])
def test_from_file_not_found(ext):
    with pytest.raises(FileNotFoundError):
        Ligands.from_file(f"nonexistent{ext}")


@pytest.mark.parametrize(
    "input_val,expected_count,error_match",
    [
        ("CCO", 1, None),
        ("not_valid_at_all", None, "Invalid input"),
    ],
    ids=["valid-smiles", "invalid"],
)
def test_parse_string_or_path(input_val, expected_count, error_match):
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            parse_fragments_from_string_or_path(input_val)
    else:
        frags = parse_fragments_from_string_or_path(input_val)
        assert len(frags) == expected_count


def test_ligands_get_names_list():
    ligands = Ligands(fragments=[Fragment(smiles="CCO", name="ethanol"), Fragment(smiles="CO")])
    assert ligands.get_names_list() == ["ethanol", None]
    assert ligands.smiles == "CCO.CO"


def test_fragment_conformers_property():
    frag = Fragment(smiles="CCO")
    assert frag.conformers == []
    frag.generate_conformers(num_conformers=1)
    assert len(frag.conformers) == 1


def test_to_pdb_basic():
    ligands = Ligands.from_smiles("CCO")
    pdb_str = ligands.to_pdb()
    assert "END" in pdb_str
    assert "TER" in pdb_str
    atom_lines = [line for line in pdb_str.split("\n") if line.startswith(("HETATM", "ATOM"))]
    assert len(atom_lines) > 0


def test_to_pdb_two_fragments():
    ligands = Ligands(fragments=[Fragment(smiles="CCO"), Fragment(smiles="CO")])
    pdb_str = ligands.to_pdb()
    lines = pdb_str.split("\n")
    chain_ids = {line[21] for line in lines if line.startswith(("HETATM", "ATOM"))}
    assert chain_ids == {"A", "B"}


def test_to_pdb_empty_raises():
    with pytest.raises(ValueError, match="no fragments"):
        Ligands(fragments=[]).to_pdb()


# ── Constructor shorthands ──────────────────────────────────────────────


def test_ligands_constructor_from_smiles_kwarg():
    """Dot-separated SMILES kwarg expands to N fragments."""
    ligands = Ligands(smiles="CCO.CO")
    assert len(ligands) == 2
    assert {f.smiles for f in ligands} == {"CCO", "CO"}


def test_ligands_constructor_from_ccd_codes_kwarg():
    """List of CCD codes kwarg expands to one fragment per code."""
    ligands = Ligands(ccd_codes=["ATP", "MG", "MG"])
    assert len(ligands) == 3
    assert [f.ccd_code for f in ligands] == ["ATP", "MG", "MG"]


def test_ligands_constructor_combines_kwargs():
    """smiles=, ccd_codes=, and explicit fragments= all combine into one collection."""
    ligands = Ligands(
        fragments=[Fragment(smiles="CO")],
        smiles="CCO",
        ccd_codes=["MG"],
    )
    assert len(ligands) == 3
    smis = {f.smiles for f in ligands}
    assert "CCO" in smis
    assert "CO" in smis
