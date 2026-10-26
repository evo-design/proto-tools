"""tests/inverse_folding_tests/conftest.py.

Fixtures shared by the inverse-folding tests, chiefly structures whose residue numbering
exercises the staging every one of these models needs.
"""

from pathlib import Path

import pytest

from proto_tools.entities.structures.structure import Structure

STAGING_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"


def with_insertion_code(structure: Structure, after: int, code: str = "A") -> Structure:
    """Relabel residue ``after + 1`` as ``<after><code>`` and shift every later residue down by one.

    The chain keeps its residue count, but residue ``after`` and its insertion-coded neighbour
    now share a number, which only an insertion-code-aware mapping keeps apart.

    Args:
        structure (Structure): The structure to relabel.
        after (int): Residue number whose successor takes the insertion code.
        code (str): The insertion code to apply.

    Returns:
        Structure: The relabelled structure, as PDB.
    """
    lines = []
    for line in structure.structure_pdb.splitlines():
        relabelled = line
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) >= 27:
            number = int(line[22:26])
            if number == after + 1:
                relabelled = line[:22] + f"{after:>4}{code}" + line[27:]
            elif number > after + 1:
                relabelled = line[:22] + f"{number - 1:>4}" + line[26:]
        lines.append(relabelled)
    return Structure(structure="\n".join(lines) + "\n", structure_format="pdb", source="test")


@pytest.fixture(scope="module")
def staging_structure() -> Structure:
    """A plain single-chain backbone, numbered 1..65."""
    return Structure.from_file(STAGING_PDB_FILE)


@pytest.fixture(scope="module", params=["pdb", "cif"])
def insertion_coded_structure(request, staging_structure) -> Structure:
    """The backbone numbered 1..5, 5A, 6..64, as PDB and as CIF: position 6 is residue ``5A``."""
    structure = with_insertion_code(staging_structure, after=5)
    if request.param == "cif":
        return Structure(structure=structure.structure_cif, structure_format="cif", source="test")
    return structure


@pytest.fixture(scope="module")
def two_character_chain_structure(staging_structure) -> Structure:
    """The backbone as a CIF whose only chain is ``AB``, an ID PDB format cannot hold."""
    gemmi_struct = staging_structure.gemmi_struct.clone()
    for chain in gemmi_struct[0]:
        chain.name = "AB"
    return Structure(structure=gemmi_struct.make_mmcif_document().as_string(), structure_format="cif", source="test")
