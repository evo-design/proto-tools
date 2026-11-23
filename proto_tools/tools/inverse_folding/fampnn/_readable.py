"""proto_tools/tools/inverse_folding/fampnn/_readable.py.

Which residues FAMPNN actually reads from a staged structure.

Mirrors upstream commit ``aaf788b``, the pin in ``standalone/setup.sh``.
``fampnn.data.protein.read_pdb`` parses the staged PDB with Biopython, and applies two gates.

1. A ``HETATM`` residue is skipped unless its name is in ``ncaa_mapping``
   (``fampnn/data/protein.py:130-134``, table at ``fampnn/data/residue_constants.py:1302``).
   Biopython takes the hetero flag from the record type alone
   (``Bio/PDB/PDBParser.py:189-195``), so selenomethionine and phosphoserine are read, while
   GFP's chromophore is skipped. Only membership is tested: a listed residue still falls
   through to ``restype_3to1.get(resname, "X")``, so even ``MSE`` reads as ``X``, not ``M``.
2. A residue with no atom named in the 37-atom AlphaFold set is skipped: ``np.sum(mask) <
   0.5`` at ``fampnn/data/protein.py:167-169``, over a mask only atoms passing
   ``fampnn/data/protein.py:144-145`` fill. That is what erases a nucleic chain, whose atoms
   are ``P``/``O5'``/``N9``.

Unlike ProteinMPNN, an incomplete backbone keeps its residue: missing atoms populate
``missing_atom_mask`` (``fampnn/data/data.py:46-50``) and the residue stays in the sequence.
An unrecognized residue name is kept as ``X`` (``fampnn/data/protein.py:136-139``).

When updating the pinned FAMPNN commit, check both gates against upstream.
"""

from typing import Any

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.shared_data_models import readable_positions_where

#: fampnn/data/residue_constants.py:1302-1315 at aaf788b. Membership alone decides whether a
#: HETATM residue is read; the letter each name maps to is not consulted by the parser.
FAMPNN_NCAA = frozenset({"MSE", "CME", "CSO", "TPO", "SEP", "PTR", "MLY", "HYP", "PCA", "ACE", "GTS", "GSH"})

#: fampnn/data/residue_constants.py:535-573 at aaf788b, the 37 AlphaFold heavy-atom names.
FAMPNN_ATOM_TYPES = frozenset(
    {
        "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1",
        "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1",
        "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
    }
)  # fmt: skip


def fampnn_readable_positions(structure: Structure) -> dict[str, list[int]]:
    """Positions FAMPNN will read, 1-indexed, per chain.

    Args:
        structure (Structure): The structure that will be staged for the model.

    Returns:
        dict[str, list[int]]: Ascending readable positions per polymer chain ID.
    """

    def is_read(residue: Any) -> bool:
        if residue.is_hetatm and residue.name not in FAMPNN_NCAA:
            return False  # gate 1: a HETATM outside the NCAA table never reaches the features
        return not FAMPNN_ATOM_TYPES.isdisjoint(residue.atom_names)  # gate 2: all-zero mask

    return readable_positions_where(structure, is_read)
