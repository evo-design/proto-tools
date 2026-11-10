"""proto_tools/tools/inverse_folding/esm_if1/_readable.py.

Which residues ESM-IF1 actually reads from a staged structure.

Mirrors fair-esm 2.0.0 as ``standalone/setup.sh`` patches it, together with biotite, which
``standalone/requirements.txt`` leaves unpinned. Two gates decide it, and they run in order:

1. ``load_structure`` keeps only atoms named ``N``, ``CA``, ``C`` or ``O``
   (``esm/inverse_folding/util.py:41``). ``standalone/setup.sh`` supplies that line, because
   biotite >= 1.0 has no ``filter_backbone``, and it keeps only the atom-name half of what
   that function tested. A residue carrying none of those four names loses every atom and so
   ceases to exist. GFP's chromophore is named ``N1``/``CA1``/``C1``..., never a bare
   ``N``/``CA``/``C``/``O``, and goes here.
2. ``standalone/inference.py`` then keeps only residues whose name
   ``biotite.sequence.ProteinSequence.convert_letter_3to1`` maps to one letter: the 20
   standard names plus ``ASX``, ``GLX``, ``XLE``, ``UNK``, ``SEC`` and ``MSE``. The ``HETATM``
   flag plays no part, unlike ProteinMPNN, so ``MSE`` written as ``HETATM`` reads as
   methionine while ``SEP`` written as ``ATOM`` is dropped. Waters pass gate 1 on their lone
   ``O`` and are removed here, which matters because staging numbers them into the polymer's
   own chain.

Surviving both gates is all a residue needs to hold a position. A missing ``N``, ``CA`` or
``C`` leaves it in place with NaN coordinates and the model still emits a token for it
(``esm/inverse_folding/util.py:94-100``).

Keep this in step with the pins: a bump to fair-esm or biotite should be checked against both
gates, and gate 1 against ``setup.sh`` as well.
"""

from typing import Any

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.shared_data_models import readable_positions_where

#: esm/inverse_folding/util.py:41, the atom-name mask setup.sh patches in.
ESM_IF1_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})

#: biotite/sequence/seqtypes.py:543-575 at biotite 2.0.0, which inverts the one-letter table
#: and then adds SEC and MSE. The table's " * " key is not a residue name, so it is left out.
BIOTITE_THREE_TO_ONE = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "ASX", "CYS", "GLN", "GLU", "GLX", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SEC", "SER", "THR", "TRP", "TYR", "UNK",
        "VAL", "XLE",
    }
)  # fmt: skip


def esm_if1_readable_positions(structure: Structure) -> dict[str, list[int]]:
    """Positions ESM-IF1 will read, 1-indexed, per chain.

    Args:
        structure (Structure): The structure that will be staged for the model.

    Returns:
        dict[str, list[int]]: Ascending readable positions per polymer chain ID.
    """

    def is_read(residue: Any) -> bool:
        if ESM_IF1_BACKBONE_ATOMS.isdisjoint(residue.atom_names):
            return False  # gate 1: every atom is masked out, so the residue disappears
        return residue.name in BIOTITE_THREE_TO_ONE  # gate 2: biotite has no code for it

    return readable_positions_where(structure, is_read)
