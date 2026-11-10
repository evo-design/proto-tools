"""proto_tools/tools/inverse_folding/proteinmpnn/_readable.py.

Which residues ProteinMPNN actually reads from a staged structure.

Mirrors ColabDesign v1.1.3, the pin in ``standalone/requirements.txt``. Two gates decide it,
and both are easy to get wrong from the outside:

1. ``pdb_to_string`` (``colabdesign/shared/protein.py:52-58``) rewrites a ``HETATM`` line to
   ``ATOM`` when its residue name is in the ``MODRES`` table below, giving it the parent
   standard residue's name. A ``HETATM`` residue whose name is *not* in the table never
   reaches the parser at all. So selenomethionine is read as methionine, while GFP's
   chromophore is skipped.
2. ``prep_pdb`` runs with ``ignore_missing=True`` (``colabdesign/mpnn/model.py:56-57``), which
   keeps only residues whose backbone ``N`` is present --
   ``all_atom_mask[:,0] == 1`` at ``colabdesign/af/prep.py:437-440``, where
   ``atom_types[0] == "N"``. That drops ordinary ``ATOM`` residues with an incomplete
   backbone, and it is why the chromophore would be skipped even after gate 1: its atoms are
   named ``N1``/``CA1``/``C1``, not ``N``/``CA``/``C``.

Keep this in step with the pin: a bump to ColabDesign should be checked against both gates.
"""

from typing import Any

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.shared_data_models import readable_positions_where

# colabdesign/shared/protein.py:8-22 at v1.1.3; staged PDBs carry no MODRES records to extend it.
COLABDESIGN_MODRES = frozenset(
    {
        "4BF", "AGM", "ALY", "B3A", "B3D", "B3E", "B3K", "B3Y", "BAL", "CME", "CSD", "CSO",
        "CSX", "DAL", "DAS", "DBZ", "DCY", "DLY", "DPR", "DSN", "DTH", "DVA", "FME", "GL3",
        "GPL", "HIC", "HSK", "HY3", "HYP", "HZP", "IAS", "KCX", "KYN", "LLP", "M3L", "MGN",
        "MHS", "MLY", "MLZ", "MSE", "NLE", "OCS", "PCA", "PHI", "PTR", "SAC", "SEC", "SEP",
        "SMC", "TPO", "TRQ", "TYS", "YCM",
    }
)  # fmt: skip


def proteinmpnn_readable_positions(structure: Structure) -> dict[str, list[int]]:
    """Positions ProteinMPNN will read, 1-indexed, per chain.

    Args:
        structure (Structure): The structure that will be staged for the model.

    Returns:
        dict[str, list[int]]: Ascending readable positions per polymer chain ID.
    """

    def is_read(residue: Any) -> bool:
        if residue.is_hetatm and residue.name not in COLABDESIGN_MODRES:
            return False  # gate 1: never rewritten to ATOM, so the parser never sees it
        return "N" in residue.atom_names  # gate 2: ignore_missing drops a missing backbone N

    return readable_positions_where(structure, is_read)
