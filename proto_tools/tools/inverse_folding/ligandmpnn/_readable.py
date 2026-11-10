"""proto_tools/tools/inverse_folding/ligandmpnn/_readable.py.

Which residues LigandMPNN designs, on the Foundry ``model_type="ligand_mpnn"`` path.

Mirrors rc-foundry 0.2.0 and atomworks 2.2.1, the pins in ``standalone/requirements.txt``.
The model emits one sequence letter per *non-atomized* token of the processed atom array
(``mpnn/inference_engines/mpnn.py:403-423``). An atomized residue is still read, as ligand
context atoms (``mpnn/pipelines/mpnn.py:144-147``), but it holds no sequence position, so it
is absent from this rule. Three gates decide what is left:

1. ``_parse_from_pdb`` moves every ``HETATM`` atom of a mixed chain into a synthetic chain of
   its own (``atomworks/io/parser.py:808-813``); a chain whose atoms are all ``HETATM`` is
   typed ``NON_POLYMER`` (``atomworks/io/utils/non_rcsb.py:159-163``), and
   ``FlagNonPolymersForAtomization`` atomizes it (``atomworks/ml/transforms/atomize.py:35``).
   Staging writes a PDB, so this is the branch that runs and a ``HETATM`` residue never holds
   a sequence position. Note this gate also decides ``MSE``: ``convert_mse_to_met=True``
   renames it, but at ``atomworks/io/parser.py:609-611``, after the chain split has already
   set ``is_polymer=False``. The rename therefore only saves an ``MSE`` written as ``ATOM``.
2. ``AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=STANDARD_AA + (UNKNOWN_AA,))``
   (``mpnn/pipelines/mpnn.py:119-124``) atomizes every remaining residue whose name is not one
   of the 20 standard CCD amino acids or ``UNK``. That catches a modified residue written as
   ``ATOM``, and every nucleotide.
3. ``MaskResiduesWithSpecificUnresolvedAtoms`` zeroes a protein residue's occupancy when it is
   missing any of ``N``, ``CA``, ``C``, ``O`` (``mpnn/pipelines/mpnn.py:126-136``) and
   ``RemoveUnresolvedTokens`` drops it (``atomworks/ml/transforms/filters.py:46-70``). The
   parser pads each residue out to its CCD template with ``occupancy=0.0`` placeholder atoms
   (``atomworks/io/template.py:366-373``), so a missing backbone atom drops the residue
   outright rather than leaving a gap.

Unlike ProteinMPNN there is no MODRES rewrite: a modified residue is read as ligand context,
never as its parent amino acid.

Two upstream behaviours cannot be derived from residue identity, so this rule does not model
them. ``FlagAndReassignCovalentModifications`` (``mpnn/pipelines/mpnn.py:115``) atomizes a
standard residue covalently bonded to a non-polymer unit, which depends on the bond graph;
and a residue whose backbone atom is present but written with zero occupancy is dropped
upstream. Both make the rule over-count, which surfaces as a length mismatch in
``DesignedComplex.from_design`` rather than a design placed on the wrong residues.

Keep this in step with the pins: a bump to rc-foundry or atomworks should be checked against
all three gates.
"""

from typing import Any

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.shared_data_models import readable_positions_where

#: atomworks/constants.py:386-388 (STANDARD_AA) and :347 (UNKNOWN_AA), plus MSE, which the
#: parser rewrites to MET before gate 2 whenever gate 1 has not already excluded it.
LIGANDMPNN_RESIDUE_NAMES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "UNK", "VAL",
    }
)  # fmt: skip

#: mpnn/pipelines/mpnn.py:127-133: MPNN needs the backbone plus the carbonyl oxygen.
LIGANDMPNN_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})

#: Residue names ``ligandmpnn_sample`` stages as ``ATOM`` so gate 1 does not lose them.
#: Only ``MSE`` earns a place: AtomWorks renames it to ``MET`` itself
#: (``convert_mse_to_met`` at ``atomworks/io/parser.py:72``), so once the chain split leaves
#: it in the polymer it clears gate 2 as methionine. Any other modified residue would clear
#: gate 1 only to be atomized by name, which is where it ends up anyway.
LIGANDMPNN_ATOM_RECORD_RESIDUES = frozenset({"MSE"})


def ligandmpnn_readable_positions(structure: Structure) -> dict[str, list[int]]:
    """Positions LigandMPNN designs, 1-indexed, per chain.

    Args:
        structure (Structure): The structure that will be staged for the model.

    Returns:
        dict[str, list[int]]: Ascending readable positions per polymer chain ID.
    """

    def is_read(residue: Any) -> bool:
        if residue.is_hetatm and residue.name not in LIGANDMPNN_ATOM_RECORD_RESIDUES:
            return False  # gate 1: split into a non-polymer chain, so read only as context
        if residue.name not in LIGANDMPNN_RESIDUE_NAMES:
            return False  # gate 2: atomized by name, so read only as context
        # gate 3: an incomplete backbone leaves the residue unresolved
        return LIGANDMPNN_BACKBONE_ATOMS.issubset(residue.atom_names)

    return readable_positions_where(structure, is_read)
