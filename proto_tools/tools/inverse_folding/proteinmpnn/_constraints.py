"""proto_tools/tools/inverse_folding/proteinmpnn/_constraints.py.

Resolving which residues ProteinMPNN is allowed to redesign.

Three things decide it, and they are resolved together here so the model, the output and the
warnings cannot disagree:

- the residues the parser processes (``_readable``),
- the residues carrying a CCD modification, which are fixed by default rather than redesigned,
- the residues the caller fixed explicitly.
"""

import logging
from dataclasses import dataclass

from proto_tools.entities.structures import ResidueSelection, Structure
from proto_tools.tools.inverse_folding.proteinmpnn._readable import proteinmpnn_readable_positions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProteinMPNNConstraints:
    """Resolved per-structure constraints, shared by dispatch and reconstruction.

    Attributes:
        readable (dict[str, list[int]]): Positions the parser processes, per redesigned chain.
        worker_fixed (ResidueSelection | None): Fixed positions the model can resolve. Excluded
            positions are absent: they have no counterpart in the parsed structure.
        effective_fixed (dict[str, list[int]]): Positions that will not be redesigned, per chain,
            whether the caller asked or the modification policy did.
    """

    readable: dict[str, list[int]]
    worker_fixed: ResidueSelection | None
    effective_fixed: dict[str, list[int]]


def resolve_constraints(
    structure: Structure,
    redesign_ids: list[str],
    user_fixed: ResidueSelection | None,
    *,
    tool: str,
) -> ProteinMPNNConstraints:
    """Resolve the fixed positions for one input structure, warning about automatic ones.

    A modified residue is kept fixed by default: the model either cannot read it, or reads it
    as its parent amino acid and would otherwise design it away. A position the caller fixed
    explicitly is honoured silently, since that is what they asked for.

    Args:
        structure (Structure): The input structure.
        redesign_ids (list[str]): Chains selected for redesign.
        user_fixed (ResidueSelection | None): Positions the caller fixed.
        tool (str): Tool key named in warnings.

    Returns:
        ProteinMPNNConstraints: The mapping and the constraints derived from it.
    """
    readable_all = proteinmpnn_readable_positions(structure)
    residues = structure.get_chain_residue_atoms()
    requested = (user_fixed.chains if user_fixed is not None else {}) or {}

    readable: dict[str, list[int]] = {}
    worker: dict[str, list[int]] = {}
    effective: dict[str, list[int]] = {}

    for chain_id in redesign_ids:
        chain_readable = set(readable_all.get(chain_id, ()))
        all_positions = set(range(1, len(residues.get(chain_id, ())) + 1))
        requested_fixed = set(requested.get(chain_id, ()))
        modified = {m.position for m in structure.get_chain(chain_id).modifications}

        effective_fixed = requested_fixed | modified
        worker_fixed = effective_fixed & chain_readable
        excluded = all_positions - chain_readable
        automatic = (modified | excluded) - requested_fixed

        readable[chain_id] = sorted(chain_readable)
        if worker_fixed:
            worker[chain_id] = sorted(worker_fixed)
        effective[chain_id] = sorted(effective_fixed | excluded)
        _warn_automatic(tool, structure, chain_id, sorted(automatic), modified, excluded)

    return ProteinMPNNConstraints(
        readable=readable,
        worker_fixed=ResidueSelection(chains=worker) if worker else None,
        effective_fixed=effective,
    )


def _warn_automatic(
    tool: str,
    structure: Structure,
    chain_id: str,
    automatic: list[int],
    modified: set[int],
    excluded: set[int],
) -> None:
    """Report positions kept fixed without being asked, grouped by why, one line per group."""
    if not automatic:
        return
    residues = structure.get_chain_residue_atoms(chain_id)[chain_id]

    def names(positions: list[int]) -> str:
        return ", ".join(f"{p} ({residues[p - 1].name})" for p in positions)

    unreadable_modified = [p for p in automatic if p in modified and p in excluded]
    readable_modified = [p for p in automatic if p in modified and p not in excluded]
    unreadable_standard = [p for p in automatic if p not in modified]

    if unreadable_modified:
        logger.warning(
            "%s: Chain %s, position %s is a modified residue that ProteinMPNN cannot redesign. "
            "It will be kept fixed in the output.",
            tool,
            chain_id,
            names(unreadable_modified),
        )
    if readable_modified:
        logger.warning(
            "%s: Chain %s, position %s is a modified residue. It will be kept fixed during design.",
            tool,
            chain_id,
            names(readable_modified),
        )
    if unreadable_standard:
        missing_n = [p for p in unreadable_standard if "N" not in residues[p - 1].atom_names]
        other = [p for p in unreadable_standard if p not in set(missing_n)]
        if missing_n:
            logger.warning(
                "%s: Chain %s, position %s cannot be processed because its backbone nitrogen is "
                "missing. It will be preserved unchanged in the output.",
                tool,
                chain_id,
                names(missing_n),
            )
        if other:
            logger.warning(
                "%s: Chain %s, position %s is excluded by the model's structure parser. It will "
                "be preserved unchanged in the output.",
                tool,
                chain_id,
                names(other),
            )


def require_matching_protomers(
    structure: Structure, protomer_ids: list[str], readable: dict[str, list[int]], *, tool: str
) -> None:
    """Reject tied design when the parser processes different positions in each protomer.

    Equal length and equal numbering are not enough: two protomers can exclude different
    positions, and tying them would associate different structural residues.

    Args:
        structure (Structure): The input structure.
        protomer_ids (list[str]): The chains to be tied.
        readable (dict[str, list[int]]): Positions the parser processes, per chain.
        tool (str): Tool key named in the error.

    Raises:
        ValueError: If two protomers have different readable positions.
    """

    def excluded_of(chain_id: str) -> list[int]:
        total = len(structure.get_chain_residue_atoms(chain_id)[chain_id])
        return sorted(set(range(1, total + 1)) - set(readable.get(chain_id, ())))

    def names(chain_id: str, positions: list[int]) -> str:
        residues = structure.get_chain_residue_atoms(chain_id)[chain_id]
        return ", ".join(f"{p} ({residues[p - 1].name})" for p in positions) or "none"

    reference_id = protomer_ids[0]
    reference = readable.get(reference_id, [])
    for chain_id in protomer_ids[1:]:
        if readable.get(chain_id, []) == reference:
            continue
        raise ValueError(
            f"{tool}: Cannot tie chains {reference_id} and {chain_id} because ProteinMPNN "
            f"processes different residue positions in each chain. Chain {reference_id} excludes "
            f"{names(reference_id, excluded_of(reference_id))}, while chain {chain_id} excludes "
            f"{names(chain_id, excluded_of(chain_id))}. Tied design requires matching residue "
            "mappings across protomers."
        )
