"""proto_tools/tools/inverse_folding/proteinmpnn/_constraints.py.

Resolving which residues ProteinMPNN is allowed to redesign.

A residue is redesignable unless the caller fixed it or the parser cannot read it. A CCD
modification does not fix anything on its own; it only decides which annotations survive
reconstruction and how a warning is worded.

Resolved in one place so the model, the output and the warnings cannot disagree.
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
        excluded = all_positions - chain_readable

        # A modification does not fix a position; only the caller or the parser does.
        effective_fixed = requested_fixed | excluded
        worker_fixed = requested_fixed & chain_readable

        readable[chain_id] = sorted(chain_readable)
        if worker_fixed:
            worker[chain_id] = sorted(worker_fixed)
        effective[chain_id] = sorted(effective_fixed)
        _warn_excluded(tool, structure, chain_id, sorted(excluded - requested_fixed))
        _warn_fixed_modifications(tool, structure, chain_id, sorted((requested_fixed & chain_readable) & modified))

    return ProteinMPNNConstraints(
        readable=readable,
        worker_fixed=ResidueSelection(chains=worker) if worker else None,
        effective_fixed=effective,
    )


def _warn_excluded(tool: str, structure: Structure, chain_id: str, excluded: list[int]) -> None:
    """Report positions the parser skipped that the caller did not already fix."""
    if not excluded:
        return
    residues = structure.get_chain_residue_atoms(chain_id)[chain_id]
    modified = {m.position for m in structure.get_chain(chain_id).modifications}

    def names(positions: list[int]) -> str:
        return ", ".join(f"{p} ({residues[p - 1].name})" for p in positions)

    groups = (
        ([p for p in excluded if p in modified], "is a modified residue that ProteinMPNN cannot redesign"),
        (
            [p for p in excluded if p not in modified and "N" not in residues[p - 1].atom_names],
            "cannot be processed because its backbone nitrogen is missing",
        ),
        (
            [p for p in excluded if p not in modified and "N" in residues[p - 1].atom_names],
            "is excluded by the model's structure parser",
        ),
    )
    for positions, reason in groups:
        if positions:
            logger.warning(
                "%s: Chain %s, position %s %s. The original residue will be preserved in the output.",
                tool,
                chain_id,
                names(positions),
                reason,
            )


def _warn_fixed_modifications(tool: str, structure: Structure, chain_id: str, positions: list[int]) -> None:
    """Explain that a fixed modified residue reads as its parent amino acid in the sequence."""
    if not positions:
        return
    codes = {m.position: m.modification_code for m in structure.get_chain(chain_id).modifications}
    sequence = structure.get_chain_sequence(chain_id)
    for position in positions:
        logger.warning(
            "%s: Chain %s, position %s (%s) is fixed. The output sequence contains %s at this "
            "position, and the original %s annotation is retained in the output's "
            "chains[].modifications field.",
            tool,
            chain_id,
            position,
            codes[position],
            sequence[position - 1],
            codes[position],
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


def require_compatible_fixed_residues(
    structure: Structure, protomer_ids: list[str], fixed: dict[str, list[int]], *, tool: str
) -> None:
    """Reject tied design when a fixed position holds different residues in different protomers.

    Fixing a position pins it to whatever that chain already has. Tied sampling produces one
    shared sequence, so two protomers demanding different residues at the same position cannot
    both be satisfied. Positions that differ but are not fixed are fine: the shared sequence is
    free to design them.

    Args:
        structure (Structure): The input structure.
        protomer_ids (list[str]): The chains being tied.
        fixed (dict[str, list[int]]): Positions kept unchanged, per chain.
        tool (str): Tool key named in the error.

    Raises:
        ValueError: If a fixed position holds different residues across protomers.
    """
    sequences = {chain_id: structure.get_chain_sequence(chain_id) for chain_id in protomer_ids}
    positions = sorted({p for chain_id in protomer_ids for p in fixed.get(chain_id, ())})
    for position in positions:
        holders = {
            chain_id: sequences[chain_id][position - 1]
            for chain_id in protomer_ids
            if position in set(fixed.get(chain_id, ())) and position <= len(sequences[chain_id])
        }
        if len(set(holders.values())) > 1:
            detail = ", ".join(f"{chain_id} requires {aa}" for chain_id, aa in sorted(holders.items()))
            raise ValueError(
                f"{tool}: Cannot tie these protomers because fixed position {position} requires "
                f"different residues ({detail}). Tied design produces one shared sequence, so a "
                "fixed position must hold the same residue in every protomer."
            )
