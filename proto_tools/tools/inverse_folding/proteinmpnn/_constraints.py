"""proto_tools/tools/inverse_folding/proteinmpnn/_constraints.py.

Resolving which residues ProteinMPNN is allowed to redesign.

A residue is redesignable unless the caller fixed it or the parser cannot read it. A CCD
modification does not fix anything on its own; it only decides which annotations survive
reconstruction and how a warning is worded.

Resolved in one place so the model, the output and the warnings cannot disagree.
"""

import logging
from dataclasses import dataclass
from typing import Any

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


def parser_excluded_positions(structure: Structure, chain_ids: list[str]) -> dict[str, list[int]]:
    """Positions ProteinMPNN's structure parser cannot process, per chain, ascending.

    Pure detection: it reads the structure and reports, without warning or altering anything,
    so sampling and scoring can each decide what the result means.

    Args:
        structure (Structure): The structure to read.
        chain_ids (list[str]): Chains to report on.

    Returns:
        dict[str, list[int]]: Skipped positions per chain; chains with none are omitted.
    """
    readable = proteinmpnn_readable_positions(structure)
    residues = structure.get_chain_residue_atoms()
    skipped = {}
    for chain_id in chain_ids:
        processed = set(readable.get(chain_id, ()))
        total = len(residues.get(chain_id, ()))
        missing = [p for p in range(1, total + 1) if p not in processed]
        if missing:
            skipped[chain_id] = missing
    return skipped


def describe_positions(structure: Structure, chain_id: str, positions: list[int]) -> str:
    """Render positions as ``64 (CRO)`` for a diagnostic."""
    residues = structure.get_chain_residue_atoms(chain_id)[chain_id]
    return ", ".join(f"{p} ({residues[p - 1].name})" for p in positions)


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
    skipped = parser_excluded_positions(structure, redesign_ids)
    requested = (user_fixed.chains if user_fixed is not None else {}) or {}

    readable: dict[str, list[int]] = {}
    worker: dict[str, list[int]] = {}
    effective: dict[str, list[int]] = {}

    for chain_id in redesign_ids:
        chain_readable = set(readable_all.get(chain_id, ()))
        requested_fixed = set(requested.get(chain_id, ()))
        modified = {m.position for m in structure.get_chain(chain_id).modifications}
        excluded = set(skipped.get(chain_id, ()))

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
        return describe_positions(structure, chain_id, positions)

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
    skipped = parser_excluded_positions(structure, protomer_ids)

    def names(chain_id: str) -> str:
        return describe_positions(structure, chain_id, skipped.get(chain_id, [])) or "none"

    reference_id = protomer_ids[0]
    reference = readable.get(reference_id, [])
    for chain_id in protomer_ids[1:]:
        if readable.get(chain_id, []) == reference:
            continue
        raise ValueError(
            f"{tool}: Cannot tie chains {reference_id} and {chain_id} because ProteinMPNN "
            f"processes different residue positions in each chain. Chain {reference_id} excludes "
            f"{names(reference_id)}, while chain {chain_id} excludes "
            f"{names(chain_id)}. Tied design requires matching residue "
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


def restrict_sequence_to_readable(
    structure: Structure,
    chain_ids: list[str],
    sequence: str,
    *,
    tool: str,
    allow_excluded: bool = True,
) -> str:
    """Drop the characters at positions ProteinMPNN's parser excludes.

    A scored sequence covers every observed residue, but the model only parses the residues in
    its own mapping, so the two lengths disagree whenever a chain holds a modified residue or
    one missing a backbone N. Trimming keeps the sequence aligned with what is scored.

    Args:
        structure (Structure): The structure being scored against.
        chain_ids (list[str]): Chains in the order the sequence concatenates them.
        sequence (str): One character per observed residue across ``chain_ids``.
        tool (str): Tool key named in the warning.
        allow_excluded (bool): When false, excluded positions are an error rather than a
            warning. Callers asking for per-position logits need them, since logits are
            declared as one row per sequence position and cannot describe a trimmed sequence.

    Returns:
        str: The sequence restricted to the positions the model processes.

    Raises:
        ValueError: If ``sequence`` does not cover the chains' observed residues, or if a
            position is excluded and ``allow_excluded`` is false.
    """
    readable = proteinmpnn_readable_positions(structure)
    residues = structure.get_chain_residue_atoms()
    expected = sum(len(residues.get(chain_id, ())) for chain_id in chain_ids)
    if len(sequence) != expected:
        raise ValueError(
            f"{tool}: the sequence has {len(sequence)} residues but chains {chain_ids} hold "
            f"{expected}. Each chain's residues must appear in order."
        )

    dropped = parser_excluded_positions(structure, chain_ids)
    kept: list[str] = []
    offset = 0
    for chain_id in chain_ids:
        processed = set(readable.get(chain_id, ()))
        total = len(residues.get(chain_id, ()))
        kept.extend(sequence[offset + p - 1] for p in range(1, total + 1) if p in processed)
        offset += total

    for chain_id, excluded in dropped.items():
        named = describe_positions(structure, chain_id, excluded)
        if not allow_excluded:
            raise ValueError(
                f"{tool}: Chain {chain_id}, position {named} is excluded by the model's structure "
                "parser, so per-position logits cannot be aligned to the input sequence. Score "
                "without return_logits, or remove the excluded residues from the structure."
            )
        logger.warning(
            "%s: Chain %s, position %s is excluded by the model's structure parser and is not included in the score.",
            tool,
            chain_id,
            named,
        )
    return "".join(kept)


def worker_excluded_positions(
    structure: Structure, selection: Any, chain_ids: list[str]
) -> dict[str, list[int]] | None:
    """User scoring exclusions, minus the ones the parser already drops, as residue numbers.

    A position the parser never read has no counterpart in the model's input, so sending it
    would fail to resolve there. Filtering happens in ordinal space before the selection is
    translated, so the remaining positions still name the residues the caller meant.

    Args:
        structure (Structure): The structure being scored against.
        selection (Any): The caller's ``excluded_positions``, or ``None``.
        chain_ids (list[str]): Chains in scoring order.

    Returns:
        dict[str, list[int]] | None: Per-chain residue numbers for the worker, or ``None``.
    """
    if selection is None:
        return None
    skipped = parser_excluded_positions(structure, chain_ids)
    kept = {
        chain_id: [p for p in positions if p not in set(skipped.get(chain_id, ()))]
        for chain_id, positions in selection.chains.items()
    }
    kept = {chain_id: positions for chain_id, positions in kept.items() if positions}
    if not kept:
        return None
    translated: dict[str, list[int]] = type(selection)(chains=kept).to_residue_numbers(structure)
    return translated
