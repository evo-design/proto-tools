"""Shared data models and input-prep helpers for PyRosetta scoring tools."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from proto_tools.entities.structures import ChainSelection, StructureInputBase

if TYPE_CHECKING:
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import PyRosettaRelaxConfig

logger = logging.getLogger(__name__)

# PDB format stores chain IDs in a single character column, which gemmi's
# ``shorten_chain_names()`` maps into the pool of printable single chars
# (A-Z, a-z, 0-9). Structures with more unique chain labels than this pool
# cannot be represented in PDB at all, so PyRosetta scoring is impossible.
MAX_CHAINS_FOR_PDB = 62


class ScoringStructureInput(StructureInputBase):
    """Bundles a structure with optional chain selection for scoring.

    Base input model for PyRosetta-backed scoring tools (energy, SAP, SASA) and
    structure refinement (relax). Wraps a protein structure with an optional
    chain selection. Chain IDs are validated against and exposed using the
    structure's native chain labels (including multi-character mmCIF labels
    like ``"Heavy"``); the tool layer internally shortens them to fit PDB
    format when dispatching to PyRosetta and restores the originals in the
    output.

    Attributes:
        structure (Structure): Protein structure (file path, PDB string,
            ``Structure``, or ``Structure.model_dump`` dict).
        chains_to_score (ChainSelection | None): Chains to include in scoring. ``None``
            means include every chain. Accepts shorthand ``"A"`` or
            ``["A", "B"]`` at construction.

    Examples:
        >>> inp = ScoringStructureInput(structure="/path/to/protein.pdb")
        >>> inp = ScoringStructureInput(
        ...     structure="/path/to/complex.pdb",
        ...     chains_to_score=["A", "B"],
        ... )
        >>> inp = ScoringStructureInput(
        ...     structure="/path/to/antibody.cif",
        ...     chains_to_score=["Heavy", "Light"],
        ... )
    """

    chains_to_score: ChainSelection | None = Field(
        default=None,
        title="Chains to Score",
        description="Chains to include in scoring. None = include every chain.",
    )

    @model_validator(mode="after")
    def _reject_too_many_chains(self) -> ScoringStructureInput:
        # PyRosetta runs on PDB-format poses, which cap chain IDs at 62 unique
        # single-character labels. Reject up front so a clearer error is raised
        # before the tool layer attempts the gemmi shortening.
        n_chains = len(self.structure.get_chain_ids())
        if n_chains > MAX_CHAINS_FOR_PDB:
            raise ValueError(
                f"Structure has {n_chains} chains, but PyRosetta scoring "
                f"requires PDB format which supports at most {MAX_CHAINS_FOR_PDB} "
                f"single-character chain IDs.",
            )
        return self

    @property
    def chain_ids_to_score(self) -> list[str] | None:
        """Resolved list of chain IDs to score, or None if every chain is included.

        Returns:
            list[str] | None: Explicit list of chain IDs when ``chains_to_score`` is set;
                ``None`` when no selection was provided (signals "all chains" to
                the dispatch helpers downstream).
        """
        if self.chains_to_score is None:
            return None
        return list(self.chains_to_score.chains)


# ============================================================================
# Chain-ID mapping helpers
# ============================================================================
def prepare_pdb_and_chain_maps(
    inputs: list[ScoringStructureInput],
) -> tuple[list[str], list[list[str] | None], list[dict[str, str]]]:
    """Convert each input structure to PDB content and compute chain ID mappings.

    PDB format restricts chain IDs to a single character, while mmCIF permits
    arbitrary-length chain labels. For CIF structures with multi-character
    chain labels, ``Structure.to_pdb_with_chain_mapping()`` emits a shortened
    PDB plus a mmCIF→PDB chain ID map. This helper applies that conversion to
    every input, translates the user's ``chains_to_score`` chain selection into the PDB
    namespace (which is what PyRosetta sees), and returns a parallel list of
    reverse (PDB→mmCIF) maps so per-residue output chain IDs can be restored
    to their original labels.

    Args:
        inputs (list[ScoringStructureInput]): Scoring inputs to prepare.

    Returns:
        tuple[list[str], list[list[str] | None], list[dict[str, str]]]: A triple
            ``(pdb_contents, pdb_chain_ids_list, pdb_to_mmcif_maps)`` of parallel
            lists indexed by input. ``pdb_contents`` is PDB-format content ready
            for PyRosetta; ``pdb_chain_ids_list`` is the user's chain selection
            translated into PDB labels (or ``None`` if no selection);
            ``pdb_to_mmcif_maps`` is the reverse mapping for each input so callers
            can remap per-residue output chain IDs back to their original labels.
    """
    pdb_contents: list[str] = []
    pdb_chain_ids_list: list[list[str] | None] = []
    pdb_to_mmcif_maps: list[dict[str, str]] = []
    for inp in inputs:
        pdb_content, mmcif_to_pdb = inp.structure.to_pdb_with_chain_mapping()
        pdb_contents.append(pdb_content)
        if inp.chains_to_score is not None:
            pdb_chain_ids_list.append([mmcif_to_pdb[c] for c in inp.chains_to_score.chains])
        else:
            pdb_chain_ids_list.append(None)
        pdb_to_mmcif_maps.append({v: k for k, v in mmcif_to_pdb.items()})
    return pdb_contents, pdb_chain_ids_list, pdb_to_mmcif_maps


def remap_per_residue_chain_ids(
    results: list[dict[str, Any]],
    pdb_to_mmcif_maps: list[dict[str, str]],
) -> None:
    """Rewrite per-residue chain_id fields in-place from PDB labels back to original.

    The PyRosetta dispatch returns per-residue records tagged with PDB (short)
    chain IDs because that is what the pose sees. Before constructing the
    Pydantic output models, this helper walks each result's ``per_residue`` list
    and replaces each ``chain_id`` with its original mmCIF label using the
    reverse maps from :func:`prepare_pdb_and_chain_maps`.

    Args:
        results (list[dict[str, Any]]): Raw result dicts from the PyRosetta
            dispatch. Each dict must have a ``"per_residue"`` list of residue
            dicts with ``"chain_id"`` keys.
        pdb_to_mmcif_maps (list[dict[str, str]]): Parallel reverse maps from
            :func:`prepare_pdb_and_chain_maps`. One map per result.
    """
    for result, reverse_map in zip(results, pdb_to_mmcif_maps, strict=True):
        for res in result.get("per_residue", []):
            res["chain_id"] = reverse_map.get(res["chain_id"], res["chain_id"])


def relax_inputs_via_pyrosetta(
    inputs: list[ScoringStructureInput],
    relax_config: PyRosettaRelaxConfig,
) -> list[ScoringStructureInput]:
    """Run pyrosetta-relax on each input structure; return new ScoringStructureInputs.

    Used by the ``preprocess`` hook on scoring tool configs (energy, SAP, SASA)
    to opt into FastRelax preprocessing without re-implementing FastRelax. One
    dispatch handles the whole batch; ``chains_to_score`` on each input is preserved
    unchanged because ``run_pyrosetta_relax`` restores original chain labels on
    the returned ``Structure`` (via :meth:`Structure.with_renamed_chains`).

    Args:
        inputs (list[ScoringStructureInput]): Structures + chain selections.
        relax_config (PyRosettaRelaxConfig): How to relax.

    Returns:
        list[ScoringStructureInput]: New inputs with the same ``chains_to_score`` selection,
            pointing at the relaxed structures (chain labels match input).
    """
    # Lazy import to break the circular dependency: pyrosetta_relax imports
    # from this module, and we'd otherwise import it back here.
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import (
        PyRosettaRelaxInput,
        run_pyrosetta_relax,
    )

    relax_out = run_pyrosetta_relax(
        PyRosettaRelaxInput(
            inputs=[ScoringStructureInput(structure=inp.structure) for inp in inputs],
        ),
        relax_config,
    )

    return [
        ScoringStructureInput(
            structure=relax_result.relax.relaxed_structure,
            chains_to_score=inp.chains_to_score,
        )
        for inp, relax_result in zip(inputs, relax_out.results, strict=True)
    ]


def warn_about_dropped_residues(results: list[dict[str, Any]]) -> None:
    """Emit a log warning summarizing residues PyRosetta silently dropped.

    PyRosetta is initialized with ``-ignore_unrecognized_res true`` and
    ``-remember_unrecognized_res true``, which silently drops residues it
    cannot parse (non-standard amino acids, modified residues, exotic
    ligands, etc.) while remembering their names in ``PDBInfo``. The
    standalone dispatch reads that record via
    :meth:`PyRosettaScorer._find_dropped_residues` and attaches a
    ``"dropped_residues"`` list of three-letter codes to each result dict.
    This helper pops that list and emits a single
    :func:`logging.Logger.warning` per affected input, aggregated by
    residue name so repeated names collapse into a readable summary like
    ``"ATP (1), MG (2)"``. Crystallographic waters are already filtered
    upstream by Rosetta's separate ``-ignore_waters`` flag (default true)
    and never reach this helper.

    ``dropped_residues`` is transport-only state and must not reach the
    Pydantic output model.

    Args:
        results (list[dict[str, Any]]): Raw result dicts from the PyRosetta
            dispatch. Each dict may have a ``"dropped_residues"`` key
            containing ``list[str]`` (three-letter residue codes); the key
            is popped.
    """
    for i, result in enumerate(results):
        dropped = result.pop("dropped_residues", None)
        if not dropped:
            continue
        counts = Counter(dropped)
        breakdown = ", ".join(f"{name} ({count})" for name, count in counts.most_common())
        logger.warning(
            "PyRosetta ignored %d unrecognized residue(s) in input %d: %s",
            len(dropped),
            i,
            breakdown,
        )
