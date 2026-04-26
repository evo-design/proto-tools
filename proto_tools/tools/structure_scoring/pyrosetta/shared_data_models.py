"""Shared data models and input-prep helpers for PyRosetta scoring tools."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from proto_tools.entities.structures import Structure

if TYPE_CHECKING:
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import PyRosettaRelaxConfig

logger = logging.getLogger(__name__)

# PDB format stores chain IDs in a single character column, which gemmi's
# ``shorten_chain_names()`` maps into the pool of printable single chars
# (A-Z, a-z, 0-9). Structures with more unique chain labels than this pool
# cannot be represented in PDB at all, so PyRosetta scoring is impossible.
MAX_CHAINS_FOR_PDB = 62


class ScoringStructureInput(BaseModel):
    """Bundles a structure with optional chain selection for scoring.

    Base input model for structure scoring tools. Wraps a protein structure
    with optional chain filtering. Chain IDs are validated against and exposed
    using the structure's native chain labels (including multi-character
    mmCIF labels like ``"Heavy"``); the tool layer internally shortens them
    to fit PDB format when dispatching to PyRosetta and restores the originals
    in the output.

    Attributes:
        structure (Structure): The protein structure. Accepts file path, PDB content
            string, or Structure object.
        chain_ids (list[str] | None): Optional list of chain IDs to include in
            scoring. If None, all chains are included.

    Examples:
        >>> inp = ScoringStructureInput(structure="/path/to/protein.pdb")
        >>> inp = ScoringStructureInput(
        ...     structure="/path/to/complex.pdb",
        ...     chain_ids=["A", "B"],
        ... )
        >>> inp = ScoringStructureInput(
        ...     structure="/path/to/antibody.cif",
        ...     chain_ids=["Heavy", "Light"],
        ... )
    """

    structure: Structure = Field(description="Protein structure (file path, PDB string, or Structure object).")
    chain_ids: list[str] | None = Field(
        default=None,
        description="Chain IDs to include in scoring. If None, all chains.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_and_validate(cls, data: Any) -> Any:
        """Load structure, validate chain IDs, and reject structures with too many chains."""
        if isinstance(data, (str, Path, Structure)):
            data = {"structure": data}

        if isinstance(data, dict):
            structure = data.get("structure")
            chain_ids = data.get("chain_ids")
        else:
            structure = getattr(data, "structure", None)
            chain_ids = getattr(data, "chain_ids", None)

        if isinstance(structure, (str, Path)):
            resolved_structure = Structure(structure=str(structure))
        elif isinstance(structure, Structure):
            resolved_structure = structure
        elif isinstance(structure, dict):
            # JSON round-trip case: model_dump serializes a Structure as a dict.
            resolved_structure = Structure(**structure)
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")

        available_chains = set(resolved_structure.get_chain_ids())

        if len(available_chains) > MAX_CHAINS_FOR_PDB:
            raise ValueError(
                f"Structure has {len(available_chains)} chains, but PyRosetta scoring "
                f"requires PDB format which supports at most {MAX_CHAINS_FOR_PDB} "
                f"single-character chain IDs."
            )

        if chain_ids is not None:
            requested_chains = set(chain_ids)
            if not requested_chains.issubset(available_chains):
                missing = requested_chains - available_chains
                raise ValueError(f"Chain IDs {missing} not found in structure. Available chains: {available_chains}")

        result = {"structure": resolved_structure, "chain_ids": chain_ids}
        if isinstance(data, dict):
            for k, v in data.items():
                if k not in result:
                    result[k] = v
        return result


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
    every input, translates the user's ``chain_ids`` into the PDB namespace
    (which is what PyRosetta sees), and returns a parallel list of reverse
    (PDB→mmCIF) maps so per-residue output chain IDs can be restored to their
    original labels.

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
        if inp.chain_ids is not None:
            pdb_chain_ids_list.append([mmcif_to_pdb[c] for c in inp.chain_ids])
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
    dispatch handles the whole batch; ``chain_ids`` on each input are preserved
    unchanged because ``run_pyrosetta_relax`` restores original chain labels on
    the returned ``Structure`` (via :meth:`Structure.with_renamed_chains`).

    Args:
        inputs (list[ScoringStructureInput]): Structures + chain selections.
        relax_config (PyRosettaRelaxConfig): How to relax.

    Returns:
        list[ScoringStructureInput]: New inputs with the same ``chain_ids``,
            pointing at the relaxed structures (chain labels match input).

    Raises:
        RuntimeError: If the relax dispatch fails.
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
    if not relax_out.success:
        raise RuntimeError(f"FastRelax preprocess failed: {relax_out.errors}")

    return [
        ScoringStructureInput(
            structure=relax_result.relax.relaxed_structure,
            chain_ids=inp.chain_ids,
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
    Pydantic output model, which uses ``extra="forbid"``.

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
