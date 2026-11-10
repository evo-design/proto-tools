"""proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py.

LigandMPNN sampling tool.
"""

import logging
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.ligandmpnn._readable import (
    LIGANDMPNN_ATOM_RECORD_RESIDUES,
    ligandmpnn_readable_positions,
)
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedComplex,
    DesignSet,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
    split_fixed_positions,
    summarize_positions,
    unreadable_positions,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import AminoAcid, ConfigField, ToolInstance
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.sequence import normalize_excluded_amino_acids
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

LigandMPNNModelType = Literal[
    "ligand_mpnn",
    # The original LigandMPNN implementation. Its sampler emits packed full-atom structures and
    # its scorer runs the same original code; code and weights are auto-provisioned on first use.
    # Foundry is the default otherwise.
    "original",
    # Membrane variants need transmembrane-label inputs that are not wired; disabled for now.
    # "per_residue_label_membrane_mpnn",
    # "global_label_membrane_mpnn",
]

# model_type values that run the original implementation instead of Foundry.
LIGANDMPNN_LEGACY_MODEL_TYPES = frozenset({"original"})


# ============================================================================
# Data Models
# ============================================================================
# Input:
LigandMPNNSampleInput = InverseFoldingInput


# Config:
class LigandMPNNSampleConfig(InverseFoldingConfig):
    """Configuration for LigandMPNN sampling.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits.
        excluded_amino_acids (list[AminoAcid] | None): One-letter codes of amino acids to exclude.
        seed (int): Random seed to use for sampling.
        model_type (LigandMPNNModelType): LigandMPNN implementation. ``ligand_mpnn`` (default) is
            the Foundry implementation; ``original`` runs the original LigandMPNN code and weights,
            auto-provisioned on first use, and emits packed full-atom structures.
        checkpoint_path (str | None): Optional explicit LigandMPNN checkpoint path.
        use_atom_context (bool): Whether ligand-aware variants encode ligand atom context.
        use_side_chain_context (bool): Whether to condition on fixed-residue sidechain atoms.
        cutoff_for_score (float): Ligand-residue distance cutoff (Å) for the ligand-interface
            recovery score.
        sc_num_denoising_steps (int): Side-chain denoising steps. Only used when model_type
            is ``original``.
        sc_num_samples (int): Side-chain packing samples. Only used when model_type is
            ``original``.
    """

    model_type: LigandMPNNModelType = ConfigField(
        title="Model Type",
        default="ligand_mpnn",
        description="Implementation: 'ligand_mpnn' (Foundry, default) or 'original' (packs full-atom structures).",
        reload_on_change=True,
    )
    use_atom_context: bool = ConfigField(
        title="Use Ligand Atom Context",
        default=True,
        description="Encode ligand atom context in the message-passing graph",
    )
    checkpoint_path: str | None = ConfigField(
        title="Checkpoint Path",
        default=None,
        description="Optional explicit LigandMPNN checkpoint path.",
        reload_on_change=True,
    )
    use_side_chain_context: bool = ConfigField(
        title="Use Sidechain Context",
        default=False,
        description="Condition on sidechain atoms of fixed residues",
    )
    cutoff_for_score: float = ConfigField(
        title="Ligand Cutoff for Score",
        default=8.0,
        gt=0.0,
        description="Ligand-residue distance cutoff (Å) for interface recovery score",
    )
    excluded_amino_acids: list[AminoAcid] | None = ConfigField(
        title="Excluded Amino Acids",
        default=None,
        description="Single-letter codes of amino acids to exclude (e.g. ['C'] to forbid cysteine)",
        examples=[["C"]],
    )

    sc_num_denoising_steps: int = ConfigField(
        title="Sidechain Denoising Steps",
        default=8,
        ge=1,
        description="Side-chain denoising steps; only used with model_type='original'.",
    )
    sc_num_samples: int = ConfigField(
        title="Sidechain Samples",
        default=1,
        ge=1,
        description="Side-chain packing samples; only used with model_type='original'.",
    )

    @field_validator("excluded_amino_acids", mode="before")
    @classmethod
    def validate_excluded_amino_acids(cls, value: Any) -> Any:
        """Normalize and validate excluded amino-acid codes."""
        return normalize_excluded_amino_acids(value)

    @model_validator(mode="after")
    def _validate_legacy_only_fields(self) -> "LigandMPNNSampleConfig":
        """Reject non-default packing/packer fields unless the legacy implementation is selected.

        Compares against each field's default (rather than ``model_fields_set``) so configs
        survive a ``model_dump``/``model_validate`` round-trip, which re-sets defaulted fields.

        Returns:
            LigandMPNNSampleConfig: The validated config.
        """
        if self.model_type not in LIGANDMPNN_LEGACY_MODEL_TYPES:
            fields = type(self).model_fields
            misused = [
                name
                for name in ("sc_num_denoising_steps", "sc_num_samples")
                if getattr(self, name) != fields[name].default
            ]
            if misused:
                raise ValueError(
                    f"{', '.join(misused)} only apply when model_type='original'; got model_type={self.model_type!r}."
                )
        return self

    def readable_positions(self, inp: Any) -> dict[str, list[int]] | None:
        """Positions LigandMPNN designs, which excludes every HETATM residue.

        Only the Foundry path has a declared rule. ``model_type='original'`` parses the
        structure with the original implementation's own ``parse_PDB``, whose behaviour is
        not mirrored here, so it keeps strict length matching.

        Args:
            inp (Any): One structure input.

        Returns:
            dict[str, list[int]] | None: Ascending readable positions per chain ID, or
                ``None`` for the original implementation.
        """
        if self.model_type in LIGANDMPNN_LEGACY_MODEL_TYPES:
            return None
        return ligandmpnn_readable_positions(inp.structure)


class LigandMPNNDesignMetrics(Metrics):
    """Per-design recovery metrics emitted by LigandMPNN sampling.

    Metrics documented in ``metric_spec``:
        sequence_recovery (float): Fraction of chains_to_redesign residues
            matching the input structure's reference sequence (0.0-1.0).
            Present when at least one redesigned residue is scored.
        ligand_interface_sequence_recovery (float): Recovery restricted to
            ligand-interface residues (0.0-1.0). Present only when a ligand
            interface is present; ``NaN`` or absent when the input structure
            has no ligand interface.
        pmpnn (float): Mean sequence probability reported by LigandMPNN.
            Higher values indicate higher model probability for the emitted
            sequence.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "sequence_recovery": {
            "availability": "when at least one redesigned residue is scored",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "ligand_interface_sequence_recovery": {
            "availability": "when a ligand interface is present",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "pmpnn": {
            "availability": "when LigandMPNN reports a finite sequence probability",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="sequence_recovery",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class LigandMPNNDesign(DesignedComplex):
    """One LigandMPNN-designed complex produced from a single input structure.

    Holds every protein chain of the input structure (redesigned chains plus
    fixed context chains carried over unchanged), in input structure chain
    order, plus this design's scalar recovery metrics.

    Attributes:
        chains (list[Chain | Fragment]): All polymer chains of the design, in input
            structure chain order, each followed by its ligands as context.
        metrics (LigandMPNNDesignMetrics): Per-design recovery metrics
            (``sequence_recovery`` and ``ligand_interface_sequence_recovery``).
    """

    metrics: LigandMPNNDesignMetrics = Field(
        default_factory=LigandMPNNDesignMetrics,
        title="Metrics",
        description="Per-design recovery metrics for this complex.",
    )
    structure: Structure | None = Field(
        default=None,
        title="Designed Structure",
        description="Optional LigandMPNN full-atom structure for this design.",
    )


class LigandMPNNDesignSet(DesignSet):
    """All LigandMPNN complexes produced for a single input structure.

    Attributes:
        complexes (list[LigandMPNNDesign]): The complexes generated for one input
            structure, each a complete multi-chain complex with recovery metrics.
    """

    complexes: list[LigandMPNNDesign] = Field(  # type: ignore[assignment]
        title="Complexes",
        description="LigandMPNN complexes generated for one input structure, each a complete complex.",
    )


class LigandMPNNSampleOutput(InverseFoldingOutput):
    """Output of the LigandMPNN sampling tool.

    Attributes:
        design_sets (list[LigandMPNNDesignSet]): One ``LigandMPNNDesignSet``
            per input structure, in input order.
    """

    design_sets: list[LigandMPNNDesignSet] = Field(  # type: ignore[assignment]
        title="Design Sets",
        description="One LigandMPNNDesignSet per input structure, in input order.",
    )


def _resolvable_fixed_positions(inp: Any, readable: dict[str, list[int]] | None) -> Any:
    """Drop fixed positions LigandMPNN's parser never puts in the designed chain.

    AtomWorks moves every unlisted ``HETATM`` residue into a non-polymer chain, so a position
    naming one no longer exists under the chain the caller asked for.
    ``MPNNInferenceInput._mask_from_ids`` raises on an ID matching nothing rather than
    ignoring it, so the whole run fails on a constraint that is already satisfied: the model
    never redesigns that residue. Kept positions are not renumbered, because the model
    matches a fixed position by residue id, which staging leaves at the chain's own ordinal.

    Args:
        inp (Any): One structure input, exposing ``fixed_positions``.
        readable (dict[str, list[int]] | None): Positions the model designs, per chain ID, or
            ``None`` for the original implementation, whose parser is not mirrored here.

    Returns:
        Any: A ``ResidueSelection`` the model can resolve, or ``None`` when nothing is left.
    """
    if readable is None:
        return inp.fixed_positions
    resolved, dropped = split_fixed_positions(inp.fixed_positions, readable, renumber=False)
    for chain_id, positions in dropped.items():
        logger.warning(
            "ligandmpnn-sample: Chain %s, position %s is read as ligand context rather than a design "
            "position, so it cannot be fixed at the model. The original residue is preserved in the "
            "output regardless.",
            chain_id,
            summarize_positions(inp.structure, chain_id, positions),
        )
    return resolved


def _fixed_positions_cover_structure(inp: InverseFoldingStructureInput) -> bool:
    """Return whether fixed_positions cover every residue in every structure chain."""
    if inp.fixed_positions is None:
        return False
    for chain_id in inp.structure.get_chain_ids():
        selected = set(inp.fixed_positions.chains.get(chain_id, []))
        if selected != set(range(1, len(inp.structure.get_chain_positions(chain_id)) + 1)):
            return False
    return True


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return LigandMPNNSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="ligandmpnn-sample",
    label="LigandMPNN Sampling",
    category="inverse_folding",
    input_class=LigandMPNNSampleInput,
    config_class=LigandMPNNSampleConfig,
    output_class=LigandMPNNSampleOutput,
    description="Sample protein sequences using LigandMPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="design_sets",
    max_chunk_size=32,
    cacheable=True,
    stochastic=True,
)
def run_ligandmpnn_sample(
    inputs: LigandMPNNSampleInput,
    config: LigandMPNNSampleConfig,
    instance: Any = None,
) -> LigandMPNNSampleOutput:
    """Sample protein sequences using LigandMPNN.

    Args:
        inputs (LigandMPNNSampleInput): LigandMPNNSampleInput containing a list of structure inputs,
            each with optional ``chains_to_redesign`` and ``fixed_positions`` selections.
        config (LigandMPNNSampleConfig): Configuration for sampling (temperature, batch_size, etc.).

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        LigandMPNNSampleOutput: LigandMPNNSampleOutput with one design set per input structure.
    """
    design_sets: list[LigandMPNNDesignSet] = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advances across every dispatch (inputs x chunks) so duplicate items get distinct seeds.
    dispatch_idx = 0

    # Local venv execution
    for inp in progress_bar(
        inputs.inputs,
        desc="LigandMPNN sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_chain_sequences: list[list[dict[str, str]]] = []
        all_recovery: list[float] = []
        all_interface_recovery: list[float] = []
        all_pmpnn: list[float | None] = []
        all_pdb_strings: list[str | None] = []
        remaining = config.num_sequences_per_structure
        readable = config.readable_positions(inp)
        worker_fixed = _resolvable_fixed_positions(inp, readable)
        # Staged once per input and reused across chunks, numbered 1..N per chain so LigandMPNN,
        # which addresses residues by number alone, sees every insertion-coded residue.
        with inp.staged(atom_record_residues=LIGANDMPNN_ATOM_RECORD_RESIDUES) as staged:
            while remaining > 0:
                chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
                all_positions_fixed = _fixed_positions_cover_structure(inp)
                input_dict = {
                    "operation": "sample",
                    "pdb_path": str(staged.path),
                    "chain_ids": staged.chains(inp.chain_ids_to_redesign),
                    "chains_explicitly_set": inp.chains_to_redesign is not None and not all_positions_fixed,
                    "batch_size": chunk,
                    "temperature": config.temperature,
                    "fixed_positions": staged.positions(worker_fixed),
                    "excluded_amino_acids": config.excluded_amino_acids,
                    "seed": base_seed + dispatch_idx,
                    "device": config.device,
                    "verbose": config.verbose,
                    "model_type": config.model_type,
                    "checkpoint_path": config.checkpoint_path,
                    "ligand_mpnn_use_atom_context": config.use_atom_context,
                    "ligand_mpnn_use_side_chain_context": config.use_side_chain_context,
                    "ligand_mpnn_cutoff_for_score": config.cutoff_for_score,
                    "sc_num_denoising_steps": config.sc_num_denoising_steps,
                    "sc_num_samples": config.sc_num_samples,
                }
                result = ToolInstance.dispatch(
                    "ligandmpnn",
                    input_dict,
                    instance=instance,
                    config=config,
                )
                all_chain_sequences.extend(result["chain_sequences"])
                all_recovery.extend(m["sequence_recovery"] for m in result["metrics"])
                all_interface_recovery.extend(
                    m.get("ligand_interface_sequence_recovery", math.nan) for m in result["metrics"]
                )
                all_pmpnn.extend(m.get("pmpnn") for m in result["metrics"])
                all_pdb_strings.extend(result.get("pdb_strings", [None] * len(result["chain_sequences"])))
                dispatch_idx += 1
                remaining -= chunk  # type: ignore[operator]

        # The standalone emits ordered per-chain (chain_id, sequence) pairs.
        redesign_ids = set(inp.chain_ids_to_redesign)
        redesigned_ids = set() if _fixed_positions_cover_structure(inp) else redesign_ids

        preserved = unreadable_positions(readable, inp.structure) if readable is not None else None
        complexes: list[LigandMPNNDesign] = []
        for chain_seqs, recovery, interface_recovery, pmpnn, pdb_string in zip(
            all_chain_sequences, all_recovery, all_interface_recovery, all_pmpnn, all_pdb_strings, strict=True
        ):
            complexes.append(
                LigandMPNNDesign.from_design(
                    inp.structure,
                    {
                        cid: entry["sequence"]
                        for entry in chain_seqs
                        if (cid := staged.original_chain(entry["id"])) in redesign_ids
                    },
                    redesigned_ids,
                    tool="ligandmpnn-sample",
                    include_ligands=True,
                    readable_positions=readable,
                    fixed_positions=preserved,
                    metrics=LigandMPNNDesignMetrics(
                        sequence_recovery=(recovery if math.isfinite(recovery) else None),
                        # Foundry returns NaN when no ligand interface; route to None so the conditional key is dropped.
                        ligand_interface_sequence_recovery=(
                            interface_recovery if math.isfinite(interface_recovery) else None
                        ),
                        pmpnn=(pmpnn if pmpnn is not None and math.isfinite(pmpnn) else None),
                    ),
                    structure=(
                        staged.restore(pdb_string, source="ligandmpnn-sample", readable_positions=readable)
                        if pdb_string
                        else None
                    ),
                )
            )
        design_sets.append(LigandMPNNDesignSet(complexes=complexes))

    return LigandMPNNSampleOutput(design_sets=design_sets)
