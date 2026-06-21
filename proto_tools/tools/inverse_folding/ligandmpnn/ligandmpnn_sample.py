"""proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py.

LigandMPNN sampling tool.
"""

import logging
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from proto_tools.entities.complex import Chain
from proto_tools.entities.ligands import Fragment
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedComplex,
    DesignSet,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import AminoAcid, ConfigField, ToolInstance
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

LigandMPNNModelType = Literal[
    "ligand_mpnn",
    # Membrane variants need transmembrane-label inputs that are not wired; disabled for now.
    # "per_residue_label_membrane_mpnn",
    # "global_label_membrane_mpnn",
]


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
        model_type (LigandMPNNModelType): LigandMPNN variant to load.
        ligand_mpnn_use_atom_context (bool): Whether ligand-aware variants encode ligand atom context.
        ligand_mpnn_use_side_chain_context (bool): Whether to condition on fixed-residue sidechain atoms.
        ligand_mpnn_cutoff_for_score (float): Ligand-residue distance cutoff (Å) for the ligand-interface
            recovery score.
    """

    model_type: LigandMPNNModelType = ConfigField(
        title="Model Type",
        default="ligand_mpnn",
        description="LigandMPNN model variant (ligand-aware weights).",
        reload_on_change=True,
    )
    ligand_mpnn_use_atom_context: bool = ConfigField(
        title="Use Ligand Atom Context",
        default=True,
        description="Encode ligand atom context in the message-passing graph",
    )
    ligand_mpnn_use_side_chain_context: bool = ConfigField(
        title="Use Sidechain Context",
        default=False,
        description="Condition on sidechain atoms of fixed residues",
    )
    ligand_mpnn_cutoff_for_score: float = ConfigField(
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


class LigandMPNNDesignMetrics(Metrics):
    """Per-design recovery metrics emitted by LigandMPNN sampling.

    Metrics documented in ``metric_spec``:
        sequence_recovery (float): Fraction of chains_to_redesign residues
            matching the input structure's reference sequence (0.0-1.0).
            Always present.
        ligand_interface_sequence_recovery (float): Recovery restricted to
            ligand-interface residues (0.0-1.0). Present only when a ligand
            interface is present; ``NaN`` or absent when the input structure
            has no ligand interface.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "sequence_recovery": {
            "availability": "always",
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
        chains (list[Chain | Fragment]): All protein chains of the design, in input
            structure chain order.
        metrics (LigandMPNNDesignMetrics): Per-design recovery metrics
            (``sequence_recovery`` and ``ligand_interface_sequence_recovery``).
    """

    metrics: LigandMPNNDesignMetrics = Field(
        default_factory=LigandMPNNDesignMetrics,
        title="Metrics",
        description="Per-design recovery metrics for this complex.",
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
        remaining = config.num_sequences_per_structure
        # Materialize the Structure to a tempfile once per input — reused across chunks.
        with inp.structure.temp_file() as pdb_path:
            while remaining > 0:
                chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
                input_dict = {
                    "operation": "sample",
                    "pdb_path": str(pdb_path),
                    "chain_ids": inp.chain_ids_to_redesign,
                    "chains_explicitly_set": inp.chains_to_redesign is not None,
                    "batch_size": chunk,
                    "temperature": config.temperature,
                    "fixed_positions": inp.fixed_positions.chains if inp.fixed_positions is not None else None,
                    "excluded_amino_acids": config.excluded_amino_acids,
                    "seed": base_seed + dispatch_idx,
                    "device": config.device,
                    "verbose": config.verbose,
                    "model_type": config.model_type,
                    "ligand_mpnn_use_atom_context": config.ligand_mpnn_use_atom_context,
                    "ligand_mpnn_use_side_chain_context": config.ligand_mpnn_use_side_chain_context,
                    "ligand_mpnn_cutoff_for_score": config.ligand_mpnn_cutoff_for_score,
                }
                result = ToolInstance.dispatch(
                    "ligandmpnn",
                    input_dict,
                    instance=instance,
                    config=config,
                )
                all_chain_sequences.extend(result["chain_sequences"])
                all_recovery.extend(m["sequence_recovery"] for m in result["metrics"])
                all_interface_recovery.extend(m["ligand_interface_sequence_recovery"] for m in result["metrics"])
                dispatch_idx += 1
                remaining -= chunk  # type: ignore[operator]

        # The standalone emits ordered per-chain (chain_id, sequence) pairs.
        redesigned_ids = set(inp.chain_ids_to_redesign)

        complexes: list[LigandMPNNDesign] = []
        for chain_seqs, recovery, interface_recovery in zip(
            all_chain_sequences, all_recovery, all_interface_recovery, strict=True
        ):
            chains: list[Chain | Fragment] = [Chain(id=entry["id"], sequence=entry["sequence"]) for entry in chain_seqs]
            designed: list[bool] = [entry["id"] in redesigned_ids for entry in chain_seqs]
            complexes.append(
                LigandMPNNDesign(
                    chains=chains,
                    designed=designed,
                    metrics=LigandMPNNDesignMetrics(
                        sequence_recovery=recovery,
                        # Foundry returns NaN when no ligand interface; route to None so the conditional key is dropped.
                        ligand_interface_sequence_recovery=(
                            interface_recovery if math.isfinite(interface_recovery) else None
                        ),
                    ),
                )
            )
        design_sets.append(LigandMPNNDesignSet(complexes=complexes))

    return LigandMPNNSampleOutput(design_sets=design_sets)
