"""proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample.py.

ProteinMPNN sampling tool.
"""

import logging
import random
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
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

# ============================================================================
# Data Models
# ============================================================================
# Input:
ProteinMPNNSampleInput = InverseFoldingInput


# Config:
class ProteinMPNNSampleConfig(InverseFoldingConfig):
    """Configuration for ProteinMPNN sampling.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits.
        excluded_amino_acids (list[AminoAcid] | None): One-letter codes of amino acids to exclude.
        seed (int): Random seed to use for sampling.
        model_choice (Literal['proteinmpnn', 'v_48_002', 'v_48_010', 'v_48_030', 'abmpnn', 'soluble']): Model
            weights. ``"proteinmpnn"`` is ColabDesign's default ``v_48_020`` (medium training noise). The
            ``v_48_*`` variants are the same architecture trained at different noise levels (002 / 010 / 030).
            ``"abmpnn"`` is antibody-optimized; ``"soluble"`` is soluble-protein-trained.
        backbone_noise (float): Gaussian noise (A) added to backbone coordinates before each forward pass.
    """

    model_choice: Literal["proteinmpnn", "v_48_002", "v_48_010", "v_48_030", "abmpnn", "soluble"] = ConfigField(
        title="Model Choice",
        default="proteinmpnn",
        description="Weights: proteinmpnn (=v_48_020), v_48_{002,010,030} noise variants, abmpnn, soluble",
        reload_on_change=True,
        examples=["proteinmpnn", "v_48_010", "abmpnn", "soluble"],
    )
    backbone_noise: float = ConfigField(
        title="Backbone Noise",
        default=0.0,
        ge=0.0,
        description="Gaussian noise (A) on backbone coords; raise (e.g. 0.02) for diversity",
    )
    excluded_amino_acids: list[AminoAcid] | None = ConfigField(
        title="Excluded Amino Acids",
        default=None,
        description="Single-letter codes of amino acids to exclude (e.g. ['C'] to forbid cysteine)",
        examples=[["C"]],
    )


class ProteinMPNNDesignMetrics(Metrics):
    """Per-design metrics emitted by ProteinMPNN sampling.

    Metrics documented in ``metric_spec``:
        perplexity (float): Perplexity of the design from the ProteinMPNN model.
            Always present. Range ``[1, ∞)``.
        sequence_recovery (float): Fraction of redesigned-chain residues matching
            the PDB prompt reference for this design (0.0-1.0). Always present.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "perplexity": {
            "availability": "always",
            "type": "float",
            "min": 1.0,
            "max": None,
            "better_values_are": "lower",
        },
        "sequence_recovery": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="perplexity",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class ProteinMPNNDesign(DesignedComplex):
    """One ProteinMPNN-designed complex with its per-design scalar metrics.

    Attributes:
        chains (list[Chain | Fragment]): All protein chains of the design, in input
            structure chain order. Redesigned chains carry the model output;
            context chains carry the original input sequence unchanged.
        metrics (ProteinMPNNDesignMetrics): Per-design metrics (perplexity and
            sequence recovery) for this design.
    """

    metrics: ProteinMPNNDesignMetrics = Field(
        default_factory=ProteinMPNNDesignMetrics,
        title="Metrics",
        description="Per-design metrics (perplexity and sequence recovery).",
    )


class ProteinMPNNDesignSet(DesignSet):
    """All ProteinMPNN complexes produced for a single input structure.

    Attributes:
        complexes (list[ProteinMPNNDesign]): The complexes generated for one input
            structure, each a complete multi-chain complex with per-design metrics.
    """

    complexes: list[ProteinMPNNDesign] = Field(  # type: ignore[assignment]
        title="Complexes",
        description="ProteinMPNN complexes for one input structure, each a complete complex.",
    )


class ProteinMPNNSampleOutput(InverseFoldingOutput):
    """Output of the ProteinMPNN sampling tool.

    Attributes:
        design_sets (list[ProteinMPNNDesignSet]): One ``ProteinMPNNDesignSet``
            per input structure, in input order. Entry ``i`` holds all complexes for
            input structure ``i``.
    """

    design_sets: list[ProteinMPNNDesignSet] = Field(  # type: ignore[assignment]
        title="Design Sets",
        description="One ProteinMPNNDesignSet per input structure, in input order.",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="proteinmpnn-sample",
    label="ProteinMPNN Sampling",
    category="inverse_folding",
    input_class=ProteinMPNNSampleInput,
    config_class=ProteinMPNNSampleConfig,
    output_class=ProteinMPNNSampleOutput,
    description="Sample protein sequences using ProteinMPNN",
    uses_gpu=True,
    pin_visible_devices=True,
    stochastic=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="design_sets",
)
def run_proteinmpnn_sample(
    inputs: ProteinMPNNSampleInput,
    config: ProteinMPNNSampleConfig,
    instance: Any = None,
) -> ProteinMPNNSampleOutput:
    """Sample protein sequences using ProteinMPNN.

    Args:
        inputs (ProteinMPNNSampleInput): ProteinMPNNSampleInput containing a list of structure inputs,
            each with optional ``chains_to_redesign`` and ``fixed_positions`` selections.
        config (ProteinMPNNSampleConfig): Configuration for sampling (temperature, batch_size, etc.).

        instance (Any): A ToolInstance, or a string referencing one pre-registered via
            ToolInstance.get/persist_tool (unknown names raise); None runs one-shot.

    Returns:
        ProteinMPNNSampleOutput: ProteinMPNNSampleOutput with one design set per input structure.

    Note:
        Each design is a ``ProteinMPNNDesign`` whose ``chains`` cover every chain
        of the input structure in order: redesigned chains carry the model output
        with ``redesigned=True``, context chains are carried over unchanged with
        ``redesigned=False``.
    """
    design_sets: list[ProteinMPNNDesignSet] = []

    # Local venv execution
    logger.debug("Using local venv for ProteinMPNN sampling")

    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Draw a fresh seed per chunk so identical structures across inputs do not collide.
    seed_rng = random.Random(base_seed)  # noqa: S311 -- non-cryptographic

    for inp in progress_bar(
        inputs.inputs,
        desc="ProteinMPNN sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        all_seqs: list[str] = []
        all_perp: list[float] = []
        all_seq_recovery: list[float] = []
        remaining = config.num_sequences_per_structure
        # Materialize the Structure to a tempfile once per input, reused across chunks.
        with inp.structure.temp_file() as pdb_path:
            while remaining > 0:
                chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
                input_dict = {
                    "operation": "sample",
                    "pdb_path": str(pdb_path),
                    "chain_ids": inp.chain_ids_to_redesign,
                    "batch_size": chunk,
                    "temperature": config.temperature,
                    "fixed_positions": inp.fixed_positions.chains if inp.fixed_positions is not None else None,
                    "excluded_amino_acids": config.excluded_amino_acids,
                    "seed": seed_rng.randint(0, 2**31 - 1),
                    "device": config.device,
                    "model_choice": config.model_choice,
                    "verbose": config.verbose,
                    "return_logits": False,
                    "backbone_noise": config.backbone_noise,
                }
                result = ToolInstance.dispatch(
                    "proteinmpnn",
                    input_dict,
                    instance=instance,
                    config=config,
                )
                all_seqs.extend(result["seq"])
                all_perp.extend(np.exp(result["score"]).tolist())
                all_seq_recovery.extend(result["seqid"])
                remaining -= chunk  # type: ignore[operator]

        redesign_ids = inp.chain_ids_to_redesign
        redesign_set = set(redesign_ids)
        all_chain_ids = inp.structure.get_chain_ids()
        complexes: list[ProteinMPNNDesign] = []
        for raw_seq, perp, recovery in zip(all_seqs, all_perp, all_seq_recovery, strict=True):
            # Segments are the redesigned chains in chain_ids_to_redesign order.
            segments = raw_seq.split("/")
            if len(segments) != len(redesign_ids):
                raise ValueError(
                    f"ProteinMPNN returned {len(segments)} sequence segment(s) but "
                    f"{len(redesign_ids)} chain(s) were requested for redesign "
                    f"({redesign_ids}). Raw sequence: {raw_seq!r}."
                )
            redesigned_seq_by_chain = dict(zip(redesign_ids, segments, strict=True))
            # Build all chains in input order; context chains keep the input sequence.
            chains: list[Chain | Fragment] = [
                Chain(
                    id=cid,
                    sequence=redesigned_seq_by_chain[cid]
                    if cid in redesign_set
                    else inp.structure.get_chain_sequence(cid),
                )
                for cid in all_chain_ids
            ]
            designed: list[bool] = [cid in redesign_set for cid in all_chain_ids]
            complexes.append(
                ProteinMPNNDesign(
                    chains=chains,
                    designed=designed,
                    metrics=ProteinMPNNDesignMetrics(perplexity=perp, sequence_recovery=recovery),
                )
            )
        design_sets.append(ProteinMPNNDesignSet(complexes=complexes))
    return ProteinMPNNSampleOutput(design_sets=design_sets)
