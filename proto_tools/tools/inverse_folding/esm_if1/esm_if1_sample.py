"""proto_tools/tools/inverse_folding/esm_if1/esm_if1_sample.py.

ESM-IF1/ProteinDPO sampling tool.
"""

import logging
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
from proto_tools.utils import ConfigField
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
ESMIF1SampleInput = InverseFoldingInput


class ESMIF1SampleConfig(InverseFoldingConfig):
    """Configuration for ESM-IF1/ProteinDPO sequence sampling.

    Attributes:
        weights_variant (Literal['esmif', 'protein_dpo']): Which model weights to use. 'esmif' loads vanilla ESM-IF1,
            'protein_dpo' loads DPO-aligned weights optimized for protein stability.
        num_sequences_per_structure (int): Total number of sequences to generate per structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
        temperature (float): Sampling temperature; ESM-IF1's tuned default is 1.0.
        seed (int): Random seed for sampling reproducibility.
    """

    weights_variant: Literal["esmif", "protein_dpo"] = ConfigField(
        title="Weights Variant",
        default="protein_dpo",
        description="'esmif' for vanilla ESM-IF1, 'protein_dpo' for DPO-aligned weights",
        reload_on_change=True,
        examples=["esmif", "protein_dpo"],
    )
    # ESM-IF1's reference inference script uses temperature=1.0; the base 0.1 default is too greedy here.
    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=1.0,
        ge=0.0,
        description="Sampling temperature; lower = greedier, higher = more diverse",
        examples=[0.5, 1.0, 2.0],
    )


class ESMIF1DesignMetrics(Metrics):
    """Per-design metrics emitted by ESM-IF1/ProteinDPO sampling.

    Metrics documented in ``metric_spec``:
        log_likelihood (float): Sum of per-position log-likelihoods of the
            designed target chain under the model (<= 0). Always present.
        avg_log_likelihood (float): Mean per-position log-likelihood (<= 0).
            Always present.
        perplexity (float): exp(-avg_log_likelihood) (>= 1). Always present.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "avg_log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "perplexity": {
            "availability": "always",
            "type": "float",
            "min": 1.0,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="perplexity",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class ESMIF1Design(DesignedComplex):
    """One ESM-IF1/ProteinDPO design over a single input structure.

    Holds every protein chain of the input structure; only the target chain is
    redesigned, all other chains are carried over as fixed context.

    Attributes:
        chains (list[Chain | Fragment]): All protein chains of the design, in input
            structure chain order.
        metrics (ESMIF1DesignMetrics): Per-design metrics, including the average
            per-position log-likelihood of the redesigned target chain.
    """

    metrics: ESMIF1DesignMetrics = Field(
        default_factory=ESMIF1DesignMetrics,
        title="Metrics",
        description="Per-design ESM-IF1/ProteinDPO metrics (average log-likelihood).",
    )


class ESMIF1DesignSet(DesignSet):
    """All ESM-IF1/ProteinDPO complexes produced for a single input structure.

    Attributes:
        complexes (list[ESMIF1Design]): The complexes generated for one input,
            each a complete multi-chain complex with a per-design log-likelihood.
    """

    complexes: list[ESMIF1Design] = Field(  # type: ignore[assignment]
        title="Complexes",
        description="ESM-IF1/ProteinDPO complexes for one input structure, each a complete complex.",
    )


class ESMIF1SampleOutput(InverseFoldingOutput):
    """Output of the ESM-IF1 sampling tool.

    Attributes:
        design_sets (list[ESMIF1DesignSet]): One ``ESMIF1DesignSet`` per
            input structure, in input order.
    """

    design_sets: list[ESMIF1DesignSet] = Field(  # type: ignore[assignment]
        title="Design Sets",
        description="One ESMIF1DesignSet per input structure, in input order.",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESMIF1SampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="esm-if1-sample",
    label="ESM-IF1 Sampling",
    category="inverse_folding",
    input_class=ESMIF1SampleInput,
    config_class=ESMIF1SampleConfig,
    output_class=ESMIF1SampleOutput,
    description=(
        "Sample protein sequences conditioned on backbone structure using "
        "ESM-IF1 or ProteinDPO (DPO-aligned for stability). Supports "
        "multi-chain complexes."
    ),
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="design_sets",
    cacheable=True,
    stochastic=True,
)
def run_esm_if1_sample(
    inputs: ESMIF1SampleInput,
    config: ESMIF1SampleConfig,
    instance: Any = None,
) -> ESMIF1SampleOutput:
    """Sample protein sequences using ESM-IF1/ProteinDPO.

    Args:
        inputs (ESMIF1SampleInput): Structure inputs with optional chain/fixed_positions position constraints.
        config (ESMIF1SampleConfig): Configuration including weights variant, temperature, etc.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMIF1SampleOutput: ESMIF1SampleOutput with one design set per input structure.
    """
    design_sets = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advances across every dispatch (inputs x chunks) so duplicate items get distinct seeds.
    dispatch_idx = 0

    for inp in progress_bar(
        inputs.inputs,
        desc="ESM-IF1 sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_seqs, all_metrics = [], []
        remaining = config.num_sequences_per_structure
        # Materialize the Structure to a tempfile once per input — reused across chunks.
        with inp.structure.temp_file() as pdb_path:
            while remaining > 0:
                chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
                input_dict = {
                    "operation": "sample",
                    "pdb_path": str(pdb_path),
                    "chain_ids": inp.chain_ids_to_redesign,
                    "batch_size": chunk,
                    "temperature": config.temperature,
                    "seed": base_seed + dispatch_idx,
                    "device": config.device,
                    "weights_variant": config.weights_variant,
                    "verbose": config.verbose,
                    "fixed_positions": inp.fixed_positions.chains if inp.fixed_positions is not None else None,
                }
                result = ToolInstance.dispatch(
                    "esm_if1",
                    input_dict,
                    instance=instance,
                    config=config,
                )
                all_seqs.extend(result["sequences"])
                all_metrics.extend(result["metrics"])
                dispatch_idx += 1
                remaining -= chunk  # type: ignore[operator]

        # ESM-IF1 complexes exactly one chain; standalone falls back to chain_ids[0].
        target = inp.chain_ids_to_redesign[0]
        structure_chain_ids = inp.structure.get_chain_ids()
        context_seqs = {cid: inp.structure.get_chain_sequence(cid) for cid in structure_chain_ids if cid != target}

        complexes = []
        for designed_seq, metrics in zip(all_seqs, all_metrics, strict=True):
            chains: list[Chain | Fragment] = [
                Chain(id=cid, sequence=designed_seq if cid == target else context_seqs[cid])
                for cid in structure_chain_ids
            ]
            designed: list[bool] = [cid == target for cid in structure_chain_ids]
            complexes.append(ESMIF1Design(chains=chains, designed=designed, metrics=ESMIF1DesignMetrics(**metrics)))
        design_sets.append(ESMIF1DesignSet(complexes=complexes))

    return ESMIF1SampleOutput(design_sets=design_sets)
