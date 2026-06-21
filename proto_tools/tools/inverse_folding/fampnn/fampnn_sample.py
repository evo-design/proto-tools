"""proto_tools/tools/inverse_folding/fampnn/fampnn_sample.py.

FAMPNN sequence sampling tool with full-atom sidechain co-generation.
"""

import logging
import statistics
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from proto_tools.entities.complex import Chain
from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import ResidueSelection, Structure
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedComplex,
    DesignSet,
    InverseFoldingConfig,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec


class FAMPNNStructureInput(InverseFoldingStructureInput):
    """FAMPNN structure input with optional sidechain-conditioning selection.

    Extends :class:`InverseFoldingStructureInput` with ``fixed_sidechain_positions`` for
    conditioning on known sidechain conformations during design/packing.

    FAMPNN distinguishes two kinds of fixed-residue constraints:

    - ``fixed_positions`` (inherited): residue positions whose amino acid identity is kept
      fixed during sequence design (the model will not redesign these positions).
    - ``fixed_sidechain_positions``: residue positions whose sidechain atom coordinates
      are used as structural context (the model conditions on their 3D geometry).

    Attributes:
        fixed_sidechain_positions (ResidueSelection | None): Per-chain residue positions
            whose sidechain coordinates condition the model during
            sampling/packing (1-indexed). Accepts shorthand ``{"A": [1, 2]}`` at
            construction.
    """

    fixed_sidechain_positions: ResidueSelection | None = Field(
        default=None,
        title="Fixed Sidechain Positions",
        description="Per-chain positions whose sidechain coordinates condition the model (1-indexed).",
    )


logger = logging.getLogger(__name__)


class FAMPNNSampleInput(BaseToolInput):
    """Input for FAMPNN sequence sampling.

    Attributes:
        inputs (list[FAMPNNStructureInput]): Per-structure inputs, each
            containing a structure and optional ``chains_to_redesign`` / ``fixed_positions`` /
            ``fixed_sidechain_positions`` selections.
    """

    inputs: list[FAMPNNStructureInput] = InputField(
        title="Structure Inputs",
        description="List of structure inputs for sequence design.",
    )


class FAMPNNSampleConfig(InverseFoldingConfig):
    """Configuration for FAMPNN sequence sampling.

    Extends InverseFoldingConfig with FAMPNN-specific parameters for
    iterative masked language modeling and sidechain diffusion.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits.
        seed (int | None): Random seed; None draws a random seed per run.
        model_variant (str): FAMPNN checkpoint variant. '0.3' for sequence design
            (PDB-trained, 0.3 Å noise), '0.0' for sidechain packing (PDB-trained,
            0.0 Å noise), '0.3_cath' for mutation scoring (CATH-trained).
        num_steps (int): Number of iterative unmasking steps for sequence design.
            More steps yield higher quality but slower inference. 10 steps is
            sufficient for high self-consistency; 100 for best quality.
        seq_only (bool): If True, skip sidechain generation during sampling.
        repack_last (bool): If True, repack sidechains after final sequence is determined.
        psce_threshold (float): Only condition on sidechains with predicted sidechain
            error below this threshold during iterative sampling.
        scn_diffusion_steps (int): Number of sidechain diffusion denoising steps.
        scn_step_scale (float): Step scale for sidechain diffusion (eta parameter).
    """

    model_variant: str = ConfigField(
        title="Model Variant",
        default="0.3",
        description="FAMPNN checkpoint: '0.3' (design), '0.0' (packing), '0.3_cath' (scoring)",
        examples=["0.3", "0.0", "0.3_cath"],
        reload_on_change=True,
    )
    num_steps: int = ConfigField(
        title="Unmasking Steps",
        default=100,
        ge=1,
        description="Number of iterative unmasking steps for sequence design",
        examples=[10, 50, 100],
    )
    seq_only: bool = ConfigField(
        title="Sequence Only",
        default=False,
        description="If True, skip sidechain generation during sampling",
    )
    repack_last: bool = ConfigField(
        title="Repack Last",
        default=True,
        description="Repack sidechains after final sequence is determined",
    )
    psce_threshold: float = ConfigField(
        title="pSCE Threshold",
        default=0.3,
        ge=0.0,
        description="Only keep sidechains below this predicted error threshold during design",
        examples=[0.3, 0.5, 1.0],
    )
    scn_diffusion_steps: int = ConfigField(
        title="Sidechain Diffusion Steps",
        default=50,
        ge=1,
        description="Number of sidechain diffusion denoising steps",
    )
    scn_step_scale: float = ConfigField(
        title="Sidechain Step Scale",
        default=1.5,
        gt=0.0,
        description="Step scale (eta) for sidechain diffusion",
    )


class FAMPNNDesignMetrics(Metrics):
    """Per-design full-atom sidechain confidence metrics for FAMPNN sampling.

    Metrics documented in ``metric_spec``:
        avg_psce (float): Mean of the per-residue predicted sidechain error
            (``psce``) over all residues in the design, in Angstroms. Lower is
            more confident. Always present.

    Attributes:
        psce (list[float]): Per-residue predicted sidechain error (mean over
            atoms) in Angstroms, concatenated across all chains in input chain
            order.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_psce": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
    }
    primary_metric: str | None = Field(
        default="avg_psce",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    psce: list[float] = Field(
        default_factory=list,
        title="pSCE",
        description="Per-residue predicted sidechain error (Angstroms), concatenated in input chain order.",
    )


class FAMPNNDesign(DesignedComplex):
    """One FAMPNN-designed complex with full-atom sidechain confidence.

    Attributes:
        chains (list[Chain | Fragment]): All protein chains of the design, in input
            structure chain order. Inherited from :class:`DesignedComplex`.
        structure (Structure): Designed full-atom structure with packed sidechain
            coordinates co-generated for this design.
        metrics (FAMPNNDesignMetrics): Per-design full-atom sidechain confidence
            metrics (``avg_psce`` plus the per-residue ``psce`` array).
    """

    structure: Structure = Field(
        title="Designed Structure",
        description="Designed full-atom structure with packed sidechain coordinates.",
    )
    metrics: FAMPNNDesignMetrics = Field(
        default_factory=FAMPNNDesignMetrics,
        title="Metrics",
        description="Per-design full-atom sidechain confidence metrics (avg_psce plus per-residue psce).",
    )


class FAMPNNDesignSet(DesignSet):
    """All FAMPNN complexes produced for a single input structure.

    Attributes:
        complexes (list[FAMPNNDesign]): The FAMPNN complexes generated for one input,
            each a complete multi-chain complex.
    """

    complexes: list[FAMPNNDesign] = Field(  # type: ignore[assignment]
        title="Complexes",
        description="FAMPNN complexes generated for one input structure, each a complete complex.",
    )


class FAMPNNSampleOutput(InverseFoldingOutput):
    """Output of the FAMPNN sampling tool.

    Attributes:
        design_sets (list[FAMPNNDesignSet]): One ``FAMPNNDesignSet`` per
            input structure, in input order.
    """

    design_sets: list[FAMPNNDesignSet] = Field(  # type: ignore[assignment]
        title="Design Sets",
        description="One FAMPNNDesignSet per input structure, in input order.",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return FAMPNNSampleInput(
        inputs=[
            FAMPNNStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="fampnn-sample",
    label="FAMPNN Sampling",
    category="inverse_folding",
    input_class=FAMPNNSampleInput,
    config_class=FAMPNNSampleConfig,
    output_class=FAMPNNSampleOutput,
    description="Design protein sequences with full-atom sidechain co-generation using FAMPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="design_sets",
    cacheable=True,
    stochastic=True,
)
def run_fampnn_sample(
    inputs: FAMPNNSampleInput,
    config: FAMPNNSampleConfig,
    instance: Any = None,
) -> FAMPNNSampleOutput:
    """Design protein sequences with full-atom sidechain co-generation using FAMPNN.

    FAMPNN iteratively unmasks sequence and sidechain tokens, jointly generating
    amino acid identities and sidechain conformations. The output includes
    full-atom PDB structures with predicted sidechain coordinates and per-residue
    confidence scores (pSCE).

    Args:
        inputs (FAMPNNSampleInput): FAMPNNSampleInput containing structure inputs with optional
            ``chains_to_redesign``, ``fixed_positions``, and ``fixed_sidechain_positions`` selections.
        config (FAMPNNSampleConfig): Configuration for sampling (temperature, num_steps, etc.).
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        FAMPNNSampleOutput: FAMPNNSampleOutput with one FAMPNNDesignSet per input structure.
    """
    design_sets: list[FAMPNNDesignSet] = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advances across every dispatch (inputs x chunks) so duplicate items get distinct seeds.
    dispatch_idx = 0

    for inp in progress_bar(
        inputs.inputs,
        desc="FAMPNN sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        chain_ids = inp.structure.get_chain_ids()
        redesign_set = set(inp.chain_ids_to_redesign)
        all_chain_seqs: list[list[str]] = []
        all_psce: list[list[float]] = []
        all_pdbs: list[str] = []
        remaining = config.num_sequences_per_structure
        # Materialize the Structure to a tempfile once per input, reused across chunks.
        with inp.structure.temp_file() as pdb_path:
            while remaining > 0:
                chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
                input_dict = {
                    "operation": "sample",
                    "pdb_path": str(pdb_path),
                    "chain_ids": inp.chain_ids_to_redesign,
                    "num_sequences": chunk,
                    "temperature": config.temperature,
                    "num_steps": config.num_steps,
                    "seq_only": config.seq_only,
                    "repack_last": config.repack_last,
                    "psce_threshold": config.psce_threshold,
                    "scn_diffusion_steps": config.scn_diffusion_steps,
                    "scn_step_scale": config.scn_step_scale,
                    "seed": base_seed + dispatch_idx,
                    "model_variant": config.model_variant,
                    "device": config.device,
                    "verbose": config.verbose,
                    "fixed_positions": inp.fixed_positions.chains if inp.fixed_positions is not None else None,
                    "fixed_sidechain_positions": (
                        inp.fixed_sidechain_positions.chains if inp.fixed_sidechain_positions is not None else None
                    ),
                }
                result = ToolInstance.dispatch(
                    "fampnn",
                    input_dict,
                    instance=instance,
                    config=config,
                )
                all_chain_seqs.extend(result["chain_sequences"])
                all_psce.extend(result["psce"])
                all_pdbs.extend(result["pdb_strings"])
                dispatch_idx += 1
                remaining -= chunk  # type: ignore[operator]

        complexes: list[FAMPNNDesign] = []
        for chain_seqs, psce, pdb in zip(all_chain_seqs, all_psce, all_pdbs, strict=True):
            if len(chain_seqs) != len(chain_ids):
                raise ValueError(
                    f"fampnn-sample: model returned {len(chain_seqs)} chains but the input "
                    f"structure has {len(chain_ids)} chains ({chain_ids})"
                )
            if sum(len(s) for s in chain_seqs) != len(psce):
                raise ValueError(
                    f"fampnn-sample: total designed length {sum(len(s) for s in chain_seqs)} "
                    f"does not match per-residue pSCE length {len(psce)}"
                )
            chains: list[Chain | Fragment] = [
                Chain(id=cid, sequence=seq) for cid, seq in zip(chain_ids, chain_seqs, strict=True)
            ]
            designed: list[bool] = [cid in redesign_set for cid in chain_ids]
            structure = Structure(structure=pdb, structure_format="pdb", source="fampnn-sample")
            avg_psce = statistics.fmean(psce) if psce else 0.0
            complexes.append(
                FAMPNNDesign(
                    chains=chains,
                    designed=designed,
                    structure=structure,
                    metrics=FAMPNNDesignMetrics(avg_psce=avg_psce, psce=psce),
                )
            )

        design_sets.append(FAMPNNDesignSet(complexes=complexes))

    return FAMPNNSampleOutput(design_sets=design_sets)
