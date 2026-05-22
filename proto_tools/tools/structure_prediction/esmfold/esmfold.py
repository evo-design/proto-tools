"""proto_tools/tools/structure_prediction/esmfold/esmfold.py.

Protein structure prediction using ESMFold.

This module provides standardized interfaces for protein structure prediction
using ESMFold from Meta AI.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_prediction.esmfold.helpers import (
    relabel_chains as _relabel_chains,
)
from proto_tools.tools.structure_prediction.esmfold.helpers import (
    split_into_safe_batches as _split_into_safe_batches,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    ConfigField,
    GradientInput,
    GradientOutput,
    InputField,
    ToolInstance,
    return_invalid_protein_chars,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

_VALID_GRADIENT_LOSS_KEYS = frozenset({"plddt", "ptm", "pae"})


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ESMFoldInput(StructurePredictionInput):
    """Input object for ESMFold structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins
    using ESMFold, a fast protein structure prediction model from Meta AI.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more protein chains. The linked length actually
            folded (summed chain residues plus the inter-chain ``chain_linker``,
            i.e. ``len(chain_linker) * (num_chains - 1)``) must not exceed 2,400.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.

    Note:
        ESMFold only supports protein sequences (amino acids). DNA, RNA, ligands,
        and glycans are not supported. Sequences can include 'X' for unknown amino
        acids. Entity types are automatically inferred if not provided. The 2,400
        residue limit is a hard constraint of the ESMFold architecture.
    """

    # ESMFold only supports proteins
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes", check_fields=False)
    @classmethod
    def validate_complexes(cls, complexes: list[StructurePredictionComplex]) -> list[StructurePredictionComplex]:
        """Ensures that complexes are valid inputs for ESMFold.

        Args:
            complexes (list[StructurePredictionComplex]): Complexes to validate.

        Checks:
        - Valid protein characters (including 'X' for unknown)
        - Bare chain-residue sum ≤ 2400 (coarse early guard; the linker-aware
          2,400 cap is enforced in ``prepare_complexes`` where the configured
          ``chain_linker`` is known)

        Note:
            Entity type validation (proteins only) is handled automatically
            by the base class using SUPPORTED_ENTITY_TYPES.
        """
        for comp_idx, comp in enumerate(complexes):
            # Validate characters
            for chain_idx, chain_seq in enumerate(comp.chain_sequences):
                invalid_chars = return_invalid_protein_chars(chain_seq, additional_valid_chars="X")
                if invalid_chars:
                    raise ValueError(
                        f"Invalid protein characters in complex {comp_idx}, chain {chain_idx}: "
                        f"{', '.join(sorted(invalid_chars))}"
                    )

            # Check ESMFold length limit
            if comp.sum_of_chain_lengths() > 2400:
                raise ValueError(f"Complex {comp_idx} too long ({comp.sum_of_chain_lengths()} positions, max 2400)")

        return complexes

    def prepare_complexes(self, chain_linker: str) -> list[dict[str, Any]]:
        """Prepares complexes for ESMFold inference."""
        prepared_complexes = []
        for comp_idx, comp in enumerate(self.complexes):
            chain_sequences = comp.chain_sequences  # SUPPORTED_ENTITY_TYPES rejects ligands upstream
            seq_lengths = [len(s) for s in chain_sequences]
            linked_seq = chain_linker.join(chain_sequences)
            # Enforce the 2,400 cap and batch budget against the linked length, not the bare sum.
            linked_len = len(linked_seq)
            if linked_len > 2400:
                raise ValueError(
                    f"Complex {comp_idx} too long ({linked_len} positions after linking "
                    f"{comp.num_chains()} chains with a {len(chain_linker)}-residue linker, max 2400)"
                )
            prepared_complexes.append(
                {
                    "complex_idx": comp_idx,
                    "chains": chain_sequences,
                    "linked_seq": linked_seq,
                    "seq_lengths": seq_lengths,
                    "total_residues": linked_len,
                    "num_chains": comp.num_chains(),
                }
            )
        return prepared_complexes


class ESMFoldMetrics(Metrics):
    """Per-structure metrics emitted by ESMFold prediction.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Average predicted LDDT score (0-1). Always present.
        ptm (float): Predicted TM-score (0-1). Depends on model output.
        avg_pae (float): Average predicted aligned error. Depends on model output.
        pae (list[list[float]]): Full per-residue PAE matrix in Å. Present when include_pae_matrix=True and model emits PAE.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "ptm": {"availability": "depends on model output", "type": "float", "min": 0.0, "max": 1.0},
        "avg_pae": {"availability": "depends on model output", "type": "float", "min": 0.0, "max": None},
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
        },
    }
    primary_metric: str | None = "avg_plddt"


# Output:
class ESMFoldOutput(StructurePredictionOutput):
    """ESMFold prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying an
            :class:`ESMFoldMetrics` instance on ``.metrics``.
    """


# Config:
class ESMFoldConfig(StructurePredictionConfig):
    """Configuration object for ESMFold structure prediction.

    This class defines configuration parameters for running ESMFold, a fast
    protein structure prediction model that does not require multiple sequence
    alignments (MSAs).

    Inherits from ``StructurePredictionConfig``.

    Attributes:
        residue_idx_offset (int): Residue numbering gap between chains in multi-chain
            structures. Used to ensure proper chain separation in the output PDB/CIF
            files. Higher values create larger gaps in residue numbering between
            chains. Must be at least 0. Default: 512.

        chain_linker (str): Amino acid sequence used to link chains internally for
            multi-chain prediction. ESMFold predicts multi-chain complexes by linking
            chains with a flexible linker sequence (typically glycines). The linker
            is removed in the final output. Default: 25 glycines (``"G" * 25``).

        max_batch_residues (int): Maximum total residues to process in a single
            inference batch. Used to prevent GPU memory overflow when predicting
            multiple structures. Complexes are automatically split into safe batches
            based on this limit. Must be at least 100. Default: 1200.

        num_recycles (int): Recycling iterations through the structure module.

        device (str): Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        include_pae_matrix (bool): Inherited. Default: ``False``.

        verbose: Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

    Note:
        ESMFold has a maximum total sequence length of 2,400 residues per complex.
        Unlike AlphaFold-based models, ESMFold does not use MSAs, making it much
        faster but potentially less accurate for some targets. A precomputed
        ``msas`` input is ignored and logs a single warning.
    """

    residue_idx_offset: int = ConfigField(
        title="Residue Index Offset",
        default=512,
        ge=0,
        description="Residue numbering gap between chains in multi-chain structures",
    )
    chain_linker: str = ConfigField(
        title="Chain Linker",
        default="G" * 25,
        description="Sequence to link chains (default: 25 glycines)",
    )
    max_batch_residues: int = ConfigField(
        title="Max Batch Residues",
        default=1200,
        ge=100,
        description="Maximum total residues per inference batch (to prevent GPU memory overflow)",
    )
    num_recycles: int = ConfigField(
        title="Recycling Iterations",
        default=4,
        ge=1,
        description="Iterative refinement passes through ESMFold. Higher = more accurate but slower.",
    )

    def preprocess(self, inputs: ESMFoldInput) -> ESMFoldInput:  # type: ignore[override]
        """Warn once if MSAs were supplied (ESMFold ignores them), then no-op."""
        # ESMFoldGradientConfig inherits this; its input has no msas field.
        msas = getattr(inputs, "msas", None)
        if msas:
            logger.warning(
                "ESMFold is single-sequence and does not use MSAs; the %d supplied MSA(s) were ignored.",
                len(msas),
            )
        return inputs


class ESMFoldGradientInput(GradientInput):
    """Input for differentiable ESMFold confidence scoring.

    Attributes:
        logits (list[list[float]]): Target-chain logits in proto amino-acid order.
        temperature (float): Softmax temperature for the target-chain relaxed sequence.
        chains (list[str]): Complete complex chain sequences. Entries listed in
            ``target_chain_indices`` are replaced by the hard decode of ``logits``
            before folding, but their lengths must match ``len(logits)``.
        target_chain_indices (list[int]): Chain positions that should receive
            the relaxed target logits. Repeated target segments in proto-language
            should pass each occurrence once; gradients are summed through the
            shared logits tensor.
    """

    chains: list[str] = InputField(
        title="Chains",
        description="Complete protein-chain sequences for the ESMFold complex.",
        examples=[["EVQLV"]],
    )
    target_chain_indices: list[int] = InputField(
        default=[0],
        title="Target Chain Indices",
        description="Zero-based chain indices that receive the relaxed input logits.",
    )

    @field_validator("chains")
    @classmethod
    def validate_chains(cls, chains: list[str]) -> list[str]:
        """Ensure chains are non-empty ESMFold-compatible protein sequences."""
        if not chains:
            raise ValueError("chains must contain at least one protein sequence")
        total_length = 0
        for idx, chain in enumerate(chains):
            if not chain:
                raise ValueError(f"chains[{idx}] must be non-empty")
            invalid = return_invalid_protein_chars(chain, additional_valid_chars="X")
            if invalid:
                raise ValueError(f"Invalid protein characters in chain {idx}: {', '.join(sorted(invalid))}")
            total_length += len(chain)
        if total_length > 2400:
            raise ValueError(f"ESMFold gradient input too long ({total_length} positions, max 2400)")
        return chains

    @model_validator(mode="after")
    def validate_target_chains(self) -> Self:
        """Target-chain indices must be in bounds and logits-length compatible."""
        if not self.target_chain_indices:
            raise ValueError("target_chain_indices must contain at least one index")
        if len(set(self.target_chain_indices)) != len(self.target_chain_indices):
            raise ValueError("target_chain_indices must not contain duplicate indices")
        bad = [idx for idx in self.target_chain_indices if idx < 0 or idx >= len(self.chains)]
        if bad:
            raise ValueError(f"target_chain_indices out of bounds for {len(self.chains)} chains: {bad}")
        logit_len = len(self.logits)
        mismatched = [idx for idx in self.target_chain_indices if len(self.chains[idx]) != logit_len]
        if mismatched:
            raise ValueError(
                f"target chains {mismatched} must have length {logit_len} to match logits; "
                f"got {[len(self.chains[idx]) for idx in mismatched]}"
            )
        return self


class ESMFoldGradientConfig(ESMFoldConfig):
    """Configuration for one differentiable ESMFold confidence pass.

    Attributes:
        include_pae_matrix (bool): Attach the full per-residue PAE matrix.
        residue_idx_offset (int): Residue numbering gap between linked chains.
        chain_linker (str): Sequence inserted between chains before folding.
        max_batch_residues (int): Maximum residues per ESMFold inference batch.
        num_recycles (int): Structure module recycling iterations.
        loss_weights (dict[str, float]): Weights for pLDDT, pTM, and pAE losses.
        soft (float): Soft probability blend for relaxed target sequence.
        hard (float): Straight-through hard-forward blend for relaxed target sequence.
        compute_gradient (bool): Whether to return the gradient with respect to logits.
    """

    loss_weights: dict[str, float] = ConfigField(
        default_factory=lambda: {"plddt": 1.0},
        title="Loss Weights",
        description="ESMFold confidence loss weights. Valid keys: plddt, ptm, pae.",
    )
    soft: float = ConfigField(
        title="Soft Mixing",
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend hard argmax one-hot (0) to softmax probabilities (1).",
    )
    hard: float = ConfigField(
        title="Hard Mixing",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Straight-through hard-forward coefficient.",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
    )

    @model_validator(mode="after")
    def validate_loss_weights(self) -> Self:
        """Reject unknown or negative confidence loss weights."""
        unknown_keys = set(self.loss_weights) - _VALID_GRADIENT_LOSS_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown loss_weights keys: {unknown_keys}. Valid keys: {sorted(_VALID_GRADIENT_LOSS_KEYS)}"
            )
        negative = {key: weight for key, weight in self.loss_weights.items() if weight < 0.0}
        if negative:
            raise ValueError(f"loss_weights must be non-negative; got {negative}")
        return self


class ESMFoldGradientOutput(GradientOutput):
    """Differentiable ESMFold confidence output.

    Attributes:
        gradient (list[list[float]] | None): Gradient matrix matching the input logits shape.
        loss (float): Scalar weighted confidence objective value.
        metrics (dict[str, Any]): Confidence metrics and per-term unweighted losses.
        vocab (list[str]): Amino-acid column ordering for logits and gradient.
        structure (Structure): Predicted ESMFold complex structure.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )
    structure: Structure = Field(
        title="Predicted Structure",
        description="Predicted ESMFold complex structure.",
    )

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write gradient JSON plus a PDB sidecar."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        pdb_path = base.parent / f"{base.name}.pdb"
        json_path = base.parent / f"{base.name}.json"
        self.structure.write_pdb(pdb_path)
        payload = self.model_dump(include={"gradient", "loss", "metrics", "vocab"}) | {"structure_pdb": pdb_path.name}
        json_path.write_text(json.dumps(payload, indent=2))


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESMFoldInput(complexes=["MKTL"])  # type: ignore[list-item]


def example_gradient_input() -> ESMFoldGradientInput:
    """Minimal valid input for gradient tool testing and examples."""
    from proto_tools.utils import one_hot_protein_logits

    return ESMFoldGradientInput(logits=one_hot_protein_logits("MKTL", sharpness=2.0), chains=["MKTL"])


@tool(
    key="esmfold-prediction",
    label="ESMFold Structure Prediction",
    category="structure_prediction",
    input_class=ESMFoldInput,
    config_class=ESMFoldConfig,
    output_class=ESMFoldOutput,
    metrics_class=ESMFoldMetrics,
    description="Protein structure prediction using ESMFold",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
)
def run_esmfold(
    inputs: ESMFoldInput,
    config: ESMFoldConfig,
    instance: Any = None,
) -> ESMFoldOutput:
    """Predict protein 3D structures using ESMFold.

    Uses ESMFold, a fast transformer-based protein structure prediction model from
    Meta AI, to predict 3D structures without requiring multiple sequence alignments.
    Supports local GPU execution with automatic batching for memory efficiency.

    Args:
        inputs (ESMFoldInput): Validated input containing one or more protein complexes
            to predict structures for. Each complex's linked length (chain residues
            plus inter-chain linkers) must be ≤ 2,400.
        config (ESMFoldConfig): Validated ESMFold configuration specifying chain linking,
            batching, and execution options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMFoldOutput: Structured output containing:
            - ``structures``: List of ``Structure`` instances, one per input complex
            - Each structure includes coordinates and the following confidence metrics:
                    avg_plddt: Average per-residue confidence (pLDDT) across all residues.
                        Range: 0.0-1.0 (normalized scale, unlike AlphaFold's 0-100). Interpretation:

                        - ``> 0.9``: Very high confidence
                        - ``0.7-0.9``: High confidence
                        - ``0.5-0.7``: Low confidence
                        - ``< 0.5``: Very low confidence

                        This is the primary quality metric for ESMFold predictions.

                    ptm: Predicted Template Modeling score measuring overall
                        structural accuracy. Range: 0.0-1.0. Higher values indicate better
                        predicted structures. May be ``None`` for some predictions.

    Raises:
        ValueError: If total residues exceed 2,400, if sequences contain invalid
            amino acids (except 'X'), if entity types are not protein, or if
            sequences are empty.
        RuntimeError: If model loading or prediction fails, or if GPU memory is
            insufficient even after batching.
        ImportError: If required dependencies (``esm``, ``torch``, ``biotite``) are not installed.

    See Also:
        - ESMFold paper: https://doi.org/10.1126/science.ade2574
        - ESM GitHub: https://github.com/facebookresearch/esm

    Example:
        >>> inputs = ESMFoldInput(complexes=["MVLSPADKTNVKAAW"])
        >>> config = ESMFoldConfig(verbose=True)
        >>> result = run_esmfold(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0].metrics.avg_plddt:.2f}")

    Note:
        - Maximum 2,400 linked residues per complex (chain residues plus
          inter-chain linkers; hard limit)
        - Multi-chain complexes are predicted by linking chains
    """
    # Prepare complexes for inference
    prepared_complexes = inputs.prepare_complexes(chain_linker=config.chain_linker)

    # Split into memory-safe sub-batches
    sub_batches = _split_into_safe_batches(
        prepared_complexes,
        max_residues=config.max_batch_residues,
    )

    logger.debug(f"Processing {len(prepared_complexes)} complex(es) in {len(sub_batches)} sub-batch(es)...")

    # Run inference on all prepared complexes
    all_results = []

    # Local GPU execution via venv subprocess
    logger.debug("Using local GPU for ESMFold structure prediction...")

    # Process each sub-batch through standalone script
    # Use tqdm to show progress over individual structures, not batches
    pbar = progress_bar(
        total=len(prepared_complexes),
        desc="Folding structures (ESMFold)",
        unit="structure",
    )

    for batch_idx, sub_batch in enumerate(sub_batches):
        if len(sub_batches) > 1:
            logger.debug(f"  Sub-batch {batch_idx + 1}/{len(sub_batches)}: {len(sub_batch)} complexes")

        # Prepare input data for inference script
        input_data = {
            "operation": "predict",
            "batch_data": sub_batch,
            "residue_idx_offset": config.residue_idx_offset,
            "chain_linker": config.chain_linker,
            "include_pae_matrix": config.include_pae_matrix,
            "num_recycles": config.num_recycles,
        }

        # Call the inference script
        input_data["device"] = config.device
        input_data["verbose"] = config.verbose
        output_data = ToolInstance.dispatch(
            "esmfold",
            input_data,
            instance=instance,
            config=config,
        )

        all_results.extend(output_data["results"])

        # Update progress bar by the number of structures in this batch
        pbar.update(len(sub_batch))

    pbar.close()

    # Post-process: relabel chains and convert to CIF
    structure_outputs = []
    for result, metadata in zip(all_results, prepared_complexes, strict=False):
        # Relabel chains (A, B, C, ...)
        pdb_output = _relabel_chains(result["pdb"], metadata["seq_lengths"])

        structure_outputs.append(
            Structure(
                structure=pdb_output,
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                metrics=ESMFoldMetrics(
                    avg_plddt=result["avg_plddt"],
                    ptm=result["ptm"],
                    avg_pae=result["avg_pae"],
                    pae=result.get("pae"),
                ),
                source="esmfold-prediction",
            )
        )

    return ESMFoldOutput(
        structures=structure_outputs,
        metadata={
            "num_complexes": len(structure_outputs),
            "total_chains": sum(s.num_chains for s in structure_outputs),
        },
    )


@tool(
    key="esmfold-gradient",
    label="ESMFold Gradient",
    category="structure_prediction",
    input_class=ESMFoldGradientInput,
    config_class=ESMFoldGradientConfig,
    output_class=ESMFoldGradientOutput,
    metrics_class=ESMFoldMetrics,
    description="Differentiable ESMFold confidence loss and gradient w.r.t. target-chain logits",
    uses_gpu=True,
    example_input=example_gradient_input,
    cacheable=False,
    stochastic=True,
)
def run_esmfold_gradient(
    inputs: ESMFoldGradientInput,
    config: ESMFoldGradientConfig,
    instance: Any = None,
) -> ESMFoldGradientOutput:
    """Run one differentiable ESMFold confidence pass.

    This is the gradient counterpart to :func:`run_esmfold`: one target-chain
    logit matrix is relaxed into ESMFold's sequence pathway, all requested
    confidence terms are summed into a single weighted loss, and one backward
    pass returns ``d(loss) / d(logits)``.
    """
    complex_input = ESMFoldInput(
        complexes=[
            StructurePredictionComplex(chains=[Chain(sequence=chain, entity_type="protein") for chain in inputs.chains])
        ]
    )
    prepared_complex = complex_input.prepare_complexes(chain_linker=config.chain_linker)[0]

    output_data = ToolInstance.dispatch(
        "esmfold",
        {
            "operation": "compute_gradient",
            "complex_data": prepared_complex,
            "logits": inputs.logits,
            "target_chain_indices": inputs.target_chain_indices,
            "temperature": inputs.temperature,
            "soft": config.soft,
            "hard": config.hard,
            "loss_weights": config.loss_weights,
            "compute_gradient": config.compute_gradient,
            "residue_idx_offset": config.residue_idx_offset,
            "chain_linker": config.chain_linker,
            "include_pae_matrix": config.include_pae_matrix,
            "num_recycles": config.num_recycles,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    pdb_output = _relabel_chains(output_data["pdb"], prepared_complex["seq_lengths"])
    metrics = output_data["metrics"]
    return ESMFoldGradientOutput(
        gradient=output_data["gradient"],
        loss=output_data["loss"],
        metrics=metrics,
        vocab=list(PROTEIN_AMINO_ACIDS),
        structure=Structure(
            structure=pdb_output,
            b_factor_type=BFactorType.NORMALIZED_PLDDT,
            metrics=ESMFoldMetrics(
                avg_plddt=metrics["avg_plddt"],
                ptm=metrics.get("ptm"),
                avg_pae=metrics.get("avg_pae"),
                pae=metrics.get("pae"),
            ),
            source="esmfold-gradient",
        ),
    )
