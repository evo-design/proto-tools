"""proto_tools/tools/structure_prediction/boltz2/boltz2.py.

Protein structure prediction using Boltz2.

Example:
    >>> from proto_tools.tools.structure_prediction.boltz2 import run_boltz2, Boltz2Config
    >>> config = Boltz2Config(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
    >>> result = run_boltz2(config)
    >>> print(f"Confidence: {result.confidence_score:.2f}")
"""

import os
import tempfile
import warnings
from logging import getLogger
from typing import Any, ClassVar

from proto_tools.entities.structures import BFactorType, Structure, StructureMetrics
from proto_tools.tools.structure_prediction.boltz2.helpers import (
    complex_to_yaml,
    write_msa_csv,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    MetricSpec,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance, extract_msa_sequences
from proto_tools.utils.progress import progress_bar

logger = getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class Boltz2Input(StructurePredictionInput):
    """Input object for Boltz2 structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins,
    DNA, RNA, and ligands using Boltz2, a multi-modal structure prediction model.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, DNA, RNA, and/or ligands.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.

    Note:
        Boltz2 supports entity types: ``"protein"``, ``"dna"``, ``"rna"``, and ``"ligand"``.
        Entity types are automatically inferred if not explicitly provided. Invalid
        entity types will raise a validation error.
    """

    # Boltz2 supports all standard entity types except glycan
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False


# Output:
class Boltz2Output(StructurePredictionOutput):
    """Boltz2 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures with confidence metrics.

    Metrics:
        confidence_score (float): Overall confidence score. Always present.
        ptm (float): Predicted TM-score. Always present.
        iptm (float): Interface predicted TM-score. Always present.
        chains_ptm (list[float]): Per-chain pTM scores. Always present.
        pair_chains_iptm (list[list[float]]): Pairwise chain ipTM scores. Always present.
        ligand_iptm (float): Ligand interface pTM. Depends on complex composition.
        protein_iptm (float): Protein interface pTM. Depends on complex composition.
        complex_plddt (float): Complex predicted LDDT. Depends on complex composition.
        complex_iplddt (float): Complex interface predicted LDDT. Depends on complex composition.
        complex_pde (float): Complex predicted distance error. Depends on complex composition.
        complex_ipde (float): Complex interface PDE. Depends on complex composition.
    """

    METRICS: ClassVar[dict[str, MetricSpec]] = {
        "confidence_score": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
        "ptm": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
        "iptm": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
        "chains_ptm": {"availability": "always", "type": list, "min": 0.0, "max": 1.0},
        "pair_chains_iptm": {"availability": "always", "type": list, "min": 0.0, "max": 1.0},
        "ligand_iptm": {"availability": "depends on complex composition", "type": float, "min": 0.0, "max": 1.0},
        "protein_iptm": {"availability": "depends on complex composition", "type": float, "min": 0.0, "max": 1.0},
        "complex_plddt": {"availability": "depends on complex composition", "type": float, "min": 0.0, "max": 1.0},
        "complex_iplddt": {"availability": "depends on complex composition", "type": float, "min": 0.0, "max": 1.0},
        "complex_pde": {"availability": "depends on complex composition", "type": float, "min": 0.0, "max": None},
        "complex_ipde": {"availability": "depends on complex composition", "type": float, "min": 0.0, "max": None},
    }
    PRIMARY_METRIC: ClassVar[str] = "confidence_score"


# Config:
class Boltz2Config(MSAStructurePredictionConfig):
    """Configuration object for Boltz2 structure prediction.

    This class defines configuration parameters for running Boltz2, a multi-modal
    structure prediction model supporting proteins, DNA, RNA, and ligands.

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        recycling_steps (int): Number of iterative refinement passes through the
            model. Higher values produce more refined structures but increase
            computation time. Typical range: 3-20. Must be at least 0.
            Default: 10.

        sampling_steps (int): Number of denoising steps in the diffusion process.
            Higher values produce more refined structures but are slower. Typical
            range: 100-500. Must be at least 1. Default: 200.

        diffusion_samples (int): Number of independent structure samples to generate
            per complex. Only the best sample (by confidence score) is returned.
            Higher values explore more of the conformational space but increase
            computation time. Must be at least 1. Default: 25.

        num_workers (int): Number of CPU workers for parallel processing during
            prediction. Automatically set to the minimum of available CPU cores or 4.
            Must be at least 1. Default: ``min(cpu_count, 4)``.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig | None): Configuration for
            ColabFold MSA search. Only used when ``use_msa=True``. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``None``.

        verbose: Whether to print status messages during execution including
            MSA generation, model loading, and prediction progress. Inherited from
            ``StructurePredictionConfig``. Default: ``False``.

        timeout (int): Maximum execution time in seconds. Default: 1200.

    """

    recycling_steps: int = ConfigField(
        title="Number of Recycling Steps",
        default=10,
        ge=0,
        description="Number of iterative refinement passes (higher=more refined structures but slower)",
        advanced=True,
    )
    sampling_steps: int = ConfigField(
        title="Number of Sampling Steps",
        default=200,
        ge=1,
        description="Number of denoising steps in the diffusion process (higher=more refined but slower)",
        advanced=True,
    )
    diffusion_samples: int = ConfigField(
        title="Number of Diffusion Samples",
        default=25,
        ge=1,
        description="Number of independent structure samples to generate (Only best is returned for each complex)",
        advanced=True,
    )
    num_workers: int = ConfigField(
        title="Number of Workers",
        default=min(os.cpu_count() or 4, 4),
        ge=1,
        description="Number of workers for prediction",
        hidden=True,
    )
    timeout: int = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds",
        hidden=True,
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return Boltz2Input(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="boltz2-prediction",
    label="Boltz2 Structure Prediction",
    category="structure_prediction",
    input_class=Boltz2Input,
    config_class=Boltz2Config,
    output_class=Boltz2Output,
    description="Multi-modal structure prediction using Boltz2",
    uses_gpu=True,
    device_count="1-2",
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
)
def run_boltz2(inputs: Boltz2Input, config: Boltz2Config, instance: Any = None) -> Boltz2Output:
    """Predict 3D structures using Boltz2 multi-modal model.

    Uses Boltz2, a diffusion-based deep learning model, to predict 3D structures
    of proteins, DNA, RNA, ligands, and their complexes. Runs via local GPU
    execution in isolated Python environments.

    Args:
        inputs (Boltz2Input): Validated input containing one or more complexes to
            predict structures for.
        config (Boltz2Config): Validated Boltz2 configuration specifying MSA settings,
            refinement parameters, and execution options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Boltz2Output: Structured output containing:
            - ``structures``: List of ``Boltz2Structure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:
                    confidence_score: Primary ranking score used to select the best
                        structure from multiple samples. For multi-chain complexes, this is
                        ``iptm``; for single chains, this is ``ptm``. Range: 0.0-1.0 (higher
                        is better).

                    ptm: Predicted Template Modeling score measuring overall structural
                        accuracy. Range: 0.0-1.0. Interpretation:

                        - ``> 0.7``: High quality structure
                        - ``0.5-0.7``: Moderate quality
                        - ``< 0.5``: Low confidence

                    iptm: Interface PTM score measuring confidence in inter-chain
                        interfaces and relative orientations. Range: 0.0-1.0. Interpretation:

                        - ``> 0.85``: High confidence in interface
                        - ``0.7-0.85``: Moderate confidence
                        - ``< 0.7``: Low confidence

                    ligand_iptm: Protein-ligand interface PTM score. Only
                        present for complexes containing ligands. Range: 0.0-1.0. Higher values
                        indicate more confident protein-ligand binding predictions.

                    protein_iptm: Protein-protein interface PTM score. Only
                        present for multi-protein complexes. Range: 0.0-1.0. Higher values
                        indicate more confident protein-protein interactions.

                    complex_plddt: Average per-residue confidence (pLDDT)
                        across all residues in the complex. Range: 0-1. Interpretation:

                        - ``> 0.9``: Very high confidence
                        - ``0.7-0.9``: High confidence
                        - ``0.5-0.7``: Low confidence
                        - ``< 0.5``: Very low confidence

                    complex_iplddt: Average pLDDT for interface residues only.
                        Range: 0-1. Useful for assessing confidence specifically in the
                        interaction regions.

                    complex_pde: Average Predicted Aligned Error (PAE) in
                        Angstroms across all residue pairs. Lower values indicate more confident
                        relative positioning. From 0 to 31.75 Å.

                    complex_ipde: PAE for interface residue pairs only, in
                        Angstroms. Lower values indicate more confident interface geometry.

                    chains_ptm: Individual PTM scores for each chain.
                        Useful for identifying which chains are predicted with high vs. low
                        confidence.

                    pair_chains_iptm: Pairwise ipTM scores between
                        all chain pairs. Shape: ``(num_chains, num_chains)``. Symmetric matrix
                        with diagonal values representing self-interactions.

    See Also:
        - Boltz2 GitHub: https://github.com/jwohlwend/boltz
        - Boltz2 paper: https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
        - Boltz2 Website: https://boltz.bio/boltz2

    Example:
        >>> inputs = Boltz2Input(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
        >>> config = Boltz2Config(recycling_steps=10, sampling_steps=200, diffusion_samples=25, verbose=True)
        >>> result = run_boltz2(inputs, config)
        >>> print(f"Confidence: {result.structures[0].confidence_score:.2f}")

    Note:
        - Boltz2 processes each complex independently and sequentially
        - MSA generation modes:
            - ``use_msa=False``: Single-sequence mode without MSAs
            - ``use_msa=True`` (default): Use ColabFold search tool for MSA generation
        - Higher ``recycling_steps`` and ``sampling_steps`` improve quality but increase runtime
        - Supports both local and remote ColabFold search modes when ``use_msa=True``
    """
    results = [
        run_boltz2_on_complex(
            config=config,
            sp_complex=comp,
            msas=inputs.msas,
            instance=instance,
        )
        for comp in progress_bar(
            inputs.complexes, desc="Folding structures (Boltz-2)", unit="complex", total=len(inputs.complexes)
        )
    ]
    return Boltz2Output(structures=results)


def _msa_to_csv_file(msa: Any, csv_path: str, query_index: int = 0) -> None:
    """Write an MSA object to Boltz's CSV format with pairing keys."""
    sequences, _ids = extract_msa_sequences(msa, query_index)
    write_msa_csv(sequences, csv_path)


def run_boltz2_on_complex(
    config: Boltz2Config,
    sp_complex: Any,
    msas: dict[str, Any] | None = None,
    instance: Any = None,
) -> Structure:
    """Run Boltz2 structure prediction on a single complex. This function is wrapped.

    by ``run_boltz2`` to sequentially predict all complexes in the input.

    Args:
        config (Boltz2Config): Boltz2 configuration
        sp_complex (Any): StructurePredictionComplex instance containing chain information
        msas (dict[str, Any] | None): Pre-computed MSAs keyed by protein sequence
        instance (Any): Optional ToolInstance for persistent execution
    """
    if config.verbose:
        logger.info("Using local GPU for Boltz2 structure prediction...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "boltz2_output")
        os.makedirs(output_dir)

        # Build chain_msa_paths for complex_to_yaml
        chain_msa_paths: dict[str, str] | None = None
        if config.use_msa:
            chain_msa_paths = {}
            protein_seqs, protein_chain_ids = sp_complex.extract_protein_chains()
            if protein_seqs and msas:
                msa_dir = os.path.join(temp_dir, "msas")
                os.makedirs(msa_dir, exist_ok=True)
                for seq, chain_id in zip(protein_seqs, protein_chain_ids, strict=False):
                    if seq in msas:
                        csv_path = os.path.join(msa_dir, f"{chain_id}.csv")
                        _msa_to_csv_file(msa=msas[seq], csv_path=csv_path, query_index=0)
                        chain_msa_paths[chain_id] = csv_path
                        if config.verbose:
                            logger.info(f"Assigned MSA to chain {chain_id} ({len(msas[seq])} sequences)")

            # Warn for protein chains without MSAs
            for chain_id in protein_chain_ids:
                if chain_id not in chain_msa_paths:
                    warnings.warn(
                        f"No homologs found for chain {chain_id} - setting msa='empty'.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Generate YAML directly with MSA paths baked in
        chain_dicts = [{"entity_type": chain.entity_type, "sequence": chain.sequence} for chain in sp_complex.chains]
        yaml_content = complex_to_yaml(chain_dicts, chain_msa_paths=chain_msa_paths)

        input_yaml_path = os.path.join(temp_dir, "boltz2_input.yaml")
        with open(input_yaml_path, "w") as f:
            f.write(yaml_content)

        # Prepare input data for inference script
        input_data = {
            "input_yaml_path": str(input_yaml_path),
            "output_dir": str(output_dir),
            "recycling_steps": config.recycling_steps,
            "sampling_steps": config.sampling_steps,
            "diffusion_samples": config.diffusion_samples,
            "num_workers": config.num_workers,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.resolved_seed,
        }

        # Call the inference script
        output_data = ToolInstance.dispatch(
            "boltz2",
            input_data,
            instance=instance,
            config=config,
        )

        cif_output = output_data["structure_cif_output"]
        formatted_metrics = output_data["metrics"]

    # Extract metrics
    metrics_dict: dict[str, Any] = {
        "confidence_score": float(formatted_metrics["confidence_score"]),
        "ptm": float(formatted_metrics["ptm"]),
        "iptm": float(formatted_metrics["iptm"]),
        "chains_ptm": formatted_metrics["chains_ptm"],
        "pair_chains_iptm": formatted_metrics["pair_chains_iptm"],
    }
    for metric in ("ligand_iptm", "protein_iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
        if metric in formatted_metrics:
            metrics_dict[metric] = float(formatted_metrics[metric])

    return Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=StructureMetrics(primary_metric="confidence_score", **metrics_dict),
        source="boltz2-prediction",
    )
