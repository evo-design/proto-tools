"""proto_tools/tools/structure_prediction/chai1/chai1.py.

Protein structure prediction using Chai1.
"""

import logging
import os
import tempfile
from typing import Any, ClassVar

from pydantic import field_validator

from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)

from proto_tools.entities.structures.structure import BFactorType, Structure, StructureMetrics
from proto_tools.tools.structure_prediction.chai1.helpers import (
    complex_to_fasta,
    write_msa_pqt,
)
from proto_tools.tools.structure_prediction.chai1.helpers import (
    hash_sequence as _hash_sequence,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    MetricSpec,
    MSAStructurePredictionConfig,
    StructurePredictionComplex,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance, extract_msa_sequences

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"


# ============================================================================
# Data Models
# ============================================================================
# Input:
class Chai1Input(StructurePredictionInput):
    """Input object for Chai1 structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins,
    ligands, and glycans using Chai1, a multi-modal structure prediction model.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, ligands, and/or glycans. Total
            length across all chains in a complex must not exceed 2,048 residues.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.

    Note:
        Chai1 supports entity types: ``"protein"``, ``"ligand"``, and ``"glycan"``.
        DNA and RNA are not supported. Entity types are automatically inferred if
        not explicitly provided. The 2,048 residue limit is a hard constraint of
        the Chai1 architecture.
    """

    # Chai1 supports proteins, ligands, and glycans (no DNA/RNA)
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "ligand", "glycan"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes")
    @classmethod
    def validate_sequence_length(cls, complexes: list[StructurePredictionComplex]) -> list[StructurePredictionComplex]:
        """Validate total sequence length doesn't exceed Chai1 limit (2048 residues)."""
        for comp_idx, comp in enumerate(complexes):
            if comp.sum_of_chain_lengths() > 2048:
                raise ValueError(f"Complex {comp_idx} too long ({comp.sum_of_chain_lengths()} positions, max 2048)")
        return complexes


# Output:
class Chai1Output(StructurePredictionOutput):
    """Chai-1 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures with confidence metrics.

    Metrics:
        avg_plddt (float): Average predicted LDDT score (0-1). Always present.
        ptm (float): Predicted TM-score (0-1). Always present.
        iptm (float): Interface predicted TM-score (0-1). Always present.
        avg_pae (float): Average predicted aligned error. Always present.
        confidence_score (float): Chai-1 confidence score. Always present.
    """

    METRICS: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
        "ptm": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
        "iptm": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
        "avg_pae": {"availability": "always", "type": float, "min": 0.0, "max": None},
        "confidence_score": {"availability": "always", "type": float, "min": 0.0, "max": 1.0},
    }
    PRIMARY_METRIC: ClassVar[str] = "avg_plddt"


# Config:
class Chai1Config(MSAStructurePredictionConfig):
    """Configuration object for Chai1 structure prediction.

    This class defines configuration parameters for running Chai1, a multi-modal
    structure prediction model supporting proteins, ligands, and glycans.

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        use_esm_embeddings (bool): Whether to use ESM (Evolutionary Scale Modeling)
            embeddings for improved predictions. ESM embeddings provide evolutionary
            context from large-scale protein language models, typically improving
            prediction quality. Default: ``True``.

        num_trunk_recycles (int): Number of iterative refinement passes through
            the trunk network. Higher values produce more refined structures but
            increase computation time. Typical range: 0-10. Must be at least 0.
            Default: 3.

        num_diffn_timesteps (int): Number of denoising steps in the diffusion process.
            Higher values produce more refined structures but are slower. Typical
            range: 100-500. Must be at least 1. Default: 200.

        num_diffn_samples (int): Number of independent structure samples to generate
            per complex via the diffusion process. Only the best sample (by confidence)
            is returned. Higher values explore more conformational space but increase
            computation time. Must be at least 1. Default: 1.

        num_trunk_samples (int): Number of independent trunk forward passes per
            diffusion sample. Increases diversity in structure generation. Must be
            at least 1. Default: 1.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig | None): Configuration for
            ColabFold MSA search. Only used when ``use_msa=True``. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``None``.

        device: Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose: Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

        timeout (int): Maximum execution time in seconds. Default: 1200.

    Note:
        Chai1 has a maximum total sequence length of 2,048 residues per complex.
        Higher refinement parameters (``num_trunk_recycles``, ``num_diffn_timesteps``)
        improve quality but significantly increase runtime.
    """

    use_esm_embeddings: bool = ConfigField(
        title="Use ESM Embeddings",
        default=True,
        description="Whether to use ESM embeddings for improved predictions",
        advanced=True,
    )

    num_trunk_recycles: int = ConfigField(
        title="Number of Trunk Recycles",
        default=3,
        ge=0,
        description="Number of iterative refinement passes through the trunk network",
        advanced=True,
    )
    num_diffn_timesteps: int = ConfigField(
        title="Number of Diffusion Timesteps",
        default=200,
        ge=1,
        description="Number of denoising steps in the diffusion process (higher=more refined structures but slower)",
        advanced=True,
    )
    num_diffn_samples: int = ConfigField(
        title="Number of Diffusion Samples",
        default=1,
        ge=1,
        description="Number of independent structure samples to generate (Only best is returned for each complex)",
        advanced=True,
    )
    num_trunk_samples: int = ConfigField(
        title="Number of Trunk Samples",
        default=1,
        ge=1,
        description="Number of independent trunk forward passes per diffusion sample",
        advanced=True,
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
    return Chai1Input(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="chai1-prediction",
    label="Chai1 Structure Prediction",
    category="structure_prediction",
    input_class=Chai1Input,
    config_class=Chai1Config,
    output_class=Chai1Output,
    description="Multi-modal structure prediction using Chai1",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
)
def run_chai1(inputs: Chai1Input, config: Chai1Config, instance: Any = None) -> Chai1Output:
    """Predict 3D structures using Chai1 multi-modal model.

    Uses Chai1, a diffusion-based model, to predict 3D structures of proteins,
    ligands, glycans, and their complexes. Runs via local GPU execution in
    isolated Python environments.

    Args:
        inputs (Chai1Input): Validated input containing one or more complexes to
            predict structures for. Each complex must be ≤ 2,048 residues total.
        config (Chai1Config): Validated Chai1 configuration specifying ESM embeddings,
            MSA settings, refinement parameters, and execution options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Chai1Output: Structured output containing:
            - ``structures``: List of ``ChaiStructure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:
                avg_plddt: Average per-residue confidence (pLDDT) across all residues.
                    Range: 0-1. Interpretation:

                    - ``> 0.9``: Very high confidence
                    - ``0.7-0.9``: High confidence
                    - ``0.5-0.7``: Low confidence
                    - ``< 0.5``: Very low confidence

                    This is the primary quality metric for Chai1 predictions.

                ptm: Predicted Template Modeling score measuring overall
                    structural accuracy. Range: 0.0-1.0. Higher values indicate better
                    predicted structures. May be ``None`` for some predictions.

                iptm: Interface PTM score measuring confidence in inter-chain
                    interfaces. Range: 0.0-1.0. Higher values indicate more confident
                    predictions of chain-chain interactions. Only meaningful for multi-chain
                    complexes. May be ``None`` for single-chain predictions.

                confidence_score: Overall confidence score combining
                    multiple quality metrics. Higher values indicate more reliable predictions.
                    May be ``None`` if not computed.

    Raises:
        ValueError: If total residues exceed 2,048, if entity types are invalid
            (only ``"protein"``, ``"ligand"``, ``"glycan"`` supported), or if
            sequences are empty.
        RuntimeError: If model loading, embedding generation, or prediction fails.
        ImportError: If required dependencies (``chai-lab``, ``torch``) are not installed.

    See Also:
        - Chai1 GitHub: https://github.com/chaidiscovery/chai-lab
        - Chai1 paper: https://www.biorxiv.org/content/10.1101/2024.10.10.615955

    Example:
        >>> inputs = Chai1Input(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
        >>> config = Chai1Config(use_esm_embeddings=True, num_trunk_recycles=3, verbose=True)
        >>> result = run_chai1(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0].avg_plddt:.2f}")

    Note:
        - Chai1 processes each complex independently and sequentially
        - ESM embeddings generally improve prediction quality
        - Does not support DNA or RNA (use Boltz2 for nucleic acids)
    """
    results = [
        run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)
        for comp in progress_bar(
            inputs.complexes, desc="Folding structures (Chai-1)", unit="complex", total=len(inputs.complexes)
        )
    ]
    return Chai1Output(
        structures=results,
    )


def _msa_to_pqt_file(msa: Any, pqt_path: str, query_index: int = 0, source_database: str = "uniref90") -> None:
    """Write an MSA object to Chai1's .aligned.pqt Parquet format."""
    sequences, seq_ids = extract_msa_sequences(msa, query_index)
    write_msa_pqt(sequences, pqt_path, source_database=source_database, comments=seq_ids)


def _generate_fasta_content(comp: StructurePredictionComplex) -> str:
    """Generate FASTA content from a typed StructurePredictionComplex."""
    chain_dicts = [{"entity_type": chain.entity_type, "sequence": chain.sequence} for chain in comp.chains]
    return complex_to_fasta(chain_dicts)


def run_chai1_on_complex(
    comp: StructurePredictionComplex,
    config: Chai1Config,
    msas: dict[str, Any] | None = None,
    instance: Any = None,
) -> Structure:
    """Run Chai1 structure prediction on a single complex. This function is wrapped.

    by ``run_chai1`` to sequentially predict all complexes in the input.
    """
    # Local GPU execution via venv subprocess
    logger.debug("Using local GPU for Chai1 structure prediction...")

    # Create temporary directory for inputs and outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate FASTA content
        fasta_content = _generate_fasta_content(comp)

        # Write input file
        input_file = os.path.join(temp_dir, "input.fasta")
        with open(input_file, "w") as f:
            f.write(fasta_content)

        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Write pre-computed MSAs to Parquet files
        msa_directory = None
        if config.use_msa and msas:
            pqt_dir = os.path.join(temp_dir, "msa_pqt")
            os.makedirs(pqt_dir, exist_ok=True)
            for chain in comp.chains:
                if chain.entity_type == "protein" and chain.sequence in msas:
                    seq_hash = _hash_sequence(chain.sequence.upper())
                    pqt_path = os.path.join(pqt_dir, f"{seq_hash}.aligned.pqt")
                    _msa_to_pqt_file(
                        msa=msas[chain.sequence],
                        pqt_path=pqt_path,
                        query_index=0,
                    )
                    if config.verbose:
                        logger.info(f"Assigned MSA to protein chain ({len(msas[chain.sequence])} sequences)")
            if os.listdir(pqt_dir):
                msa_directory = pqt_dir

        # Prepare input data for inference script
        input_data = {
            "fasta_file": input_file,
            "output_dir": output_dir,
            "use_esm_embeddings": config.use_esm_embeddings,
            "msa_directory": msa_directory,
            "num_trunk_recycles": config.num_trunk_recycles,
            "num_diffn_timesteps": config.num_diffn_timesteps,
            "num_diffn_samples": config.num_diffn_samples,
            "num_trunk_samples": config.num_trunk_samples,
            "seed": config.seed,
        }

        # Call the inference script with the venv activated
        input_data["device"] = config.device
        input_data["verbose"] = config.verbose
        result = ToolInstance.dispatch(
            "chai1",
            input_data,
            instance=instance,
            config=config,
        )

        cif_output = result["cif_output"]

    return Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=StructureMetrics(primary_metric="avg_plddt", **result["metrics"]),
        source="chai1-prediction",
    )
