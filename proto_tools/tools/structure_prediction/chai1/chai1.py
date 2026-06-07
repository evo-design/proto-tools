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

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_prediction.chai1.helpers import (
    complex_to_fasta,
    count_chai1_tokens,
    write_msa_pqt,
)
from proto_tools.tools.structure_prediction.chai1.helpers import (
    hash_sequence as _hash_sequence,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    normalize_output_chain_ids,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance, extract_msa_sequences
from proto_tools.utils.tool_io import Metrics, MetricSpec

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
        complexes (list[Complex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, ligands, and/or glycans. Total
            token count per complex must not exceed 2,048 (see ``Note`` below).
        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one
            entry per complex. Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by
            chain index); ``paired=True`` marks rows taxonomy-aligned across chains. Populated by preprocess() or supplied directly.
            Default: None.

    Note:
        Chai1 supports entity types: ``"protein"``, ``"ligand"``, and ``"glycan"``.
        DNA and RNA are not supported. Entity types are automatically inferred if
        not explicitly provided. The 2,048-token cap is a hard model constraint;
        ligands and glycans cost 1 token per heavy atom.
    """

    # Chai1 supports proteins, ligands, and glycans (no DNA/RNA)
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "ligand", "glycan"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes")
    @classmethod
    def validate_token_count(cls, complexes: list[Complex]) -> list[Complex]:
        """Reject complexes that exceed Chai-1's 2048-token limit."""
        for comp_idx, comp in enumerate(complexes):
            n_tokens = count_chai1_tokens(comp.chains)
            if n_tokens > 2048:
                raise ValueError(f"Complex {comp_idx} has {n_tokens} tokens, exceeding Chai-1's 2048-token limit.")
        return complexes


class Chai1Metrics(Metrics):
    """Per-structure metrics emitted by Chai-1 prediction.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Average predicted LDDT score (0-1). Always present.
        ptm (float): Predicted TM-score (0-1). Always present.
        iptm (float): Interface predicted TM-score (0-1). Always present.
        avg_pae (float): Average predicted aligned error. Always present.
        pae (list[list[float]]): Full per-residue PAE matrix in Å. Present when include_pae_matrix=True.
        confidence_score (float): Chai-1 confidence score. Always present.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "avg_pae": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
            "better_values_are": "lower",
        },
        "confidence_score": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = "avg_plddt"


# Output:
class Chai1Output(StructurePredictionOutput):
    """Chai-1 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying a
            :class:`Chai1Metrics` instance on ``.metrics``.
    """


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
            prediction quality. Independent of ``use_msa``; both can be enabled
            together and Chai-1 conditions on the ESM embeddings and the MSA
            simultaneously. Default: ``True``.

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
            computation time. Must be at least 1. Default: 5.

        num_trunk_samples (int): Number of independent trunk forward passes per
            diffusion sample. Increases diversity in structure generation. Must be
            at least 1. Default: 1.

        low_memory (bool): Stream MSA + template features per sample to reduce
            peak GPU memory at the cost of speed. Default: True.

        recycle_msa_subsample (int): Stochastically subsample MSA across recycles
            for diversity. 0 disables (default).

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using MMseqs2 homology search. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        pair_heterocomplex_msas (bool): Whether heterocomplex protein chains
            should use taxonomy-paired MSA generation. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        msa_search_config (Mmseqs2HomologySearchConfig | None): Configuration for
            MMseqs2 homology search (MSA generation). Only used when ``use_msa=True``.
            Inherited from ``MSAStructurePredictionConfig``. Default: ``None``.

        include_pae_matrix (bool): Inherited. Default: ``False``.

        device: Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose: Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

        timeout (int | None): Maximum execution time in seconds. ``None`` waits indefinitely. Default: 1200.

    Note:
        Chai-1 caps each complex at 2,048 tokens (1 per amino acid, 1 per heavy
        atom for ligands and glycans). Higher refinement parameters
        (``num_trunk_recycles``, ``num_diffn_timesteps``) improve quality but
        significantly increase runtime.
    """

    use_esm_embeddings: bool = ConfigField(
        title="Use ESM Embeddings",
        default=True,
        description="Whether to use ESM embeddings for improved predictions",
    )

    num_trunk_recycles: int = ConfigField(
        title="Number of Trunk Recycles",
        default=3,
        ge=0,
        description="Iterative refinement passes through the trunk network. Higher = more accurate but slower.",
    )
    num_diffn_timesteps: int = ConfigField(
        title="Number of Diffusion Timesteps",
        default=200,
        ge=1,
        description="Denoising steps in the diffusion process. Higher = more refined but slower.",
    )
    num_diffn_samples: int = ConfigField(
        title="Number of Diffusion Samples",
        default=5,
        ge=1,
        description="Structure samples per complex; best by confidence is kept. Higher = more thorough but slower.",
    )
    num_trunk_samples: int = ConfigField(
        title="Number of Trunk Samples",
        default=1,
        ge=1,
        description="Independent trunk forward passes per diffusion sample (adds sample diversity).",
    )
    low_memory: bool = ConfigField(
        title="Low Memory Mode",
        default=True,
        description="Stream features per sample to reduce peak GPU memory at the cost of speed.",
    )
    recycle_msa_subsample: int = ConfigField(
        title="Recycle MSA Subsample",
        default=0,
        ge=0,
        description="Randomly subsample the MSA across recycles for diversity. 0 disables.",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds",
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
    metrics_class=Chai1Metrics,
    description="Multi-modal structure prediction using Chai1",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
)
def run_chai1(inputs: Chai1Input, config: Chai1Config, instance: Any = None) -> Chai1Output:
    """Predict 3D structures using Chai1 multi-modal model.

    Uses Chai1, a diffusion-based model, to predict 3D structures of proteins,
    ligands, glycans, and their complexes. Runs via local GPU execution in
    isolated Python environments.

    Args:
        inputs (Chai1Input): Validated input containing one or more complexes to
            predict structures for. Each complex must be ≤ 2,048 tokens total.
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
        ValueError: If total tokens exceed 2,048, if entity types are invalid
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
        >>> print(f"Average pLDDT: {result.structures[0].metrics.avg_plddt:.2f}")

    Note:
        - Chai1 processes each complex independently and sequentially
        - ESM embeddings generally improve prediction quality
        - Does not support DNA or RNA (use Boltz2 for nucleic acids)
    """
    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advance the seed per complex so duplicate inputs get distinct seeds.
    results = [
        run_chai1_on_complex(
            comp=comp,
            config=config,
            complex_msas=inputs.msas[dispatch_idx] if inputs.msas else None,
            instance=instance,
            seed=base_seed + dispatch_idx,
        )
        for dispatch_idx, comp in enumerate(
            progress_bar(
                inputs.complexes, desc="Folding structures (Chai-1)", unit="complex", total=len(inputs.complexes)
            )
        )
    ]
    return Chai1Output(
        structures=results,
    )


def _msa_to_pqt_file(
    msa: Any,
    pqt_path: str,
    query_index: int = 0,
    source_database: str = "uniref90",
    paired: bool = False,
) -> None:
    """Write an MSA object to Chai1's .aligned.pqt Parquet format.

    When ``paired=True``, each non-query row receives a pairing_key equal to its
    row index so chai_lab matches rows across chains by index.
    """
    sequences, seq_ids = extract_msa_sequences(msa, query_index)
    pairing_keys = None
    if paired:
        # Row 0 is the query; chai_lab pairs rows whose pairing_key is non-empty and equal.
        pairing_keys = [""] + [str(i) for i in range(1, len(sequences))]
    write_msa_pqt(
        sequences,
        pqt_path,
        source_database=source_database,
        comments=seq_ids,
        pairing_keys=pairing_keys,
    )


def _generate_fasta_content(comp: Complex) -> str:
    """Generate FASTA content from a typed Complex."""
    return complex_to_fasta(comp.chains)


def run_chai1_on_complex(
    comp: Complex,
    config: Chai1Config,
    complex_msas: Any = None,
    instance: Any = None,
    seed: int | None = None,
) -> Structure:
    """Run Chai1 structure prediction on a single complex. This function is wrapped.

    by ``run_chai1`` to sequentially predict all complexes in the input.

    Args:
        comp (Complex): The complex to fold.
        config (Chai1Config): Chai1 configuration.
        complex_msas (Any): Pre-computed ``ComplexMSAs`` for this complex, keyed by
            chain index. Row N across chains pairs by taxonomy when ``paired``.
        instance (Any): Optional ToolInstance for persistent execution.
        seed (int | None): Per-complex seed forwarded to the standalone. When ``None``,
            falls back to ``config.seed``.
    """
    if seed is None:
        seed = config.seed
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

        from proto_tools.tools.structure_prediction.shared_data_models import unwrap_complex_msas

        per_chain_msas, is_paired = unwrap_complex_msas(complex_msas)

        # Honor MSAs whenever present: auto-generated when use_msa=True, or supplied
        # by the caller (always respected regardless of use_msa).
        msa_directory = None
        if per_chain_msas:
            pqt_dir = os.path.join(temp_dir, "msa_pqt")
            os.makedirs(pqt_dir, exist_ok=True)
            for ch_idx, chain in enumerate(comp.chains):
                if not isinstance(chain, Chain) or chain.entity_type != "protein":
                    continue
                msa = per_chain_msas.get(ch_idx)
                if msa is None:
                    continue
                seq_hash = _hash_sequence(chain.sequence.upper())
                pqt_path = os.path.join(pqt_dir, f"{seq_hash}.aligned.pqt")
                # Only set pairing_keys when MSAs are actually row-aligned across chains.
                _msa_to_pqt_file(
                    msa=msa,
                    pqt_path=pqt_path,
                    query_index=0,
                    paired=is_paired,
                )
                if config.verbose:
                    logger.info(f"Assigned MSA to protein chain {ch_idx} ({len(msa)} sequences)")
            if os.listdir(pqt_dir):
                msa_directory = pqt_dir

        # Prepare input data for inference script
        input_data = {
            "operation": "predict",
            "fasta_file": input_file,
            "output_dir": output_dir,
            "use_esm_embeddings": config.use_esm_embeddings,
            "msa_directory": msa_directory,
            "num_trunk_recycles": config.num_trunk_recycles,
            "num_diffn_timesteps": config.num_diffn_timesteps,
            "num_diffn_samples": config.num_diffn_samples,
            "num_trunk_samples": config.num_trunk_samples,
            "low_memory": config.low_memory,
            "recycle_msa_subsample": config.recycle_msa_subsample,
            "seed": seed,
            "include_pae_matrix": config.include_pae_matrix,
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

    structure = Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=Chai1Metrics(**result["metrics"]),
        source="chai1-prediction",
    )
    return normalize_output_chain_ids(structure, comp.chains)
