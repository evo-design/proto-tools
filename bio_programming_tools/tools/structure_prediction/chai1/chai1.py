"""
chai1.py

Protein structure prediction using Chai1.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import List, Optional

from pydantic import field_validator, model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)

from bio_programming_tools.entities.structures.structure import BFactorType, Structure
from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    run_colabfold_search,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField
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
        complexes (List[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, ligands, and/or glycans. Total
            length across all chains in a complex must not exceed 2,048 residues.

    Note:
        Chai1 supports entity types: ``"protein"``, ``"ligand"``, and ``"glycan"``.
        DNA and RNA are not supported. Entity types are automatically inferred if
        not explicitly provided. The 2,048 residue limit is a hard constraint of
        the Chai1 architecture.
    """

    # Chai1 supports proteins, ligands, and glycans (no DNA/RNA)
    SUPPORTED_ENTITY_TYPES = {"protein", "ligand", "glycan"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes")
    @classmethod
    def validate_sequence_length(
        cls, complexes: List[StructurePredictionComplex]
    ) -> List[StructurePredictionComplex]:
        """Validate total sequence length doesn't exceed Chai1 limit (2048 residues)."""

        for comp_idx, comp in enumerate(complexes):
            if comp.sum_of_chain_lengths() > 2048:
                raise ValueError(
                    f"Complex {comp_idx} too long ({comp.sum_of_chain_lengths()} positions, max 2048)"
                )
        return complexes

# Output:
Chai1Output = StructurePredictionOutput

# Config:
class Chai1Config(StructurePredictionConfig):
    """Configuration object for Chai1 structure prediction.

    This class defines configuration parameters for running Chai1, a multi-modal
    structure prediction model supporting proteins, ligands, and glycans.

    Inherits from ``StructurePredictionConfig``.

    Attributes:
        use_esm_embeddings (bool): Whether to use ESM (Evolutionary Scale Modeling)
            embeddings for improved predictions. ESM embeddings provide evolutionary
            context from large-scale protein language models, typically improving
            prediction quality. Default: ``True``.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. If ``False``, runs without MSAs.
            Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig): Configuration for ColabFold
            MSA search. Controls search mode (local/remote), database paths, and other
            MSA generation parameters. Only used when ``use_msa=True``.
            Default: Uses ColabfoldSearchConfig defaults.

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

        seed (Optional[int]): Random seed for reproducible results. Set to a fixed
            value for deterministic predictions or ``None`` for random behavior.
            Default: 42.

        device (str): Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

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

    use_msa: bool = ConfigField(
        title="Use MSA",
        default=True,
        description="Whether to generate and use MSAs for protein chains using ColabFold search",
    )

    colabfold_search_config: Optional[ColabfoldSearchConfig] = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description="Nested configuration for ColabFold MSA search. If None, uses default settings.",
        hidden=True,
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
    seed: Optional[int] = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed for reproducible results",
        hidden=True,
    )

    @model_validator(mode="after")
    def sync_nested_config(self):
        """Sync verbose flag with nested colabfold_search_config."""
        if self.colabfold_search_config is None:
            self.colabfold_search_config = ColabfoldSearchConfig()
        self.colabfold_search_config.verbose = self.verbose
        return self

# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return Chai1Input(complexes=["MKTL"])


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
def run_chai1(inputs: Chai1Input, config: Chai1Config | None = None, instance=None) -> Chai1Output:
    """Predict 3D structures using Chai1 multi-modal model.

    Uses Chai1, a diffusion-based model, to predict 3D structures of proteins,
    ligands, glycans, and their complexes. Runs via local GPU execution in
    isolated Python environments.

    Args:
        inputs (Chai1Input): Validated input containing one or more complexes to
            predict structures for. Each complex must be ≤ 2,048 residues total.
        config (Chai1Config): Validated Chai1 configuration specifying ESM embeddings,
            MSA settings, refinement parameters, and execution options.

    Returns:
        Chai1Output: Structured output containing:
            - ``structures``: List of ``ChaiStructure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:
                avg_plddt (float): Average per-residue confidence (pLDDT) across all residues.
                    Range: 0-1. Interpretation:

                    - ``> 0.9``: Very high confidence
                    - ``0.7-0.9``: High confidence
                    - ``0.5-0.7``: Low confidence
                    - ``< 0.5``: Very low confidence

                    This is the primary quality metric for Chai1 predictions.

                ptm (Optional[float]): Predicted Template Modeling score measuring overall
                    structural accuracy. Range: 0.0-1.0. Higher values indicate better
                    predicted structures. May be ``None`` for some predictions.

                iptm (Optional[float]): Interface PTM score measuring confidence in inter-chain
                    interfaces. Range: 0.0-1.0. Higher values indicate more confident
                    predictions of chain-chain interactions. Only meaningful for multi-chain
                    complexes. May be ``None`` for single-chain predictions.

                confidence_score (Optional[float]): Overall confidence score combining
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
        >>> inputs = Chai1Input(
        ...     complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]]
        ... )
        >>> config = Chai1Config(
        ...     use_esm_embeddings=True,
        ...     num_trunk_recycles=3,
        ...     verbose=True
        ... )
        >>> result = run_chai1(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0].avg_plddt:.2f}")

    Note:
        - Chai1 processes each complex independently and sequentially
        - ESM embeddings generally improve prediction quality
        - Does not support DNA or RNA (use Boltz2 for nucleic acids)
    """

    results = []

    for comp in tqdm(inputs.complexes, desc="Folding structures (Chai-1)", unit="complex", total=len(inputs.complexes)):
        results.append(run_chai1_on_complex(comp=comp, config=config, instance=instance))
    return Chai1Output(
        structures=results,
    )

# ============================================================================
# Helper Functions
# ============================================================================
def _hash_sequence(seq: str) -> str:
    """Compute SHA-256 hash of a sequence.

    This matches the implementation in chai_lab.data.parsing.msas.aligned_pqt.hash_sequence,
    which is used to generate MSA filenames that Chai1's run_inference expects.

    Args:
        seq: Protein sequence string

    Returns:
        Hexadecimal SHA-256 hash string
    """
    hash_object = hashlib.sha256(seq.encode())
    return hash_object.hexdigest()


def _msa_to_pqt_file(
    msa, pqt_path: str, query_index: int = 0, source_database: str = "uniref90"
) -> None:
    """Writes an MSA to Chai1's .aligned.pqt (Parquet) format.

    The .aligned.pqt format is a Parquet file containing a DataFrame with
    four columns: sequence, source_database, pairing_key, and comment.
    This function streams sequences one at a time to avoid loading large MSAs
    into memory.

    Args:
        msa: MSA object to export
        pqt_path: Path where the .pqt file will be written (should end in .aligned.pqt).
        query_index: Index of the sequence to use as the query (default: 0).
        source_database: Name of the source database for non-query sequences
            (default: "uniref90"). Valid values: "query", "uniref90", "uniprot",
            "bfd_uniclust", "mgnify".

    Raises:
        IndexError: If query_index is out of range.
        ImportError: If pandas is not available.
    """
    if query_index < 0 or query_index >= msa.num_sequences:
        raise IndexError(
            f"Query index {query_index} out of range [0, {msa.num_sequences})"
        )

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for msa_to_pqt_file(). Install with: pip install pandas"
        )

    # Build records one sequence at a time to avoid loading entire MSA
    records = []
    for idx, (seq_id, seq) in enumerate(msa.iter_with_ids()):
        # First sequence (query) gets special treatment
        if idx == query_index:
            source = "query"
        else:
            source = source_database

        record = {
            "sequence": seq,
            "source_database": source,
            "pairing_key": "",  # Empty pairing key (not critical for prediction)
            "comment": seq_id,
        }
        records.append(record)

    # Ensure query is first row (required by Chai1)
    if query_index != 0:
        # Swap query to first position
        records[0], records[query_index] = records[query_index], records[0]

    # Create DataFrame and save as Parquet
    df = pd.DataFrame(records)
    df.to_parquet(pqt_path, engine="pyarrow", index=False)


def _generate_fasta_content(comp: StructurePredictionComplex) -> str:
    """Generate FASTA content from a complex in the format expected by Chai1.

    Args:
        comp: Complex containing chains and entity types

    Returns:
        FASTA-formatted string
    """
    fasta_content = ""
    for i, chain in enumerate(comp.chains):
        e_type = chain.entity_type
        seq = chain.sequence
        fasta_content += f">{e_type}|name={e_type}_{i+1}\n"
        fasta_content += f"{seq.upper()}\n"
    return fasta_content


def _generate_msa_pqt_files(
    comp: StructurePredictionComplex,
    config: Chai1Config,
    output_dir: str,
) -> Optional[str]:
    """Generate MSA files in Chai1's .pqt format using ColabFold search.

    Args:
        comp: Complex containing chains and entity types
        config: Chai1 configuration with MSA settings
        output_dir: Directory where .pqt files will be written

    Returns:
        Path to directory containing .pqt files, or None if no MSAs generated
    """
    # Extract protein sequences for MSA generation
    protein_seqs = []
    for chain in comp.chains:
        if chain.entity_type == "protein":
            protein_seqs.append(chain.sequence)

    if not protein_seqs:
        return None

    # Use ColabFold search tool to generate MSAs
    logger.debug(
        f"Generating MSAs for {len(protein_seqs)} protein chain(s) using ColabFold search..."
    )

    # Create queries with hash of uppercased sequence
    # NOTE: Defines the sequence ID as the hash of the sequence
    queries = [(seq, _hash_sequence(seq.upper())) for seq in protein_seqs]
    colabfold_input = ColabfoldSearchInput(queries=queries)

    try:
        colabfold_output = run_colabfold_search(colabfold_input, config.colabfold_search_config)
    except Exception as e:
        raise RuntimeError(f"ColabFold MSA search failed: {e}") from e

    # Convert MSAs to Chai1's .pqt format
    pqt_dir = os.path.join(output_dir, "msa_pqt")
    os.makedirs(pqt_dir, exist_ok=True)

    for query, result in zip(colabfold_input.queries, colabfold_output.results):
        if result.msa is not None:
            # Convert MSA directly to Chai1's .pqt format
            # NOTE: Chai1 expects the filename to be the hash of the sequence
            pqt_path = os.path.join(pqt_dir, f"{query.sequence_id}.aligned.pqt")
            _msa_to_pqt_file(
                msa=result.msa,
                pqt_path=pqt_path,
                query_index=0,
                source_database="uniref90",
            )

            logger.debug(
                f"Generated MSA for {query.sequence_id}: {result.num_homologs_found} homologs found"
            )
        else:
            logger.debug(f"Warning: No homologs found for {query.sequence_id}")

    return pqt_dir


def _serialize_pqt_files(pqt_dir: str) -> dict[str, bytes]:
    """Read .pqt files from directory and serialize them as bytes.

    Args:
        pqt_dir: Directory containing .aligned.pqt files

    Returns:
        Dictionary mapping filenames to file contents as bytes
    """
    import glob

    msa_pqt_files = {}
    for pqt_file in glob.glob(os.path.join(pqt_dir, "*.aligned.pqt")):
        with open(pqt_file, "rb") as f:
            filename = os.path.basename(pqt_file)
            msa_pqt_files[filename] = f.read()
    return msa_pqt_files


def run_chai1_on_complex(
    comp: StructurePredictionComplex,
    config: Chai1Config,
    instance=None,
) -> Structure:
    """
    Run Chai1 structure prediction on a single complex. This function is wrapped
    by ``run_chai1`` to sequentially predict all complexes in the input.
    """
    # Local GPU execution via venv subprocess
    logger.debug("Using local GPU for Chai1 structure prediction...")

    from bio_programming_tools.utils.tool_instance import ToolInstance

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

        # Handle ColabFold MSA generation if needed
        msa_directory = None
        if config.use_msa:
            # Configure output directory for MSAs (local mode requires this)
            msa_config = config.colabfold_search_config.model_copy(deep=True)
            msa_config.output_dir = temp_dir

            # Temporarily swap config to use the modified one for local execution
            original_config = config.colabfold_search_config
            config.colabfold_search_config = msa_config
            try:
                msa_directory = _generate_msa_pqt_files(comp, config, temp_dir)
            finally:
                # Restore original config
                config.colabfold_search_config = original_config

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
        metrics = result["metrics"]

    return Structure(
        structure_filepath_or_content=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=metrics,
        source="chai1-prediction",
    )
