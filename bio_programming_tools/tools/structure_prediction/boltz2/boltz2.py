"""
boltz2.py
Protein structure prediction using Boltz2.

Example:
    >>> from bio_programming_tools.tools.structure_prediction.boltz2 import run_boltz2, Boltz2Config
    >>> config = Boltz2Config(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
    >>> result = run_boltz2(config)
    >>> print(f"Confidence: {result.confidence_score:.2f}")

"""

from __future__ import annotations

import os
import string
import tempfile
import warnings
from logging import getLogger
from typing import Optional

import yaml
from pydantic import model_validator
from tqdm import tqdm

from bio_programming_tools.entities.structures import BFactorType, Structure
from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    run_colabfold_search,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField
from bio_programming_tools.utils.tool_cache import tool_cache_iterable

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
        complexes (List[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, DNA, RNA, and/or ligands.

    Note:
        Boltz2 supports entity types: ``"protein"``, ``"dna"``, ``"rna"``, and ``"ligand"``.
        Entity types are automatically inferred if not explicitly provided. Invalid
        entity types will raise a validation error.
    """

    # Boltz2 supports all standard entity types except glycan
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    _CHAIN_IDS = list(string.ascii_uppercase)

    def to_yaml(self, complex_idx: int, single_sequence_mode: bool = False) -> str:
        """Convert a complex to Boltz2 YAML input format.

        Args:
            complex_idx: Index of the complex to convert
            single_sequence_mode: Whether to run without MSA (sets msa="empty")

        Returns:
            YAML formatted string for Boltz2 input
        """
        comp = self.complexes[complex_idx]
        yaml_entries = []

        for i, chain in enumerate(comp.chains):
            entry = {"id": self._CHAIN_IDS[i]}
            e_type = chain.entity_type
            seq = chain.sequence

            if e_type in ["protein", "dna", "rna"]:
                entry["sequence"] = seq
            elif e_type == "ligand":
                entry["smiles"] = seq

            if single_sequence_mode:
                entry["msa"] = "empty"

            yaml_entries.append({e_type: entry})

        return yaml.dump(
            {"sequences": yaml_entries, "predict": {"structure": {"enabled": True}}},
            sort_keys=False,
            default_flow_style=False,
        )

# Output:
Boltz2Output = StructurePredictionOutput

# Config:
class Boltz2Config(StructurePredictionConfig):
    """Configuration object for Boltz2 structure prediction.

    This class defines configuration parameters for running Boltz2, a multi-modal
    structure prediction model supporting proteins, DNA, RNA, and ligands.

    Inherits from ``StructurePredictionConfig``.

    Attributes:
        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. If ``False``, runs in single-sequence
            mode without MSAs. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig): Configuration for ColabFold
            MSA search. Controls search mode (local/remote), database paths, and other
            MSA generation parameters. Only used when ``use_msa=True``.
            Default: Uses ColabfoldSearchConfig defaults.

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

        verbose (bool): Whether to print status messages during execution including
            MSA generation, model loading, and prediction progress. Inherited from
            ``StructurePredictionConfig``. Default: ``False``.

    """

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
    return Boltz2Input(complexes=["MKTL"])


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
)
@tool_cache_iterable(
    input_iterable_field="complexes",
    output_iterable_field="structures",
    tool_name="boltz2-prediction",
)
def run_boltz2(inputs: Boltz2Input, config: Boltz2Config | None = None, instance=None) -> Boltz2Output:
    """Predict 3D structures using Boltz2 multi-modal model.

    Uses Boltz2, a diffusion-based deep learning model, to predict 3D structures
    of proteins, DNA, RNA, ligands, and their complexes. Runs via local GPU
    execution in isolated Python environments.

    Args:
        inputs (Boltz2Input): Validated input containing one or more complexes to
            predict structures for.
        config (Boltz2Config): Validated Boltz2 configuration specifying MSA settings,
            refinement parameters, and execution options.

    Returns:
        Boltz2Output: Structured output containing:
            - ``structures``: List of ``Boltz2Structure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:
                    confidence_score (float): Primary ranking score used to select the best
                        structure from multiple samples. For multi-chain complexes, this is
                        ``iptm``; for single chains, this is ``ptm``. Range: 0.0-1.0 (higher
                        is better).

                    ptm (float): Predicted Template Modeling score measuring overall structural
                        accuracy. Range: 0.0-1.0. Interpretation:

                        - ``> 0.7``: High quality structure
                        - ``0.5-0.7``: Moderate quality
                        - ``< 0.5``: Low confidence

                    iptm (float): Interface PTM score measuring confidence in inter-chain
                        interfaces and relative orientations. Range: 0.0-1.0. Interpretation:

                        - ``> 0.85``: High confidence in interface
                        - ``0.7-0.85``: Moderate confidence
                        - ``< 0.7``: Low confidence

                    ligand_iptm (Optional[float]): Protein-ligand interface PTM score. Only
                        present for complexes containing ligands. Range: 0.0-1.0. Higher values
                        indicate more confident protein-ligand binding predictions.

                    protein_iptm (Optional[float]): Protein-protein interface PTM score. Only
                        present for multi-protein complexes. Range: 0.0-1.0. Higher values
                        indicate more confident protein-protein interactions.

                    complex_plddt (Optional[float]): Average per-residue confidence (pLDDT)
                        across all residues in the complex. Range: 0-1. Interpretation:

                        - ``> 0.9``: Very high confidence
                        - ``0.7-0.9``: High confidence
                        - ``0.5-0.7``: Low confidence
                        - ``< 0.5``: Very low confidence

                    complex_iplddt (Optional[float]): Average pLDDT for interface residues only.
                        Range: 0-1. Useful for assessing confidence specifically in the
                        interaction regions.

                    complex_pde (Optional[float]): Average Predicted Aligned Error (PAE) in
                        Angstroms across all residue pairs. Lower values indicate more confident
                        relative positioning. From 0 to 31.75 Å.

                    complex_ipde (Optional[float]): PAE for interface residue pairs only, in
                        Angstroms. Lower values indicate more confident interface geometry.

                    chains_ptm (Optional[List[float]]): Individual PTM scores for each chain.
                        Useful for identifying which chains are predicted with high vs. low
                        confidence.

                    pair_chains_iptm (Optional[List[List[float]]]): Pairwise ipTM scores between
                        all chain pairs. Shape: ``(num_chains, num_chains)``. Symmetric matrix
                        with diagonal values representing self-interactions.

    See Also:
        - Boltz2 GitHub: https://github.com/jwohlwend/boltz
        - Boltz2 paper: https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
        - Boltz2 Website: https://boltz.bio/boltz2

    Example:
        >>> inputs = Boltz2Input(
        ...     complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]]
        ... )
        >>> config = Boltz2Config(
        ...     recycling_steps=10,
        ...     sampling_steps=200,
        ...     diffusion_samples=25,
        ...     verbose=True
        ... )
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
    results = []

    for i in tqdm(range(len(inputs.complexes)), desc="Folding structures (Boltz-2)", unit="complex", total=len(inputs.complexes)):
        # Determine single_sequence_mode based on use_msa
        single_sequence_mode = not config.use_msa
        yaml_content = inputs.to_yaml(i, single_sequence_mode=single_sequence_mode)
        results.append(
            run_boltz2_on_complex(
                yaml_content=yaml_content, config=config, sp_complex=inputs.complexes[i],
                instance=instance,
            )
        )
    return Boltz2Output(structures=results)

# ============================================================================
# Helper Functions
# ============================================================================
def _msa_to_csv_file(msa, csv_path: str, query_index: int = 0) -> None:
    """Writes an MSA to Boltz's CSV format with pairing keys.

    The CSV format contains two columns: 'sequence' and 'key'.
    The 'key' is the row index, which serves as a simple pairing identifier.
    Sequences at the same row index across different MSAs are considered paired.

    Args:
        msa: MSA object to export
        csv_path: Path where the CSV file will be written.
        query_index: Index of the sequence to use as the query (default: 0).
            The query sequence is always placed first with key=0.

    Raises:
        IndexError: If query_index is out of range.
        ImportError: If pandas is not available.

    Note:
        This uses row-index based pairing. For more sophisticated taxonomy-based
        pairing, additional metadata would need to be extracted from A3M headers.
    """
    if query_index < 0 or query_index >= msa.num_sequences:
        raise IndexError(f"Query index {query_index} out of range [0, {msa.num_sequences})")

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for msa_to_csv_file(). Install with: pip install pandas"
        )

    # Build records with row index as pairing key
    records = []
    for idx, (seq_id, seq) in enumerate(msa.iter_with_ids()):
        record = {
            "sequence": seq,
            "key": idx,
        }
        records.append(record)

    # Ensure query is first row (required by Boltz)
    if query_index != 0:
        # Swap query to first position
        records[0], records[query_index] = records[query_index], records[0]
        # Reset keys so query always has key=0
        for idx, record in enumerate(records):
            record["key"] = idx

    # Create DataFrame and save as CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)


def _extract_protein_sequences_and_chain_ids(sp_complex) -> tuple[list[str], list[str]]:
    """Extract protein sequences and their chain IDs from a complex.

    Args:
        sp_complex: StructurePredictionComplex instance containing chain information

    Returns:
        Tuple of (protein_seqs, protein_chain_ids) where chain IDs are uppercase letters
    """
    all_chain_ids = list(string.ascii_uppercase)
    protein_seqs = []
    protein_chain_ids = []
    for i, chain in enumerate(sp_complex.chains):
        if chain.entity_type == "protein":
            protein_seqs.append(chain.sequence)
            protein_chain_ids.append(all_chain_ids[i])
    return protein_seqs, protein_chain_ids


def _generate_msa_csv_files(
    protein_seqs: list[str],
    protein_chain_ids: list[str],
    config: Boltz2Config,
    output_dir: str,
) -> Optional[dict[str, str]]:
    """Generate MSA files in Boltz's CSV format using ColabFold search.

    Args:
        protein_seqs: List of protein sequences
        protein_chain_ids: List of chain IDs (A, B, C...) corresponding to sequences
        config: Boltz2 configuration with MSA settings
        output_dir: Directory where CSV files will be written

    Returns:
        Dictionary mapping chain_id -> csv_path, or None if no protein sequences
    """
    if not protein_seqs:
        return None

    # Use ColabFold search tool to generate MSAs
    if config.verbose:
        logger.info(
            f"Generating MSAs for {len(protein_seqs)} protein chain(s) using ColabFold search..."
        )

    queries = [(seq, name) for seq, name in zip(protein_seqs, protein_chain_ids)]
    colabfold_input = ColabfoldSearchInput(queries=queries)

    try:
        colabfold_output = run_colabfold_search(colabfold_input, config.colabfold_search_config)
    except Exception as e:
        raise RuntimeError(f"ColabFold MSA search failed: {e}") from e

    # Convert MSAs to Boltz's CSV format
    msa_dir = os.path.join(output_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)

    msa_paths = {}
    for result in colabfold_output.results:
        if result.msa is not None:
            # Write MSA in CSV format with pairing keys
            csv_path = os.path.join(msa_dir, f"{result.sequence_id}.csv")
            _msa_to_csv_file(msa=result.msa, csv_path=csv_path, query_index=0)

            # Store path for later use
            msa_paths[result.sequence_id] = csv_path

            if config.verbose:
                logger.info(
                    f"Generated MSA for chain {result.sequence_id}: {result.num_homologs_found} homologs found"
                )
        else:
            if config.verbose:
                logger.warning(f"No homologs found for chain {result.sequence_id}")

    return msa_paths if msa_paths else None


def _serialize_csv_files(msa_paths: dict[str, str]) -> dict[str, bytes]:
    """Read CSV files and serialize them as bytes.

    Args:
        msa_paths: Dictionary mapping chain_id -> csv_path

    Returns:
        Dictionary mapping filename -> file contents as bytes
    """
    msa_csv_files = {}
    for chain_id, csv_path in msa_paths.items():
        with open(csv_path, "rb") as f:
            filename = os.path.basename(csv_path)
            msa_csv_files[filename] = f.read()
    return msa_csv_files


def run_boltz2_on_complex(
    yaml_content: str,
    config: Boltz2Config,
    sp_complex,
    instance=None,
) -> Structure:
    """
    Run Boltz2 structure prediction on a single complex. This function is wrapped
    by ``run_boltz2`` to sequentially predict all complexes in the input.

    Args:
        yaml_content: YAML formatted string for Boltz2 input
        config: Boltz2 configuration
        sp_complex: StructurePredictionComplex instance containing chain information
    """
    if config.verbose:
        logger.info("Using local GPU for Boltz2 structure prediction...")

    from bio_programming_tools.utils.tool_instance import ToolInstance

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "boltz2_output")
        os.makedirs(output_dir)

        # Handle ColabFold MSA generation if needed
        msa_paths = None
        if config.use_msa:
            # Extract protein sequences and generate MSAs
            protein_seqs, protein_chain_ids = _extract_protein_sequences_and_chain_ids(sp_complex)

            if protein_seqs:
                # Configure output directory for MSAs (local mode requires this)
                msa_config = config.colabfold_search_config.model_copy(deep=True)
                msa_config.output_dir = temp_dir

                # Temporarily swap config to use the modified one for local execution
                original_config = config.colabfold_search_config
                config.colabfold_search_config = msa_config
                try:
                    msa_paths = _generate_msa_csv_files(
                        protein_seqs, protein_chain_ids, config, temp_dir
                    )
                finally:
                    # Restore original config
                    config.colabfold_search_config = original_config

                # Update YAML content to include MSA paths or set empty for chains without MSAs
                yaml_dict = yaml.safe_load(yaml_content)
                for seq_entry in yaml_dict["sequences"]:
                    for entity_type in seq_entry:
                        chain_id = seq_entry[entity_type]["id"]
                        # Handle both string and list formats for chain ID
                        if isinstance(chain_id, list):
                            chain_id = chain_id[0] if chain_id else None

                        # Only set MSA for protein chains
                        if entity_type == "protein" and chain_id:
                            if msa_paths and chain_id in msa_paths:
                                # Use the generated MSA
                                seq_entry[entity_type]["msa"] = msa_paths[chain_id]
                            else:
                                # No MSA found for this chain, set to empty
                                seq_entry[entity_type]["msa"] = "empty"
                                print(f"No Homologs were found by ColabFold for chain {chain_id} - setting msa='empty'. ")
                                warnings.warn(
                                    f"No Homologs were found by ColabFold for chain {chain_id} - setting msa='empty'. ",
                                    UserWarning,
                                    stacklevel=2
                                )

                yaml_content = yaml.dump(
                    yaml_dict, sort_keys=False, default_flow_style=False
                )

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
        }

        # Call the inference script
        output_data = ToolInstance.dispatch(
            "boltz2",
            input_data,
            instance=instance,
            verbose=config.verbose,
            timeout=config.timeout,
        )

        cif_output = output_data["structure_cif_output"]
        formatted_metrics = output_data["metrics"]

    # Extract metrics
    metrics = {
        "confidence_score": float(formatted_metrics["confidence_score"]),
        "ptm": float(formatted_metrics["ptm"]),
        "iptm": float(formatted_metrics["iptm"]),
        "chains_ptm": formatted_metrics["chains_ptm"],
        "pair_chains_iptm": formatted_metrics["pair_chains_iptm"],
    }
    optional_metrics = [
        "ligand_iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
        "complex_pde",
        "complex_ipde",
    ]
    for metric in optional_metrics:
        if metric in formatted_metrics:
            metrics[metric] = float(formatted_metrics[metric])
        else:
            metrics[metric] = None

    # Add these for consistency across structure predictor metrics.
    metrics["avg_plddt"] = metrics.get("complex_plddt")
    # Boltz does not natively store PAE, use PDE instead:
    metrics["avg_pae"] = metrics.get("complex_pde")

    return Structure(
        structure_filepath_or_content=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=metrics,
        source="boltz2-prediction",
    )


def _format_output(value) -> any:
    """
    Prepare Boltz output dictionaries by unpacking them into lists.

    Boltz outputs nested dicts with string keys "0", "1", "2", etc.
    This function recursively converts them to lists.

    Example:
        {"0": 1.5, "1": 2.0, "2": 3.5} -> [1.5, 2.0, 3.5]
        {"0": {"0": 1, "1": 2}, "1": {"0": 3, "1": 4}} -> [[1, 2], [3, 4]]
    """
    if isinstance(value, dict):
        list_value = []
        for k in range(len(value)):
            list_value.append(_format_output(value[str(k)]))
        return list_value
    else:
        return value
