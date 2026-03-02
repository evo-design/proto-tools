"""
alphafold2.py

Protein structure prediction using AlphaFold2 via ColabDesign.

This module provides standardized interfaces for protein structure prediction
using the original AlphaFold2 model through the ColabDesign JAX wrapper.
"""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, List, Optional

from pydantic import field_validator, model_validator
from tqdm import tqdm

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
from bio_programming_tools.utils import ConfigField, return_invalid_protein_chars
from bio_programming_tools.utils.tool_cache import tool_cache_iterable

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class AlphaFold2Input(StructurePredictionInput):
    """Input object for AlphaFold2 structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins
    using AlphaFold2 via the ColabDesign JAX wrapper.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (List[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more protein chains.

    Note:
        AlphaFold2 only supports protein sequences (amino acids). DNA, RNA, ligands,
        and glycans are not supported. Sequences can include 'X' for unknown amino
        acids. Entity types are automatically inferred if not provided.
    """

    SUPPORTED_ENTITY_TYPES = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes", check_fields=False)
    @classmethod
    def validate_complexes(
        cls, complexes: List[StructurePredictionComplex]
    ) -> List[StructurePredictionComplex]:
        """Validate that complexes contain valid protein sequences.

        Checks:
        - Valid protein characters (including 'X' for unknown)

        Note:
            Entity type validation (proteins only) is handled automatically
            by the base class using SUPPORTED_ENTITY_TYPES.
        """
        for comp_idx, comp in enumerate(complexes):
            for chain_idx, chain_seq in enumerate(comp.chain_sequences):
                invalid_chars = return_invalid_protein_chars(
                    chain_seq, additional_valid_chars="X"
                )
                if invalid_chars:
                    raise ValueError(
                        f"Invalid protein characters in complex {comp_idx}, chain {chain_idx}: "
                        f"{', '.join(sorted(invalid_chars))}"
                    )

        return complexes

    def prepare_complexes(self) -> List[Dict[str, Any]]:
        """Prepare complexes for AlphaFold2 inference.

        Returns:
            List of dicts with keys: chains, seq_lengths, num_chains, total_residues
        """
        prepared_complexes = []
        for comp_idx, comp in enumerate(self.complexes):
            seq_lengths = [len(chain.sequence) for chain in comp.chains]
            chain_sequences = [chain.sequence for chain in comp.chains]
            prepared_complexes.append(
                {
                    "complex_idx": comp_idx,
                    "chains": chain_sequences,
                    "seq_lengths": seq_lengths,
                    "num_chains": comp.num_chains(),
                    "total_residues": comp.sum_of_chain_lengths(),
                }
            )
        return prepared_complexes


# Output:
AlphaFold2Output = StructurePredictionOutput


# Config:
class AlphaFold2Config(StructurePredictionConfig):
    """Configuration object for AlphaFold2 structure prediction.

    This class defines configuration parameters for running AlphaFold2 via
    the ColabDesign JAX wrapper.

    Inherits from ``StructurePredictionConfig``.

    Attributes:
        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. If ``False``, runs in single-sequence
            mode without MSAs. Default: ``True``.

        colabfold_search_config (Optional[ColabfoldSearchConfig]): Configuration for
            ColabFold MSA search. Only used when ``use_msa=True``.
            Default: Uses ColabfoldSearchConfig defaults.

        num_recycles (int): Number of recycling iterations through the model.
            Higher values can improve accuracy at the cost of computation time.
            Range: 0-48. Default: 3.

        model_num (int): Which AlphaFold2 model parameter set to use (1-5).
            AF2 ships 5 independently trained parameter sets. Different sets can
            produce different predictions. Mutually exclusive with
            ``num_ensemble_models > 1`` — set one or the other. Default: 1.

        num_ensemble_models (int): Number of model parameter sets to run and average.
            Running multiple models and averaging their outputs can improve prediction
            quality at the cost of increased computation time. Mutually exclusive with
            ``model_num`` — when ensembling, models are selected from the full pool
            (models 1 through N). Range: 1-5. Default: 1.

        seed (Optional[int]): Random seed for reproducibility. If ``None``, uses
            non-deterministic initialization. Default: ``None``.

        device (str): Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during execution. Inherited
            from ``BaseConfig``. Default: ``False``.
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
    num_recycles: int = ConfigField(
        title="Number of Recycles",
        default=3,
        ge=0,
        le=48,
        description="Number of recycling iterations (higher=more refined but slower)",
        advanced=True,
    )
    model_num: int = ConfigField(
        title="Model Number",
        default=1,
        ge=1,
        le=5,
        description="Which AlphaFold2 model parameter set to use (1-5)",
        advanced=True,
    )
    num_ensemble_models: int = ConfigField(
        title="Ensemble Models",
        default=1,
        ge=1,
        le=5,
        description="Number of model parameter sets to run and average (higher=better but slower)",
    )
    seed: Optional[int] = ConfigField(
        title="Random Seed",
        default=None,
        description="Random seed for reproducibility. If None, uses non-deterministic initialization.",
        advanced=True,
    )

    @model_validator(mode="after")
    def validate_model_selection(self):
        """Validate model_num and num_ensemble_models are not both set."""
        if self.model_num != 1 and self.num_ensemble_models > 1:
            raise ValueError(
                "model_num and num_ensemble_models are mutually exclusive. "
                "Use model_num to select a specific parameter set, or "
                "num_ensemble_models to average multiple models."
            )
        if self.colabfold_search_config is None:
            self.colabfold_search_config = ColabfoldSearchConfig()
        self.colabfold_search_config.verbose = self.verbose
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return AlphaFold2Input(complexes=["MKTL"])


@tool(
    key="alphafold2-prediction",
    label="AlphaFold2 Structure Prediction",
    category="structure_prediction",
    input_class=AlphaFold2Input,
    config_class=AlphaFold2Config,
    output_class=AlphaFold2Output,
    description="Protein structure prediction using AlphaFold2 via ColabDesign",
    uses_gpu=True,
    example_input=example_input,
)
@tool_cache_iterable(
    input_iterable_field="complexes",
    output_iterable_field="structures",
    tool_name="alphafold2-prediction",
)
def run_alphafold2(
    inputs: AlphaFold2Input,
    config: AlphaFold2Config | None = None,
    instance=None,
) -> AlphaFold2Output:
    """Predict protein 3D structures using AlphaFold2.

    Uses the original AlphaFold2 model via the ColabDesign JAX wrapper to predict
    3D structures of protein sequences. Supports optional MSA generation via
    ColabFold search for improved accuracy.

    Args:
        inputs (AlphaFold2Input): Validated input containing one or more protein
            complexes to predict structures for.
        config (AlphaFold2Config): Validated AlphaFold2 configuration specifying
            MSA settings, recycling, and model parameters.
        instance: Optional tool instance name for persistent workers.

    Returns:
        AlphaFold2Output: Structured output containing:
            - ``structures``: List of ``Structure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:
                    avg_plddt (float): Average per-residue confidence (pLDDT).
                        Range: 0.0-1.0. This is the primary quality metric.

                    ptm (float): Predicted Template Modeling score measuring overall
                        structural accuracy. Range: 0.0-1.0.

                    iptm (Optional[float]): Interface PTM score for multi-chain
                        complexes. Range: 0.0-1.0. None for single-chain predictions.

                    avg_pae (Optional[float]): Average Predicted Aligned Error.
                        Lower values indicate more confident relative positioning.

    See Also:
        - AlphaFold2 paper: https://doi.org/10.1038/s41586-021-03819-2
        - ColabDesign: https://github.com/sokrypton/ColabDesign

    Example:
        >>> inputs = AlphaFold2Input(complexes=["MVLSPADKTNVKAAW"])
        >>> config = AlphaFold2Config(use_msa=False, verbose=True)
        >>> result = run_alphafold2(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0].avg_plddt:.2f}")
    """
    from bio_programming_tools.utils.tool_instance import ToolInstance

    prepared_complexes = inputs.prepare_complexes()

    structure_outputs = []

    for complex_data in tqdm(
        prepared_complexes,
        desc="Folding structures (AlphaFold2)",
        unit="structure",
    ):
        # Generate MSA if requested (single-chain only)
        msa_a3m_content = None
        if config.use_msa:
            if complex_data["num_chains"] > 1:
                logger.info(
                    "MSA not yet supported for multi-chain complexes, "
                    "running without MSA"
                )
            else:
                protein_seqs, protein_chain_ids = (
                    _extract_protein_sequences_and_chain_ids(
                        inputs.complexes[complex_data["complex_idx"]]
                    )
                )
                if protein_seqs:
                    msa_dict = _generate_msa_for_alphafold2(
                        protein_seqs, protein_chain_ids, config
                    )
                    if msa_dict:
                        # Single-chain: extract the A3M string directly
                        msa_a3m_content = next(iter(msa_dict.values()))

        # Prepare input data for standalone dispatch
        input_data = {
            "complex_data": complex_data,
            "num_recycles": config.num_recycles,
            "model_num": config.model_num,
            "num_ensemble_models": config.num_ensemble_models,
            "seed": config.seed,
            "msa_a3m_content": msa_a3m_content,
            "device": config.device,
            "verbose": config.verbose,
        }

        # Dispatch to standalone subprocess
        output_data = ToolInstance.dispatch(
            "alphafold2",
            input_data,
            instance=instance,
            verbose=config.verbose,
        )

        # Post-process: create Structure from PDB output
        metrics = {
            "avg_plddt": output_data["avg_plddt"],
            "ptm": output_data["ptm"],
            "iptm": output_data.get("iptm"),
            "avg_pae": output_data.get("avg_pae"),
        }

        structure_outputs.append(
            Structure(
                structure_filepath_or_content=output_data["pdb"],
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                metrics=metrics,
                source="alphafold2-prediction",
            )
        )

    return AlphaFold2Output(
        structures=structure_outputs,
        metadata={
            "num_complexes": len(structure_outputs),
            "total_chains": sum(s.num_chains for s in structure_outputs),
        },
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _extract_protein_sequences_and_chain_ids(
    sp_complex: StructurePredictionComplex,
) -> tuple[list[str], list[str]]:
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


def _generate_msa_for_alphafold2(
    protein_seqs: list[str],
    protein_chain_ids: list[str],
    config: AlphaFold2Config,
) -> Optional[dict[str, str]]:
    """Generate MSAs for protein sequences using ColabFold search.

    Returns a dict mapping chain_id -> A3M string content, or None if no
    protein sequences or no homologs found.

    Args:
        protein_seqs: List of protein sequences
        protein_chain_ids: List of chain IDs (A, B, C...) corresponding to sequences
        config: AlphaFold2 configuration with MSA settings

    Returns:
        Dictionary mapping chain_id -> A3M content string, or None
    """
    if not protein_seqs:
        return None

    if config.verbose:
        logger.info(
            f"Generating MSAs for {len(protein_seqs)} protein chain(s) using ColabFold search..."
        )

    queries = [(seq, name) for seq, name in zip(protein_seqs, protein_chain_ids)]
    colabfold_input = ColabfoldSearchInput(queries=queries)

    try:
        colabfold_output = run_colabfold_search(
            colabfold_input, config.colabfold_search_config
        )
    except Exception as e:
        raise RuntimeError(f"ColabFold MSA search failed: {e}") from e

    # Serialize MSAs as A3M strings
    msa_a3m_content = {}
    for result in colabfold_output.results:
        if result.msa is not None:
            msa_a3m_content[result.sequence_id] = result.msa.to_a3m_string()

            if config.verbose:
                logger.info(
                    f"Generated MSA for chain {result.sequence_id}: "
                    f"{result.num_homologs_found} homologs found"
                )
        else:
            if config.verbose:
                logger.warning(f"No homologs found for chain {result.sequence_id}")

    return msa_a3m_content if msa_a3m_content else None
