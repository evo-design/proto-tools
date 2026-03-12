"""
alphafold2.py

Protein structure prediction using AlphaFold2 via ColabDesign.

This module provides standardized interfaces for protein structure prediction
using the original AlphaFold2 model through the ColabDesign JAX wrapper.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import field_validator, model_validator
from tqdm import tqdm

from bio_programming_tools.entities.structures.structure import BFactorType, Structure
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField, return_invalid_protein_chars

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
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.

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
class AlphaFold2Config(MSAStructurePredictionConfig):
    """Configuration object for AlphaFold2 structure prediction.

    This class defines configuration parameters for running AlphaFold2 via
    the ColabDesign JAX wrapper.

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
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
        reload_on_change=True,
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
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
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
        # Read pre-computed MSA (single-chain only)
        msa_a3m_content = None
        if inputs.msas:
            if complex_data["num_chains"] > 1:
                logger.info(
                    "MSA not yet supported for multi-chain complexes, "
                    "running without MSA"
                )
            else:
                protein_seq = inputs.complexes[complex_data["complex_idx"]].chains[0].sequence
                msa = inputs.msas.get(protein_seq)
                if msa is not None:
                    msa_a3m_content = msa.to_a3m_string()
                    if config.verbose:
                        logger.info(
                            f"Loaded MSA for complex {complex_data['complex_idx']} "
                            f"({len(msa)} sequences)"
                        )

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
            config=config,
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

