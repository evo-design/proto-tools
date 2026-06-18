"""Protein structure prediction using AlphaFold2 via ColabDesign.

This module provides standardized interfaces for protein structure prediction
using the original AlphaFold2 model through the ColabDesign JAX wrapper.
"""

import logging
from typing import Any, ClassVar

from pydantic import field_validator, model_validator

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    normalize_output_chain_ids,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    ToolInstance,
    return_invalid_protein_chars,
)
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

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
        complexes (list[Complex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more protein chains.
        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one
            entry per complex. Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by
            chain index); ``paired=True`` marks rows taxonomy-aligned across chains. Populated by preprocess() or supplied directly.
            Default: None.

    Note:
        AlphaFold2 only supports protein sequences (amino acids). DNA, RNA, ligands,
        and glycans are not supported. Sequences can include 'X' for unknown amino
        acids. Entity types are automatically inferred if not provided.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes", check_fields=False)
    @classmethod
    def validate_complexes(cls, complexes: list[Complex]) -> list[Complex]:
        """Validate that complexes contain valid protein sequences.

        Args:
            complexes (list[Complex]): Complexes to validate.

        Checks:
        - Valid protein characters (including 'X' for unknown)

        Note:
            Entity type validation (proteins only) is handled automatically
            by the base class using SUPPORTED_ENTITY_TYPES.
        """
        for comp_idx, comp in enumerate(complexes):
            for chain_idx, chain_seq in enumerate(comp.chain_sequences):
                invalid_chars = return_invalid_protein_chars(chain_seq, additional_valid_chars="X")
                if invalid_chars:
                    raise ValueError(
                        f"Invalid protein characters in complex {comp_idx}, chain {chain_idx}: "
                        f"{', '.join(sorted(invalid_chars))}"
                    )

        return complexes

    def prepare_complexes(self) -> list[dict[str, Any]]:
        """Prepare complexes for AlphaFold2 inference.

        Returns:
            list[dict[str, Any]]: List of dicts with keys: complex_idx, chains, seq_lengths, num_chains, total_residues
        """
        prepared_complexes = []
        for comp_idx, comp in enumerate(self.complexes):
            chain_sequences = comp.chain_sequences  # SUPPORTED_ENTITY_TYPES rejects ligands upstream
            seq_lengths = [len(s) for s in chain_sequences]
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


class AlphaFold2Metrics(Metrics):
    """Per-structure metrics emitted by AlphaFold2 prediction.

    Attributes:
        primary_metric (str | None): Defaults to ``"avg_plddt"`` for AlphaFold2.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Average predicted LDDT score (0-1). Always present.
        ptm (float): Predicted TM-score (0-1). Always present.
        iptm (float): Interface predicted TM-score (0-1). Multi-chain input only.
        avg_pae (float): Average predicted aligned error. Always present.
        pae (list[list[float]]): Full per-residue PAE matrix in Å. Present when include_pae_matrix=True.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {
            "availability": "multi-chain input only",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "avg_pae": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "avg_plddt"


# Output:
class AlphaFold2Output(StructurePredictionOutput):
    """AlphaFold2 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying an
            :class:`AlphaFold2Metrics` instance on ``.metrics``.
    """


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
            ``num_ensemble_models > 1``; set one or the other. Default: 1.

        num_ensemble_models (int): Number of model parameter sets to run and average.
            Running multiple models and averaging their outputs can improve prediction
            quality at the cost of increased computation time. Mutually exclusive with
            ``model_num``; when ensembling, models are selected from the full pool
            (models 1 through N). Range: 1-5. Default: 1.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using MMseqs2 homology search. Supplied MSAs are always
            used and override ``use_msa=False``. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        pair_heterocomplex_msas (bool): Whether heterocomplex protein chains
            should use taxonomy-paired MSA generation. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        msa_search_config (Mmseqs2HomologySearchConfig | None): Configuration for
            MMseqs2 homology search (MSA generation). Only used when ``use_msa=True``.
            Inherited from ``MSAStructurePredictionConfig``. Default: ``None``.

        device: Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        include_pae_matrix (bool): Inherited. Default: ``False``.

        verbose: Verbosity level (0=quiet, 1=info, 2=debug, 3=raw subprocess
            stderr). Inherited from ``BaseConfig``. Default: ``0``.
    """

    num_recycles: int = ConfigField(
        title="Number of Recycles",
        default=3,
        ge=0,
        description="Recycling iterations through the model. Higher = more accurate but slower.",
    )
    model_num: int = ConfigField(
        title="Model Number",
        default=1,
        ge=1,
        le=5,
        description="Which of AlphaFold2's 5 trained parameter sets to use.",
        reload_on_change=True,
    )
    num_ensemble_models: int = ConfigField(
        title="Ensemble Models",
        default=1,
        ge=1,
        le=5,
        description="Number of parameter sets to run and average. Higher = more accurate but slower.",
    )

    @model_validator(mode="after")
    def validate_model_selection(self) -> Any:
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
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return AlphaFold2Input(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="alphafold2-prediction",
    label="AlphaFold2 Structure Prediction",
    category="structure_prediction",
    input_class=AlphaFold2Input,
    config_class=AlphaFold2Config,
    output_class=AlphaFold2Output,
    metrics_class=AlphaFold2Metrics,
    description="Protein structure prediction using AlphaFold2 via ColabDesign",
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    cacheable=True,
)
def run_alphafold2(
    inputs: AlphaFold2Input,
    config: AlphaFold2Config,
    instance: Any = None,
) -> AlphaFold2Output:
    """Predict protein 3D structures using AlphaFold2.

    Uses the original AlphaFold2 model via the ColabDesign JAX wrapper to predict
    3D structures of protein sequences. Supports optional MSA generation via
    MMseqs2 homology search for improved accuracy.

    Args:
        inputs (AlphaFold2Input): Validated input containing one or more protein
            complexes to predict structures for.
        config (AlphaFold2Config): Validated AlphaFold2 configuration specifying
            MSA settings, recycling, and model parameters.
        instance (Any): A ToolInstance, or a string referencing one pre-registered via
            ToolInstance.get/persist_tool (unknown names raise); None runs one-shot.

    Returns:
        AlphaFold2Output: Structured output containing:
            - ``structures``: List of ``Structure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:
                    avg_plddt: Average per-residue confidence (pLDDT).
                        Range: 0.0-1.0. This is the primary quality metric.

                    ptm: Predicted Template Modeling score measuring overall
                        structural accuracy. Range: 0.0-1.0.

                    iptm: Interface PTM score for multi-chain
                        complexes. Range: 0.0-1.0. None for single-chain predictions.

                    avg_pae: Average Predicted Aligned Error.
                        Lower values indicate more confident relative positioning.

    See Also:
        - AlphaFold2 paper: https://doi.org/10.1038/s41586-021-03819-2
        - ColabDesign: https://github.com/sokrypton/ColabDesign

    Example:
        >>> inputs = AlphaFold2Input(complexes=["MVLSPADKTNVKAAW"])
        >>> config = AlphaFold2Config(use_msa=False, verbose=True)
        >>> result = run_alphafold2(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0].metrics.avg_plddt:.2f}")
    """
    prepared_complexes = inputs.prepare_complexes()

    structure_outputs = []

    for complex_data in progress_bar(
        prepared_complexes,
        desc="Folding structures (AlphaFold2)",
        unit="structure",
    ):
        # Attach MSAs: single-chain/homo-oligomer pass one A3M (ColabDesign tiles it); heteromultimers pass per-chain A3Ms stitched into a block-diagonal MSA downstream.
        msa_a3m_content: str | None = None
        per_chain_msas_a3m: list[str | None] | None = None
        unpaired_per_chain_msas_a3m: list[str | None] | None = None
        is_paired = False
        if inputs.msas:
            from proto_tools.tools.structure_prediction.shared_data_models import unwrap_complex_msas

            per_chain, unpaired_per_chain, is_paired = unwrap_complex_msas(inputs.msas[complex_data["complex_idx"]])
            num_chains = complex_data["num_chains"]
            is_homooligomer = num_chains > 1 and len(set(complex_data["chains"])) == 1

            if num_chains == 1 or is_homooligomer:
                msa = per_chain.get(0)
                if msa is not None:
                    msa_a3m_content = msa.to_a3m_string()
                    if config.verbose:
                        logger.info(f"Loaded MSA for complex {complex_data['complex_idx']} ({len(msa)} sequences)")
            else:
                per_chain_msas_a3m = [
                    per_chain[i].to_a3m_string() if i in per_chain else None for i in range(num_chains)
                ]
                # Deep per-chain unpaired MSAs ride alongside the paired rows for block-diagonal depth.
                if unpaired_per_chain:
                    unpaired_per_chain_msas_a3m = [
                        unpaired_per_chain[i].to_a3m_string() if i in unpaired_per_chain else None
                        for i in range(num_chains)
                    ]
                if config.verbose:
                    depths = [len(per_chain[i]) if i in per_chain else 0 for i in range(num_chains)]
                    logger.info(
                        f"Loaded per-chain MSAs for complex {complex_data['complex_idx']} "
                        f"(depths={depths}, paired={is_paired})"
                    )

        # Prepare input data for standalone dispatch
        input_data = {
            "operation": "predict",
            "complex_data": complex_data,
            "num_recycles": config.num_recycles,
            "model_num": config.model_num,
            "num_ensemble_models": config.num_ensemble_models,
            "seed": config.seed,
            "msa_a3m_content": msa_a3m_content,
            "per_chain_msas_a3m": per_chain_msas_a3m,
            "is_paired": is_paired,
            "unpaired_per_chain_msas_a3m": unpaired_per_chain_msas_a3m,
            "device": config.device,
            "verbose": config.verbose,
            "include_pae_matrix": config.include_pae_matrix,
        }

        # Dispatch to standalone subprocess
        output_data = ToolInstance.dispatch(
            "alphafold2",
            input_data,
            instance=instance,
            config=config,
        )

        # Post-process: create Structure from PDB output
        metrics = AlphaFold2Metrics(
            avg_plddt=output_data["avg_plddt"],
            ptm=output_data["ptm"],
            iptm=output_data.get("iptm"),
            avg_pae=output_data.get("avg_pae"),
            pae=output_data.get("pae"),
        )

        comp = inputs.complexes[complex_data["complex_idx"]]
        structure = Structure(
            structure=output_data["pdb"],
            # ColabDesign's save_pdb writes B-factors as 100 * plddt (0-100 scale).
            b_factor_type=BFactorType.PLDDT,
            metrics=metrics,
            source="alphafold2-prediction",
        )
        structure_outputs.append(normalize_output_chain_ids(structure, comp.chains))

    return AlphaFold2Output(
        structures=structure_outputs,
        metadata={
            "num_complexes": len(structure_outputs),
            "total_chains": sum(s.num_chains for s in structure_outputs),
        },
    )
