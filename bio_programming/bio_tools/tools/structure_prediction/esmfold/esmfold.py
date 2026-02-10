"""
esmfold.py

Protein structure prediction using ESMFold.

This module provides standardized interfaces for protein structure prediction
using ESMFold from Meta AI.
"""

from __future__ import annotations

import io
import logging
import string
from pathlib import Path
from typing import Any, Dict, List

from Bio import PDB
from pydantic import field_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)

from bio_programming.bio_tools.tools.utils import ConfigField, return_invalid_protein_chars
from bio_programming.bio_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from bio_programming.bio_tools.entities.structures.structure import BFactorType, Structure
from bio_programming.bio_tools.tools.infra.tool_cache import tool_cache_iterable
from bio_programming.bio_tools.tools.tool_registry import tool
from bio_programming.bio_tools.tools.utils import use_cloud_gpu


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
        complexes (List[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more protein chains. Total length across all chains
            in a complex must not exceed 2,400 residues.

    Note:
        ESMFold only supports protein sequences (amino acids). DNA, RNA, ligands,
        and glycans are not supported. Sequences can include 'X' for unknown amino
        acids. Entity types are automatically inferred if not provided. The 2,400
        residue limit is a hard constraint of the ESMFold architecture.
    """

    # ESMFold only supports proteins
    SUPPORTED_ENTITY_TYPES = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes", check_fields=False)
    @classmethod
    def validate_complexes(
        cls, complexes: List[StructurePredictionComplex]
    ) -> List[StructurePredictionComplex]:
        """
        Ensures that complexes are valid inputs for ESMFold.

        Checks:
        - Valid protein characters (including 'X' for unknown)
        - Total residues per complex ≤ 2400 (ESMFold limit)

        Note:
            Entity type validation (proteins only) is handled automatically
            by the base class using SUPPORTED_ENTITY_TYPES.
        """
        for comp_idx, comp in enumerate(complexes):
            # Validate characters
            for chain_idx, chain_seq in enumerate(comp.chain_sequences):
                invalid_chars = return_invalid_protein_chars(
                    chain_seq, additional_valid_chars="X"
                )
                if invalid_chars:
                    raise ValueError(
                        f"Invalid protein characters in complex {comp_idx}, chain {chain_idx}: "
                        f"{', '.join(sorted(invalid_chars))}"
                    )

            # Check ESMFold length limit
            if comp.sum_of_chain_lengths() > 2400:
                raise ValueError(
                    f"Complex {comp_idx} too long ({comp.sum_of_chain_lengths()} positions, max 2400)"
                )

        return complexes

    def prepare_complexes(self, chain_linker: str) -> List[Dict[str, Any]]:
        """
        Prepares complexes for ESMFold inference
        """
        prepared_complexes = []
        for comp_idx, comp in enumerate(self.complexes):
            seq_lengths = [len(chain.sequence) for chain in comp.chains]
            chain_sequences = [chain.sequence for chain in comp.chains]
            linked_seq = chain_linker.join(chain_sequences)
            prepared_complexes.append(
                {
                    "complex_idx": comp_idx,
                    "chains": chain_sequences,
                    "linked_seq": linked_seq,
                    "seq_lengths": seq_lengths,
                    "total_residues": comp.sum_of_chain_lengths(),
                    "num_chains": comp.num_chains(),
                }
            )
        return prepared_complexes

# Output:
ESMFoldOutput = StructurePredictionOutput

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

        device (str): Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

    Note:
        ESMFold has a maximum total sequence length of 2,400 residues per complex.
        Unlike AlphaFold-based models, ESMFold does not use MSAs, making it much
        faster but potentially less accurate for some targets.
    """

    residue_idx_offset: int = ConfigField(
        title="Residue Index Offset",
        default=512,
        ge=0,
        description="Residue numbering gap between chains in multi-chain structures",
        hidden=True,
    )
    chain_linker: str = ConfigField(
        title="Chain Linker",
        default="G" * 25,
        description="Sequence to link chains (default: 25 glycines)",
        advanced=True,
    )
    # TODO: We need to think about how to handle this for the client.
    max_batch_residues: int = ConfigField(
        title="Max Batch Residues",
        default=1200,
        ge=100,
        description="Maximum total residues per inference batch (to prevent GPU memory overflow)",
        hidden=True,
    )

# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="esmfold",
    label="ESMFold Structure Prediction",
    input=ESMFoldInput,
    config=ESMFoldConfig,
    output=ESMFoldOutput,
    description="Protein structure prediction using ESMFold",
)
@tool_cache_iterable(
    input_iterable_field="complexes",
    output_iterable_field="structures",
    tool_name="esmfold",
)
def run_esmfold(
    inputs: ESMFoldInput, config: ESMFoldConfig
) -> ESMFoldOutput:
    """Predict protein 3D structures using ESMFold.

    Uses ESMFold, a fast transformer-based protein structure prediction model from
    Meta AI, to predict 3D structures without requiring multiple sequence alignments.
    Supports both the cloud runtime cloud execution and local GPU execution with automatic
    batching for memory efficiency.

    Args:
        inputs (ESMFoldInput): Validated input containing one or more protein complexes
            to predict structures for. Each complex must be ≤ 2,400 residues total.
        config (ESMFoldConfig): Validated ESMFold configuration specifying chain linking,
            batching, and execution options.

    Returns:
        ESMFoldOutput: Structured output containing:
            - ``structures``: List of ``Structure`` instances, one per input complex
            - Each structure includes coordinates and the following confidence metrics:
                    avg_plddt (float): Average per-residue confidence (pLDDT) across all residues.
                        Range: 0.0-1.0 (normalized scale, unlike AlphaFold's 0-100). Interpretation:

                        - ``> 0.9``: Very high confidence
                        - ``0.7-0.9``: High confidence
                        - ``0.5-0.7``: Low confidence
                        - ``< 0.5``: Very low confidence

                        This is the primary quality metric for ESMFold predictions.

                    ptm (Optional[float]): Predicted Template Modeling score measuring overall
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
        >>> inputs = ESMFoldInput(
        ...     complexes=["MVLSPADKTNVKAAW"]
        ... )
        >>> config = ESMFoldConfig(verbose=True)
        >>> result = run_esmfold(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0].avg_plddt:.2f}")

    Note:
        - Maximum 2,400 residues per complex (hard limit)
        - Multi-chain complexes are predicted by linking chains
    """

    # Prepare complexes for inference
    prepared_complexes = inputs.prepare_complexes(chain_linker=config.chain_linker)

    # Split into memory-safe sub-batches
    sub_batches = _split_into_safe_batches(
        complexes_to_batch=prepared_complexes, max_residues=config.max_batch_residues
    )

    logger.debug(
        f"Processing {len(prepared_complexes)} complex(es) in {len(sub_batches)} sub-batch(es)..."
    )

    # Run inference on all prepared complexes
    all_results = []

    if use_cloud_gpu():
        # the cloud runtime cloud execution
        logger.debug("Using the cloud runtime for ESMFold structure prediction...")

        import _gpu_runtime
        ESMFoldService = _gpu_runtime.Cls.from_name("bio-programming", "ESMFoldService")

        # Use tqdm to show progress over individual structures, not batches
        pbar = tqdm(
            total=len(prepared_complexes),
            desc="Folding structures (ESMFold)",
            unit="structure",
        )

        for batch_idx, sub_batch in enumerate(sub_batches):
            if len(sub_batches) > 1:
                logger.debug(f"  Sub-batch {batch_idx + 1}/{len(sub_batches)}: {len(sub_batch)} complexes")

            batch_results = ESMFoldService().predict.remote(
                batch_data=sub_batch,
                residue_idx_offset=config.residue_idx_offset,
                chain_linker=config.chain_linker,
                verbose=config.verbose,
            )
            all_results.extend(batch_results)

            # Update progress bar by the number of structures in this batch
            pbar.update(len(sub_batch))

        pbar.close()
    else:
        # Use local GPU execution via venv subprocess
        logger.debug("Using local GPU for ESMFold structure prediction...")

        from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

        venv_manager = EnvManager(model_name="esmfold")

        # Process each sub-batch through standalone script
        # Use tqdm to show progress over individual structures, not batches
        pbar = tqdm(
            total=len(prepared_complexes),
            desc="Folding structures (ESMFold)",
            unit="structure",
        )

        for batch_idx, sub_batch in enumerate(sub_batches):
            if len(sub_batches) > 1:
                logger.debug(f"  Sub-batch {batch_idx + 1}/{len(sub_batches)}: {len(sub_batch)} complexes")

            # Prepare input data for inference script
            input_data = {
                "batch_data": sub_batch,
                "residue_idx_offset": config.residue_idx_offset,
                "chain_linker": config.chain_linker,
            }

            # Call the inference script
            output_data = venv_manager.call_standalone_script_in_venv(
                script_path=Path(__file__).parent / "standalone" / "inference.py",
                input_dict=input_data,
                device=config.device,
                verbose=config.verbose,
            )

            all_results.extend(output_data["results"])

            # Update progress bar by the number of structures in this batch
            pbar.update(len(sub_batch))

        pbar.close()

    # Post-process: relabel chains and convert to CIF
    structure_outputs = []
    for result, metadata in zip(all_results, prepared_complexes):
        # Relabel chains (A, B, C, ...)
        pdb_output = _relabel_chains(result["pdb"], metadata["seq_lengths"])

        structure_outputs.append(
            Structure(
                structure_filepath_or_content=pdb_output,
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                metrics={
                    "avg_plddt": result["avg_plddt"],
                    "ptm": result["ptm"],
                    "avg_pae": result["avg_pae"],
                },
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


# ============================================================================
# Helper Functions
# ============================================================================
def _split_into_safe_batches(
    complexes_to_batch: List[Dict[str, Any]], max_residues: int
) -> List[List[Dict[str, Any]]]:
    """
    Split complexes into sub-batches with <= max_residues per sub-batch.

    Single large complexes (>max_residues) get their own sub-batch.
    """
    safe_batches = []
    current_batch = []
    current_residues = 0

    for item in complexes_to_batch:
        item_residues = item["total_residues"]

        # Large complex: put in its own batch
        if item_residues > max_residues:
            # Flush current batch first
            if current_batch:
                safe_batches.append(current_batch)
                current_batch = []
                current_residues = 0

            # Add large complex alone
            safe_batches.append([item])
            continue

        # Would exceed limit: start new batch
        if current_residues + item_residues > max_residues:
            safe_batches.append(current_batch)
            current_batch = []
            current_residues = 0

        # Add to current batch
        current_batch.append(item)
        current_residues += item_residues

    # Don't forget last batch
    if current_batch:
        safe_batches.append(current_batch)

    return safe_batches


def _relabel_chains(pdb_str: str, chain_lengths: List[int]) -> str:
    """
    Relabel chains in PDB from single chain to A, B, C, ... based on lengths.

    Args:
        pdb_str: PDB file content (assumed single chain)
        chain_lengths: Number of residues per desired chain

    Returns:
        PDB content with chains relabeled
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", io.StringIO(pdb_str))
    model = structure[0]

    # Get original single chain and all residues
    original_chain = list(model.get_chains())[0]
    all_residues = list(original_chain.get_residues())

    # Split residues into new chains
    chain_ids = list(string.ascii_uppercase)
    new_chains = []
    start = 0

    for idx, length in enumerate(chain_lengths):
        new_chain = PDB.Chain.Chain(chain_ids[idx])
        for residue in all_residues[start : start + length]:
            new_chain.add(residue)
        new_chains.append(new_chain)
        start += length

    # Replace original chain with new chains
    model.detach_child(original_chain.id)
    for chain in new_chains:
        model.add(chain)

    # Write back to PDB string
    output = io.StringIO()
    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(output)

    return output.getvalue()
