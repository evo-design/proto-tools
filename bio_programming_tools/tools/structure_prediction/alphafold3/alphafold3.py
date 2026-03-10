"""
alphafold3.py

Protein structure prediction using AlphaFold3.

This module provides standardized interfaces for protein structure prediction
using AlphaFold3 from Google DeepMind.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)

from bio_programming_tools.entities.ligands import map_smiles_to_ccd_code
from bio_programming_tools.entities.structures.structure import BFactorType, Structure
from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance

# Type alias for AlphaFold3 JSON format
AlphaFold3JSON = Dict[str, Any]


# ============================================================================
# Data Models
# ============================================================================
# Input:
class AlphaFold3Input(StructurePredictionInput):
    """Input object for AlphaFold3 structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins,
    nucleic acids, and ligands using AlphaFold3.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (List[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more sequences of proteins, DNA, RNA, or ligands.
    """

    # AlphaFold3 supports all standard entity types except glycan
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True

# Output:
AlphaFold3Output = StructurePredictionOutput

# Config:
class AlphaFold3Config(StructurePredictionConfig):
    """Configuration object for AlphaFold3 structure prediction.

    This class defines configuration parameters for running AlphaFold3, a state-of-the-art
    multi-modal structure predictor.

    Inherits from ``StructurePredictionConfig``.

    Attributes:
        name (str): Name of the folding job. Default: ``"af3_job"``.

        seeds (List[int]): Seeds to use for AlphaFold3. Default: ``[0]``.
            Note: AlphaFold3 will do five diffusion samples per seed, so this often can be
            set to a single seed. More seeds are required for complex docking tasks,
            such as antibody-antigen docking.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains. If ``False``, skips MSA generation. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig): Configuration for ColabFold
            MSA search. Controls search mode (local/remote), database paths, and other
            MSA generation parameters. Only used if ``use_msa`` is ``True``.
            Default: Uses ColabfoldSearchConfig defaults.

        output_dir (Optional[str]): Path prefix for the AlphaFold3 output directory.
            Appends ``_af3_results`` to the provided string. If ``None`` (default),
            uses a temporary directory that is automatically cleaned up after inference.
            If specified, creates a persistent directory at the given path that will
            NOT be automatically deleted. Default: ``None``.

        repo_path (str): Local path to the cloned AlphaFold3 repository.
            Required for execution.

        sif_path (str): Local path to the AlphaFold3 Singularity image file (.sif).
            Required for container execution.

        model_dir (str): Local path to the directory containing AlphaFold3
            model parameters/weights.

        db_dir (str): Local path to the AlphaFold3 genetic databases.

        device (str): Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

    """

    name: str = ConfigField(
        title="AlphaFold3 Job Name",
        default="af3_job",
        description="Name of the AlphaFold3 folding job",
        advanced=True,
    )

    seeds: List[int] = ConfigField(
        title="AlphaFold3 Seeds",
        default=[0],
        description="Seeds to use for AlphaFold3",
        advanced=True,
    )

    use_msa: bool = ConfigField(
        title="Use MSA",
        default=True,
        description="Whether to generate and use MSAs for protein chains",
    )

    colabfold_search_config: Optional[ColabfoldSearchConfig] = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description="Nested configuration for ColabFold MSA search. If None, uses default settings.",
        hidden=True,
    )

    output_dir: Optional[str] = ConfigField(
        title="Output Directory Prefix",
        default=None,
        description="Prefix for the AlphaFold3 output directory. If None, uses temp directory with auto-cleanup.",
        hidden=True,
    )

    repo_path: str = ConfigField(
        title="AlphaFold3 Repo Path",
        default="/large_storage/hielab/brk/models/alphafold3",
        description="Path to AlphaFold3 repository",
        hidden=True,
    )

    sif_path: str = ConfigField(
        title="AlphaFold3 SIF Path",
        default="/large_storage/hielab/brk/models/alphafold3/alphafold3_latest.sif",
        description="Path to AlphaFold3 Singularity container",
        hidden=True,
    )

    model_dir: str = ConfigField(
        title="AlphaFold3 Model Path",
        default="/large_storage/hielab/brk/models/af3_weights",
        description="Path to AlphaFold3 model weights",
        hidden=True,
    )

    db_dir: str = ConfigField(
        title="AlphaFold3 Database Path",
        default="/large_storage/hielab/brk/databases/af3_dbs",
        description="Path to AlphaFold3 sequence database",
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
    return AlphaFold3Input(complexes=["MKTL"])


@tool(
    key="alphafold3-prediction",
    label="AlphaFold3 Structure Prediction",
    category="structure_prediction",
    input_class=AlphaFold3Input,
    config_class=AlphaFold3Config,
    output_class=AlphaFold3Output,
    description="Protein structure prediction using AlphaFold3",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
)
def run_alphafold3(
    inputs: AlphaFold3Input, config: AlphaFold3Config | None = None,
    instance=None,
) -> AlphaFold3Output:
    """Predict protein 3D structures using AlphaFold3."""

    output_structures: List[Structure] = []

    for comp_idx, comp in tqdm(enumerate(inputs.complexes), desc="Folding structures (AlphaFold3)", unit="complex", total=len(inputs.complexes)):
        input_json = _create_input_json_from_complex(
            comp,
            f"{config.name}_{comp_idx}",
            config.seeds,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Determine output directory
            if config.output_dir is None:
                # Create inside temp directory for auto-cleanup
                output_dir = os.path.join(temp_dir, f"{config.name}_{comp_idx}_af3_results")
            else:
                # Create at specified path (persists after execution)
                output_dir = f"{config.output_dir}_af3_results"

            # Create input directory for MSAs
            input_dir = os.path.join(output_dir, "af3_inputs")
            os.makedirs(input_dir, exist_ok=True)

            # Generate MSAs if requested
            if config.use_msa:
                input_json = _generate_msas(
                    input_json,
                    input_dir,
                    config.colabfold_search_config,
                    verbose=config.verbose,
                )
            else:
                logger.debug("MSA generation skipped (use_msa=False).")

            # Write input JSON to file for worker protocol
            input_json_path = os.path.join(input_dir, f"{config.name}_{comp_idx}.json")
            with open(input_json_path, "w") as f:
                json.dump(input_json, f, indent=2)

            # Prepare dispatch input
            input_data = {
                "input_json_path": input_json_path,
                "output_dir": output_dir,
                "device": config.device,
                "repo_path": config.repo_path,
                "sif_path": config.sif_path,
                "model_dir": config.model_dir,
                "db_dir": config.db_dir,
                "verbose": config.verbose,
            }

            # Dispatch to worker (goes through DeviceManager)
            output_data = ToolInstance.dispatch(
                "alphafold3",
                input_data,
                instance=instance,
                config=config,
            )

            # Extract results from dict
            pdb_path = output_data["structure_pdb"]
            alphafold3_scores = output_data["metrics"]

            output_structures.append(
                Structure(
                    structure_filepath_or_content=pdb_path,
                    b_factor_type=BFactorType.PLDDT,
                    metrics=alphafold3_scores,
                    source="alphafold3-prediction",
                )
            )

    return AlphaFold3Output(
        structures=output_structures,
        metadata={
            "num_complexes": len(output_structures),
            "total_chains": sum(s.num_chains for s in output_structures),
        },
    )

# ============================================================================
# Helper Functions
# ============================================================================
def _generate_msas(
    input_json_data: AlphaFold3JSON,
    input_dir: str,
    colabfold_search_config: ColabfoldSearchConfig,
    verbose: bool = False,
) -> AlphaFold3JSON:
    """
    Generate multiple sequence alignments (MSAs) for protein chains using ColabFold search.

    Args:
        input_json_data: AlphaFold3 input JSON dictionary to update with MSA paths.
        input_dir: Directory for MSA output files (AlphaFold3 input directory).
        colabfold_search_config: Configuration for ColabFold MSA search.
        verbose: Whether to print progress messages.

    Returns:
        Updated input_json_data with MSA paths populated.
    """
    from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
        ColabfoldSearchInput,
        run_colabfold_search,
    )

    name = input_json_data["name"]

    # Collect unique sequences and track which chain indices use each.
    seq_to_indices: Dict[str, List[int]] = {}
    seq_to_name: Dict[str, str] = {}

    for seq_idx, seq_entry in enumerate(input_json_data["sequences"]):
        if "protein" not in seq_entry:
            continue
        sequence = seq_entry["protein"]["sequence"]
        if sequence not in seq_to_indices:
            chain_id = seq_entry["protein"]["id"]
            if isinstance(chain_id, list):
                chain_id = chain_id[0]
            seq_to_indices[sequence] = []
            seq_to_name[sequence] = f"{name}_{chain_id}_{seq_idx}"
        seq_to_indices[sequence].append(seq_idx)

    if not seq_to_indices:
        logger.debug("No protein sequences found, skipping MSA generation.")
        return input_json_data

    unique_seqs = list(seq_to_indices.keys())
    unique_names = [seq_to_name[s] for s in unique_seqs]

    # Create queries for ColabFold search
    queries = [(seq, name) for seq, name in zip(unique_seqs, unique_names)]
    colabfold_input = ColabfoldSearchInput(queries=queries)

    # Configure output directory for MSAs within the AlphaFold3 input directory
    msa_config = colabfold_search_config.model_copy(deep=True)
    msa_config.output_dir = input_dir
    msa_config.verbose = verbose

    logger.debug(
        f"Generating MSAs for {len(unique_seqs)} unique protein sequence(s) using ColabFold search..."
    )

    # Run ColabFold search
    try:
        colabfold_output = run_colabfold_search(colabfold_input, msa_config)
    except Exception as e:
        raise RuntimeError(f"ColabFold MSA search failed: {e}") from e

    # Process results and assign MSA paths to chains
    msa_paths: Dict[str, str] = {}

    for result in colabfold_output.results:
        if result.msa is not None:
            # Write the MSA to A3M format for AlphaFold3
            # The MSA class automatically handles insertion removal when loading A3M files
            # and to_a3m_file() writes clean A3M without lowercase insertions
            a3m_path = os.path.join(input_dir, "msas", f"{result.sequence_id}.a3m")
            os.makedirs(os.path.dirname(a3m_path), exist_ok=True)
            result.msa.to_a3m_file(a3m_path, query_index=0)

            # Store relative path from input_dir
            rel_path = os.path.relpath(a3m_path, input_dir)
            msa_paths[result.sequence_id] = rel_path

            logger.debug(
                f"Generated MSA for {result.sequence_id}: {result.num_homologs_found} homologs found"
            )
        else:
            logger.debug(f"Warning: No homologs found for {result.sequence_id}")

    # Assign MSA paths back to all chains that use each sequence
    for sequence, indices in seq_to_indices.items():
        seq_name = seq_to_name[sequence]
        if seq_name in msa_paths:
            for idx in indices:
                input_json_data["sequences"][idx]["protein"]["unpairedMsaPath"] = (
                    msa_paths[seq_name]
                )
                chain_id = input_json_data["sequences"][idx]["protein"]["id"]
                logger.debug(f"Assigned MSA '{msa_paths[seq_name]}' to chain {chain_id}")
        else:
            logger.debug(
                f"Warning: No MSA available for sequence used in chain indices {indices}"
            )

    return input_json_data


def _create_input_json_from_complex(
    sp_complex: StructurePredictionComplex,
    name: str,
    seed: int | List[int],
) -> AlphaFold3JSON:
    """
    Create input JSON data for AlphaFold3 inference from a list of components.

    The "alphafold3" JSON dialect is documented here:
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

    Also converts SMILES strings to CCD code using data from here:
    https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi

    Args:
        sp_complex: Complex to predict.
        name: Name identifier for this prediction job.
        seed: Random seed(s) for structure prediction.

    Returns:
        Dictionary formatted for AlphaFold3 input JSON.
    """
    if isinstance(seed, int):
        seed = [seed]

    input_json_data = {
        "name": name,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 2,
        "sequences": [],
    }

    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(sp_complex.chains) > len(chain_ids):
        # This is a hard limit on the PDB file format.
        # Consider moving to mmCIF if this becomes an issue.
        raise ValueError(f"Cannot provide more than {len(chain_ids)} chains")

    for idx, chain in enumerate(sp_complex.chains):
        mol_type = chain.entity_type  # Currently, we use the same conventions as AlphaFold3.
        sequence = chain.sequence

        if mol_type == "ligand":
            # AlphaFold3 does not allow MSA fields for ligands.
            ccd_code = map_smiles_to_ccd_code(sequence)
            if ccd_code is None:
                raise ValueError(
                    f"Unable to map SMILES to CCD code: {sequence}. Please ensure the SMILES is valid and in the CCD database."
                )
            else:
                # AlphaFold3 prefers CCD codes.
                sequence_entry = {
                    mol_type: {
                        "id": chain_ids[idx],
                        "ccdCodes": [ccd_code],
                    }
                }

        elif mol_type == "dna" or mol_type == "rna":
            # Ignore MSA fields for DNA and RNA.
            sequence_entry = {
                mol_type: {
                    "id": chain_ids[idx],
                    "sequence": sequence,
                }
            }

        else:
            sequence_entry = {
                mol_type: {
                    "id": chain_ids[idx],
                    "sequence": sequence,
                    "pairedMsa": "",
                    "unpairedMsa": "",
                }
            }

        # Add modifications from Chain object if present
        if chain.modifications and mol_type != "ligand":
            # Convert ChainModification objects to AlphaFold3 JSON format
            # AlphaFold3 uses different field names for different entity types
            alphafold3_modifications = []
            for mod in chain.modifications:
                if mol_type == "protein":
                    # Protein uses ptmType and ptmPosition
                    alphafold3_modifications.append({
                        "ptmType": mod.modification_code,
                        "ptmPosition": mod.position
                    })
                elif mol_type in ("dna", "rna"):
                    # DNA and RNA use modificationType and basePosition
                    alphafold3_modifications.append({
                        "modificationType": mod.modification_code,
                        "basePosition": mod.position
                    })

            # Add modifications to the sequence entry
            sequence_entry[mol_type]["modifications"] = alphafold3_modifications

        input_json_data["sequences"].append(sequence_entry)

    return input_json_data
