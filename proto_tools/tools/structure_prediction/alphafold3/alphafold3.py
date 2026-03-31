"""
proto_tools/tools/structure_prediction/alphafold3/alphafold3.py

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

from tqdm import tqdm

logger = logging.getLogger(__name__)

from proto_tools.entities.ligands import map_smiles_to_ccd_code
from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    MSAStructurePredictionConfig,
    StructurePredictionComplex,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance

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
        complexes (list[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more sequences of proteins, DNA, RNA, or ligands.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.
    """

    # AlphaFold3 supports all standard entity types except glycan
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True

# Output:
AlphaFold3Output = StructurePredictionOutput

# Config:
class AlphaFold3Config(MSAStructurePredictionConfig):
    """Configuration object for AlphaFold3 structure prediction.

    This class defines configuration parameters for running AlphaFold3, a state-of-the-art
    multi-modal structure predictor.

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        name (str): Name of the folding job. Default: ``"af3_job"``.

        seeds (list[int]): Seeds to use for AlphaFold3. Default: ``[0]``.
            Note: AlphaFold3 will do five diffusion samples per seed, so this often can be
            set to a single seed. More seeds are required for complex docking tasks,
            such as antibody-antigen docking.

        output_dir (str | None): Path prefix for the AlphaFold3 output directory.
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

            # Write pre-computed MSAs to A3M files
            if inputs.msas:
                input_json = _assign_msas_to_input_json(
                    input_json, inputs.msas, input_dir, config.verbose
                )

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
def _assign_msas_to_input_json(
    input_json_data: AlphaFold3JSON,
    msas: dict[str, object],
    input_dir: str,
    verbose: bool = False,
) -> AlphaFold3JSON:
    """Write pre-computed MSAs to A3M files and assign paths to input JSON.

    Args:
        input_json_data (AlphaFold3JSON): AlphaFold3 input JSON dictionary to update with MSA paths.
        msas (dict[str, object]): Pre-computed MSAs keyed by sequence string.
        input_dir (str): Directory for MSA output files.
        verbose (bool): Whether to print progress messages.

    Returns:
        AlphaFold3JSON: Updated input_json_data with MSA paths populated.
    """
    msa_dir = os.path.join(input_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)

    for seq_idx, seq_entry in enumerate(input_json_data["sequences"]):
        if "protein" not in seq_entry:
            continue
        sequence = seq_entry["protein"]["sequence"]
        msa = msas.get(sequence)
        if msa is None:
            continue

        chain_id = seq_entry["protein"]["id"]
        if isinstance(chain_id, list):
            chain_id = chain_id[0]

        a3m_path = os.path.join(msa_dir, f"chain_{chain_id}_{seq_idx}.a3m")
        msa.to_a3m_file(a3m_path, query_index=0)

        rel_path = os.path.relpath(a3m_path, input_dir)
        seq_entry["protein"]["unpairedMsaPath"] = rel_path

        if verbose:
            logger.info(
                f"Assigned MSA to chain {chain_id} ({len(msa)} sequences)"
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
        sp_complex (StructurePredictionComplex): Complex to predict.
        name (str): Name identifier for this prediction job.
        seed (int | list[int]): Random seed(s) for structure prediction.

    Returns:
        AlphaFold3JSON: Dictionary formatted for AlphaFold3 input JSON.
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
