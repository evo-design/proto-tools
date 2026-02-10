"""
alphafold3.py

Protein structure prediction using AlphaFold3.

This module provides standardized interfaces for protein structure prediction
using AlphaFold3 from Google DeepMind.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Optional

from pydantic import model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)

from bio_programming.bio_tools.tools.utils import ConfigField
from bio_programming.bio_tools.entities.ligands import map_smiles_to_ccd_code
from bio_programming.bio_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
)
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

from .inference import AlphaFold3JSON, alphafold3_inference


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
@tool(
    key="alphafold3",
    label="AlphaFold3 Structure Prediction",
    input=AlphaFold3Input,
    config=AlphaFold3Config,
    output=AlphaFold3Output,
    description="Protein structure prediction using AlphaFold3",
)
@tool_cache_iterable(
    input_iterable_field="complexes",
    output_iterable_field="structures",
    tool_name="alphafold3",
)
def run_alphafold3(
    inputs: AlphaFold3Input, config: AlphaFold3Config
) -> AlphaFold3Output:
    """Predict protein 3D structures using AlphaFold3."""

    # NOTE: the cloud runtime deployment will not be supported for AlphaFold3.
    if use_cloud_gpu():
        raise ValueError(
            "AlphaFold3 is not supported for the cloud runtime deployment. Please run on a local GPU."
        )

    output_structures: List[Structure] = []

    for comp_idx, comp in tqdm(enumerate(inputs.complexes), desc="Folding structures (AlphaFold3)", unit="complex", total=len(inputs.complexes)):
        input_json = _create_input_json_from_complex(
            comp,
            f"{config.name}_{comp_idx}",
            config.seeds,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            if config.output_dir is None:
                # Create inside temp directory for auto-cleanup
                output_dir = os.path.join(temp_dir, f"{config.name}_{comp_idx}_af3_results")
            else:
                # Create at specified path (persists after execution)
                output_dir = f"{config.output_dir}_af3_results"

            pdb_path, alphafold3_scores = alphafold3_inference(
                input_json=input_json,
                output_dir=output_dir,
                use_msa=config.use_msa,
                colabfold_search_config=config.colabfold_search_config,
                repo_path=config.repo_path,
                sif_path=config.sif_path,
                model_dir=config.model_dir,
                db_dir=config.db_dir,
                verbose=config.verbose,
            )

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
