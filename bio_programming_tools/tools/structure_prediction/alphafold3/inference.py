"""
AlphaFold3 Structure Prediction Pipeline.

This script provides utilities for running AlphaFold3 structure predictions
via Singularity containers, with support for MSA generation via ColabFold search.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np
from Bio import PDB

logger = logging.getLogger(__name__)

from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    run_colabfold_search,
)


class AlphaFold3ExecutionError(Exception):
    """Raised when AlphaFold3 execution fails."""

    pass


AlphaFold3JSON = Dict[str, Any]


def _generate_msas(
    input_json_data: dict,
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
        raise AlphaFold3ExecutionError(f"ColabFold MSA search failed: {e}") from e

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


def _extract_structure_and_scores(
    output_dir: str,
    name: str,
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract predicted structure and confidence scores from AlphaFold3 output.

    Args:
        output_dir: Directory containing AlphaFold3 output.
        name: Name of the prediction job.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (pdb_path, scores_dict).
    """
    alphafold3_results_folder = os.path.join(output_dir, name)
    alphafold3_structure = os.path.join(alphafold3_results_folder, f"{name}_model.cif")

    # Convert mmCIF structure file to PDB format.
    pdb_path = os.path.join(output_dir, f"{name}_af3.pdb")
    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", alphafold3_structure)
    io.set_structure(structure)
    io.save(pdb_path)

    # Extract confidence scores from AlphaFold3 JSON output files.
    summary_confidences_path = os.path.join(
        alphafold3_results_folder, f"{name}_summary_confidences.json"
    )
    full_confidences_path = os.path.join(alphafold3_results_folder, f"{name}_confidences.json")

    with open(summary_confidences_path, "r") as f:
        summary_metrics = json.load(f)
    with open(full_confidences_path, "r") as f:
        full_metrics = json.load(f)

    alphafold3_scores: Dict[str, Any] = {}
    alphafold3_scores["avg_plddt"] = float(np.mean(full_metrics["atom_plddts"]))
    alphafold3_scores["avg_pae"] = float(np.mean(np.array(full_metrics["pae"])))
    alphafold3_scores["ptm"] = summary_metrics.get("ptm")
    alphafold3_scores["iptm"] = summary_metrics.get("iptm")
    alphafold3_scores["ranking_score"] = summary_metrics.get("ranking_score")

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(alphafold3_scores, f, indent=2)

    return pdb_path, alphafold3_scores


def alphafold3_inference(
    input_json: AlphaFold3JSON,
    output_dir: str,
    use_msa: bool,
    colabfold_search_config: ColabfoldSearchConfig,
    repo_path: str,
    sif_path: str,
    model_dir: str,
    db_dir: str,
    verbose: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Execute AlphaFold3 structure prediction via Singularity container.

    Args:
        input_json: AlphaFold3 input JSON as a Python dictionary.
        output_dir: Directory for output files.
        use_msa: Whether to generate and use MSAs for protein chains.
        colabfold_search_config: Configuration for ColabFold MSA search.
        repo_path: Path to AlphaFold3 repository.
        sif_path: Path to Singularity image.
        model_dir: Path to model weights.
        db_dir: Path to databases.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (pdb_path, scores_dict).

    Raises:
        AlphaFold3ExecutionError: If AlphaFold3 execution fails.
    """
    original_output_dir = output_dir
    counter = 1
    # Check if dir exists, if so, append .1, .2, etc.
    while os.path.exists(output_dir):
        output_dir = f"{original_output_dir}.{counter}"
        counter += 1
    os.makedirs(output_dir)
    if counter > 1:
        logger.debug(f"Output dir existed, created new directory: {output_dir}")

    # Prepare input directory and generate MSAs.

    input_dir = os.path.join(output_dir, "af3_inputs")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{input_json['name']}.json")

    if use_msa:
        input_json = _generate_msas(
            input_json,
            input_dir,
            colabfold_search_config,
            verbose=verbose,
        )
    else:
        logger.debug("MSA generation skipped (use_msa=False).")

    with open(input_path, "w") as f:
        json.dump(input_json, f, indent=2)

    for name, path in [
        ("repo_path", repo_path),
        ("model_dir", model_dir),
        ("db_dir", db_dir),
        ("sif_path", sif_path),
    ]:
        if not os.path.exists(path):
            raise AlphaFold3ExecutionError(f"Path does not exist for {name}: {path}")

    # Actually run AlphaFold3.
    run_cmds = [
        "singularity",
        "exec",
        "--nv",
        "--env",
        "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        # Mount required directories into the container.
        "--bind",
        f"{output_dir}:/root/af_output",
        "--bind",
        f"{input_dir}:/root/af_input",
        "--bind",
        f"{model_dir}:/root/models",
        "--bind",
        f"{db_dir}:/root/public_databases",
        "--bind",
        f"{repo_path}:/root/alphafold3",
        sif_path,
        "python",
        "/root/alphafold3/run_alphafold.py",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
        f"--json_path=/root/af_input/{os.path.basename(input_path)}",
    ]

    logger.debug(f"Executing AlphaFold3 via Singularity...")
    logger.debug(f"  Input: {input_path}")
    logger.debug(f"  Output: {output_dir}")

    process = subprocess.Popen(
        run_cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        error_msg = (
            f"AlphaFold3 failed with return code {process.returncode}\n"
            f"Command: {' '.join(run_cmds)}\n"
            f"Stderr:\n{stderr}"
        )
        raise AlphaFold3ExecutionError(error_msg)

    logger.debug("AlphaFold3 execution completed successfully.")

    # Process data and return.

    return _extract_structure_and_scores(
        output_dir,
        input_json["name"],
        verbose=verbose,
    )
