"""Prodigal standalone runner for ToolInstance venv execution.

Handles prokaryotic gene prediction via pyrodigal Python bindings.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import sys
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Any

import pyrodigal
from standalone_helpers import get_logger

logger = get_logger(__name__)


# =============================================================================
# Gene Prediction
# =============================================================================
def _process_sequence(
    gene_finder: pyrodigal.GeneFinder,
    seq_idx_and_seq: tuple[int, str],
) -> tuple[int, list[dict[str, Any]]]:
    """Process a single sequence using pyrodigal.

    Returns:
        Tuple of (sequence_index, list of ORF dicts)
    """
    seq_idx, sequence = seq_idx_and_seq
    sequence_bytes = sequence.encode("utf-8")
    genes = gene_finder.find_genes(sequence_bytes)
    orfs = []

    for i, gene in enumerate(genes, start=1):
        protein_seq = gene.translate().rstrip("*")
        nucleotide_seq = gene.sequence()

        description = (
            f"{i} # {gene.begin} # {gene.end} # {gene.strand} # "
            f"ID={i};partial={gene.partial_begin:02d}_{gene.partial_end:02d};"
            f"start_type={gene.start_type};rbs_motif={gene.rbs_motif};"
            f"rbs_spacer={gene.rbs_spacer};gc_cont={gene.gc_cont:.3f}"
        )

        metrics = {
            "start_type": gene.start_type,
            "rbs_motif": gene.rbs_motif,
            "rbs_spacer": gene.rbs_spacer,
            "partial_begin": int(gene.partial_begin),
            "partial_end": int(gene.partial_end),
            "description": description,
            "partial": f"{gene.partial_begin:02d}_{gene.partial_end:02d}",
        }

        orf_dict = {
            "parent_id": f"seq_{seq_idx}",
            "orf_id": f"gene_{i}",
            "strand": "+" if gene.strand == 1 else "-",
            # Forward (+): translation starts at gene.begin
            # Reverse (-): translation starts at gene.end (reads backward)
            "frame": ((gene.begin - 1) % 3) + 1 if gene.strand == 1 else ((gene.end - 1) % 3) + 1,
            "amino_acid_sequence": protein_seq,
            "nucleotide_sequence": nucleotide_seq,
            "amino_acid_length": len(protein_seq),
            "nucleotide_length": len(nucleotide_seq),
            "nucleotide_start": int(gene.begin),
            "nucleotide_end": int(gene.end),
            "metrics": metrics,
        }
        orfs.append(orf_dict)

    return seq_idx, orfs


# =============================================================================
# Main Entry Point
# =============================================================================
def run_prodigal(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run Prodigal gene prediction on one or more sequences.

    Args:
        input_data: Dict with keys: sequences, config

    Returns:
        Dict with key: predicted_orfs (list of list of ORF dicts)
    """
    sequences = input_data["sequences"]
    config = input_data.get("config", {})

    meta_mode = config["meta_mode"]
    closed_ends = config["closed_ends"]
    mask = config["mask"]
    min_gene = config["min_gene"]
    num_threads = config["num_threads"]
    translation_table = config["translation_table"]

    if meta_mode and translation_table != 11:
        logger.warning(
            "translation_table=%s is ignored in meta mode. "
            "Metagenomic models use their own built-in translation tables. "
            "Set meta_mode=False to use a custom translation table.",
            translation_table,
        )

    # Initialize Prodigal gene finder
    gene_finder = pyrodigal.GeneFinder(
        meta=meta_mode,
        closed=closed_ends,
        mask=mask,
        min_gene=min_gene,
    )

    if not meta_mode:
        # Concatenate all sequences for training
        training_seq = "".join(sequences)
        gene_finder.train(training_seq.encode("utf-8"), translation_table=translation_table)

    # Process in parallel if multiple sequences and threads > 1
    if num_threads > 1 and len(sequences) > 1:
        with ThreadPool(num_threads) as pool:
            process_with_finder = partial(_process_sequence, gene_finder)
            results = pool.map(process_with_finder, enumerate(sequences))
        all_results = [orfs for _, orfs in results]
    else:
        # Sequential
        all_results = [_process_sequence(gene_finder, (i, seq))[1] for i, seq in enumerate(sequences)]

    return {"predicted_orfs": all_results}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    return run_prodigal(input_dict)


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================


def to_device(device: str) -> dict[str, Any]:
    """Passthrough - tool does not maintain persistent state."""
    return {"success": True, "device": device}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"prodigal: usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path) as f:
        input_data = json.load(f)

    output_data = run_prodigal(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
