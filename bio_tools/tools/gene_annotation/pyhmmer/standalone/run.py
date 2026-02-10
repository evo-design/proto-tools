"""
PyHMMER standalone runner for EnvManager venv execution.

Handles hmmsearch, hmmscan, phmmer, nhmmer, and jackhmmer operations.
Communicates via JSON input/output files (EnvManager pattern).

Usage (called by EnvManager, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

import pyhmmer
import pyhmmer.easel
import pyhmmer.plan7

# ============================================================================
# Helpers
# ============================================================================


def _create_hmms_from_file(hmm_file_path: str) -> List[pyhmmer.plan7.HMM]:
    """Create PyHMMER HMM objects from file path."""
    hmms = []
    with pyhmmer.plan7.HMMFile(hmm_file_path) as hmm_file:
        hmms.extend(list(hmm_file))
    return hmms


def _create_sequences_from_strings(
    sequences: List[str],
    alphabet: str = "amino",
) -> List[pyhmmer.easel.DigitalSequence]:
    """Create PyHMMER digital sequence objects from strings."""
    if alphabet == "amino":
        alphabet_obj = pyhmmer.easel.Alphabet.amino()
    elif alphabet == "dna":
        alphabet_obj = pyhmmer.easel.Alphabet.dna()
    elif alphabet == "rna":
        alphabet_obj = pyhmmer.easel.Alphabet.rna()
    else:
        raise ValueError(f"Unsupported alphabet: {alphabet}")

    digital_sequences = []

    for i, seq_str in enumerate(sequences):
        text_seq = pyhmmer.easel.TextSequence(
            name=f"{i}".encode("utf-8"),
            sequence=seq_str,
        )
        digital_sequences.append(text_seq.digitize(alphabet_obj))

    return digital_sequences


def _convert_query_field(field) -> str:
    """Convert a query field to a string."""
    if field is None:
        return "-"
    if isinstance(field, bytes):
        return field.decode("utf-8")
    elif isinstance(field, str):
        return field
    else:
        return "-"


def _convert_hits_to_dicts(
    hits: list,
    queries: list,
) -> Tuple[List[Dict], List[Dict]]:
    """Convert PyHMMER Hit objects to lists of dicts (JSON-serializable).

    Args:
        hits: List of lists of PyHMMER TopHit objects.
        queries: List of query HMM or DigitalSequence objects.

    Returns:
        Tuple of (sequence_hits, domain_hits) as lists of dicts.
    """
    if len(hits) != len(queries):
        raise ValueError("The number of TopHit objects and queries must be the same")

    sequence_data = []
    domain_data = []

    for query_idx, query_object in enumerate(queries):
        query_name = _convert_query_field(query_object.name)
        query_accession = _convert_query_field(query_object.accession)
        query_description = _convert_query_field(query_object.description)

        for hit in hits[query_idx]:
            # Sequence-level information
            seq_row = {
                "query_name": query_name,
                "query_accession": query_accession,
                "query_description": query_description,
                "query_idx": query_idx,
                "target_name": _convert_query_field(hit.name),
                "target_accession": (
                    _convert_query_field(hit.accession) if hit.accession else "-"
                ),
                "target_description": (
                    _convert_query_field(hit.description) if hit.description else "-"
                ),
                "evalue": float(hit.evalue),
                "score": float(hit.score),
                "bias": float(hit.bias),
                "sum_score": float(hit.sum_score),
                "reported": bool(hit.reported),
                "included": bool(hit.included),
                "pvalue": float(hit.pvalue),
                "num_domains": len(hit.domains),
            }
            sequence_data.append(seq_row)

            # Domain-level information
            for i, domain in enumerate(hit.domains):
                domain_row = {
                    "query_name": query_name,
                    "query_accession": query_accession,
                    "query_description": query_description,
                    "query_idx": query_idx,
                    "target_name": _convert_query_field(hit.name),
                    "target_accession": (
                        _convert_query_field(hit.accession) if hit.accession else "-"
                    ),
                    "target_description": (
                        _convert_query_field(hit.description) if hit.description else "-"
                    ),
                    "hmm_length": int(domain.alignment.hmm_length),
                    "hmm_from": int(domain.alignment.hmm_from),
                    "hmm_to": int(domain.alignment.hmm_to),
                    "target_from": int(domain.alignment.target_from),
                    "target_to": int(domain.alignment.target_to),
                    "target_length": int(domain.alignment.target_length),
                    "c_evalue": float(domain.c_evalue),
                    "i_evalue": float(domain.i_evalue),
                    "domain_score": float(domain.score),
                    "domain_bias": float(domain.bias),
                    "domain_idx": i,
                    "env_from": int(domain.env_from),
                    "env_to": int(domain.env_to),
                    "envelope_score": float(domain.envelope_score),
                    "domain_included": bool(domain.included),
                    "domain_reported": bool(domain.reported),
                    "domain_pvalue": float(domain.pvalue),
                }
                domain_data.append(domain_row)

    return sequence_data, domain_data


# ============================================================================
# Implementation
# ============================================================================


def run_hmmsearch(input_data: dict) -> dict:
    """Run PyHMMER hmmsearch (HMM profiles vs sequences).

    Args:
        input_data: Dict with keys: hmm_path, sequences, num_threads,
                    evalue_threshold, score_threshold, domain_evalue_threshold,
                    domain_score_threshold

    Returns:
        Dict with keys: sequence_hits (list of dicts), domain_hits (list of dicts)
    """
    hmms = _create_hmms_from_file(input_data["hmm_path"])
    sequences = _create_sequences_from_strings(
        input_data["sequences"], alphabet="amino"
    )

    query_hits = list(
        pyhmmer.hmmsearch(
            queries=hmms,
            sequences=sequences,
            cpus=input_data.get("num_threads", 0),
            E=input_data.get("evalue_threshold", 10.0),
            T=input_data.get("score_threshold"),
            domE=input_data.get("domain_evalue_threshold", 10.0),
            domT=input_data.get("domain_score_threshold"),
        )
    )

    sequence_hits, domain_hits = _convert_hits_to_dicts(
        hits=query_hits, queries=hmms
    )

    return {
        "sequence_hits": sequence_hits,
        "domain_hits": domain_hits,
        "num_hmms": len(hmms),
        "num_sequences": len(sequences),
    }


def run_hmmscan(input_data: dict) -> dict:
    """Run PyHMMER hmmscan (sequences vs HMM database).

    Args:
        input_data: Dict with keys: hmm_db_path, sequences, num_threads,
                    evalue_threshold, score_threshold, domain_evalue_threshold,
                    domain_score_threshold

    Returns:
        Dict with keys: sequence_hits (list of dicts), domain_hits (list of dicts)
    """
    hmms = _create_hmms_from_file(input_data["hmm_db_path"])
    sequences = _create_sequences_from_strings(
        input_data["sequences"], alphabet="amino"
    )

    query_hits = list(
        pyhmmer.hmmscan(
            queries=sequences,
            profiles=hmms,
            cpus=input_data.get("num_threads", 0),
            E=input_data.get("evalue_threshold", 10.0),
            T=input_data.get("score_threshold"),
            domE=input_data.get("domain_evalue_threshold", 10.0),
            domT=input_data.get("domain_score_threshold"),
        )
    )

    sequence_hits, domain_hits = _convert_hits_to_dicts(
        hits=query_hits, queries=sequences
    )

    return {
        "sequence_hits": sequence_hits,
        "domain_hits": domain_hits,
        "num_hmms": len(hmms),
        "num_queries": len(sequences),
    }


def run_phmmer(input_data: dict) -> dict:
    """Run PyHMMER phmmer (sequences vs sequences).

    Args:
        input_data: Dict with keys: sequences, target_sequences, num_threads,
                    evalue_threshold, score_threshold, domain_evalue_threshold,
                    domain_score_threshold

    Returns:
        Dict with keys: sequence_hits (list of dicts), domain_hits (list of dicts)
    """
    query_sequences = _create_sequences_from_strings(
        input_data["sequences"], alphabet="amino"
    )
    target_sequences = _create_sequences_from_strings(
        input_data["target_sequences"], alphabet="amino"
    )

    query_hits = list(
        pyhmmer.phmmer(
            queries=query_sequences,
            sequences=target_sequences,
            cpus=input_data.get("num_threads", 0),
            E=input_data.get("evalue_threshold", 10.0),
            T=input_data.get("score_threshold"),
            domE=input_data.get("domain_evalue_threshold", 10.0),
            domT=input_data.get("domain_score_threshold"),
        )
    )

    sequence_hits, domain_hits = _convert_hits_to_dicts(
        hits=query_hits, queries=query_sequences
    )

    return {
        "sequence_hits": sequence_hits,
        "domain_hits": domain_hits,
        "num_query_sequences": len(query_sequences),
        "num_target_sequences": len(target_sequences),
    }


def run_nhmmer(input_data: dict) -> dict:
    """Run PyHMMER nhmmer (nucleotide sequences vs nucleotide sequences)."""
    query_sequences = _create_sequences_from_strings(
        input_data["sequences"], alphabet="dna"
    )
    target_sequences = _create_sequences_from_strings(
        input_data["target_sequences"], alphabet="dna"
    )

    query_hits = list(
        pyhmmer.nhmmer(
            queries=query_sequences,
            sequences=target_sequences,
            cpus=input_data.get("num_threads", 0),
            E=input_data.get("evalue_threshold", 10.0),
            T=input_data.get("score_threshold"),
            domE=input_data.get("domain_evalue_threshold", 10.0),
            domT=input_data.get("domain_score_threshold"),
        )
    )

    sequence_hits, domain_hits = _convert_hits_to_dicts(
        hits=query_hits, queries=query_sequences
    )

    return {
        "sequence_hits": sequence_hits,
        "domain_hits": domain_hits,
        "num_query_sequences": len(query_sequences),
        "num_target_sequences": len(target_sequences),
    }


def run_jackhmmer(input_data: dict) -> dict:
    """Run PyHMMER jackhmmer (iterative protein sequence search)."""
    query_sequences = _create_sequences_from_strings(
        input_data["sequences"], alphabet="amino"
    )
    target_sequences = _create_sequences_from_strings(
        input_data["target_sequences"], alphabet="amino"
    )

    iteration_results = list(
        pyhmmer.jackhmmer(
            queries=query_sequences,
            sequences=target_sequences,
            max_iterations=input_data.get("max_iterations", 5),
            checkpoints=False,
            cpus=input_data.get("num_threads", 0),
            E=input_data.get("evalue_threshold", 10.0),
            T=input_data.get("score_threshold"),
            domE=input_data.get("domain_evalue_threshold", 10.0),
            domT=input_data.get("domain_score_threshold"),
        )
    )

    final_hits = [result.hits for result in iteration_results]
    sequence_hits, domain_hits = _convert_hits_to_dicts(
        hits=final_hits, queries=query_sequences
    )

    return {
        "sequence_hits": sequence_hits,
        "domain_hits": domain_hits,
        "num_query_sequences": len(query_sequences),
        "num_target_sequences": len(target_sequences),
        "iterations_per_query": [int(result.iteration) for result in iteration_results],
        "converged_per_query": [bool(result.converged) for result in iteration_results],
    }


# =============================================================================
# Entry point (called by EnvManager.call_standalone_script_in_venv)
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    operation = input_data["operation"]

    if operation == "hmmsearch":
        output_data = run_hmmsearch(input_data)
    elif operation == "hmmscan":
        output_data = run_hmmscan(input_data)
    elif operation == "phmmer":
        output_data = run_phmmer(input_data)
    elif operation == "nhmmer":
        output_data = run_nhmmer(input_data)
    elif operation == "jackhmmer":
        output_data = run_jackhmmer(input_data)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
