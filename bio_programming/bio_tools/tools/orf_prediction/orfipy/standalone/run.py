"""
Orfipy standalone runner for EnvManager venv execution.

Handles ORF prediction via the orfipy CLI tool and FASTA output parsing.
Communicates via JSON input/output files (EnvManager pattern).

Usage (called by EnvManager, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Parsing Utilities
# =============================================================================
def _parse_orfipy_header(header: str) -> Optional[Dict[str, Any]]:
    """
    Parse Orfipy FASTA header to extract ORF information.

    Args:
        header: Orfipy FASTA header string.

    Returns:
        Dictionary containing parsed ORF information, or None if parsing fails.
        Keys include: parent_id, orf_id, start, end, strand, frame.
    """
    # Split header into ID and metadata parts
    parts = header.split(" ", 1)
    if len(parts) < 2:
        return None

    orf_full_id = parts[0]
    metadata_part = parts[1]

    # Extract parent ID and ORF tag
    if "_" in orf_full_id:
        parent_id, orf_tag = orf_full_id.rsplit("_", 1)
    else:
        parent_id, orf_tag = orf_full_id, ""

    # Extract coordinates [start-end](strand)
    coord_match = re.search(r"\[(\d+)-(\d+)\]\(([+-])\)", metadata_part)
    if not coord_match:
        return None

    start_pos = int(coord_match.group(1))
    end_pos = int(coord_match.group(2))
    strand = coord_match.group(3)

    # Extract frame number
    frame_match = re.search(r"frame:(\d+)", metadata_part)
    frame = int(frame_match.group(1)) if frame_match else 1

    return {
        "parent_id": parent_id,
        "orf_id": orf_tag,
        "start": start_pos,
        "end": end_pos,
        "strand": strand,
        "frame": frame,
    }


def _parse_orfipy_results(
    aa_fasta: str | Path, nt_fasta: str | Path, seq_id: str
) -> List[Dict[str, Any]]:
    """
    Parse Orfipy peptide and nucleotide FASTA files into a list of ORF dicts.

    Args:
        aa_fasta: Path to Orfipy amino acid FASTA output.
        nt_fasta: Path to Orfipy nucleotide FASTA output.
        seq_id: Sequence identifier to assign as parent_id.

    Returns:
        List of dicts, each representing an ORF with keys: parent_id, orf_id,
        amino_acid_sequence, nucleotide_sequence, amino_acid_length,
        nucleotide_length, nucleotide_start, nucleotide_end, strand, frame.

    Raises:
        ValueError: If the number of amino acid and nucleotide records don't match.
    """
    from Bio import SeqIO

    aa_records = list(SeqIO.parse(str(aa_fasta), "fasta"))
    nt_records = list(SeqIO.parse(str(nt_fasta), "fasta"))

    # Ensure we have matching records
    if len(aa_records) != len(nt_records):
        raise ValueError(
            f"Mismatch between amino acid ({len(aa_records)}) and nucleotide ({len(nt_records)}) records"
        )

    data = []

    for aa_record, nt_record in zip(aa_records, nt_records):
        parsed_info = _parse_orfipy_header(aa_record.description)
        if parsed_info:  # Skip malformed headers
            # Remove only trailing stop codon marker (*) if present
            clean_aa_sequence = str(aa_record.seq).rstrip("*")
            nt_sequence = str(nt_record.seq)

            orf_dict = {
                "parent_id": seq_id,
                "orf_id": parsed_info["orf_id"],
                "strand": parsed_info["strand"],
                "frame": parsed_info["frame"],
                "amino_acid_sequence": clean_aa_sequence,
                "nucleotide_sequence": nt_sequence,
                "amino_acid_length": len(clean_aa_sequence),
                "nucleotide_length": len(nt_sequence),
                "nucleotide_start": parsed_info["start"] + 1,  # orfipy native 0-indexed -> 1-indexed
                "nucleotide_end": parsed_info["end"],  # orfipy native 0-indexed exclusive -> 1-indexed inclusive
            }
            data.append(orf_dict)

    return data


# =============================================================================
# Subprocess Execution
# =============================================================================
DNA_NUCLEOTIDES = {"A", "T", "C", "G"}


def _run_single_orfipy(sequence: str, seq_id: str, config: dict) -> List[Dict[str, Any]]:
    """Run Orfipy on a single DNA sequence string and return list of ORF dicts."""

    # Create temporary directory for isolated execution
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create input FASTA for single sequence
        input_fasta = temp_dir / "input.fasta"
        clean_seq = "".join(c for c in sequence.upper() if c in DNA_NUCLEOTIDES)
        with open(input_fasta, "w") as f:
            f.write(f">seq_0\n{clean_seq}\n")

        # Define output file paths (temp)
        aa_path = temp_dir / "orfipy_aa.faa"
        nt_path = temp_dir / "orfipy_nt.fna"

        # Build command (use venv's orfipy binary via sys.prefix)
        orfipy_bin = str(Path(sys.prefix) / "bin" / "orfipy")
        cmd = [
            orfipy_bin,
            str(input_fasta),
            "--procs",
            str(config.get("threads", 4)),
            "--start",
            config.get("start_codons", "ATG,GTG,TTG"),
            "--stop",
            config.get("stop_codons", "TAA,TAG,TGA"),
            "--strand",
            config.get("strand", "b"),
            "--min",
            str(config.get("min_len", 0)),
            "--max",
            str(config.get("max_len", 10000)),
            "--dna",
            str(nt_path),
            "--pep",
            str(aa_path),
        ]

        if config.get("include_stop", True):
            cmd.append("--include-stop")
        if config.get("translation_table") is not None:
            cmd.extend(["--table", str(config["translation_table"])])

        # Execute orfipy
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Orfipy failed for sequence '{seq_id}' with code {proc.returncode}: {proc.stderr}"
            )

        # Parse results
        orfs = _parse_orfipy_results(aa_path, nt_path, seq_id)
        return orfs


# =============================================================================
# Main Entry Point
# =============================================================================
def run_orfipy(input_data: dict) -> dict:
    """Run Orfipy ORF prediction on one or more sequences.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, config

    Returns:
        Dict with key: predicted_orfs (list of list of ORF dicts)
    """
    sequences = input_data["sequences"]
    sequence_ids = input_data["sequence_ids"]
    config = input_data.get("config", {})

    predicted_orfs = []
    for seq, seq_id in zip(sequences, sequence_ids):
        result = _run_single_orfipy(seq, seq_id, config)
        predicted_orfs.append(result)

    return {"predicted_orfs": predicted_orfs}


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

    output_data = run_orfipy(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
