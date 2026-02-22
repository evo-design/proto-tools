"""
Orfipy standalone runner for ToolInstance venv execution.

Handles ORF prediction via the orfipy CLI tool and FASTA output parsing.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
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


# =============================================================================
# Subprocess Execution
# =============================================================================
DNA_NUCLEOTIDES = {"A", "T", "C", "G"}


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

    if not sequences:
        return {"predicted_orfs": []}

    # Batch all sequences into a single orfipy call for performance.
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Write all sequences to one FASTA file
        input_fasta = temp_dir / "input.fasta"
        with open(input_fasta, "w") as f:
            for i, seq in enumerate(sequences):
                clean_seq = "".join(c for c in seq.upper() if c in DNA_NUCLEOTIDES)
                f.write(f">seq_{i}\n{clean_seq}\n")

        aa_path = temp_dir / "orfipy_aa.faa"
        nt_path = temp_dir / "orfipy_nt.fna"

        orfipy_bin = str(Path(sys.prefix) / "bin" / "orfipy")
        outdir = temp_dir / "orfipy_out"
        cmd = [
            orfipy_bin,
            str(input_fasta),
            "--outdir",
            str(outdir),
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

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Orfipy failed with code {proc.returncode}: {proc.stderr}"
            )

        # Parse all results and group by parent sequence
        from Bio import SeqIO

        aa_records = list(SeqIO.parse(str(aa_path), "fasta")) if aa_path.exists() else []
        nt_records = list(SeqIO.parse(str(nt_path), "fasta")) if nt_path.exists() else []

        if len(aa_records) != len(nt_records):
            raise ValueError(
                f"Mismatch between amino acid ({len(aa_records)}) and nucleotide ({len(nt_records)}) records"
            )

        # Group ORFs by parent sequence index
        orfs_by_seq: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(sequences))}
        for aa_record, nt_record in zip(aa_records, nt_records):
            parsed_info = _parse_orfipy_header(aa_record.description)
            if not parsed_info:
                continue
            parent_id = parsed_info["parent_id"]  # e.g. "seq_0"
            try:
                seq_idx = int(parent_id.replace("seq_", ""))
            except (ValueError, AttributeError):
                continue

            clean_aa_sequence = str(aa_record.seq).rstrip("*")
            nt_sequence = str(nt_record.seq)
            orf_dict = {
                "parent_id": sequence_ids[seq_idx],
                "orf_id": parsed_info["orf_id"],
                "strand": parsed_info["strand"],
                "frame": parsed_info["frame"],
                "amino_acid_sequence": clean_aa_sequence,
                "nucleotide_sequence": nt_sequence,
                "amino_acid_length": len(clean_aa_sequence),
                "nucleotide_length": len(nt_sequence),
                "nucleotide_start": parsed_info["start"] + 1,
                "nucleotide_end": parsed_info["end"],
            }
            orfs_by_seq[seq_idx].append(orf_dict)

        predicted_orfs = [orfs_by_seq[i] for i in range(len(sequences))]

    return {"predicted_orfs": predicted_orfs}


def dispatch(input_dict: dict) -> dict:
    """Entry point for persistent-worker execution."""
    return run_orfipy(input_dict)


# =============================================================================
# Entry point (called by ToolInstance)
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
