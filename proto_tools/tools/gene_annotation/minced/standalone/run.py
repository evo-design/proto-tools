"""
MinCED standalone runner for ToolInstance venv execution.

Handles CRISPR array detection via the minced CLI tool and output parsing.
Communicates via JSON input/output files.

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# Output Parsing
# =============================================================================
def _process_minced_output(output: str) -> List[List[Dict[str, Any]]]:
    """Parse MinCED stdout into structured CRISPR array data.

    Args:
        output: Raw stdout from minced binary.

    Returns:
        List of CRISPR arrays, each containing a list of repeat-spacer dicts.
    """
    lines = output.split("\n")
    all_arrays = []
    current_array = []
    parse = False

    for line in lines:
        if "CRISPR" in line:
            if current_array:
                all_arrays.append(current_array)
                current_array = []
            continue

        if line.startswith("-------"):
            parse = not parse
            continue

        if parse and line.strip():
            parts = line.split()
            if not parts:
                continue

            position = int(parts[0]) if parts[0].isdigit() else 0
            repeat = parts[1] if len(parts) > 1 else ""
            spacer = None
            repeat_length = None
            spacer_length = None

            if "[" in line:
                spacer_index = line.find("[")
                spacer_parts = line[:spacer_index].split()[2:]
                spacer = " ".join(spacer_parts).strip() or None
                length_str = line[spacer_index:].strip("[]")
                length_parts = length_str.split(", ")
                repeat_length = int(length_parts[0]) if len(length_parts) > 0 and length_parts[0].strip().isdigit() else None
                spacer_length = int(length_parts[1]) if len(length_parts) > 1 and length_parts[1].strip().isdigit() else None
            else:
                spacer = " ".join(parts[2:]) if len(parts) > 2 else None

            current_array.append({
                "position": position,
                "repeat": repeat,
                "spacer": spacer,
                "repeat_length": repeat_length,
                "spacer_length": spacer_length,
            })

    if current_array:
        all_arrays.append(current_array)

    return all_arrays


def _run_single_minced(
    sequence: str, seq_id: str, config: dict
) -> Dict[str, Any]:
    """Run MinCED on a single nucleotide sequence.

    Args:
        sequence: Nucleotide sequence string.
        seq_id: Sequence identifier.
        config: Configuration dict with min_num_repeats and min_repeat_length.

    Returns:
        Dict with sequence_id and crispr_arrays.
    """
    min_nr = config.get("min_num_repeats", 3)
    min_rl = config.get("min_repeat_length", 27)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_fasta = temp_dir_path / "input.fasta"

        clean_seq = "".join(c for c in sequence.upper() if c in "ATCGN")
        with open(input_fasta, "w") as f:
            f.write(f">{seq_id}\n{clean_seq}\n")

        minced_bin = str(Path(sys.prefix) / "bin" / "minced")
        cmd = [
            minced_bin,
            "-minNR", str(min_nr),
            "-minRL", str(min_rl),
            str(input_fasta),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            return {
                "sequence_id": seq_id,
                "crispr_arrays": [],
            }

        if proc.returncode != 0:
            return {
                "sequence_id": seq_id,
                "crispr_arrays": [],
            }

        arrays_data = _process_minced_output(proc.stdout)
        crispr_arrays = [
            {"repeats_and_spacers": array} for array in arrays_data
        ]

        return {
            "sequence_id": seq_id,
            "crispr_arrays": crispr_arrays,
        }


# =============================================================================
# Main Entry Point
# =============================================================================
def run_minced(input_data: dict) -> dict:
    """Run MinCED CRISPR detection on one or more sequences.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, config.

    Returns:
        Dict with key: results (list of per-sequence result dicts).
    """
    sequences = input_data["sequences"]
    sequence_ids = input_data["sequence_ids"]
    config = input_data.get("config", {})

    results = []
    for seq, seq_id in zip(sequences, sequence_ids):
        result = _run_single_minced(seq, seq_id, config)
        results.append(result)

    return {"results": results}


def dispatch(input_dict: dict) -> dict:
    """Entry point for persistent-worker execution."""
    return run_minced(input_dict)


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================

def to_device(device: str) -> dict:
    """Passthrough for CLI tool - automatically unloads after each call."""
    # CLI tool that spawns subprocesses and naturally unloads after each call
    # This is a passthrough for standardization with other tools
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


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

    output_data = run_minced(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
