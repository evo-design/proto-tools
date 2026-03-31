"""
TMalign standalone runner for ToolInstance venv execution.

Pairwise protein structure alignment via TM-score.
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


def run_tmalign_alignment(pdb_text_1: str, pdb_text_2: str) -> dict:
    """Run TMalign on two PDB text blobs and parse TM-scores.

    Args:
        pdb_text_1: PDB content of structure 1 (query / candidate).
        pdb_text_2: PDB content of structure 2 (reference / target).

    Returns:
        Dict with tm_score_chain_1 and tm_score_chain_2.
    """
    binary = Path(sys.executable).parent / "TMalign"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        path_1 = tmp_path / "structure_1.pdb"
        path_2 = tmp_path / "structure_2.pdb"
        path_1.write_text(pdb_text_1)
        path_2.write_text(pdb_text_2)

        result = subprocess.run(
            [str(binary), str(path_1), str(path_2)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"TMalign failed with code {result.returncode}: {result.stderr}")

    output = result.stdout

    # Parse both TM-scores from output
    # TMalign outputs (format varies by version):
    #   TM-score= X.XXXX (normalized by length of Structure_1: L=N, d0=X.XX)
    #   TM-score= X.XXXX (normalized by length of Structure_2: L=N, d0=X.XX)
    m1 = re.search(
        r"TM-score=\s*([0-9.]+)\s+\((?:if )?normalized by length of (?:Chain|Structure)_1",
        output,
    )
    m2 = re.search(
        r"TM-score=\s*([0-9.]+)\s+\((?:if )?normalized by length of (?:Chain|Structure)_2",
        output,
    )

    if not m1 or not m2:
        raise RuntimeError(
            f"Could not parse TM-scores from TMalign output:\n{output}"
        )

    return {
        "tm_score_chain_1": float(m1.group(1)),
        "tm_score_chain_2": float(m2.group(1)),
    }


def dispatch(input_dict: dict) -> dict:
    """Entry point for persistent-worker execution."""
    return run_tmalign_alignment(
        pdb_text_1=input_dict["pdb_text_1"],
        pdb_text_2=input_dict["pdb_text_2"],
    )


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    output_data = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f, indent=2)
