"""USalign standalone runner for ToolInstance venv execution.

Universal structure alignment (monomers, multimers, nucleic acids) via TM-score.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def run_usalign_alignment(pdb_text_1: str, pdb_text_2: str) -> dict[str, Any]:
    """Run USalign on two PDB text blobs and parse TM-scores.

    Calls USalign with ``-mm 1 -ter 1`` for multimer support.

    Args:
        pdb_text_1: PDB content of structure 1 (query / candidate).
        pdb_text_2: PDB content of structure 2 (reference / target).

    Returns:
        Dict with tm_score_structure_1 and tm_score_structure_2.
    """
    binary = Path(sys.executable).parent / "USalign"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        path_1 = tmp_path / "structure_1.pdb"
        path_2 = tmp_path / "structure_2.pdb"
        path_1.write_text(pdb_text_1)
        path_2.write_text(pdb_text_2)

        try:
            result = subprocess.run(
                [str(binary), str(path_1), str(path_2), "-mm", "1", "-ter", "1"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
            raise RuntimeError(
                f"usalign: alignment failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
            ) from e

    output = result.stdout

    # Parse both TM-scores from output (format varies by version):
    #   TM-score= X.XXXX (normalized by length of Structure_1: ...)
    #   TM-score= X.XXXX (normalized by length of Structure_2: ...)
    m1 = re.search(
        r"TM-score=\s*([0-9.]+)\s+\((?:if )?normalized by length of (?:Chain|Structure)_1",
        output,
    )
    m2 = re.search(
        r"TM-score=\s*([0-9.]+)\s+\((?:if )?normalized by length of (?:Chain|Structure)_2",
        output,
    )

    if not m1 or not m2:
        raise RuntimeError(f"usalign: could not parse TM-scores from USalign output: {output[-500:]}")

    return {
        "tm_score_structure_1": float(m1.group(1)),
        "tm_score_structure_2": float(m2.group(1)),
    }


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    return run_usalign_alignment(
        pdb_text_1=input_dict["pdb_text_1"],
        pdb_text_2=input_dict["pdb_text_2"],
    )


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usalign: usage: python run.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    output_data = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f, indent=2)
