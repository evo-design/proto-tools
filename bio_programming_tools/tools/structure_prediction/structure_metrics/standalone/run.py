"""
Structure metrics standalone runner for ToolInstance venv execution.

Computes longest alpha helix length and gyration radius from PDB files
using biotite for SSE annotation and structural analysis.

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import numpy as np


# =============================================================================
# Structure Metrics Computation
# =============================================================================
def _longest_alpha(sse: np.ndarray) -> int:
    """Compute the length of the longest alpha helix from SSE annotation.

    Args:
        sse: Array of SSE annotations where 'a' indicates alpha helix.

    Returns:
        Length of the longest contiguous alpha helix segment.
    """
    if len(sse) == 0:
        return 0

    lens = []
    curr_len = 0
    for el in sse:
        if el == "a":
            curr_len += 1
        else:
            if curr_len > 0:
                lens.append(curr_len)
            curr_len = 0

    if curr_len > 0:
        lens.append(curr_len)

    return max(lens, default=0)


def _compute_metrics(pdb_path: str) -> Dict[str, Any]:
    """Compute structure metrics for a single PDB file.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Dict with keys: pdb_path, longest_alpha_helix, gyration_radius.
    """
    import biotite.structure.io as strucio
    from biotite.structure import annotate_sse, gyration_radius

    array = strucio.load_structure(pdb_path)
    sse = annotate_sse(array)
    alpha_len = _longest_alpha(sse)
    gyr_rad = float(gyration_radius(array))

    return {
        "pdb_path": pdb_path,
        "longest_alpha_helix": alpha_len,
        "gyration_radius": gyr_rad,
    }


# =============================================================================
# Main Entry Point
# =============================================================================
def run_structure_metrics(input_data: dict) -> dict:
    """Compute structure metrics for one or more PDB files.

    Args:
        input_data: Dict with key: pdb_paths (list of PDB file paths).

    Returns:
        Dict with key: metrics (list of metric dicts).
    """
    pdb_paths = input_data["pdb_paths"]

    metrics = []
    for pdb_path in pdb_paths:
        result = _compute_metrics(pdb_path)
        metrics.append(result)

    return {"metrics": metrics}


def dispatch(input_dict: dict) -> dict:
    """Entry point for persistent-worker execution."""
    return run_structure_metrics(input_dict)


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================

def to_device(device: str) -> dict:
    """Passthrough - tool does not maintain persistent state."""
    return {"success": True, "device": device}


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

    output_data = run_structure_metrics(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
