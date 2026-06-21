"""PyMOL RMSD standalone runner for ToolInstance venv execution."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from standalone_helpers import get_logger

logger = get_logger(__name__)

# Mirrors PyMOLRMSDConfig.method in the parent tool module. Standalone scripts
# cannot import proto_tools without defeating the isolated-tool environment.
PyMOLAlignmentMethod = Literal["cealign", "align"]

_PYMOL_STARTED = False


def _cmd() -> Any:
    """Return a clean PyMOL cmd object in quiet, headless mode."""
    global _PYMOL_STARTED

    import pymol
    from pymol import cmd

    if not _PYMOL_STARTED:
        pymol.finish_launching(["pymol", "-qc"])
        _PYMOL_STARTED = True
    cmd.reinitialize()
    return cmd


def _cealign_metrics(result: dict[str, Any], failure_rmsd: float) -> dict[str, Any]:
    return {
        "rmsd": float(result.get("RMSD", failure_rmsd)),
        "aligned_length": int(result.get("alignment_length", 0)),
    }


def _align_metrics(result: tuple[Any, ...], failure_rmsd: float) -> dict[str, Any]:
    # PyMOL returns ExecutiveRMSInfo as:
    # final_rms, final_n_atom, n_cycles_run, initial_rms,
    # initial_n_atom, raw_alignment_score, n_residues_aligned.
    return {
        "rmsd": float(result[0]) if len(result) > 0 else failure_rmsd,
        "aligned_atoms": int(result[1]) if len(result) > 1 else 0,
        "alignment_cycles": int(result[2]) if len(result) > 2 else 0,
        "pre_refinement_rmsd": float(result[3]) if len(result) > 3 else None,
        "pre_refinement_aligned_atoms": int(result[4]) if len(result) > 4 else None,
        "alignment_score": float(result[5]) if len(result) > 5 else None,
        "aligned_residues": int(result[6]) if len(result) > 6 else None,
    }


def _object_transform(cmd: Any, name: str) -> dict[str, Any]:
    """Decompose PyMOL's 4x4 object matrix (row-major, 16 floats) into rotation + translation.

    The matrix maps the aligned object's coordinates into the target frame, so the 3x3 rotation
    and length-3 translation superpose the mobile structure onto the target. ``None`` if PyMOL
    returns no matrix.
    """
    matrix = cmd.get_object_matrix(name)
    if not matrix or len(matrix) < 12:
        return {"rotation": None, "translation": None}
    m = [float(v) for v in matrix]
    return {
        "rotation": [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]],
        "translation": [m[3], m[7], m[11]],
    }


def run_pymol_rmsd_alignment(
    target_pdb_text: str,
    mobile_pdb_text: str,
    method: PyMOLAlignmentMethod = "cealign",
    target_selection: str = "target",
    mobile_selection: str = "mobile",
    failure_rmsd: float = 999.0,
) -> dict[str, Any]:
    """Run a PyMOL alignment and return RMSD metrics + the superposition transform."""
    cmd = _cmd()
    superposition: dict[str, Any] = {"rotation": None, "translation": None}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        target_path = tmp_path / "target.pdb"
        mobile_path = tmp_path / "mobile.pdb"
        target_path.write_text(target_pdb_text)
        mobile_path.write_text(mobile_pdb_text)

        try:
            cmd.load(str(target_path), "target")
            cmd.load(str(mobile_path), "mobile")

            if method == "cealign":
                result = cmd.cealign(target_selection, mobile_selection)
                metrics = _cealign_metrics(result, failure_rmsd)
            elif method == "align":
                result = cmd.align(mobile_selection, target_selection)
                metrics = _align_metrics(result, failure_rmsd)
            else:
                raise ValueError(f"Unsupported PyMOL alignment method: {method}")

            # Best-effort: the alignment moved the "mobile" object onto the target. Capture its
            # transform without letting a failure here clobber the alignment metrics.
            try:
                superposition = _object_transform(cmd, "mobile")
            except Exception:
                logger.warning("pymol_rmsd: could not capture superposition transform")
        except Exception as exc:
            metrics = {
                "rmsd": failure_rmsd,
                "alignment_error": str(exc),
            }
            if method == "cealign":
                metrics["aligned_length"] = 0
            else:
                metrics["aligned_atoms"] = 0
                metrics["alignment_cycles"] = 0
        finally:
            cmd.reinitialize()

    return {"method": method, **metrics, **superposition}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    return run_pymol_rmsd_alignment(
        target_pdb_text=input_dict["target_pdb_text"],
        mobile_pdb_text=input_dict["mobile_pdb_text"],
        method=input_dict.get("method", "cealign"),
        target_selection=input_dict.get("target_selection", "target"),
        mobile_selection=input_dict.get("mobile_selection", "mobile"),
        failure_rmsd=input_dict.get("failure_rmsd", 999.0),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("pymol_rmsd: usage: python run.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    output_data = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f, indent=2)
