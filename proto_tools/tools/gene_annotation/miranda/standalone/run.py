"""miRanda standalone runner for ToolInstance venv execution.

Builds the query/target FASTA files, runs the miranda binary, and returns its raw
stdout (the tool layer parses it). Communicates via JSON files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


def run_miranda(
    mirna_sequences: list[str],
    target_sequences: list[str],
    score_threshold: int,
    energy_threshold: int,
    scale: float,
    gap_open: int,
    gap_extend: int,
    strict: bool,
    compute_energy: bool,
    trim: int,
) -> dict[str, Any]:
    """Run miRanda on microRNA queries against target sequences.

    Args:
        mirna_sequences: microRNA query sequences (written as q0, q1, ...).
        target_sequences: target sequences (written as t0, t1, ...).
        score_threshold: minimum alignment score (-sc).
        energy_threshold: maximum free energy in kcal/mol (-en).
        scale: scaling factor for 5' seed-region matches (-scale).
        gap_open: gap-open penalty, <= 0 (-go).
        gap_extend: gap-extension penalty, <= 0 (-ge).
        strict: when True, pass -strict to enforce the strict 5' duplex heuristics.
        compute_energy: when False, pass -noenergy to skip folding.
        trim: when > 0, trim reference sequences to this length (-trim).

    Returns:
        Dict with the raw stdout produced by the miranda binary.
    """
    binary = Path(sys.executable).parent / "miranda"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        query_path = tmp_path / "q.fa"
        target_path = tmp_path / "t.fa"

        query_path.write_text("".join(f">q{i}\n{seq}\n" for i, seq in enumerate(mirna_sequences)))
        target_path.write_text("".join(f">t{j}\n{seq}\n" for j, seq in enumerate(target_sequences)))

        argv = [
            str(binary),
            str(query_path),
            str(target_path),
            "-sc",
            str(score_threshold),
            "-en",
            str(energy_threshold),
            "-scale",
            str(scale),
            "-go",
            str(gap_open),
            "-ge",
            str(gap_extend),
        ]
        if strict:
            argv.append("-strict")
        if not compute_energy:
            argv.append("-noenergy")
        if trim > 0:
            argv.extend(["-trim", str(trim)])

        try:
            result = subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # miRanda writes diagnostics to stdout as well as stderr, so surface both.
            diagnostics = ((e.stderr or "") + "\n" + (e.stdout or "")).strip().splitlines()
            raise RuntimeError(
                f"miranda: prediction failed (exit {e.returncode}): {' | '.join(diagnostics) or '<no output>'}"
            ) from e

    return {"stdout": result.stdout}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    return run_miranda(
        mirna_sequences=input_dict["mirna_sequences"],
        target_sequences=input_dict["target_sequences"],
        score_threshold=input_dict["score_threshold"],
        energy_threshold=input_dict["energy_threshold"],
        scale=input_dict["scale"],
        gap_open=input_dict["gap_open"],
        gap_extend=input_dict["gap_extend"],
        strict=input_dict["strict"],
        compute_energy=input_dict["compute_energy"],
        trim=input_dict["trim"],
    )


def to_device(device: str) -> dict[str, Any]:
    """DeviceManager protocol no-op: miRanda is a CPU CLI tool that auto-unloads."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """DeviceManager protocol stub: miRanda is CPU-only, so no GPU memory to report."""
    return {"available": False, "framework": "cpu", "reason": "CPU tool"}


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("miranda: usage: python run.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    output_data = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f, indent=2)
