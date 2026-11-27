"""Segmasker standalone runner for ToolInstance venv execution.

Handles low-complexity region detection via NCBI's segmasker binary.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


def _find_binary(name: str) -> str:
    """Find a binary in the venv's bin/ directory."""
    binary = Path(sys.executable).parent / name
    if not binary.exists():
        raise FileNotFoundError(
            f"segmasker: binary '{name}' not found at {binary}; re-run standalone/setup.sh to provision the venv"
        )
    return str(binary)


def run_segmasker(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run segmasker on protein sequences and return low-complexity results.

    Args:
        input_data: Dict with keys: sequences, config

    Returns:
        Dict with keys: fractions, counts, lengths
    """
    from Bio import SeqIO

    sequences = input_data["sequences"]
    config = input_data["config"]

    segmasker_path = _find_binary("segmasker")
    window = config["window"]
    locut = config["locut"]
    hicut = config["hicut"]

    seq_lengths = [len(s) for s in sequences]

    # Handle all empty sequences
    if all(length == 0 for length in seq_lengths):
        return {
            "fractions": [0.0] * len(sequences),
            "counts": [0] * len(sequences),
            "lengths": seq_lengths,
        }

    # Create temporary FASTA with all sequences
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_file:
        for seq_idx, seq_str in enumerate(sequences):
            seq_to_write = seq_str if len(seq_str) > 0 else "A"
            tmp_file.write(f">seq_{seq_idx}\n{seq_to_write}\n")
        tmp_file.flush()
        tmp_path = tmp_file.name

    try:
        cmd = [
            segmasker_path,
            "-in",
            tmp_path,
            "-outfmt",
            "fasta",
            "-window",
            str(window),
            "-locut",
            str(locut),
            "-hicut",
            str(hicut),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(
                f"segmasker: failed (exit {result.returncode}): {' | '.join((result.stderr or '').strip().splitlines()) or '<no stderr>'}"
            )

        seq_records = list(SeqIO.parse(StringIO(result.stdout), "fasta"))  # type: ignore[no-untyped-call]

        if len(seq_records) != len(sequences):
            raise RuntimeError(f"segmasker: returned {len(seq_records)} results but expected {len(sequences)}")

        fractions: list[float] = []
        counts: list[int] = []

        for original_seq, record in zip(sequences, seq_records, strict=False):
            if len(original_seq) == 0:
                fractions.append(0.0)
                counts.append(0)
                continue

            masked_seq = str(record.seq)
            lowercase_count = sum(1 for c in masked_seq if c.islower())
            fraction = lowercase_count / len(original_seq)

            fractions.append(fraction)
            counts.append(lowercase_count)

        return {
            "fractions": fractions,
            "counts": counts,
            "lengths": seq_lengths,
        }

    except subprocess.TimeoutExpired as e:
        raise RuntimeError("segmasker: execution timed out after 60 seconds") from e

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    return run_segmasker(input_dict)


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool - automatically unloads after each call."""
    # CLI tool that spawns subprocesses and naturally unloads after each call
    # This is a passthrough for standardization with other tools
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"segmasker: usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path) as f:
        input_data = json.load(f)

    output_data = run_segmasker(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
