"""Standalone runner script for MAFFT alignment.

This script runs in an isolated venv with MAFFT binaries installed.
It receives input via JSON and writes output via JSON.

Usage:
    python run.py <input.json> <output.json>

Input JSON format:
{
    "sequences": ["MVLSPADKTN", "MVLSAADKTN", ...],
    "sequence_ids": ["seq_0", "seq_1", ...],  // optional
    "align_method": "auto",  // auto, localpair, globalpair, genafpair
    "max_iterations": 0,
    "threads": 1
}

Output JSON format:
{
    "aligned_sequences": ["MVLSPADKTN", "MVLSAADKTN", ...],
    "sequence_ids": ["seq_0", "seq_1", ...],
    "alignment_length": 10,
    "num_sequences": 2
}
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# MAFFT alignment method flags
ALIGNMENT_METHOD_FLAGS = {
    "auto": ["--auto"],
    "localpair": ["--localpair"],  # L-INS-i method
    "globalpair": ["--globalpair"],  # G-INS-i method
    "genafpair": ["--genafpair"],  # E-INS-i method
}


def find_mafft_binary() -> Path:
    """Find the MAFFT binary in the venv's bin directory.

    Returns:
        Path to the mafft executable

    Raises:
        FileNotFoundError: If mafft binary is not found
    """
    # Binary is in the same directory as the Python interpreter
    bin_dir = Path(sys.executable).parent
    mafft_path = bin_dir / "mafft"

    if not mafft_path.exists():
        raise FileNotFoundError(
            f"mafft: binary not found at {mafft_path} — re-run the tool's standalone/setup.sh to provision the venv"
        )

    return mafft_path


def run_mafft_alignment(
    sequences: list[str],
    sequence_ids: list[str],
    align_method: str = "auto",
    max_iterations: int = 0,
    threads: int = 1,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run MAFFT alignment on the given sequences.

    Args:
        sequences: List of sequence strings to align
        sequence_ids: List of sequence identifiers
        align_method: Alignment method (auto, localpair, globalpair, genafpair)
        max_iterations: Maximum iterative refinement cycles
        threads: Number of CPU threads
        extra_args: Verbatim mafft CLI tokens appended before the input FASTA

    Returns:
        Dictionary with aligned_sequences, sequence_ids, alignment_length, num_sequences
    """
    import os

    mafft_binary = find_mafft_binary()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create input FASTA file
        input_fasta = tmp_path / "input.fasta"
        records = [
            SeqRecord(Seq(seq), id=seq_id, description="") for seq, seq_id in zip(sequences, sequence_ids, strict=False)
        ]
        SeqIO.write(records, input_fasta, "fasta")

        # Build MAFFT command
        cmd = [str(mafft_binary)]

        # Add method flags
        if align_method in ALIGNMENT_METHOD_FLAGS:
            cmd.extend(ALIGNMENT_METHOD_FLAGS[align_method])

        # Add max iterations if specified
        if max_iterations > 0:
            cmd.extend(["--maxiterate", str(max_iterations)])

        # Add thread count
        cmd.extend(["--thread", str(threads)])

        # Power-user escape hatch: append verbatim CLI tokens before the input FASTA.
        if extra_args:
            cmd.extend(str(arg) for arg in extra_args)

        # Add input file
        cmd.append(str(input_fasta))

        # Set up environment for MAFFT auxiliary binaries.
        # Pre-built binaries need MAFFT_BINARIES pointing to the venv's libexec/.
        # Source-compiled binaries have the correct path baked into the script,
        # so we only set the env var when the pre-built libexec layout exists.
        env = os.environ.copy()
        libexec_dir = Path(sys.executable).parent.parent / "libexec"
        libexec_mafft_dir = libexec_dir / "mafft"
        if libexec_dir.exists() and not libexec_mafft_dir.exists():
            # Pre-built layout: binaries are directly in libexec/
            env["MAFFT_BINARIES"] = str(libexec_dir)
        else:
            # Source-compiled layout (libexec/mafft/) or missing; let
            # the mafft script use its built-in path.
            env.pop("MAFFT_BINARIES", None)

        # Run MAFFT (output goes to stdout)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
            raise RuntimeError(
                f"mafft: alignment failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
            ) from e

        # Parse aligned sequences from stdout
        output_fasta = tmp_path / "output.fasta"
        output_fasta.write_text(result.stdout)

        # Parse aligned sequences (keyed by sequence ID)
        aligned_by_id = {}
        for record in SeqIO.parse(output_fasta, "fasta"):  # type: ignore[no-untyped-call]
            aligned_by_id[record.id] = str(record.seq)

    # Build output in original input order
    aligned_sequences = [aligned_by_id[sid] for sid in sequence_ids]

    return {
        "aligned_sequences": aligned_sequences,
        "sequence_ids": sequence_ids,
        "alignment_length": len(aligned_sequences[0]) if aligned_sequences else 0,
        "num_sequences": len(aligned_sequences),
    }


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    sequences = input_dict["sequences"]
    sequence_ids = input_dict.get("sequence_ids", [f"seq_{i}" for i in range(len(sequences))])
    return run_mafft_alignment(
        sequences=sequences,
        sequence_ids=sequence_ids,
        align_method=input_dict.get("align_method", "auto"),
        max_iterations=input_dict.get("max_iterations", 0),
        threads=input_dict.get("threads", 1),
        extra_args=input_dict.get("extra_args"),
    )


def main() -> None:
    """Main entry point for standalone MAFFT runner."""
    if len(sys.argv) != 3:
        print("mafft: usage: python run.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Read input JSON
    with open(input_path) as f:
        input_data = json.load(f)

    # Extract parameters
    sequences = input_data["sequences"]
    sequence_ids = input_data.get("sequence_ids")

    # Generate default IDs if not provided
    if sequence_ids is None:
        sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

    align_method = input_data.get("align_method", "auto")
    max_iterations = input_data.get("max_iterations", 0)
    threads = input_data.get("threads", 1)
    extra_args = input_data.get("extra_args")

    # Run alignment
    output_data = run_mafft_alignment(
        sequences=sequences,
        sequence_ids=sequence_ids,
        align_method=align_method,
        max_iterations=max_iterations,
        threads=threads,
        extra_args=extra_args,
    )

    # Write output JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool - automatically unloads after each call."""
    # CLI tool that spawns subprocesses and naturally unloads after each call
    # This is a passthrough for standardization with other tools
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


if __name__ == "__main__":
    main()
