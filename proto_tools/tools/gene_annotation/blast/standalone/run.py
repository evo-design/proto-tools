"""BLAST+ standalone runner for ToolInstance venv execution.

Handles local BLAST search and database creation operations.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_binary(name: str) -> str:
    """Find a BLAST+ binary in the venv's bin/ directory."""
    binary = Path(sys.executable).parent / name
    if not binary.exists():
        raise FileNotFoundError(
            f"BLAST+ binary '{name}' not found at {binary}. The standalone environment may need to be recreated."
        )
    return str(binary)


def run_local_blast(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run a local BLAST search and return raw tabular output.

    Args:
        input_data: Dict with keys: program, query_path, db, num_threads,
                    additional_params

    Returns:
        Dict with keys: stdout (raw tab-separated output)
    """
    program = _find_binary(input_data["program"])

    cmd = [
        program,
        "-query",
        input_data["query_path"],
        "-db",
        input_data["db"],
        "-num_threads",
        str(input_data["num_threads"]),
    ]

    # Add default output format only if not overridden
    additional_params = input_data.get("additional_params", {})
    if "outfmt" not in additional_params:
        cmd.extend(["-outfmt", "6"])

    # Process additional parameters into command line flags
    for key, val in additional_params.items():
        if val is None:
            continue
        flag = f"-{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Local BLAST search failed with code {proc.returncode}: {proc.stderr}")

    return {"stdout": proc.stdout}


def run_create_blast_db(input_data: dict[str, Any]) -> dict[str, Any]:
    """Create a local BLAST database from a FASTA file.

    Args:
        input_data: Dict with keys: fasta_path, dbtype, out_prefix, title,
                    additional_params

    Returns:
        Dict with keys: db_path
    """
    makeblastdb = _find_binary("makeblastdb")

    cmd = [
        makeblastdb,
        "-in",
        input_data["fasta_path"],
        "-dbtype",
        input_data["dbtype"],
        "-out",
        input_data["out_prefix"],
    ]

    title = input_data.get("title")
    if title:
        cmd.extend(["-title", title])

    for key, val in input_data.get("additional_params", {}).items():
        flag = f"-{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"BLAST database creation failed with code {proc.returncode}: {proc.stderr}")

    return {"db_path": input_data["out_prefix"]}


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
            f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path) as f:
        input_data = json.load(f)

    operation = input_data["operation"]

    if operation == "local_blast":
        output_data = run_local_blast(input_data)
    elif operation == "create_blast_db":
        output_data = run_create_blast_db(input_data)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
