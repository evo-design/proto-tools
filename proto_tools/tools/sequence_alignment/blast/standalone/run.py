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
            f"blast: binary '{name}' not found at {binary}; re-run standalone/setup.sh to provision the venv"
        )
    return str(binary)


def run_local_blast(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run a local BLAST search and return raw tabular output.

    Args:
        input_data: Dict with keys: program, query_path, db, num_threads, cli_args.
            ``cli_args`` is a pre-formatted token list (typed flags + user
            ``extra_args``) assembled by the wrapper.

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
        *input_data.get("cli_args", []),
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"blast: {Path(program).name} search failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
        ) from e

    return {"stdout": proc.stdout}


def run_create_blast_db(input_data: dict[str, Any]) -> dict[str, Any]:
    """Create a local BLAST database from a FASTA file.

    Args:
        input_data: Dict with keys: fasta_path, dbtype, out_prefix, title,
                    parse_seqids, hash_index, blastdb_version, max_file_sz,
                    taxid, extra_args.

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
        "-blastdb_version",
        str(input_data["blastdb_version"]),
        "-max_file_sz",
        input_data["max_file_sz"],
    ]

    title = input_data.get("title")
    if title:
        cmd.extend(["-title", title])

    if input_data["parse_seqids"]:
        cmd.append("-parse_seqids")

    if input_data["hash_index"]:
        cmd.append("-hash_index")

    taxid = input_data.get("taxid")
    if taxid is not None:
        cmd.extend(["-taxid", str(taxid)])

    # Power-user escape hatch for flags not exposed as typed fields.
    cmd.extend(str(arg) for arg in input_data.get("extra_args", []))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"blast: makeblastdb failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
        ) from e

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
            f"blast: usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
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
        raise ValueError(f"blast: unknown operation {operation!r}; valid: ['local_blast', 'create_blast_db']")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
