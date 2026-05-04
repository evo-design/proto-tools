"""Foldseek standalone runner for ToolInstance venv execution.

Handles single-chain search, multimer search, and clustering operations.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_binary(name: str = "foldseek") -> str:
    """Find the Foldseek binary in the venv's bin/ directory."""
    binary = Path(sys.executable).parent / name
    if not binary.exists():
        raise FileNotFoundError(
            f"foldseek: binary '{name}' not found at {binary}; re-run standalone/setup.sh to provision the venv"
        )
    return str(binary)


def _run_cmd(cmd: list[str], description: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run a subprocess command and raise on failure."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"foldseek: {description} failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
        ) from e


def _write_pdbs(structures: list[str], pdb_dir: Path, ids: list[str] | None = None) -> list[str]:
    """Write each PDB-text string to a file in pdb_dir; return assigned IDs."""
    assigned = ids or [f"structure_{i}" for i in range(len(structures))]
    for sid, text in zip(assigned, structures, strict=True):
        (pdb_dir / f"{sid}.pdb").write_text(text)
    return assigned


# Force `pident` (0-100) so local M8 matches the public server's format.
# The CLI default is `fident` (0-1), which would silently corrupt parsing.
_M8_FORMAT_PIDENT = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"


def run_easy_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-search` for a single query against a local Foldseek DB.

    Args:
        input_data: keys ``structure_text`` (PDB text), ``local_db`` (path),
            ``num_threads`` (int).

    Returns:
        ``{"stdout": <m8_text>}`` — standard 12-column BLAST M8.
    """
    foldseek = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        query_pdb = tmp_path / "query.pdb"
        query_pdb.write_text(input_data["structure_text"])
        m8_out = tmp_path / "result.m8"

        _run_cmd(
            [
                foldseek,
                "easy-search",
                str(query_pdb),
                input_data["local_db"],
                str(m8_out),
                str(tmp_path / "fs_tmp"),
                "--threads",
                str(input_data.get("num_threads", 4)),
                "--format-output",
                _M8_FORMAT_PIDENT,
            ],
            "easy-search",
        )
        return {"stdout": m8_out.read_text() if m8_out.exists() else ""}


def run_easy_cluster(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-cluster` over user-provided structures.

    Args:
        input_data: keys ``structures`` (list[str] PDB texts),
            ``structure_ids`` (list[str] | None), ``min_seq_id``, ``cov``,
            ``cov_mode``, ``num_threads``.

    Returns:
        ``{"clusters_tsv": <tsv_text>}`` — 2 cols: representative_id, member_id.
    """
    foldseek = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        _write_pdbs(input_data["structures"], pdb_dir, input_data.get("structure_ids"))

        prefix = tmp_path / "cluster"
        _run_cmd(
            [
                foldseek,
                "easy-cluster",
                str(pdb_dir),
                str(prefix),
                str(tmp_path / "fs_tmp"),
                "--min-seq-id",
                str(input_data.get("min_seq_id", 0.0)),
                "-c",
                str(input_data.get("cov", 0.8)),
                "--cov-mode",
                str(input_data.get("cov_mode", 0)),
                "--threads",
                str(input_data.get("num_threads", 4)),
            ],
            "easy-cluster",
        )
        tsv_path = prefix.with_name(prefix.name + "_cluster.tsv")
        return {"clusters_tsv": tsv_path.read_text() if tsv_path.exists() else ""}


def run_easy_multimersearch(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-multimersearch` for a multi-chain query.

    Args:
        input_data: keys ``structure_text`` (multi-chain PDB), ``local_db``,
            ``num_threads``.

    Returns:
        ``{"stdout": <m8_text>}`` — standard 12-column BLAST M8.
    """
    foldseek = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        query_pdb = tmp_path / "query.pdb"
        query_pdb.write_text(input_data["structure_text"])
        m8_out = tmp_path / "result.m8"

        _run_cmd(
            [
                foldseek,
                "easy-multimersearch",
                str(query_pdb),
                input_data["local_db"],
                str(m8_out),
                str(tmp_path / "fs_tmp"),
                "--threads",
                str(input_data.get("num_threads", 4)),
                "--format-output",
                _M8_FORMAT_PIDENT,
            ],
            "easy-multimersearch",
        )
        return {"stdout": m8_out.read_text() if m8_out.exists() else ""}


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool — automatically unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


_OPERATIONS = {
    "easy_search": run_easy_search,
    "easy_cluster": run_easy_cluster,
    "easy_multimersearch": run_easy_multimersearch,
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"foldseek: usage: python {sys.argv[0]} <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    op = input_data["operation"]
    if op not in _OPERATIONS:
        raise ValueError(f"foldseek: unknown operation {op!r}; valid: {sorted(_OPERATIONS)}")

    output_data = _OPERATIONS[op](input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f)
