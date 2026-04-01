"""MMseqs2 standalone runner for ToolInstance venv execution.

Handles protein search, genome search, and clustering operations.
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ============================================================================
# Helpers
# ============================================================================


def _find_binary(name: str = "mmseqs") -> str:
    """Find the MMseqs2 binary in the venv's bin/ directory."""
    binary = Path(sys.executable).parent / name
    if not binary.exists():
        raise FileNotFoundError(
            f"MMseqs2 binary '{name}' not found at {binary}. The standalone environment may need to be recreated."
        )
    return str(binary)


def _run_cmd(cmd: list[str], description: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run a subprocess command and raise on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{description} failed with code {proc.returncode}: {proc.stderr}")
    return proc


def _write_fasta(
    sequences: list[str],
    output_path: str,
    sequence_ids: list[str] | None = None,
    prefix: str = "seq",
) -> None:
    """Write sequences to a FASTA file."""
    ids = sequence_ids or [f"{prefix}_{i}" for i in range(len(sequences))]
    with open(output_path, "w") as f:
        f.writelines(f">{seq_id}\n{seq}\n" for seq_id, seq in zip(ids, sequences, strict=False))


# ============================================================================
# Implementation
# ============================================================================


def run_protein_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run MMseqs2 easy-search for protein sequences and return raw m8 output.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, mmseqs_db,
                    threads, split, sensitivity, m8_columns

    Returns:
        Dict with keys: stdout (raw tab-separated m8 output)
    """
    mmseqs = _find_binary()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write query FASTA
        query_fasta = str(tmp_path / "query.faa")
        _write_fasta(
            input_data["sequences"],
            query_fasta,
            input_data.get("sequence_ids"),
        )

        # Set up results directory
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        m8_path = str(results_dir / "mmseqs_results.m8")

        cmd = [
            mmseqs,
            "easy-search",
            query_fasta,
            input_data["mmseqs_db"],
            m8_path,
            str(results_dir),
            "--threads",
            str(input_data.get("threads", 96)),
            "--split",
            str(input_data.get("split", 0)),
            "-s",
            str(input_data.get("sensitivity", 4.0)),
            "--remove-tmp-files",
            "1",
            "--format-output",
            ",".join(input_data.get("m8_columns", ["query", "target", "pident", "evalue"])),
        ]

        _run_cmd(cmd, "mmseqs easy-search")

        # Read results
        m8_file = Path(m8_path)
        stdout = m8_file.read_text() if m8_file.exists() else ""

    return {"stdout": stdout}


def run_genome_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run MMseqs2 genome-to-genome search workflow and return raw m8 output.

    Args:
        input_data: Dict with keys: query_sequences, query_ids, target_sequences,
                    target_ids, search_type, threads, sensitivity, m8_columns

    Returns:
        Dict with keys: stdout (raw tab-separated m8 output)
    """
    mmseqs = _find_binary()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write query and target FASTA files
        query_fasta = str(tmp_path / "query_genomes.fna")
        target_fasta = str(tmp_path / "target_genomes.fna")
        _write_fasta(
            input_data["query_sequences"],
            query_fasta,
            input_data.get("query_ids"),
        )
        _write_fasta(
            input_data["target_sequences"],
            target_fasta,
            input_data.get("target_ids"),
            prefix="target",
        )

        # Set up paths
        query_db = str(tmp_path / "query_db")
        target_db = str(tmp_path / "target_db")
        mmseqs_tmp = str(tmp_path / "mmseqs_tmp")
        res_dir = str(tmp_path / "results")
        results_m8 = str(tmp_path / "mmseqs_results.m8")

        Path(mmseqs_tmp).mkdir()
        Path(res_dir).mkdir()

        search_type = str(input_data.get("search_type", 3))
        threads = str(input_data.get("threads", 96))
        sensitivity = str(input_data.get("sensitivity", 7.5))

        # Step 1: Create databases
        _run_cmd(
            [mmseqs, "createdb", query_fasta, query_db],
            "mmseqs createdb (query)",
        )
        _run_cmd(
            [mmseqs, "createdb", target_fasta, target_db],
            "mmseqs createdb (target)",
        )

        # Step 2: Create index
        _run_cmd(
            [
                mmseqs,
                "createindex",
                target_db,
                mmseqs_tmp,
                "--search-type",
                search_type,
                "--threads",
                threads,
            ],
            "mmseqs createindex",
        )

        # Step 3: Search
        _run_cmd(
            [
                mmseqs,
                "search",
                query_db,
                target_db,
                res_dir,
                mmseqs_tmp,
                "--search-type",
                search_type,
                "--threads",
                threads,
                "-s",
                sensitivity,
            ],
            "mmseqs search",
        )

        # Step 4: Convert results to m8 format
        m8_columns = input_data.get("m8_columns", ["query", "target", "pident", "evalue"])
        _run_cmd(
            [
                mmseqs,
                "convertalis",
                query_db,
                target_db,
                res_dir,
                results_m8,
                "--format-output",
                ",".join(m8_columns),
            ],
            "mmseqs convertalis",
        )

        # Read results
        m8_file = Path(results_m8)
        stdout = m8_file.read_text() if m8_file.exists() else ""

    return {"stdout": stdout}


def run_clustering(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run MMseqs2 clustering workflow and return cluster assignments.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, min_seq_id

    Returns:
        Dict with keys: cluster_assignments (dict mapping member_id -> representative_id)
    """
    mmseqs = _find_binary()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write input FASTA
        input_fasta = str(tmp_path / "input.faa")
        _write_fasta(
            input_data["sequences"],
            input_fasta,
            input_data.get("sequence_ids"),
        )

        # Set up paths
        db_dir = tmp_path / "mmseqs_db"
        res_dir = tmp_path / "mmseqs_results"
        mmseqs_tmp = tmp_path / "tmp"
        db_dir.mkdir()
        res_dir.mkdir()
        mmseqs_tmp.mkdir()

        min_seq_id = str(input_data.get("min_seq_id", 0.60))

        # Step 1: Create database
        _run_cmd(
            [mmseqs, "createdb", input_fasta, str(db_dir / "seqs")],
            "mmseqs createdb",
        )

        # Step 2: Cluster
        _run_cmd(
            [
                mmseqs,
                "cluster",
                str(db_dir / "seqs"),
                str(res_dir / "clusters"),
                str(mmseqs_tmp),
                "--min-seq-id",
                min_seq_id,
            ],
            "mmseqs cluster",
        )

        # Step 3: Create sub-database of representatives
        _run_cmd(
            [
                mmseqs,
                "createsubdb",
                str(res_dir / "clusters"),
                str(db_dir / "seqs"),
                str(res_dir / "rep_seqs"),
            ],
            "mmseqs createsubdb",
        )

        # Step 4: Create TSV with cluster assignments
        clusters_tsv = str(res_dir / "clusters.tsv")
        _run_cmd(
            [
                mmseqs,
                "createtsv",
                str(db_dir / "seqs"),
                str(db_dir / "seqs"),
                str(res_dir / "clusters"),
                clusters_tsv,
            ],
            "mmseqs createtsv",
        )

        # Parse cluster assignments
        cluster_assignments: dict[str, str] = {}
        tsv_path = Path(clusters_tsv)
        if tsv_path.exists():
            for line in tsv_path.read_text().strip().splitlines():
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        representative, member = parts[0], parts[1]
                        cluster_assignments[member] = representative

    return {"cluster_assignments": cluster_assignments}


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

    if operation == "protein_search":
        output_data = run_protein_search(input_data)
    elif operation == "genome_search":
        output_data = run_genome_search(input_data)
    elif operation == "clustering":
        output_data = run_clustering(input_data)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
