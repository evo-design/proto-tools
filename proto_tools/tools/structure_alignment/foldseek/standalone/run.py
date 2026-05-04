"""Foldseek standalone runner for ToolInstance venv execution.

Handles single-chain search, multimer search, clustering, multimer clustering,
and reciprocal-best-hits operations. Communicates via JSON input/output files
(ToolInstance pattern).

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


def _write_structures(structures: list[str], out_dir: Path, ids: list[str], formats: list[str]) -> None:
    """Write each structure to a file in out_dir as ``{id}.{format}``."""
    for sid, text, fmt in zip(ids, structures, formats, strict=True):
        (out_dir / f"{sid}.{fmt}").write_text(text)


# Force `pident` (0-100) so local M8 matches the public server's format.
# The CLI default is `fident` (0-1), which would silently corrupt parsing.
_M8_FORMAT_PIDENT = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"


def run_easy_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-search` for a single query against a local Foldseek DB.

    Args:
        input_data: keys ``structure_text`` (PDB text), ``local_db`` (path),
            ``evalue``, ``sensitivity``, ``max_seqs``, ``alignment_type``,
            ``tmscore_threshold``, ``lddt_threshold``, ``num_threads``.

    Returns:
        ``{"stdout": <m8_text>}`` — standard 12-column BLAST M8 (pident, 0-100).
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
                "--format-output",
                _M8_FORMAT_PIDENT,
                "-e",
                str(input_data["evalue"]),
                "-s",
                str(input_data["sensitivity"]),
                "--max-seqs",
                str(input_data["max_seqs"]),
                "--alignment-type",
                str(input_data["alignment_type"]),
                "--tmscore-threshold",
                str(input_data["tmscore_threshold"]),
                "--lddt-threshold",
                str(input_data["lddt_threshold"]),
                "--threads",
                str(input_data["num_threads"]),
            ],
            "easy-search",
        )
        return {"stdout": m8_out.read_text() if m8_out.exists() else ""}


def run_easy_cluster(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-cluster` over user-provided structures.

    Args:
        input_data: keys ``structures`` (list[str] of PDB or mmCIF text),
            ``structure_ids`` (list[str]), ``structure_formats`` (list[str],
            ``"pdb"`` or ``"cif"`` per entry), ``min_seq_id``, ``cov``,
            ``cov_mode``, ``evalue``, ``alignment_type``, ``tmscore_threshold``,
            ``lddt_threshold``, ``num_threads``.

    Returns:
        ``{"clusters_tsv": <tsv_text>}`` — 2 cols: representative_id, member_id.
    """
    foldseek = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        structures_dir = tmp_path / "structures"
        structures_dir.mkdir()
        _write_structures(
            input_data["structures"],
            structures_dir,
            input_data["structure_ids"],
            input_data["structure_formats"],
        )

        prefix = tmp_path / "cluster"
        _run_cmd(
            [
                foldseek,
                "easy-cluster",
                str(structures_dir),
                str(prefix),
                str(tmp_path / "fs_tmp"),
                "--min-seq-id",
                str(input_data["min_seq_id"]),
                "-c",
                str(input_data["cov"]),
                "--cov-mode",
                str(input_data["cov_mode"]),
                "-e",
                str(input_data["evalue"]),
                "--alignment-type",
                str(input_data["alignment_type"]),
                "--tmscore-threshold",
                str(input_data["tmscore_threshold"]),
                "--lddt-threshold",
                str(input_data["lddt_threshold"]),
                "--threads",
                str(input_data["num_threads"]),
            ],
            "easy-cluster",
        )
        tsv_path = prefix.with_name(prefix.name + "_cluster.tsv")
        return {"clusters_tsv": tsv_path.read_text() if tsv_path.exists() else ""}


def run_easy_multimersearch(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-multimersearch` for a multi-chain query.

    Args:
        input_data: keys ``structure_text`` (multi-chain PDB), ``local_db``,
            ``evalue``, ``sensitivity``, ``max_seqs``, ``alignment_type``,
            ``tmscore_threshold``, ``lddt_threshold``, ``num_threads``.

    Returns:
        ``{"stdout": <m8_text>}`` — standard 12-column BLAST M8 (pident, 0-100).
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
                "--format-output",
                _M8_FORMAT_PIDENT,
                "-e",
                str(input_data["evalue"]),
                "-s",
                str(input_data["sensitivity"]),
                "--max-seqs",
                str(input_data["max_seqs"]),
                "--alignment-type",
                str(input_data["alignment_type"]),
                "--tmscore-threshold",
                str(input_data["tmscore_threshold"]),
                "--lddt-threshold",
                str(input_data["lddt_threshold"]),
                "--threads",
                str(input_data["num_threads"]),
            ],
            "easy-multimersearch",
        )
        return {"stdout": m8_out.read_text() if m8_out.exists() else ""}


def run_easy_multimercluster(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-multimercluster` over user-provided multi-chain structures.

    Args:
        input_data: keys ``structures`` (list[str] of multi-chain PDB or mmCIF
            text), ``structure_ids`` (list[str]), ``structure_formats``
            (list[str], ``"pdb"`` or ``"cif"`` per entry),
            ``multimer_tm_threshold``, ``chain_tm_threshold``,
            ``interface_lddt_threshold``, ``alignment_type``,
            ``tmscore_threshold``, ``lddt_threshold``, ``num_threads``.

    Returns:
        ``{"clusters_tsv": <tsv_text>, "rep_seq_fasta": <fasta_text>}`` —
        TSV is 2 cols: representative_id, member_id.
    """
    foldseek = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        structures_dir = tmp_path / "structures"
        structures_dir.mkdir()
        _write_structures(
            input_data["structures"],
            structures_dir,
            input_data["structure_ids"],
            input_data["structure_formats"],
        )

        prefix = tmp_path / "multimercluster"
        _run_cmd(
            [
                foldseek,
                "easy-multimercluster",
                str(structures_dir),
                str(prefix),
                str(tmp_path / "fs_tmp"),
                "--multimer-tm-threshold",
                str(input_data["multimer_tm_threshold"]),
                "--chain-tm-threshold",
                str(input_data["chain_tm_threshold"]),
                "--interface-lddt-threshold",
                str(input_data["interface_lddt_threshold"]),
                "--alignment-type",
                str(input_data["alignment_type"]),
                "--tmscore-threshold",
                str(input_data["tmscore_threshold"]),
                "--lddt-threshold",
                str(input_data["lddt_threshold"]),
                "--threads",
                str(input_data["num_threads"]),
            ],
            "easy-multimercluster",
        )
        tsv_path = prefix.with_name(prefix.name + "_cluster.tsv")
        fasta_path = prefix.with_name(prefix.name + "_rep_seq.fasta")
        return {
            "clusters_tsv": tsv_path.read_text() if tsv_path.exists() else "",
            "rep_seq_fasta": fasta_path.read_text() if fasta_path.exists() else "",
        }


def run_easy_rbh(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldseek easy-rbh` for reciprocal-best-hits between a query and a target DB.

    Args:
        input_data: keys ``structure_text`` (PDB text), ``local_db`` (target
            DB path or directory of PDBs), ``evalue``, ``sensitivity``,
            ``max_seqs``, ``alignment_type``, ``cov``, ``cov_mode``,
            ``tmscore_threshold``, ``lddt_threshold``, ``num_threads``.

    Returns:
        ``{"stdout": <m8_text>}`` — standard 12-column BLAST M8 (pident, 0-100).
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
                "easy-rbh",
                str(query_pdb),
                input_data["local_db"],
                str(m8_out),
                str(tmp_path / "fs_tmp"),
                "--format-output",
                _M8_FORMAT_PIDENT,
                "-e",
                str(input_data["evalue"]),
                "-s",
                str(input_data["sensitivity"]),
                "--max-seqs",
                str(input_data["max_seqs"]),
                "--alignment-type",
                str(input_data["alignment_type"]),
                "-c",
                str(input_data["cov"]),
                "--cov-mode",
                str(input_data["cov_mode"]),
                "--tmscore-threshold",
                str(input_data["tmscore_threshold"]),
                "--lddt-threshold",
                str(input_data["lddt_threshold"]),
                "--threads",
                str(input_data["num_threads"]),
            ],
            "easy-rbh",
        )
        return {"stdout": m8_out.read_text() if m8_out.exists() else ""}


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool — automatically unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


_OPERATIONS = {
    "easy_search": run_easy_search,
    "easy_cluster": run_easy_cluster,
    "easy_multimersearch": run_easy_multimersearch,
    "easy_multimercluster": run_easy_multimercluster,
    "easy_rbh": run_easy_rbh,
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
