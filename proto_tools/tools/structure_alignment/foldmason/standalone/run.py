"""FoldMason standalone runner for ToolInstance venv execution.

Handles multiple structure alignment (`easy-msa`) and MSA-LDDT scoring
(`msa2lddt`). Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_binary(name: str = "foldmason") -> str:
    """Find the FoldMason binary in the venv's bin/ directory."""
    binary = Path(sys.executable).parent / name
    if not binary.exists():
        raise FileNotFoundError(
            f"foldmason: binary '{name}' not found at {binary}; re-run standalone/setup.sh to provision the venv"
        )
    return str(binary)


def _run_cmd(cmd: list[str], description: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run a subprocess command and raise on failure."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"foldmason: {description} failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
        ) from e


def _write_pdbs(structures: list[str], pdb_dir: Path, ids: list[str] | None = None) -> list[Path]:
    """Write each PDB-text string to a file in pdb_dir; return file paths in input order."""
    assigned = ids or [f"structure_{i}" for i in range(len(structures))]
    paths: list[Path] = []
    for sid, text in zip(assigned, structures, strict=True):
        path = pdb_dir / f"{sid}.pdb"
        path.write_text(text)
        paths.append(path)
    return paths


def run_easy_msa(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldmason easy-msa` on a set of structures to produce a multiple structure alignment.

    Args:
        input_data: keys ``structures`` (list[str] PDB texts),
            ``structure_ids`` (list[str] | None), ``gap_open``, ``gap_extend``,
            ``refine_iters``, ``num_threads``, ``precluster`` (bool),
            ``guide_tree_newick`` (str | None).

    Returns:
        ``{"aa_msa_fasta": <text>, "three_di_msa_fasta": <text>, "newick_tree": <text>}``.
    """
    foldmason = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        pdb_paths = _write_pdbs(input_data["structures"], pdb_dir, input_data.get("structure_ids"))

        prefix = tmp_path / "msa"
        cmd = [foldmason, "easy-msa", *(str(p) for p in pdb_paths), str(prefix), str(tmp_path / "fm_tmp")]
        cmd += [
            "--gap-open",
            str(input_data["gap_open"]),
            "--gap-extend",
            str(input_data["gap_extend"]),
            "--refine-iters",
            str(input_data["refine_iters"]),
            "--precluster",
            "1" if input_data["precluster"] else "0",
            "--threads",
            str(input_data["num_threads"]),
        ]
        guide_tree = input_data.get("guide_tree_newick")
        if guide_tree:
            tree_path = tmp_path / "guide.nw"
            tree_path.write_text(guide_tree)
            cmd += ["--guide-tree", str(tree_path)]
        _run_cmd(cmd, "easy-msa")

        # foldmason skips writing prefix.nw when --guide-tree is supplied;
        # echo the user's tree back so the output schema is uniform.
        nw_path = prefix.with_name(prefix.name + ".nw")
        newick = nw_path.read_text() if nw_path.exists() else (guide_tree or "")
        return {
            "aa_msa_fasta": prefix.with_name(prefix.name + "_aa.fa").read_text(),
            "three_di_msa_fasta": prefix.with_name(prefix.name + "_3di.fa").read_text(),
            "newick_tree": newick,
        }


def run_msa2lddt(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run `foldmason msa2lddt` to score a precomputed MSA against a structure DB.

    Args:
        input_data: keys ``structures`` (list[str] PDB texts whose order matches
            the MSA rows), ``structure_ids`` (list[str] | None),
            ``aa_msa_fasta`` (str — AA-FASTA MSA), ``pair_threshold``,
            ``only_scoring_cols``, ``num_threads``,
            ``guide_tree_newick`` (str | None).

    Returns:
        ``{"average_lddt": float, "columns_considered": int, "alignment_length": int,
        "column_scores": list[float]}``.
    """
    foldmason = _find_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        pdb_paths = _write_pdbs(input_data["structures"], pdb_dir, input_data.get("structure_ids"))

        msa_path = tmp_path / "msa.fa"
        msa_path.write_text(input_data["aa_msa_fasta"])

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        db_prefix = db_dir / "structures"
        _run_cmd(
            [foldmason, "createdb", *(str(p) for p in pdb_paths), str(db_prefix)],
            "createdb",
        )

        cmd = [
            foldmason,
            "msa2lddt",
            str(db_prefix),
            str(msa_path),
            "--pair-threshold",
            str(input_data["pair_threshold"]),
            "--only-scoring-cols",
            "1" if input_data["only_scoring_cols"] else "0",
            "--threads",
            str(input_data["num_threads"]),
        ]
        guide_tree = input_data.get("guide_tree_newick")
        if guide_tree:
            tree_path = tmp_path / "guide.nw"
            tree_path.write_text(guide_tree)
            cmd += ["--guide-tree", str(tree_path)]
        result = _run_cmd(cmd, "msa2lddt")
        return _parse_msa2lddt_stdout(result.stdout)


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool — automatically unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


_OPERATIONS = {
    "easy_msa": run_easy_msa,
    "msa2lddt": run_msa2lddt,
}


_AVERAGE_LDDT_RE = re.compile(r"Average MSA LDDT:\s*([0-9.eE+-]+)")
_COLUMNS_RE = re.compile(r"Columns considered:\s*(\d+)\s*/\s*(\d+)")
_COLUMN_SCORES_RE = re.compile(r"Column scores:[ \t]*([0-9.,eE+\-]+)")


def _parse_msa2lddt_stdout(stdout: str) -> dict[str, Any]:
    """Parse the three msa2lddt output lines into typed values."""
    avg_match = _AVERAGE_LDDT_RE.search(stdout)
    cols_match = _COLUMNS_RE.search(stdout)
    scores_match = _COLUMN_SCORES_RE.search(stdout)
    if not (avg_match and cols_match and scores_match):
        raise ValueError(f"foldmason msa2lddt stdout missing expected lines:\n{stdout}")
    return {
        "average_lddt": float(avg_match.group(1)),
        "columns_considered": int(cols_match.group(1)),
        "alignment_length": int(cols_match.group(2)),
        "column_scores": [float(x) for x in scores_match.group(1).split(",")],
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"foldmason: usage: python {sys.argv[0]} <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    op = input_data["operation"]
    if op not in _OPERATIONS:
        raise ValueError(f"foldmason: unknown operation {op!r}; valid: {sorted(_OPERATIONS)}")

    output_data = _OPERATIONS[op](input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f)
