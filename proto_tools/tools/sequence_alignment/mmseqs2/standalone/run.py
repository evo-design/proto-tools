"""Standalone runner for the unified ``mmseqs2`` toolkit.

Routes the four operations exposed by this toolkit to the right helper:

- ``protein_search`` — ``mmseqs easy-search``; optional ``--gpu 1`` when the
  payload sets ``use_gpu`` and the target DB has a ``*.idx_pad`` index.
- ``genome_search`` — full ``createdb`` + ``createindex`` + ``search`` +
  ``convertalis`` workflow for nucleotide-vs-nucleotide search. CPU only.
- ``clustering`` — ``mmseqs cluster`` greedy set-cover. CPU only.
- ``homology_search`` — ColabFold-style iterative MSA pipeline; GPU by default
  via the upstream ``colabfold_search`` CLI shipped with this env.

Communicates via JSON input/output files (one-shot CLI) or via the
persistent-worker ``dispatch`` import (see ``utils/_worker_bootstrap.py``).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ============================================================================
# Shared helpers
# ============================================================================


def _find_binary(name: str = "mmseqs") -> str:
    """Find the MMseqs2 binary in this venv's ``bin/`` directory."""
    binary = Path(sys.executable).parent / name
    if not binary.exists():
        raise FileNotFoundError(
            f"mmseqs2: binary {name!r} not found at {binary}; re-run standalone/setup.sh to provision the venv"
        )
    return str(binary)


def _run_cmd(cmd: list[str], description: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run a subprocess command and raise on failure."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            f"mmseqs2: {description} failed (exit {e.returncode}): {' | '.join(stderr_tail) or '<no stderr>'}"
        ) from e


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
# Operation — protein_search
# ============================================================================


def run_protein_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run ``mmseqs easy-search`` for protein sequences and return raw m8 output.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, mmseqs_db,
            threads, split, sensitivity, evalue, min_seq_id, coverage,
            cov_mode, max_seqs, m8_columns, use_gpu (optional).

    Returns:
        Dict with keys: stdout (raw tab-separated m8 output)
    """
    mmseqs = _find_binary()
    use_gpu = bool(input_data.get("use_gpu", False))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        query_fasta = str(tmp_path / "query.faa")
        _write_fasta(
            input_data["sequences"],
            query_fasta,
            input_data.get("sequence_ids"),
        )

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
            "--split",
            str(input_data["split"]),
            "-s",
            str(input_data["sensitivity"]),
            "-e",
            str(input_data["evalue"]),
            "--min-seq-id",
            str(input_data["min_seq_id"]),
            "-c",
            str(input_data["coverage"]),
            "--cov-mode",
            str(input_data["cov_mode"]),
            "--max-seqs",
            str(input_data["max_seqs"]),
            "--remove-tmp-files",
            "1",
            "--format-output",
            ",".join(input_data["m8_columns"]),
        ]
        # threads=0 → omit --threads entirely so mmseqs uses its built-in
        # "all available cores" auto-detection. mmseqs rejects --threads 0.
        threads = int(input_data["threads"])
        if threads > 0:
            cmd += ["--threads", str(threads)]
        if use_gpu:
            cmd += ["--gpu", "1"]
        cmd.extend(str(arg) for arg in input_data.get("extra_args", []))

        _run_cmd(cmd, "mmseqs easy-search")

        m8_file = Path(m8_path)
        stdout = m8_file.read_text() if m8_file.exists() else ""

    return {"stdout": stdout}


# ============================================================================
# Operation — genome_search
# ============================================================================


def run_genome_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run MMseqs2 genome-to-genome nucleotide search and return raw m8 output.

    Args:
        input_data: Dict with keys: query_sequences, query_ids, target_sequences,
            target_ids, search_type, threads, sensitivity, evalue, min_seq_id,
            coverage, cov_mode, max_seqs, strand, m8_columns.

    Returns:
        Dict with keys: stdout (raw tab-separated m8 output)
    """
    mmseqs = _find_binary()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

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

        query_db = str(tmp_path / "query_db")
        target_db = str(tmp_path / "target_db")
        mmseqs_tmp = str(tmp_path / "mmseqs_tmp")
        res_dir = str(tmp_path / "results")
        results_m8 = str(tmp_path / "mmseqs_results.m8")

        Path(mmseqs_tmp).mkdir()
        Path(res_dir).mkdir()

        search_type = str(input_data["search_type"])
        sensitivity = str(input_data["sensitivity"])
        evalue = str(input_data["evalue"])
        min_seq_id = str(input_data["min_seq_id"])
        coverage = str(input_data["coverage"])
        cov_mode = str(input_data["cov_mode"])
        max_seqs = str(input_data["max_seqs"])
        strand = str(input_data["strand"])

        # Threads: 0 means "let mmseqs auto-detect" (binary default behavior).
        # When the caller passes a positive value, cap it at the target DB size:
        # mmseqs splits the target DB into chunks proportional to thread count
        # and crashes with "World Size: <threads> dbSize: <num_target_seqs>" if
        # there aren't enough chunks per worker.
        requested_threads = int(input_data["threads"])
        target_db_size = len(input_data["target_sequences"])
        if requested_threads <= 0:
            # Pick a small safe value capped at target DB size; mmseqs's own
            # auto-detect would still hit the chunking bug on tiny test DBs.
            effective_threads = max(1, min((os.cpu_count() or 1), target_db_size))
        else:
            effective_threads = max(1, min(requested_threads, target_db_size))
        threads = str(effective_threads)

        _run_cmd(
            [mmseqs, "createdb", query_fasta, query_db],
            "mmseqs createdb (query)",
        )
        _run_cmd(
            [mmseqs, "createdb", target_fasta, target_db],
            "mmseqs createdb (target)",
        )
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
                "-e",
                evalue,
                "--min-seq-id",
                min_seq_id,
                "-c",
                coverage,
                "--cov-mode",
                cov_mode,
                "--max-seqs",
                max_seqs,
                "--strand",
                strand,
                *(str(arg) for arg in input_data.get("extra_args", [])),
            ],
            "mmseqs search",
        )

        m8_columns = input_data["m8_columns"]
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

        m8_file = Path(results_m8)
        stdout = m8_file.read_text() if m8_file.exists() else ""

    return {"stdout": stdout}


# ============================================================================
# Operation — clustering
# ============================================================================


def run_clustering(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run MMseqs2 clustering workflow and return cluster assignments.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, min_seq_id,
            coverage, cov_mode, evalue, cluster_mode, max_seqs, sensitivity.

    Returns:
        Dict with keys: cluster_assignments (dict mapping member_id -> representative_id)
    """
    mmseqs = _find_binary()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        input_fasta = str(tmp_path / "input.faa")
        _write_fasta(
            input_data["sequences"],
            input_fasta,
            input_data.get("sequence_ids"),
        )

        db_dir = tmp_path / "mmseqs_db"
        res_dir = tmp_path / "mmseqs_results"
        mmseqs_tmp = tmp_path / "tmp"
        db_dir.mkdir()
        res_dir.mkdir()
        mmseqs_tmp.mkdir()

        min_seq_id = str(input_data["min_seq_id"])
        coverage = str(input_data["coverage"])
        cov_mode = str(input_data["cov_mode"])
        evalue = str(input_data["evalue"])
        cluster_mode = str(input_data["cluster_mode"])
        max_seqs = str(input_data["max_seqs"])
        sensitivity = str(input_data["sensitivity"])

        _run_cmd(
            [mmseqs, "createdb", input_fasta, str(db_dir / "seqs")],
            "mmseqs createdb",
        )
        _run_cmd(
            [
                mmseqs,
                "cluster",
                str(db_dir / "seqs"),
                str(res_dir / "clusters"),
                str(mmseqs_tmp),
                "--min-seq-id",
                min_seq_id,
                "-c",
                coverage,
                "--cov-mode",
                cov_mode,
                "-e",
                evalue,
                "--cluster-mode",
                cluster_mode,
                "--max-seqs",
                max_seqs,
                "-s",
                sensitivity,
                *(str(arg) for arg in input_data.get("extra_args", [])),
            ],
            "mmseqs cluster",
        )
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


# ============================================================================
# Operation — homology_search (ColabFold MSA pipeline)
# ============================================================================


def _detect_database_name(dataset_dir: Path, db_prefix: str | None) -> str:
    """Pick the database name to pass to colabfold_search via ``--db1``.

    Args:
        dataset_dir (Path): Directory holding the indexed MMseqs2 DB files.
        db_prefix (str | None): The dataset entry's ``db_prefix`` (e.g.
            ``"uniref30_2302_db"``). When provided, validates the corresponding
            ``.dbtype`` file exists; raises otherwise.

    Returns:
        str: The DB name (without extension) for ``--db1``.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"mmseqs2-homology-search: dataset directory does not exist: {dataset_dir}")

    if db_prefix:
        if not (dataset_dir / f"{db_prefix}.dbtype").is_file():
            raise FileNotFoundError(
                f"mmseqs2-homology-search: expected MMseqs2 DB file {db_prefix}.dbtype not found in {dataset_dir}"
            )
        return db_prefix

    # Auto-detect: prefer *_db.dbtype files
    candidates = sorted(dataset_dir.glob("*_db.dbtype"))
    if not candidates:
        candidates = [
            f
            for f in sorted(dataset_dir.glob("*.dbtype"))
            if not any(f.stem.endswith(suffix) for suffix in ("_seq", "_aln", "_h", "_seq_h"))
        ]
    if not candidates:
        raise FileNotFoundError(f"mmseqs2-homology-search: no MMseqs2 DB files (*.dbtype) found in {dataset_dir}")
    return candidates[0].stem


def _resolve_executables() -> tuple[str, str]:
    """Return ``(colabfold_search_path, mmseqs_path)`` from this venv's bin dir."""
    venv_bin = Path(sys.executable).parent
    colabfold_path = venv_bin / "colabfold_search"
    if not colabfold_path.exists():
        which_colabfold = shutil.which("colabfold_search")
        if not which_colabfold:
            raise FileNotFoundError(
                "mmseqs2-homology-search: colabfold_search executable not found in venv (colabfold package missing); re-run standalone/setup.sh"
            )
        colabfold_path = Path(which_colabfold)

    mmseqs_path = venv_bin / "mmseqs"
    if not mmseqs_path.exists():
        which_mmseqs = shutil.which("mmseqs")
        if not which_mmseqs:
            raise FileNotFoundError(
                "mmseqs2-homology-search: mmseqs binary not found in venv; re-run standalone/setup.sh"
            )
        mmseqs_path = Path(which_mmseqs)

    return str(colabfold_path), str(mmseqs_path)


_INTERNAL_HEADER_PREFIX = "__q"


def _internal_header(idx: int) -> str:
    """Internal FASTA header for a query at position ``idx``.

    Decoupled from the user-supplied ``sequence_id`` so a user-chosen ID
    can never collide with colabfold_search's ``{idx}.a3m`` output naming.
    Specifically, ``colabfold_search --unpack=1`` (the default) renames
    ``{job_idx}.a3m`` to ``{safe_filename(raw_jobname)}.a3m``; if the user
    supplied ``sequence_id="1"`` for the query at idx=0, the resulting
    rename loop would clobber the second query's output. Always-prefixed
    internal headers sidestep the conflict entirely.
    """
    return f"{_INTERNAL_HEADER_PREFIX}{idx}"


def _write_query_fasta(sequences: list[str], path: Path) -> None:
    """Write sequences as a FASTA file at ``path`` with internal headers."""
    with open(path, "w") as f:
        f.writelines(f">{_internal_header(idx)}\n{seq}\n" for idx, seq in enumerate(sequences))


def _build_homology_command(
    *,
    colabfold_path: str,
    mmseqs_path: str,
    query_fasta: Path,
    dataset_dir: Path,
    output_dir: Path,
    db_name: str,
    sensitivity: float | None,
    prefilter_mode: int | None,
    max_seqs: int | None,
    extra_args: list[str],
    num_threads: int,
    use_gpu: bool,
) -> list[str]:
    """Assemble the colabfold_search CLI command from registry-driven flags."""
    cmd = [colabfold_path, "--mmseqs", mmseqs_path, "--threads", str(num_threads)]
    if sensitivity is not None:
        cmd += ["-s", str(sensitivity)]
    if prefilter_mode is not None:
        cmd += ["--prefilter-mode", str(prefilter_mode)]
    if max_seqs is not None:
        cmd += ["--max-accept", str(max_seqs)]
    # Phase 3: env DB not used (no metagenomic support yet)
    cmd += ["--use-env", "0"]
    cmd += ["--db1", db_name]
    cmd += [str(query_fasta), str(dataset_dir), str(output_dir)]
    if use_gpu:
        cmd += ["--gpu", "1"]
    cmd += list(extra_args)
    return cmd


def _backfill_empty_a3ms(query_fasta: Path, output_dir: Path) -> None:
    """Create stub A3M files for sequences that produced no hits.

    colabfold_search exits with a "prof_res does not exist" error when a
    query has no homologs at all. This helper synthesizes the trivial
    single-sequence A3M (just the query) so downstream consumers see a
    file per query regardless of hit count.
    """
    sequences: list[tuple[str, str]] = []
    current_header: str | None = None
    current_seq: list[str] = []
    for line in query_fasta.read_text().strip().split("\n"):
        if line.startswith(">"):
            if current_header is not None:
                sequences.append((current_header, "".join(current_seq)))
            current_header = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_header is not None:
        sequences.append((current_header, "".join(current_seq)))

    for header, seq in sequences:
        a3m_path = output_dir / f"{header}.a3m"
        if not a3m_path.exists():
            a3m_path.write_text(f">{header}\n{seq}\n")


def run_homology_search(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the ColabFold-style MSA pipeline against a registry-provisioned DB.

    Args:
        input_dict (dict[str, Any]): Dispatch payload built by the tool layer.
            Required keys: ``sequences`` (list[str]), ``dataset_dir`` (str),
            ``db_prefix`` (str), ``output_dir`` (str), ``num_threads`` (int).
            Optional: ``sensitivity`` (float | None), ``prefilter_mode``
            (int | None), ``max_seqs`` (int | None), ``extra_args`` (list[str]),
            ``use_gpu`` (bool, default False), ``verbose`` (bool, default False),
            ``colabfold_timeout`` (float | None).

    Returns:
        dict[str, Any]: ``{"success": bool, "output_dir": str, ...}``. On
        failure, includes ``error`` with the captured stderr/stdout.
    """
    sequences: list[str] = list(input_dict["sequences"])
    dataset_dir = Path(input_dict["dataset_dir"])
    output_dir = Path(input_dict["output_dir"])
    db_prefix: str | None = input_dict.get("db_prefix")
    num_threads: int = int(input_dict["num_threads"])
    use_gpu: bool = bool(input_dict.get("use_gpu", False))
    verbose: bool = bool(input_dict.get("verbose", False))
    sensitivity = input_dict.get("sensitivity")
    prefilter_mode = input_dict.get("prefilter_mode")
    max_seqs = input_dict.get("max_seqs")
    extra_args: list[str] = list(input_dict.get("extra_args", []))
    # Inner timeout fires before the framework's outer ToolInstance timeout so
    # we can return a structured error (with cleanup) instead of being
    # hard-killed mid-subprocess. None disables the inner timeout entirely.
    timeout: float | None = input_dict.get("colabfold_timeout")

    output_dir.mkdir(parents=True, exist_ok=True)
    colabfold_path, mmseqs_path = _resolve_executables()
    db_name = _detect_database_name(dataset_dir, db_prefix)

    query_fasta = output_dir / "query.fasta"
    _write_query_fasta(sequences, query_fasta)

    cmd = _build_homology_command(
        colabfold_path=colabfold_path,
        mmseqs_path=mmseqs_path,
        query_fasta=query_fasta,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        db_name=db_name,
        sensitivity=sensitivity,
        prefilter_mode=prefilter_mode,
        max_seqs=max_seqs,
        extra_args=extra_args,
        num_threads=num_threads,
        use_gpu=use_gpu,
    )

    try:
        # Always capture so we can inspect stderr for known recovery markers
        # (e.g. "prof_res does not exist") regardless of verbose. When verbose
        # is set, replay captured streams to our own stdout/stderr at the end.
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        if verbose:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
        return {"success": True, "output_dir": str(output_dir), "db_name": db_name}
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "output_dir": str(output_dir),
            "error": f"mmseqs2-homology-search: colabfold_search timed out after {e.timeout}s (raise Mmseqs2HomologySearchConfig.timeout if needed)",
        }
    except subprocess.CalledProcessError as e:
        if verbose:
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
        stderr = e.stderr or ""
        # "prof_res does not exist" = no hits at all; backfill empty A3Ms and
        # treat as success so the tool layer can return MSA=None for those queries.
        if "prof_res does not exist" in stderr:
            _backfill_empty_a3ms(query_fasta, output_dir)
            return {"success": True, "output_dir": str(output_dir), "db_name": db_name, "no_hits_fallback": True}
        return {
            "success": False,
            "output_dir": str(output_dir),
            "returncode": e.returncode,
            "error": f"mmseqs2-homology-search: colabfold_search failed (exit {e.returncode}): {' | '.join((stderr or '').strip().splitlines()[-10:]) or '<no stderr>'}",
        }


# ============================================================================
# Persistent-worker contract + one-shot CLI
# ============================================================================


_OPERATIONS = {
    "protein_search": run_protein_search,
    "genome_search": run_genome_search,
    "clustering": run_clustering,
    "homology_search": run_homology_search,
}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Route ``input_dict["operation"]`` to the right ``run_*`` function."""
    operation = input_dict.get("operation")
    if operation not in _OPERATIONS:
        valid = ", ".join(sorted(_OPERATIONS))
        raise ValueError(f"mmseqs2: unknown operation {operation!r}; valid: [{valid}]")
    return _OPERATIONS[operation](input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool — subprocess auto-unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"mmseqs2: usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = Path(sys.argv[1])
    output_json_path = Path(sys.argv[2])
    output_json_path.write_text(json.dumps(dispatch(json.loads(input_json_path.read_text()))))
