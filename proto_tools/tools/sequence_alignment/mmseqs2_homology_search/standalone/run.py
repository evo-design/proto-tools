"""mmseqs2-homology-search standalone subprocess.

Runs the ColabFold-style MSA pipeline via the ``colabfold_search`` CLI
(installed as part of the ``colabfold`` Python package in this tool's
isolated env). Phase 3 is intentionally a thin shim: it builds the CLI
invocation from the registry-driven dispatch dict and returns the path
to per-query A3M files for the tool layer to parse into MSA objects.

Phase 4 will replace this with direct ``mmseqs`` invocations to add the
tax-tagged paired-MSA pipeline.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _detect_database_name(dataset_dir: Path, db_prefix: str | None) -> str:
    """Pick the database name to pass to colabfold_search via --db1.

    Args:
        dataset_dir (Path): Directory holding the indexed MMseqs2 DB files.
        db_prefix (str | None): The dataset entry's ``db_prefix`` (e.g.
            ``"uniref30_2302_db"``). When provided, validates the corresponding
            ``.dbtype`` file exists; raises otherwise.

    Returns:
        str: The DB name (without extension) for ``--db1``.

    Raises:
        FileNotFoundError: If the dataset_dir is missing or the prefixed
            ``.dbtype`` file is absent.
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
    """Write sequences as a FASTA file at ``path`` with internal headers.

    User ``sequence_id`` values are intentionally not written here — see
    :func:`_internal_header` for the rationale.
    """
    with open(path, "w") as f:
        f.writelines(f">{_internal_header(idx)}\n{seq}\n" for idx, seq in enumerate(sequences))


def _build_command(
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
    file per query regardless of hit count. Only writes files that don't
    already exist (so it's safe to call after partial-success runs).

    File naming matches what colabfold writes when a query *does* have
    hits — ``{internal_header}.a3m`` (see :func:`_internal_header`). The
    tool layer maps these back to user ``sequence_id``s.
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


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution.

    Args:
        input_dict (dict[str, Any]): Dispatch payload built by the tool layer.
            Required keys: ``sequences`` (list[str]), ``dataset_dir`` (str),
            ``db_prefix`` (str), ``output_dir`` (str), ``num_threads`` (int).
            Optional: ``sensitivity`` (float | None), ``prefilter_mode``
            (int | None), ``max_seqs`` (int | None), ``extra_args`` (list[str]),
            ``use_gpu`` (bool, default False), ``verbose`` (bool, default False).
            Output A3Ms are named ``__q{idx}.a3m`` (see :func:`_internal_header`);
            the tool layer maps these to user-supplied ``sequence_id`` names.

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

    cmd = _build_command(
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("mmseqs2-homology-search: usage: python run.py <input_json_path> <output_json_path>")
    input_json_path = Path(sys.argv[1])
    output_json_path = Path(sys.argv[2])
    output_json_path.write_text(json.dumps(dispatch(json.loads(input_json_path.read_text()))))
