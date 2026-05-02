"""CRISPRtracrRNA standalone runner for ToolInstance venv execution.

Wraps the CRISPRtracrRNA.py tool from the Backofen Lab to predict
tracrRNA sequences from nucleotide CRISPR loci.

Supports parallel execution: sequences are split into batches, each
running in its own temp directory with a symlinked CWD to avoid file
contention from CRISPRtracrRNA's intermediate files.

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# =============================================================================
# CRISPRtracrRNA Invocation
# =============================================================================
def _find_crispr_tracr_rna_script() -> str:
    """Find the CRISPRtracrRNA.py script path.

    Searches in common installation locations.

    Returns:
        Path to CRISPRtracrRNA.py script.

    Raises:
        FileNotFoundError: If script cannot be found.
    """
    # Check if CRISPR_TRACR_RNA_PATH env var is set
    env_path = os.environ.get("CRISPR_TRACR_RNA_PATH")
    if env_path and Path(env_path).exists():
        script = Path(env_path) / "CRISPRtracrRNA.py"
        if script.exists():
            return str(script)

    # Check common locations (within the tool venv, not $HOME)
    candidates = [
        Path(sys.prefix) / "share" / "CRISPRtracrRNA" / "CRISPRtracrRNA.py",
        Path(sys.prefix) / "CRISPRtracrRNA" / "CRISPRtracrRNA.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Try to find it on PATH
    import shutil

    which_result = shutil.which("CRISPRtracrRNA.py")
    if which_result:
        return which_result

    raise FileNotFoundError(
        "CRISPRtracrRNA.py not found. Set CRISPR_TRACR_RNA_PATH environment variable "
        "to the CRISPRtracrRNA installation directory, or run setup.sh to install."
    )


# NA-style cells normalized to None at the parser boundary so Pydantic's
# int/float coercion doesn't choke on them.
_NULL_CSV_TOKENS = frozenset({"", "NA", "nan", "None", "null", "-"})


def _normalize_csv_value(value: Any) -> Any:
    """Return None for empty / NA-style CSV cells, otherwise the raw value."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in _NULL_CSV_TOKENS:
        return None
    return value


def _parse_tracr_results(output_dir: Path, sequence_ids: list[str]) -> list[dict[str, Any]]:
    """Parse upstream CSVs into one per-sequence result, each with all candidate rows.

    Columns pass through verbatim (typed Pydantic model on the wrapper side); NA-style
    cells normalize to None. The priority sort routes CRISPRtracrRNA_result.csv (the
    complete_run final, includes score) and complete_report.csv (model_run dedupe)
    first, then per-fasta intermediates are skipped for any accession a priority file
    already covered.

    Args:
        output_dir: Path to CRISPRtracrRNA output directory.
        sequence_ids: List of input sequence IDs.

    Returns:
        One dict per input sequence: ``{"sequence_id": str, "candidates": list[dict]}``.
    """
    candidates_by_id: dict[str, list[dict[str, Any]]] = {seq_id: [] for seq_id in sequence_ids}
    covered_by_priority: set[str] = set()

    # complete_run final ranked CSV first; then model_run's per-accession dedupe.
    priority = ("CRISPRtracrRNA_result.csv", "complete_report.csv")
    csv_files = sorted(
        output_dir.glob("*.csv"),
        key=lambda p: (priority.index(p.name) if p.name in priority else len(priority), p.name),
    )

    for csv_file in csv_files:
        is_priority = csv_file.name in priority
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    acc_id = row.get("accession_number") or row.get("acc_num") or ""
                    if not acc_id:
                        continue
                    if not is_priority and acc_id in covered_by_priority:
                        continue
                    normalized = {k: _normalize_csv_value(v) for k, v in row.items()}
                    normalized["sequence_id"] = acc_id
                    candidates_by_id.setdefault(acc_id, []).append(normalized)
                    if is_priority:
                        covered_by_priority.add(acc_id)
        except Exception as e:
            print(f"Warning: Failed to parse {csv_file}: {e}", file=sys.stderr)
            continue

    if not any(candidates_by_id.values()) and sequence_ids:
        print(
            f"Warning: No tracrRNA CSV output files found in {output_dir} for {len(sequence_ids)} input sequences",
            file=sys.stderr,
        )

    # Sort score-descending; rows with no score (NA / missing) sort last.
    for cands in candidates_by_id.values():
        cands.sort(key=lambda c: float(c["score"]) if c.get("score") else float("-inf"), reverse=True)

    return [{"sequence_id": seq_id, "candidates": candidates_by_id.get(seq_id, [])} for seq_id in sequence_ids]


# =============================================================================
# Per-batch execution
# =============================================================================
def _prepare_run_env() -> dict[str, Any]:
    """Prepare environment variables for CRISPRtracrRNA subprocess.

    Prepends conda_deps/bin to PATH so bioinformatics tools
    (IntaRNA, cmsearch, prodigal, vmatch, etc.) installed via setup.sh
    are found without polluting the base conda environment.
    """
    run_env = os.environ.copy()
    conda_deps = Path(sys.prefix) / "conda_deps"
    conda_deps_bin = conda_deps / "bin"
    if conda_deps_bin.exists():
        run_env["PATH"] = str(conda_deps_bin) + ":" + run_env.get("PATH", "")
        # vmatch needs MKVTREESMAPDIR to find symbol map files
        vmatch_trans = conda_deps / "share"
        vmatch_dirs = list(vmatch_trans.glob("vmatch-*/TRANS"))
        if vmatch_dirs:
            run_env["MKVTREESMAPDIR"] = str(vmatch_dirs[0])
    return run_env


def _create_worker_cwd(tracr_install_dir: str, worker_dir: Path) -> Path:
    """Create a worker-specific CWD with symlinks to CRISPRtracrRNA install contents.

    CRISPRtracrRNA writes intermediate files (e.g., fasta_similarity_inverted.fastab)
    to CWD with fixed names. Giving each worker its own CWD with symlinks back to
    the install directory avoids file contention between parallel workers.

    Args:
        tracr_install_dir: Path to the CRISPRtracrRNA installation directory.
        worker_dir: Temp directory for this worker.

    Returns:
        Path to the worker-specific CWD.
    """
    worker_cwd = worker_dir / "cwd"
    worker_cwd.mkdir()
    install_path = Path(tracr_install_dir)
    for item in install_path.iterdir():
        target = worker_cwd / item.name
        if not target.exists():
            os.symlink(item, target)
    return worker_cwd


# Upstream argparse defaults; flags matching these are omitted from the CLI.
_UPSTREAM_DEFAULTS: dict[str, Any] = {
    "anti_repeat_similarity_threshold": 0.7,
    "anti_repeat_coverage_threshold": 0.6,
    "weight_crispr_array_score": 0.5,
    "weight_anti_repeat_sim": 0.5,
    "weight_anti_repeat_coverage": 0.5,
    "weight_anti_sim_coverage": 0.5,
    "weight_interaction_score": 0.6,
    "weight_model_hit_score": 0.9,
    "weight_terminator_hit_score": 0.9,
    "weight_consistency_orientation": 0.1,
    "weight_consistency_anti_repeat_tail": 0.1,
    "weight_consistency_tail_terminator": 0.1,
}


def _build_upstream_flags(config: dict[str, Any]) -> list[str]:
    """Build CRISPRtracrRNA CLI flags for the optional config knobs.

    Only emits a flag when the configured value differs from upstream's
    documented default. ``perform_type_v_anti_repeat_analysis`` is special-
    cased: upstream's argparse uses ``type=bool`` (the Python footgun where
    any non-empty string evaluates True), so we only pass the flag when our
    config is True.
    """
    flags: list[str] = []
    for key, default in _UPSTREAM_DEFAULTS.items():
        value = config.get(key, default)
        if value != default:
            flags.extend([f"--{key}", str(value)])
    if config.get("perform_type_v_anti_repeat_analysis", False):
        flags.extend(["--perform_type_v_anti_repeat_analysis", "True"])
    return flags


def _run_tracr_batch(
    sequences: list[str],
    sequence_ids: list[str],
    config: dict[str, Any],
    crispr_tracr_rna_script: str,
    tracr_install_dir: str,
    run_env: dict[str, Any],
    batch_idx: int,
) -> list[dict[str, Any]]:
    """Run CRISPRtracrRNA on a batch of sequences with an isolated CWD.

    Args:
        sequences: Nucleotide sequences in this batch.
        sequence_ids: Corresponding sequence IDs.
        config: Full upstream config dict (model_type, run_type, thresholds,
            ranking weights, perform_type_v_anti_repeat_analysis).
        crispr_tracr_rna_script: Full path to CRISPRtracrRNA.py.
        tracr_install_dir: CRISPRtracrRNA installation directory.
        run_env: Environment variables dict.
        batch_idx: Batch index for logging.

    Returns:
        List of prediction dicts for this batch.
    """
    model_type = config.get("model_type", "II")
    run_type = config.get("run_type", "complete_run")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        tmp_dir = temp_path / "tmp"
        input_dir.mkdir()
        output_dir.mkdir()
        tmp_dir.mkdir()

        # Create isolated CWD for this worker
        worker_cwd = _create_worker_cwd(tracr_install_dir, temp_path)

        # Write each sequence as a separate FASTA file
        for seq_id, seq in zip(sequence_ids, sequences, strict=False):
            clean_seq = "".join(c for c in seq.upper() if c in "ATCGN")
            fasta_path = input_dir / f"{seq_id}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">{seq_id}\n{clean_seq}\n")

        cmd = [
            sys.executable,
            crispr_tracr_rna_script,
            "--input_folder",
            str(input_dir),
            "--output_folder",
            str(output_dir),
            "--output_summary_file",
            str(output_dir / "CRISPRtracrRNA_result.csv"),
            "--temp_folder_path",
            str(tmp_dir),
            "--model_type",
            model_type,
            "--run_type",
            run_type,
            *_build_upstream_flags(config),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(600, len(sequences) * 120),
                cwd=str(worker_cwd),
                env=run_env,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"CRISPRtracrRNA.py batch {batch_idx} timed out after {max(600, len(sequences) * 120)} seconds"
            ) from e

        if proc.returncode != 0:
            print(
                f"Batch {batch_idx}: CRISPRtracrRNA.py exited with code {proc.returncode}",
                file=sys.stderr,
            )
            print(proc.stderr, file=sys.stderr)

        results = _parse_tracr_results(output_dir, sequence_ids)

        if proc.returncode != 0 and not any(r["candidates"] for r in results):
            # CRISPRtracrRNA can crash on sequences where fasta36 finds no
            # anti-repeat hits (IndexError in anti_repeat_search.py). Expected
            # for some generated sequences — treat as "no tracrRNA found".
            print(
                f"WARNING: Batch {batch_idx} produced no results "
                f"(exit code {proc.returncode}); treating as no tracrRNA found",
                file=sys.stderr,
            )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================
def run_crispr_tracr_rna(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run CRISPRtracrRNA prediction on one or more sequences.

    Splits sequences into batches and runs them in parallel, each in an
    isolated working directory to avoid file contention.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, config.

    Returns:
        Dict with key: results (list of per-sequence result dicts).
    """
    sequences = input_data["sequences"]
    sequence_ids = input_data["sequence_ids"]
    config = input_data.get("config", {})
    num_workers = config["num_workers"]  # always resolved wrapper-side

    crispr_tracr_rna_script = _find_crispr_tracr_rna_script()
    tracr_install_dir = str(Path(crispr_tracr_rna_script).parent)
    run_env = _prepare_run_env()

    # Single-worker fast path
    if num_workers <= 1 or len(sequences) <= 1:
        results = _run_tracr_batch(
            sequences,
            sequence_ids,
            config,
            crispr_tracr_rna_script,
            tracr_install_dir,
            run_env,
            batch_idx=0,
        )
        return {"results": results}

    # Split sequences into batches
    num_workers = min(num_workers, len(sequences))
    batch_size = math.ceil(len(sequences) / num_workers)
    batches = [
        (
            sequences[i : i + batch_size],
            sequence_ids[i : i + batch_size],
        )
        for i in range(0, len(sequences), batch_size)
    ]

    print(
        f"Running CRISPRtracrRNA with {len(batches)} parallel workers ({batch_size} sequences/batch)",
        file=sys.stderr,
    )

    # Run batches in parallel (ThreadPoolExecutor is fine since workers
    # are subprocess-bound, not CPU-bound in Python)
    all_results = [None] * len(batches)
    batch_errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for idx, (batch_seqs, batch_ids) in enumerate(batches):
            future = executor.submit(
                _run_tracr_batch,
                batch_seqs,
                batch_ids,
                config,
                crispr_tracr_rna_script,
                tracr_install_dir,
                run_env,
                batch_idx=idx,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_results[idx] = future.result()  # type: ignore[call-overload]
            except Exception as e:
                print(
                    f"ERROR: Batch {idx} failed: {e}",
                    file=sys.stderr,
                )
                batch_errors.append(idx)
                # Empty per-sequence results so each input is still represented.
                _, batch_ids = batches[idx]
                all_results[idx] = [  # type: ignore[call-overload]
                    {"sequence_id": seq_id, "candidates": []} for seq_id in batch_ids
                ]

    if batch_errors:
        print(
            f"WARNING: {len(batch_errors)}/{len(batches)} batches failed: {batch_errors}",
            file=sys.stderr,
        )

    # Flatten in original order
    results = []
    for batch_results in all_results:
        results.extend(batch_results)  # type: ignore[arg-type]

    return {"results": results}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    return run_crispr_tracr_rna(input_dict)


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

    output_data = run_crispr_tracr_rna(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
