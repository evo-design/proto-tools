"""
CRISPRtracrRNA standalone runner for ToolInstance venv execution.

Wraps the CRISPRtracrRNA.py tool from the Backofen Lab to predict
tracrRNA sequences from nucleotide CRISPR loci.

Supports parallel execution: sequences are split into batches, each
running in its own temp directory with a symlinked CWD to avoid file
contention from CRISPRtracrRNA's intermediate files.

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# CRISPRtracrRNA Invocation
# =============================================================================
def _find_crispr_tracr_script() -> str:
    """Find the CRISPRtracrRNA.py script path.

    Searches in common installation locations.

    Returns:
        Path to CRISPRtracrRNA.py script.

    Raises:
        FileNotFoundError: If script cannot be found.
    """
    # Check if CRISPR_TRACR_PATH env var is set
    env_path = os.environ.get("CRISPR_TRACR_PATH")
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
        "CRISPRtracrRNA.py not found. Set CRISPR_TRACR_PATH environment variable "
        "to the CRISPRtracrRNA installation directory, or run setup.sh to install."
    )


def _parse_tracr_results(output_dir: Path, sequence_ids: List[str]) -> List[Dict[str, Any]]:
    """Parse CRISPRtracrRNA output CSV files.

    CRISPRtracrRNA writes results to CSV files in the output directory.
    We parse all CSV files and match results back to input sequence IDs.

    Args:
        output_dir: Path to CRISPRtracrRNA output directory.
        sequence_ids: List of input sequence IDs.

    Returns:
        List of prediction dicts, one per input sequence.
    """
    # Collect all results from CSV files in output dir
    results_by_id = {}

    for csv_file in output_dir.glob("*.csv"):
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    acc_id = row.get("accession_number", row.get("tracr_id", ""))
                    if acc_id and acc_id not in results_by_id:
                        # CRISPRtracrRNA complete_run mode uses anti_repeat_start/end
                        # and tracr_rna_sequence columns, not tracr_start/end/hit.
                        tracr_start = _safe_int(
                            row.get("anti_repeat_start", row.get("tracr_start"))
                        )
                        tracr_end = _safe_int(
                            row.get("anti_repeat_end", row.get("tracr_end"))
                        )
                        tracr_hit = (
                            row.get("tracr_rna_sequence")
                            or row.get("tracr_hit")
                        )
                        interaction_energy = _safe_float(
                            row.get("interaction_energy")
                        )
                        results_by_id[acc_id] = {
                            "sequence_id": acc_id,
                            "tracr_start": tracr_start,
                            "tracr_end": tracr_end,
                            "tracr_hit": tracr_hit,
                            "interaction_energy": interaction_energy,
                            "anti_repeat_similarity_coverage_multiplication": _safe_float(
                                row.get("anti_repeat_similarity_coverage_multiplication")
                            ),
                            "intarna_anti_repeat_interaction": row.get(
                                "intarna_anti_repeat_interaction"
                            ),
                        }
        except Exception as e:
            print(f"Warning: Failed to parse {csv_file}: {e}", file=sys.stderr)
            continue

    if not results_by_id and sequence_ids:
        print(
            f"Warning: No tracrRNA CSV output files found in {output_dir} "
            f"for {len(sequence_ids)} input sequences",
            file=sys.stderr,
        )

    # Build results list in input order
    predictions = []
    for seq_id in sequence_ids:
        if seq_id in results_by_id:
            predictions.append(results_by_id[seq_id])
        else:
            predictions.append({
                "sequence_id": seq_id,
                "tracr_start": None,
                "tracr_end": None,
                "tracr_hit": None,
                "interaction_energy": None,
                "anti_repeat_similarity_coverage_multiplication": None,
                "intarna_anti_repeat_interaction": None,
            })

    return predictions


def _safe_int(val) -> Optional[int]:
    """Safely convert to int, returning None on failure."""
    if val is None or val == "" or val == "NA" or val == "nan":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, returning None on failure."""
    if val is None or val == "" or val == "NA" or val == "nan":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Per-batch execution
# =============================================================================
def _prepare_run_env() -> dict:
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


def _run_tracr_batch(
    sequences: List[str],
    sequence_ids: List[str],
    model_type: str,
    run_type: str,
    crispr_tracr_script: str,
    tracr_install_dir: str,
    run_env: dict,
    batch_idx: int,
) -> List[Dict[str, Any]]:
    """Run CRISPRtracrRNA on a batch of sequences with an isolated CWD.

    Args:
        sequences: Nucleotide sequences in this batch.
        sequence_ids: Corresponding sequence IDs.
        model_type: CRISPR model type ("II" or "all").
        run_type: Run type ("complete_run" or "model_only").
        crispr_tracr_script: Full path to CRISPRtracrRNA.py.
        tracr_install_dir: CRISPRtracrRNA installation directory.
        run_env: Environment variables dict.
        batch_idx: Batch index for logging.

    Returns:
        List of prediction dicts for this batch.
    """
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
        for seq_id, seq in zip(sequence_ids, sequences):
            clean_seq = "".join(c for c in seq.upper() if c in "ATCGN")
            fasta_path = input_dir / f"{seq_id}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">{seq_id}\n{clean_seq}\n")

        cmd = [
            sys.executable,
            crispr_tracr_script,
            "--input_folder", str(input_dir),
            "--output_folder", str(output_dir),
            "--temp_folder_path", str(tmp_dir),
            "--model_type", model_type,
            "--run_type", run_type,
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
                f"CRISPRtracrRNA.py batch {batch_idx} timed out after "
                f"{max(600, len(sequences) * 120)} seconds"
            ) from e

        if proc.returncode != 0:
            print(
                f"Batch {batch_idx}: CRISPRtracrRNA.py exited with code "
                f"{proc.returncode}",
                file=sys.stderr,
            )
            print(proc.stderr, file=sys.stderr)

        predictions = _parse_tracr_results(output_dir, sequence_ids)

        if proc.returncode != 0 and not any(
            p.get("tracr_start") is not None for p in predictions
        ):
            # CRISPRtracrRNA can crash on sequences where fasta36 finds no
            # anti-repeat hits (IndexError in anti_repeat_search.py).  This
            # is expected for some generated sequences — treat as "no
            # tracrRNA found" rather than a hard failure.
            print(
                f"WARNING: Batch {batch_idx} produced no results "
                f"(exit code {proc.returncode}) — treating as no tracrRNA found",
                file=sys.stderr,
            )

    return predictions


# =============================================================================
# Main Entry Point
# =============================================================================
def run_crispr_tracr(input_data: dict) -> dict:
    """Run CRISPRtracrRNA prediction on one or more sequences.

    Splits sequences into batches and runs them in parallel, each in an
    isolated working directory to avoid file contention.

    Args:
        input_data: Dict with keys: sequences, sequence_ids, config.

    Returns:
        Dict with key: predictions (list of prediction dicts).
    """
    sequences = input_data["sequences"]
    sequence_ids = input_data["sequence_ids"]
    config = input_data.get("config", {})
    model_type = config.get("model_type", "II")
    run_type = config.get("run_type", "complete_run")
    num_workers = config.get("num_workers")
    if num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_workers = int(slurm_cpus) if slurm_cpus else 1

    crispr_tracr_script = _find_crispr_tracr_script()
    tracr_install_dir = str(Path(crispr_tracr_script).parent)
    run_env = _prepare_run_env()

    # Single-worker fast path
    if num_workers <= 1 or len(sequences) <= 1:
        predictions = _run_tracr_batch(
            sequences, sequence_ids, model_type, run_type,
            crispr_tracr_script, tracr_install_dir, run_env,
            batch_idx=0,
        )
        return {"predictions": predictions}

    # Split sequences into batches
    num_workers = min(num_workers, len(sequences))
    batch_size = math.ceil(len(sequences) / num_workers)
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append((
            sequences[i:i + batch_size],
            sequence_ids[i:i + batch_size],
        ))

    print(
        f"Running CRISPRtracrRNA with {len(batches)} parallel workers "
        f"({batch_size} sequences/batch)",
        file=sys.stderr,
    )

    # Run batches in parallel (ThreadPoolExecutor is fine since workers
    # are subprocess-bound, not CPU-bound in Python)
    all_predictions = [None] * len(batches)
    batch_errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for idx, (batch_seqs, batch_ids) in enumerate(batches):
            future = executor.submit(
                _run_tracr_batch,
                batch_seqs, batch_ids, model_type, run_type,
                crispr_tracr_script, tracr_install_dir, run_env,
                batch_idx=idx,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_predictions[idx] = future.result()
            except Exception as e:
                print(
                    f"ERROR: Batch {idx} failed: {e}",
                    file=sys.stderr,
                )
                batch_errors.append(idx)
                # Fill with empty predictions for this batch
                _, batch_ids = batches[idx]
                all_predictions[idx] = [
                    {
                        "sequence_id": seq_id,
                        "tracr_start": None,
                        "tracr_end": None,
                        "tracr_hit": None,
                        "interaction_energy": None,
                        "anti_repeat_similarity_coverage_multiplication": None,
                        "intarna_anti_repeat_interaction": None,
                    }
                    for seq_id in batch_ids
                ]

    if batch_errors:
        print(
            f"WARNING: {len(batch_errors)}/{len(batches)} batches failed: "
            f"{batch_errors}",
            file=sys.stderr,
        )

    # Flatten predictions in original order
    predictions = []
    for batch_preds in all_predictions:
        predictions.extend(batch_preds)

    return {"predictions": predictions}


def dispatch(input_dict: dict) -> dict:
    """Entry point for persistent-worker execution."""
    return run_crispr_tracr(input_dict)


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================

def to_device(device: str) -> dict:
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

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    output_data = run_crispr_tracr(input_data)

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
