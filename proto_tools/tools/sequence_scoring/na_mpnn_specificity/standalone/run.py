"""NA-MPNN specificity standalone runner for ToolInstance venv execution."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from standalone_helpers import get_logger, get_subprocess_device_env

logger = get_logger(__name__)

_DEFAULT_RESTYPE_TO_INT = {
    "DA": 21,
    "DC": 22,
    "DG": 23,
    "DT": 24,
}


def _resolve_weights_dir() -> str | None:
    """Resolve the managed cache dir holding the NA-MPNN checkout, if available."""
    try:
        from standalone_helpers import resolve_weights_dir

        weights_dir: str | None = resolve_weights_dir("na_mpnn_specificity")
        return weights_dir
    except Exception as exc:  # resolve_weights_dir is best-effort here
        logger.debug("resolve_weights_dir('na_mpnn_specificity') unavailable: %s", exc)
        return None


def _resolve_na_mpnn_repo() -> str:
    """Resolve the NA-MPNN checkout (with ``inference/run.py``) from the weights cache.

    The standalone setup clones the repository into the managed cache, so resolution
    is just that directory (overridable via ``PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR``).
    """
    weights_dir = _resolve_weights_dir()
    if weights_dir and os.path.isfile(os.path.join(weights_dir, "inference", "run.py")):
        return os.path.abspath(weights_dir)
    raise FileNotFoundError(
        "na-mpnn-specificity: could not locate an NA-MPNN checkout (with inference/run.py) in "
        f"the weights cache ({weights_dir or '<unresolved>'}). The standalone setup provisions it "
        "automatically; to use an existing checkout set PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR."
    )


def _resolve_checkpoint() -> str:
    """Resolve the NA-MPNN specificity checkpoint (``.pt``) from the weights cache.

    Prefers the in-repo specificity checkpoint (``models/specificity_model/*.pt``) so the
    design checkpoint is never picked up, then falls back to any top-level ``*.pt``.
    """
    weights_dir = _resolve_weights_dir()
    if weights_dir:
        for pattern in ("models/specificity_model/*.pt", "*.pt"):
            matches = sorted(str(path) for path in Path(weights_dir).glob(pattern))
            if matches:
                return os.path.abspath(matches[0])
    raise FileNotFoundError(
        "na-mpnn-specificity: could not locate a specificity checkpoint (.pt) in the weights "
        f"cache ({weights_dir or '<unresolved>'}). The standalone setup provisions it automatically; "
        "to use an existing checkout set PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR."
    )


def _load_restype_to_int(npz_data: Any) -> dict[str, int]:
    """Load the restype-to-int mapping from the NA-MPNN NPZ if present."""
    raw = npz_data.get("restype_to_int")
    if raw is None:
        return dict(_DEFAULT_RESTYPE_TO_INT)
    value = raw.item() if hasattr(raw, "item") else raw
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    logger.warning("restype_to_int present but not a dict; using defaults")
    return dict(_DEFAULT_RESTYPE_TO_INT)


def _canonicalize_npz(raw_npz_path: str, canonical_npz_path: str) -> dict[str, Any]:
    """Convert a raw NA-MPNN NPZ into the canonical DNA-only A,C,G,T format."""
    data = np.load(raw_npz_path, allow_pickle=True)

    predicted_ppm = np.asarray(data["predicted_ppm"], dtype=np.float64)
    true_sequence = np.asarray(data["true_sequence"], dtype=np.int64)
    mask = np.asarray(data["mask"], dtype=np.int64)
    dna_mask = np.asarray(data["dna_mask"], dtype=np.int64)
    chain_labels = np.asarray(data["chain_labels"], dtype=np.int64)

    restype_to_int = _load_restype_to_int(data)
    dna_cols = [
        int(restype_to_int["DA"]),
        int(restype_to_int["DC"]),
        int(restype_to_int["DG"]),
        int(restype_to_int["DT"]),
    ]

    selected = np.logical_and(mask == 1, dna_mask == 1)
    pred_dna = predicted_ppm[selected][:, dna_cols]

    row_sum = pred_dna.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    pred_dna = pred_dna / row_sum

    true_selected = true_sequence[selected]
    na_to_dna = {
        dna_cols[0]: 0,
        dna_cols[1]: 1,
        dna_cols[2]: 2,
        dna_cols[3]: 3,
    }
    true_dna = np.array([na_to_dna.get(int(x), -1) for x in true_selected], dtype=np.int64)

    chain_selected = chain_labels[selected]
    canonical_mask = np.ones(pred_dna.shape[0], dtype=np.int64)
    canonical_dna_mask = np.ones(pred_dna.shape[0], dtype=np.int64)

    # Positions whose ground-truth base isn't a known DNA token are masked out rather
    # than backfilled from the prediction (which would inflate recovery/accuracy).
    unknown = np.where(true_dna < 0)[0]
    if unknown.size:
        canonical_mask[unknown] = 0
        true_dna[unknown] = 0

    np.savez_compressed(
        canonical_npz_path,
        predicted_ppm=pred_dna,
        true_sequence=true_dna,
        mask=canonical_mask,
        dna_mask=canonical_dna_mask,
        chain_labels=chain_selected,
        source_method=np.array("na_mpnn"),
    )

    return {
        "predicted_ppm": pred_dna.tolist(),
        "true_sequence": true_dna.tolist(),
        "mask": canonical_mask.tolist(),
        "dna_mask": canonical_dna_mask.tolist(),
        "chain_labels": chain_selected.tolist(),
    }


def _run_one_structure(pdb_path: str, config: dict[str, Any], output_root: str) -> dict[str, Any]:
    """Run NA-MPNN specificity for one structure and return canonical output."""
    pdb_path = os.path.abspath(pdb_path)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB path does not exist: {pdb_path}")

    repo_path = _resolve_na_mpnn_repo()
    checkpoint_path = _resolve_checkpoint()

    run_script = os.path.join(repo_path, "inference", "run.py")
    if not os.path.exists(run_script):
        raise FileNotFoundError(f"NA-MPNN run script not found: {run_script}")

    input_name = Path(pdb_path).stem
    raw_dir = os.path.join(output_root, "na_mpnn_raw", input_name)
    os.makedirs(raw_dir, exist_ok=True)

    command = [
        sys.executable,
        run_script,
        "--model_type",
        "na_mpnn",
        "--checkpoint_na_mpnn",
        checkpoint_path,
        "--pdb_path",
        pdb_path,
        "--out_folder",
        raw_dir,
        "--number_of_batches",
        str(config["number_of_batches"]),
        "--batch_size",
        str(config["batch_size"]),
        "--temperature",
        str(config["temperature"]),
        "--omit_AA",
        str(config["omit_aa"]),
        "--design_na_only",
        str(int(bool(config["design_na_only"]))),
        "--load_residues_with_missing_atoms",
        "0",
        "--output_pdbs",
        "0",
        "--output_sequences",
        "0",
        "--output_specificity",
        "1",
        "--catch_failed_inferences",
        "0",
    ]

    logger.update_status(f"Running NA-MPNN on {input_name}")
    # Propagate the assigned device (CUDA_VISIBLE_DEVICES etc.) to the upstream
    # NA-MPNN CLI subprocess via the DeviceManager-aware helper.
    sub_env = {**get_subprocess_device_env(config.get("device", "cuda")), "PYTHONNOUSERSITE": "1"}
    verbose = bool(config.get("verbose", False))
    if verbose:
        subprocess.run(
            command,
            cwd=repo_path,
            env=sub_env,
            check=True,
        )
    else:
        proc = subprocess.run(
            command,
            cwd=repo_path,
            env=sub_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-30:]
            stdout_tail = (proc.stdout or "").strip().splitlines()[-30:]
            combined_tail = "\n".join(["[stderr]", *stderr_tail, "", "[stdout]", *stdout_tail]).strip()
            raise RuntimeError(f"NA-MPNN inference failed for {pdb_path} (exit {proc.returncode}).\n{combined_tail}")

    raw_npz = os.path.join(raw_dir, "specificity", f"{input_name}.npz")
    if not os.path.exists(raw_npz):
        raise FileNotFoundError(f"NA-MPNN specificity output not found: {raw_npz}")

    canonical_dir = os.path.join(output_root, "canonical_npz")
    os.makedirs(canonical_dir, exist_ok=True)
    canonical_npz = os.path.join(canonical_dir, f"{input_name}.npz")

    canonical = _canonicalize_npz(raw_npz, canonical_npz)

    if not bool(config.get("keep_intermediate", False)):
        shutil.rmtree(raw_dir, ignore_errors=True)

    return {
        "input_name": input_name,
        "source_method": "na_mpnn",
        "output_npz_path": canonical_npz,
        **canonical,
    }


def run_na_mpnn_specificity(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run NA-MPNN specificity across all input PDB paths."""
    output_directory = input_data.get("output_directory")
    if output_directory is None:
        output_root = tempfile.mkdtemp(prefix="na_mpnn_specificity_")
    else:
        output_root = os.path.abspath(output_directory)
        os.makedirs(output_root, exist_ok=True)

    config = {
        "batch_size": int(input_data.get("batch_size", 1)),
        "number_of_batches": int(input_data.get("number_of_batches", 1)),
        "temperature": float(input_data.get("temperature", 0.1)),
        "omit_aa": str(input_data.get("omit_aa", "")),
        "design_na_only": bool(input_data.get("design_na_only", True)),
        "keep_intermediate": bool(input_data.get("keep_intermediate", False)),
        "device": str(input_data.get("device", "cuda")),
        "verbose": bool(input_data.get("verbose", False)),
    }

    results = [_run_one_structure(pdb_path, config, output_root) for pdb_path in input_data["pdb_paths"]]

    return {"results": results}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker and one-shot execution."""
    return run_na_mpnn_specificity(input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager).

    NA-MPNN runs as a per-call subprocess of the upstream CLI, so there is no
    persistent in-process model to move; the device is honored on the next call.
    """
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("na_mpnn_specificity: usage: python run.py <input_json> <output_json>")

    with open(sys.argv[1]) as handle:
        payload = json.load(handle)

    result = dispatch(payload)

    with open(sys.argv[2], "w") as handle:
        json.dump(result, handle)
