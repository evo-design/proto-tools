"""DeepPBS specificity standalone runner for ToolInstance venv execution."""

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

_DNA_RES_TO_INT = {
    "DA": 0,
    "DC": 1,
    "DG": 2,
    "DT": 3,
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
}


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _resolve_deeppbs_repo() -> str:
    """Resolve the DeepPBS checkout (with ``run/predict.py``) from the weights cache.

    The standalone setup clones the repository (with its bundled X3DNA/DSSR binaries,
    configs, and inference weights) into the managed cache, so resolution is just that
    directory (overridable via ``PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR``).
    """
    try:
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("deeppbs_specificity")
    except Exception as exc:  # resolve_weights_dir is best-effort here
        logger.debug("resolve_weights_dir('deeppbs_specificity') unavailable: %s", exc)
        weights_dir = None

    if weights_dir and os.path.isfile(os.path.join(weights_dir, "run", "predict.py")):
        return os.path.abspath(weights_dir)
    raise FileNotFoundError(
        "deeppbs-specificity: could not locate a DeepPBS checkout (with run/predict.py) in "
        f"the weights cache ({weights_dir or '<unresolved>'}). The standalone setup provisions it "
        "automatically; to use an existing checkout set PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR."
    )


def _canonicalize_prediction(raw_npz_path: str, canonical_npz_path: str) -> dict[str, Any]:
    """Convert DeepPBS prediction NPZ into canonical DNA format."""
    raw = np.load(raw_npz_path, allow_pickle=True)

    pred = np.asarray(raw["P"], dtype=np.float64)
    seq_one_hot = np.asarray(raw["Seq"], dtype=np.float64)

    bp_pred = np.flip(np.flip(pred, axis=1), axis=0)
    bp_seq = np.flip(np.flip(seq_one_hot, axis=1), axis=0)

    pred_full = np.concatenate((pred, bp_pred), axis=0)
    seq_full = np.concatenate((seq_one_hot, bp_seq), axis=0)

    true_dna = np.argmax(seq_full, axis=1).astype(np.int64)
    chain_labels = np.concatenate(
        (
            np.zeros(pred.shape[0], dtype=np.int64),
            np.ones(bp_pred.shape[0], dtype=np.int64),
        ),
        axis=0,
    )
    mask = np.ones(pred_full.shape[0], dtype=np.int64)
    dna_mask = np.ones(pred_full.shape[0], dtype=np.int64)

    row_sum = pred_full.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    pred_full = pred_full / row_sum

    np.savez_compressed(
        canonical_npz_path,
        predicted_ppm=pred_full,
        true_sequence=true_dna,
        mask=mask,
        dna_mask=dna_mask,
        chain_labels=chain_labels,
        source_method=np.array("deeppbs"),
    )

    return {
        "predicted_ppm": pred_full.tolist(),
        "true_sequence": true_dna.tolist(),
        "mask": mask.tolist(),
        "dna_mask": dna_mask.tolist(),
        "chain_labels": chain_labels.tolist(),
    }


def _fallback_canonical_from_pdb(canonical_npz_path: str, pdb_path: str) -> dict[str, Any]:
    """Build a conservative fallback canonical output from DNA residues in a PDB.

    Emits one row per DNA residue found (per-chain labels). For a standard two-strand
    co-crystal this already matches the real path's row count (one row per base, both
    strands), with a uniform ``0.25`` PPM in place of a real prediction.
    """
    seen = set()
    chain_to_idx: dict[str, int] = {}
    true_sequence: list[int] = []
    chain_labels: list[int] = []

    with open(pdb_path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resn = line[17:20].strip().upper()
            if resn not in _DNA_RES_TO_INT:
                continue
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode, resn)
            if key in seen:
                continue
            seen.add(key)

            if chain_id not in chain_to_idx:
                chain_to_idx[chain_id] = len(chain_to_idx)
            true_sequence.append(_DNA_RES_TO_INT[resn])
            chain_labels.append(chain_to_idx[chain_id])

    if not true_sequence:
        true_sequence = [0]
        chain_labels = [0]

    length = len(true_sequence)
    predicted_ppm = np.full((length, 4), 0.25, dtype=np.float64)
    mask = np.ones(length, dtype=np.int64)
    dna_mask = np.ones(length, dtype=np.int64)
    chain_labels_arr = np.asarray(chain_labels, dtype=np.int64)
    true_arr = np.asarray(true_sequence, dtype=np.int64)

    np.savez_compressed(
        canonical_npz_path,
        predicted_ppm=predicted_ppm,
        true_sequence=true_arr,
        mask=mask,
        dna_mask=dna_mask,
        chain_labels=chain_labels_arr,
        source_method=np.array("deeppbs"),
    )

    return {
        "predicted_ppm": predicted_ppm.tolist(),
        "true_sequence": true_arr.tolist(),
        "mask": mask.tolist(),
        "dna_mask": dna_mask.tolist(),
        "chain_labels": chain_labels_arr.tolist(),
    }


def _run_one_structure(pdb_path: str, config: dict[str, Any], output_root: str) -> dict[str, Any]:
    """Run DeepPBS prediction for one input PDB and return canonical output."""
    pdb_path = os.path.abspath(pdb_path)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB path does not exist: {pdb_path}")

    repo_path = _resolve_deeppbs_repo()
    allow_fallback = bool(config.get("allow_fallback", False))
    process_script = os.path.join(repo_path, "run", "process_co_crystal.py")
    predict_script = os.path.join(repo_path, "run", "predict.py")
    if not os.path.exists(process_script):
        raise FileNotFoundError(f"DeepPBS process script missing: {process_script}")
    if not os.path.exists(predict_script):
        raise FileNotFoundError(f"DeepPBS predict script missing: {predict_script}")

    # Process/predict configs ship with the repo clone.
    process_config = os.path.abspath(os.path.join(repo_path, "run", "process", "process_config.json"))
    predict_config = os.path.abspath(
        os.path.join(repo_path, "run", "process", "pred_configs", "pred_config_deeppbs.json")
    )
    if not os.path.exists(process_config):
        raise FileNotFoundError(f"DeepPBS process config missing: {process_config}")
    if not os.path.exists(predict_config):
        raise FileNotFoundError(f"DeepPBS predict config missing: {predict_config}")

    input_name = Path(pdb_path).stem
    tmp_dir = tempfile.mkdtemp(prefix=f"deeppbs_{input_name}_")

    try:
        pdb_dir = os.path.join(tmp_dir, "pdb")
        os.makedirs(pdb_dir, exist_ok=True)
        input_filename = os.path.basename(pdb_path)
        shutil.copy(pdb_path, os.path.join(pdb_dir, input_filename))

        npz_dir = os.path.join(tmp_dir, "npz")
        os.makedirs(npz_dir, exist_ok=True)

        input_txt = os.path.join(tmp_dir, "input.txt")
        _write_text(input_txt, f"{input_filename}\n")

        predict_input_txt = os.path.join(tmp_dir, "predict_input.txt")
        _write_text(predict_input_txt, f"{input_name}.npz\n")

        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        repo_pythonpath = os.pathsep.join([repo_path, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
        env = get_subprocess_device_env(config.get("device", "cuda"))
        env["PYTHONPATH"] = repo_pythonpath
        env["PYTHONNOUSERSITE"] = "1"
        # X3DNA/DSSR binaries ship under <repo>/dependencies/bin; the canonical $X3DNA
        # env var (shared with the x3dna-fiber tool) is honored as an override.
        x3dna_env_root = os.environ.get("X3DNA")
        x3dna_bin_paths: list[str] = [os.path.abspath(os.path.join(repo_path, "dependencies", "bin"))]
        if x3dna_env_root:
            x3dna_bin_paths.append(os.path.abspath(os.path.join(x3dna_env_root, "bin")))
        # Keep order stable while deduplicating.
        x3dna_bin_paths = list(dict.fromkeys(x3dna_bin_paths))
        x3dna_home = x3dna_env_root or os.path.join(repo_path, "x3dna-v2.3-linux-64bit", "x3dna-v2.3")
        existing_bins = [path for path in x3dna_bin_paths if os.path.isdir(path)]
        if existing_bins:
            env["PATH"] = os.pathsep.join([*existing_bins, env.get("PATH", "")]).strip(os.pathsep)
        if os.path.isdir(x3dna_home):
            env["X3DNA"] = x3dna_home

        missing_bins = [
            binary for binary in ("x3dna-dssr", "analyze") if shutil.which(binary, path=env.get("PATH")) is None
        ]
        if missing_bins:
            fallback_reason = (
                "DeepPBS dependency missing: "
                + ", ".join(missing_bins)
                + " not found in PATH. "
                + "Searched bin dirs: "
                + ", ".join(existing_bins or x3dna_bin_paths)
                + "."
            )
            if not allow_fallback:
                raise RuntimeError(fallback_reason)
            canonical_dir = os.path.join(output_root, "canonical_npz")
            os.makedirs(canonical_dir, exist_ok=True)
            canonical_npz = os.path.join(canonical_dir, f"{input_name}.npz")
            fallback = _fallback_canonical_from_pdb(canonical_npz, pdb_path)
            logger.warning("deeppbs_specificity fallback for %s: %s", input_name, fallback_reason)
            return {
                "input_name": input_name,
                "source_method": "deeppbs",
                "output_npz_path": canonical_npz,
                "used_fallback": True,
                "fallback_reason": fallback_reason,
                **fallback,
            }

        verbose = bool(config.get("verbose", False))

        def _run_or_raise(cmd: list[str], step_name: str) -> dict[str, str]:
            if verbose:
                subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    env=env,
                    check=True,
                )
                return {"stdout": "", "stderr": ""}

            proc = subprocess.run(
                cmd,
                cwd=tmp_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip().splitlines()[-40:]
                stdout_tail = (proc.stdout or "").strip().splitlines()[-40:]
                combined_tail = "\n".join(["[stderr]", *stderr_tail, "", "[stdout]", *stdout_tail]).strip()
                raise RuntimeError(
                    f"DeepPBS {step_name} failed for {pdb_path} (exit {proc.returncode}).\n{combined_tail}"
                )
            return {"stdout": proc.stdout or "", "stderr": proc.stderr or ""}

        process_cmd = [
            sys.executable,
            process_script,
            input_txt,
            process_config,
            "--no_pwm",
        ]
        if bool(config.get("no_clean_protein", False)):
            process_cmd.append("--no_cleanp")
        process_io = _run_or_raise(process_cmd, step_name="preprocessing")

        canonical_dir = os.path.join(output_root, "canonical_npz")
        os.makedirs(canonical_dir, exist_ok=True)
        canonical_npz = os.path.join(canonical_dir, f"{input_name}.npz")

        preprocessed_npz = os.path.join(npz_dir, f"{input_name}.npz")
        if not os.path.exists(preprocessed_npz):
            stdout_tail = (process_io["stdout"] or "").strip().splitlines()[-4:]
            stderr_tail = (process_io["stderr"] or "").strip().splitlines()[-4:]
            details = "; ".join(
                part
                for part in [
                    ("stdout tail: " + " | ".join(stdout_tail)) if stdout_tail else "",
                    ("stderr tail: " + " | ".join(stderr_tail)) if stderr_tail else "",
                ]
                if part
            )
            fallback_reason = (
                "DeepPBS preprocessing produced no NPZ output"
                + (f"; {details}" if details else "")
                + "; hint: ensure AF3 output contains a clean protein-DNA co-crystal with both DNA strands."
            )
            if not allow_fallback:
                raise RuntimeError(fallback_reason)
            fallback = _fallback_canonical_from_pdb(canonical_npz, pdb_path)
            logger.warning("deeppbs_specificity fallback for %s: %s", input_name, fallback_reason)
            return {
                "input_name": input_name,
                "source_method": "deeppbs",
                "output_npz_path": canonical_npz,
                "used_fallback": True,
                "fallback_reason": fallback_reason,
                **fallback,
            }

        try:
            _run_or_raise(
                [
                    sys.executable,
                    predict_script,
                    predict_input_txt,
                    output_dir,
                    "-c",
                    predict_config,
                ],
                step_name="prediction",
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            fallback = _fallback_canonical_from_pdb(canonical_npz, pdb_path)
            logger.warning("deeppbs_specificity fallback for %s: prediction step failed: %s", input_name, exc)
            return {
                "input_name": input_name,
                "source_method": "deeppbs",
                "output_npz_path": canonical_npz,
                "used_fallback": True,
                "fallback_reason": "DeepPBS prediction step failed",
                **fallback,
            }

        raw_npz = os.path.join(output_dir, "npzs", f"{input_name}.npz_predict.npz")
        if not os.path.exists(raw_npz):
            fallback_reason = "DeepPBS prediction produced no output NPZ"
            if not allow_fallback:
                raise RuntimeError(fallback_reason)
            fallback = _fallback_canonical_from_pdb(canonical_npz, pdb_path)
            logger.warning("deeppbs_specificity fallback for %s: %s", input_name, fallback_reason)
            return {
                "input_name": input_name,
                "source_method": "deeppbs",
                "output_npz_path": canonical_npz,
                "used_fallback": True,
                "fallback_reason": fallback_reason,
                **fallback,
            }

        canonical = _canonicalize_prediction(raw_npz, canonical_npz)

        return {
            "input_name": input_name,
            "source_method": "deeppbs",
            "output_npz_path": canonical_npz,
            "used_fallback": False,
            "fallback_reason": None,
            **canonical,
        }
    finally:
        if not bool(config.get("keep_intermediate", False)):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_deeppbs_specificity(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run DeepPBS specificity across input PDB paths."""
    output_directory = input_data.get("output_directory")
    if output_directory is None:
        output_root = tempfile.mkdtemp(prefix="deeppbs_specificity_")
    else:
        output_root = os.path.abspath(output_directory)
        os.makedirs(output_root, exist_ok=True)

    config = {
        "keep_intermediate": bool(input_data.get("keep_intermediate", False)),
        "no_clean_protein": bool(input_data.get("no_clean_protein", False)),
        "allow_fallback": bool(input_data.get("allow_fallback", False)),
        "device": str(input_data.get("device", "cuda")),
        "verbose": bool(input_data.get("verbose", False)),
    }

    results = [_run_one_structure(pdb_path, config, output_root) for pdb_path in input_data["pdb_paths"]]

    return {"results": results}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent worker execution."""
    return run_deeppbs_specificity(input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for subprocess CLI tool - auto-unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as handle:
        payload = json.load(handle)

    result = run_deeppbs_specificity(payload)

    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(result, handle)
