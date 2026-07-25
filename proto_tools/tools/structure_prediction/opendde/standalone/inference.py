"""OpenDDE inference implementation."""

import glob
import json
import os
import re
import shutil
import sys
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

# Bundled model name -> checkpoint file under OPENDDE_ROOT_DIR (auto-downloaded by setup.sh); None = default weights.
_BUNDLED_CHECKPOINTS: dict[str, str | None] = {
    "opendde_v1": None,
    "opendde_abag": "checkpoint/opendde_abag.pt",
}


class OpenDDEExecutionError(Exception):
    """Raised when the ``opendde pred`` CLI fails for a non-OOM reason."""


def _bool_str(value: bool) -> str:
    """Render a Python bool as the lowercase ``true``/``false`` string OpenDDE's CLI expects."""
    return "true" if value else "false"


def _resolve_opendde_bin() -> str:
    """Locate the ``opendde`` console-script installed into the tool venv.

    The worker runs with the venv's interpreter, so the entry point lives next to
    ``sys.executable``; fall back to a ``PATH`` lookup for atypical layouts.

    Returns:
        str: Absolute path to the ``opendde`` executable.

    Raises:
        FileNotFoundError: If the executable cannot be located.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "opendde")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("opendde")
    if found:
        return found
    raise FileNotFoundError(
        "opendde: could not locate the 'opendde' console script; re-run setup.sh to install the 'opendde' package"
    )


def _extract_structure_and_scores(output_dir: str, include_pae_matrix: bool = False) -> dict[str, Any]:
    """Select the best-ranked sample and extract its structure + confidence metrics.

    OpenDDE writes, per diffusion sample under
    ``<output_dir>/<job_name>/seed_<seed>/predictions/``, a
    ``<job_name>_sample_<rank>.cif`` and a matching
    ``<job_name>_summary_confidence_sample_<rank>.json``. The summary already
    carries scalar confidence values (``plddt`` is the mean pLDDT on a 0-100
    scale, plus ``ptm``/``iptm``/``gpde``/``ranking_score``); The best
    sample is the one with the highest ``ranking_score``.

    When ``include_pae_matrix`` is set, the matching ``*_full_data_sample_<rank>.json``
    (emitted by ``--need_atom_confidence``) is read for the winning sample and its
    token-pair PAE matrix is returned as ``pae`` (Å) with a derived scalar ``avg_pae``.

    Args:
        output_dir (str): Directory OpenDDE wrote predictions into.
        include_pae_matrix (bool): Also read the per-token PAE matrix + avg_pae.

    Returns:
        dict[str, Any]: ``{"structure_cif_output": <cif str>, "metrics": {...}}``.

    Raises:
        FileNotFoundError: If no summary/CIF outputs are found (or the full-data
            file is missing when ``include_pae_matrix`` is set).
    """
    summary_paths = glob.glob(os.path.join(output_dir, "**", "*_summary_confidence_sample_*.json"), recursive=True)
    if not summary_paths:
        raise FileNotFoundError(
            f"opendde: no '*_summary_confidence_sample_*.json' outputs found under {output_dir}; "
            "the prediction produced no samples"
        )

    # Pick the best sample by ranking score across all seeds
    best_path: str | None = None
    best_summary: dict[str, Any] = {}
    best_score = float("-inf")
    for path in summary_paths:
        with open(path) as f:
            summary = json.load(f)
        score = summary.get("ranking_score")
        if score is None:
            logger.warning("opendde: summary %s has no 'ranking_score'; skipping for selection", path)
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_path = path
            best_summary = summary

    if best_path is None:
        raise FileNotFoundError(
            f"opendde: found {len(summary_paths)} summary file(s) under {output_dir} but none had a 'ranking_score'"
        )

    # Derive the sibling CIF path from the winning summary filename:
    #   <job_name>_summary_confidence_sample_<rank>.json  ->  <job_name>_sample_<rank>.cif
    pred_dir = os.path.dirname(best_path)
    basename = os.path.basename(best_path)
    match = re.search(r"^(?P<prefix>.+)_summary_confidence_sample_(?P<rank>.+)\.json$", basename)
    if match is None:
        raise FileNotFoundError(f"opendde: could not parse sample rank from summary filename {basename!r}")
    cif_path = os.path.join(pred_dir, f"{match.group('prefix')}_sample_{match.group('rank')}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"opendde: structure CIF not found for best sample: {cif_path}")

    metrics: dict[str, Any] = {
        "avg_plddt": float(best_summary["plddt"]),
        "ptm": float(best_summary["ptm"]),
        "iptm": float(best_summary["iptm"]),
        "gpde": float(best_summary["gpde"]),
        "ranking_score": float(best_summary["ranking_score"]),
        "has_clash": bool(best_summary["has_clash"]),
    }

    if include_pae_matrix:
        # Sibling full-data file for the winning sample (written by --need_atom_confidence).
        full_data_path = os.path.join(pred_dir, f"{match.group('prefix')}_full_data_sample_{match.group('rank')}.json")
        if not os.path.exists(full_data_path):
            raise FileNotFoundError(
                f"opendde: PAE requested but full-data file not found: {full_data_path}; "
                "the run must pass --need_atom_confidence"
            )
        with open(full_data_path) as f:
            full_data = json.load(f)
        pae = full_data.get("token_pair_pae")
        if pae is None:
            raise KeyError(f"opendde: 'token_pair_pae' missing from {full_data_path}")
        import numpy as np

        metrics["pae"] = pae
        metrics["avg_pae"] = float(np.asarray(pae, dtype=float).mean())

    with open(cif_path) as f:
        cif_output = f.read()

    return {"structure_cif_output": cif_output, "metrics": metrics}


# ============================================================================
# Worker Protocol - OpenDDE Model Class
# ============================================================================


class OpenDDEModel:
    """Wrapper for the ``opendde pred`` CLI (fresh subprocess per prediction)."""

    def __init__(self) -> None:
        """Initialize with an unresolved OpenDDE root directory."""
        self._loaded = False
        self.root_dir: str | None = None

    def load(self) -> None:
        """Resolve ``OPENDDE_ROOT_DIR`` (checkpoints + common assets) and verify it holds weights.

        An explicit ``OPENDDE_ROOT_DIR`` env var wins, otherwise the managed weights
        cache is resolved via ``resolve_weights_dir`` (PROTO_OPENDDE_WEIGHTS_DIR /
        PROTO_MODEL_CACHE / PROTO_HOME).

        Raises:
            FileNotFoundError: If no root can be resolved or no checkpoint is present.
        """
        from standalone_helpers import resolve_weights_dir

        root = os.environ.get("OPENDDE_ROOT_DIR")
        if not root:
            root = resolve_weights_dir("opendde")
        if not root:
            raise FileNotFoundError(
                "opendde: unable to resolve OPENDDE_ROOT_DIR; set OPENDDE_ROOT_DIR or "
                "PROTO_OPENDDE_WEIGHTS_DIR, or configure PROTO_MODEL_CACHE"
            )

        ckpt = os.path.join(root, "checkpoint", "opendde.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"opendde: checkpoint not found at {ckpt}; re-run setup.sh to download weights, "
                "or point OPENDDE_ROOT_DIR / PROTO_OPENDDE_WEIGHTS_DIR at a directory containing checkpoint/opendde.pt"
            )

        self.root_dir = root
        self._loaded = True

    def _build_cmd(
        self,
        input_json_path: str,
        output_dir: str,
        job_name: str,  # noqa: ARG002 -- OpenDDE derives the job name from the input JSON
        model_checkpoint: str,
        num_samples: int,
        num_steps: int,
        num_cycles: int,
        use_msa: bool,
        use_template: bool,
        use_rna_msa: bool,
        seed: int,
        device: str,
        include_pae_matrix: bool = False,
    ) -> list[str]:
        """Build the ``opendde pred`` argv."""
        assert self.root_dir is not None  # set by load()

        device_arg = "cpu" if device == "cpu" else "cuda"

        cmd = [
            _resolve_opendde_bin(),
            "pred",
            "-i",
            input_json_path,
            "-o",
            output_dir,
            "-n",
            "opendde_v1",
            "--sample",
            str(num_samples),
            "--step",
            str(num_steps),
            "--cycle",
            str(num_cycles),
            "--use_msa",
            _bool_str(use_msa),
            "--use_template",
            _bool_str(use_template),
            "--use_rna_msa",
            _bool_str(use_rna_msa),
            "--seeds",
            str(seed),
            "--device",
            device_arg,
            # Emit the full-data PAE file only when requested (it is O(n_token^2) to write).
            "--need_atom_confidence",
            _bool_str(include_pae_matrix),
        ]

        # Bundled name -> file under root (opendde_v1 -> None = default weights); anything else is an explicit path.
        if model_checkpoint in _BUNDLED_CHECKPOINTS:
            rel = _BUNDLED_CHECKPOINTS[model_checkpoint]
            resolved_ckpt = os.path.join(self.root_dir, rel) if rel else None
        else:
            resolved_ckpt = model_checkpoint

        if resolved_ckpt and not os.path.exists(resolved_ckpt):
            known = ", ".join(sorted(_BUNDLED_CHECKPOINTS))
            raise FileNotFoundError(
                f"opendde: checkpoint not found at {resolved_ckpt!r} (model_checkpoint={model_checkpoint!r}); "
                f"use a bundled model name ({known}) and re-run setup.sh to fetch weights, "
                "or pass a path to an existing .pt checkpoint"
            )
        if resolved_ckpt:
            cmd += ["--load_checkpoint_path", resolved_ckpt]

        return cmd

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        job_name: str,
        model_checkpoint: str,
        num_samples: int,
        num_steps: int,
        num_cycles: int,
        use_msa: bool,
        use_template: bool,
        use_rna_msa: bool,
        seed: int,
        device: str,
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run OpenDDE prediction and return the best sample's structure + metrics.

        Args:
            input_json_path (str): Path to the OpenDDE input JSON (MSAs already populated).
            output_dir (str): Directory for OpenDDE output files.
            job_name (str): Folding job name (drives OpenDDE's output subdirectory).
            model_checkpoint (str): Bundled model name (``opendde_v1``/``opendde_abag``) or
                a path to a custom ``.pt`` checkpoint.
            num_samples (int): Independent diffusion samples (``--sample``).
            num_steps (int): Diffusion denoising steps (``--step``).
            num_cycles (int): Recycling iterations (``--cycle``).
            use_msa (bool): Run OpenDDE's internal MSA search (``--use_msa``).
            use_template (bool): Run OpenDDE's template pipeline (``--use_template``).
            use_rna_msa (bool): Run OpenDDE's RNA MSA pipeline (``--use_rna_msa``).
            seed (int): Random seed (``--seeds``).
            device (str): Device string (e.g. ``"cuda"``, ``"cpu"``).
            verbose (bool): Tee subprocess stdout/stderr to this process.
            include_pae_matrix (bool): Pass ``--need_atom_confidence`` and return the
                per-token PAE matrix (``pae``) plus a derived ``avg_pae`` scalar.

        Returns:
            dict[str, Any]: ``{"structure_cif_output": <cif str>, "metrics": {...}}``.

        Raises:
            OpenDDEExecutionError: If the ``opendde pred`` CLI exits non-zero (non-OOM).
            GpuOutOfMemoryError: If the CLI failed with a CUDA out-of-memory error.
        """
        if not self._loaded:
            self.load()

        cmd = self._build_cmd(
            input_json_path=input_json_path,
            output_dir=output_dir,
            job_name=job_name,
            model_checkpoint=model_checkpoint,
            num_samples=num_samples,
            num_steps=num_steps,
            num_cycles=num_cycles,
            use_msa=use_msa,
            use_template=use_template,
            use_rna_msa=use_rna_msa,
            seed=seed,
            device=device,
            include_pae_matrix=include_pae_matrix,
        )

        # Pin the subprocess to the caller-specified GPU, and make sure OpenDDE reads
        # checkpoints/assets from the resolved root (env copy already carries any
        # config-supplied OPENDDE_ROOT_DIR; overwrite with the verified path).
        from standalone_helpers import get_subprocess_device_env, is_cuda_oom, raise_oom, run_teed

        env = get_subprocess_device_env(device)
        assert self.root_dir is not None
        env["OPENDDE_ROOT_DIR"] = self.root_dir

        logger.debug("Executing OpenDDE: %s", " ".join(cmd))
        returncode, _stdout, stderr = run_teed(cmd, env=env, verbose=verbose)

        if returncode != 0:
            stderr_tail = " | ".join(stderr.strip().splitlines()) or "<no stderr>"
            if is_cuda_oom(stderr_tail):
                raise_oom(
                    "opendde",
                    hint="Reduce --sample/--step/--cycle, shorten the complex, or use a GPU with more memory.",
                )
            raise OpenDDEExecutionError(f"opendde: prediction failed (exit {returncode}): {stderr_tail}")

        logger.debug("OpenDDE execution completed successfully.")
        return _extract_structure_and_scores(output_dir, include_pae_matrix=include_pae_matrix)


# ============================================================================
# Worker Protocol Entry Points
# ============================================================================

_model: OpenDDEModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution.

    Args:
        input_dict (dict[str, Any]): Input parameters from ``ToolInstance.dispatch()``.

    Returns:
        dict[str, Any]: Dict with ``structure_cif_output`` and ``metrics``.

    Raises:
        ValueError: If the requested operation is unknown.
    """
    global _model
    if _model is None:
        _model = OpenDDEModel()

    operation = input_dict["operation"]
    if operation == "predict":
        return _model(
            input_json_path=input_dict["input_json_path"],
            output_dir=input_dict["output_dir"],
            job_name=input_dict["job_name"],
            model_checkpoint=input_dict["model_checkpoint"],
            num_samples=input_dict["num_samples"],
            num_steps=input_dict["num_steps"],
            num_cycles=input_dict["num_cycles"],
            use_msa=input_dict["use_msa"],
            use_template=input_dict["use_template"],
            use_rna_msa=input_dict["use_rna_msa"],
            seed=input_dict["seed"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict.get("include_pae_matrix", False),
        )
    raise ValueError(f"opendde: unknown operation {operation!r}; valid: ['predict']")


def to_device(device: str) -> dict[str, Any]:
    """DeviceManager callback for moving the model to a different device.

    OpenDDE runs as a fresh subprocess per call (no persistent in-process state),
    so device assignment is a passthrough — CUDA_VISIBLE_DEVICES on the subprocess
    handles allocation.

    Args:
        device (str): Target device (e.g., "cpu", "cuda:0").

    Returns:
        dict[str, Any]: Success status dict.
    """
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """CLI tool, no persistent GPU state to report."""
    return {"available": False, "framework": "cli", "reason": "CLI tool, no persistent GPU state"}


# Worker protocol entry point
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("opendde: usage: python inference.py <input_json_path> <output_json_path>")
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    result = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
