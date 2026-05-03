"""AlphaFold3 Structure Prediction Pipeline.

Two execution paths, selected at runtime based on what setup.sh provisioned:

1. Sif path (preferred): invokes ``apptainer run <sif>`` against a pre-built
   AlphaFold3 container image (the sif's ``%runscript`` execs ``run_alphafold.py``).
   Robust across systems.
2. Env path (fallback): invokes ``python run_alphafold.py`` directly from a
   source clone of AlphaFold3 installed into the tool's micromamba venv.

MSAs are supplied by the caller via the input JSON (proto_tools delegates MSA
generation to the colabfold-search tool), so we pass --norun_data_pipeline and
skip the sequence databases entirely on both paths.

Worker protocol implementation for ToolInstance integration.
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any

import numpy as np
from Bio import PDB

logger = logging.getLogger(__name__)

# Import from auto-copied standalone_helpers
from standalone_helpers import get_subprocess_device_env, resolve_weights_dir


class AlphaFold3ExecutionError(Exception):
    """Raised when AlphaFold3 execution fails."""


AlphaFold3JSON = dict[str, Any]


def _venv_path() -> str:
    """Return the tool's venv directory, resolved from standard env vars."""
    venv_path = os.environ.get("VIRTUAL_ENV") or os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
    if not venv_path:
        raise FileNotFoundError(
            "alphafold3: cannot locate tool venv — VIRTUAL_ENV / TOOL_VENV_PATH / VENV_PATH not set"
        )
    return venv_path


def _resolve_sif_path(override: str | None = None) -> str | None:
    """Resolve the AlphaFold3 sif image path, or None if the env path is in use.

    Precedence:
        1. Caller-supplied override (from tool config's sif_path).
        2. ``$VENV_PATH/alphafold3.sif`` — written by setup.sh when the sif path is taken.

    Args:
        override (str | None): Optional explicit sif path from config.

    Returns:
        str | None: Absolute path to an existing sif file, or None if neither source yields one.
    """
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"alphafold3: config sif_path does not exist: {override}")
        return override
    default = os.path.join(_venv_path(), "alphafold3.sif")
    return default if os.path.exists(default) else None


def _resolve_repo_path() -> str:
    """Locate the AlphaFold3 source clone written by setup.sh (env path only).

    Returns:
        str: Absolute path to the local AlphaFold3 repository containing run_alphafold.py.

    Raises:
        FileNotFoundError: If setup.sh has not been run or the marker file is missing.
    """
    venv_path = _venv_path()
    marker = os.path.join(venv_path, "alphafold3_repo_path.txt")
    if not os.path.exists(marker):
        raise FileNotFoundError(f"alphafold3: repo marker not found at {marker}; re-run setup.sh")
    with open(marker) as f:
        repo_path = f.read().strip()
    if not os.path.exists(os.path.join(repo_path, "run_alphafold.py")):
        raise FileNotFoundError(f"alphafold3: run_alphafold.py not found in {repo_path}; re-run setup.sh")
    return repo_path


def _extract_structure_and_scores(
    output_dir: str,
    name: str,
    verbose: bool = False,  # noqa: ARG001 — required by tool interface
    include_pae_matrix: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Extract predicted structure and confidence scores from AlphaFold3 output.

    Args:
        output_dir (str): Directory containing AlphaFold3 output.
        name (str): Name of the prediction job.
        verbose (bool): Whether to print progress messages.
        include_pae_matrix (bool): Attach the full per-residue PAE matrix.

    Returns:
        tuple[str, dict[str, Any]]: Tuple of (pdb_path, scores_dict).
    """
    alphafold3_results_folder = os.path.join(output_dir, name)
    alphafold3_structure = os.path.join(alphafold3_results_folder, f"{name}_model.cif")

    # Convert mmCIF structure file to PDB format.
    pdb_path = os.path.join(output_dir, f"{name}_af3.pdb")
    parser = PDB.MMCIFParser(QUIET=True)  # type: ignore[attr-defined, no-untyped-call]
    io = PDB.PDBIO()  # type: ignore[attr-defined, no-untyped-call]
    structure = parser.get_structure("structure", alphafold3_structure)  # type: ignore[no-untyped-call]
    io.set_structure(structure)  # type: ignore[no-untyped-call]
    io.save(pdb_path)  # type: ignore[no-untyped-call]

    # Extract confidence scores from AlphaFold3 JSON output files.
    summary_confidences_path = os.path.join(alphafold3_results_folder, f"{name}_summary_confidences.json")
    full_confidences_path = os.path.join(alphafold3_results_folder, f"{name}_confidences.json")

    with open(summary_confidences_path) as f:
        summary_metrics = json.load(f)
    with open(full_confidences_path) as f:
        full_metrics = json.load(f)

    alphafold3_scores: dict[str, Any] = {}
    alphafold3_scores["avg_plddt"] = float(np.mean(full_metrics["atom_plddts"]))
    alphafold3_scores["avg_pae"] = float(np.mean(np.array(full_metrics["pae"])))
    if include_pae_matrix:
        alphafold3_scores["pae_matrix"] = full_metrics["pae"]
    alphafold3_scores["ptm"] = summary_metrics.get("ptm")
    alphafold3_scores["iptm"] = summary_metrics.get("iptm")
    alphafold3_scores["ranking_score"] = summary_metrics.get("ranking_score")

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(alphafold3_scores, f, indent=2)

    return pdb_path, alphafold3_scores


# ============================================================================
# Worker Protocol - AlphaFold3 Model Class
# ============================================================================


class AlphaFold3Model:
    """Wrapper for AlphaFold3 CLI execution (sif or env path)."""

    def __init__(self) -> None:
        """Initialize model with unresolved paths."""
        self._loaded = False
        self.sif_path: str | None = None  # set when running via apptainer
        self.repo_path: str | None = None  # set when running via env
        self.model_dir: str | None = None

    def load(self, model_dir: str | None = None, sif_path: str | None = None) -> None:
        """Resolve execution-path + weights paths and verify they exist.

        Prefers the sif path if either a config override is supplied or setup.sh
        provisioned ``$VENV_PATH/alphafold3.sif``. Falls back to the env path otherwise.

        Args:
            model_dir (str | None): Caller-provided weights directory override
                (from the tool config). Falls through to PROTO_ALPHAFOLD3_WEIGHTS_DIR /
                PROTO_MODEL_CACHE / PROTO_HOME when unset.
            sif_path (str | None): Caller-provided sif path override. Falls through
                to ``$VENV_PATH/alphafold3.sif`` when unset.
        """
        # 1. Resolve execution path (sif preferred, env fallback).
        resolved_sif = _resolve_sif_path(sif_path)
        if resolved_sif:
            self.sif_path = resolved_sif
            logger.debug("AlphaFold3: using sif path %s", self.sif_path)
        else:
            self.repo_path = _resolve_repo_path()
            logger.debug("AlphaFold3: using env path, repo at %s", self.repo_path)

        # 2. Resolve weights (same logic on both paths).
        if model_dir:
            self.model_dir = model_dir
        else:
            weights_dir = resolve_weights_dir("alphafold3")
            if not weights_dir:
                raise FileNotFoundError(
                    "alphafold3: unable to resolve weights directory; "
                    "set PROTO_ALPHAFOLD3_WEIGHTS_DIR or configure PROTO_MODEL_CACHE, "
                    "or pass model_dir via the tool config"
                )
            self.model_dir = weights_dir

        has_weights = any(
            f.endswith((".bin", ".bin.zst"))
            for f in os.listdir(self.model_dir)
            if os.path.isfile(os.path.join(self.model_dir, f))
        )
        if not has_weights:
            raise FileNotFoundError(
                f"alphafold3: no weights (.bin / .bin.zst) found in {self.model_dir}; "
                "request access via https://github.com/google-deepmind/alphafold3#obtaining-model-parameters "
                "and place the downloaded weights file there (or override "
                "with PROTO_ALPHAFOLD3_WEIGHTS_DIR)"
            )

        self._loaded = True

    def _build_cmd(self, input_json_path: str, output_dir: str) -> list[str]:
        """Build the AlphaFold3 subprocess argv for whichever path is active."""
        assert self.model_dir is not None  # set by load()

        common_args = [
            f"--json_path={input_json_path}",
            f"--output_dir={output_dir}",
            f"--model_dir={self.model_dir}",
            "--norun_data_pipeline",
        ]

        if self.sif_path is not None:
            # Sif path: `apptainer run` invokes the sif's %runscript with the
            # appended args. This avoids hardcoding the in-sif path to
            # run_alphafold.py — the sif itself encapsulates where AF3 lives
            # in its rootfs (our Singularity.def puts it at /opt/alphafold3).
            # BYO sifs must have a %runscript that accepts AF3 CLI args.
            input_dir = os.path.dirname(input_json_path)
            apptainer_bin = os.path.join(_venv_path(), "bin", "apptainer")
            return [
                apptainer_bin,
                "run",
                "--nv",
                "--bind",
                f"{input_dir}:{input_dir}",
                "--bind",
                f"{output_dir}:{output_dir}",
                "--bind",
                f"{self.model_dir}:{self.model_dir}",
                self.sif_path,
                *common_args,
            ]

        # Env path: direct python subprocess against the cloned repo.
        assert self.repo_path is not None
        return [
            sys.executable,
            os.path.join(self.repo_path, "run_alphafold.py"),
            *common_args,
        ]

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        device: str,
        model_dir: str | None = None,
        sif_path: str | None = None,
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run AlphaFold3 prediction.

        MSA generation is handled in the main process before calling this method.
        This method runs AlphaFold3 with --norun_data_pipeline via whichever
        execution path setup.sh provisioned (sif preferred, env fallback).

        Args:
            input_json_path (str): Path to AlphaFold3 input JSON file (MSAs already populated).
            output_dir (str): Directory for output files.
            device (str): Device for subprocess environment (e.g., "cuda:0").
            model_dir (str | None): Optional explicit override for the weights directory.
            sif_path (str | None): Optional explicit override for the sif image path.
            verbose (bool): Whether to print progress messages.
            include_pae_matrix (bool): Attach the full per-residue PAE matrix.

        Returns:
            dict[str, Any]: Dict with ``structure_pdb`` (path to generated PDB) and
                ``metrics`` (confidence scores).

        Raises:
            AlphaFold3ExecutionError: If AlphaFold3 execution fails.
        """
        if not self._loaded:
            self.load(model_dir=model_dir, sif_path=sif_path)

        # Load input JSON (MSAs already generated by main process)
        with open(input_json_path) as f:
            input_json = json.load(f)

        # Output directory handling: if the dir already exists, append .1, .2, etc.
        original_output_dir = output_dir
        counter = 1
        while os.path.exists(output_dir):
            output_dir = f"{original_output_dir}.{counter}"
            counter += 1
        os.makedirs(output_dir)
        if counter > 1:
            logger.debug(f"Output dir existed, created new directory: {output_dir}")

        run_cmds = self._build_cmd(input_json_path, output_dir)

        logger.debug("Executing AlphaFold3 (%s path)...", "sif" if self.sif_path else "env")
        logger.debug(f"  Input: {input_json_path}")
        logger.debug(f"  Output: {output_dir}")
        logger.debug(f"  Model dir: {self.model_dir}")

        # Pin the subprocess to the caller-specified GPU via CUDA_VISIBLE_DEVICES
        env = get_subprocess_device_env(device)

        # Inherit streams when verbose for real-time progress; capture when not
        # verbose so we can surface stderr tail on failure.
        process = subprocess.Popen(
            run_cmds,
            stdout=sys.stdout if verbose else subprocess.PIPE,
            stderr=sys.stderr if verbose else subprocess.PIPE,
            text=True,
            env=env,
        )

        _stdout, stderr = process.communicate()

        if process.returncode != 0:
            if stderr:
                stderr_tail = " | ".join(stderr.strip().splitlines()[-10:])
            else:
                # verbose=True streamed stderr to terminal — no buffer to tail.
                stderr_tail = "<streamed to terminal; rerun with verbose=False to capture>"
            raise AlphaFold3ExecutionError(f"alphafold3: failed (exit {process.returncode}): {stderr_tail}")

        logger.debug("AlphaFold3 execution completed successfully.")

        pdb_path, alphafold3_scores = _extract_structure_and_scores(
            output_dir,
            input_json["name"],
            verbose=verbose,
            include_pae_matrix=include_pae_matrix,
        )

        return {
            "structure_pdb": pdb_path,
            "metrics": alphafold3_scores,
        }


# ============================================================================
# Worker Protocol Entry Points
# ============================================================================

_model: AlphaFold3Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for ToolInstance worker protocol.

    MSA generation is handled in the main process, so this only runs AlphaFold3.

    Args:
        input_dict (dict[str, Any]): Input parameters from ToolInstance.dispatch().

    Returns:
        dict[str, Any]: Dict with structure_pdb and metrics.
    """
    global _model

    if _model is None:
        _model = AlphaFold3Model()

    return _model(
        input_json_path=input_dict["input_json_path"],
        output_dir=input_dict["output_dir"],
        device=input_dict["device"],
        model_dir=input_dict.get("model_dir"),
        sif_path=input_dict.get("sif_path"),
        verbose=input_dict["verbose"],
        include_pae_matrix=input_dict["include_pae_matrix"],
    )


def to_device(device: str) -> dict[str, Any]:
    """DeviceManager callback for moving the model to a different device.

    AlphaFold3 runs as a fresh Python subprocess per call (no persistent in-process
    state), so device assignment is a passthrough — CUDA_VISIBLE_DEVICES on the
    subprocess handles allocation.

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
        raise ValueError("alphafold3: usage: python inference.py <input_json_path> <output_json_path>")
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    result = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
