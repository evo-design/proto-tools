"""
Protenix inference implementation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _cuequivariance_available() -> bool:
    """Check if cuequivariance kernels are installed."""
    try:
        import cuequivariance_ops_torch_cu12  # noqa: F401

        return True
    except ImportError:
        return False


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """Validate that a PyTorch checkpoint file is not corrupted.

    PyTorch checkpoint files are ZIP archives. This uses PyTorch's own
    checkpoint loading to detect corruption that generic ZIP validation might miss
    (e.g., "failed finding central directory" errors from PytorchStreamReader).

    Note: This loads the checkpoint into memory which may take 10-30 seconds for
    large models, but ensures thorough validation.

    Args:
        checkpoint_path: Path to the checkpoint file to validate

    Returns:
        True if the checkpoint is valid, False if corrupted
    """
    try:
        import torch
        # Try to load the checkpoint with PyTorch's ZIP reader (PytorchStreamReader)
        # This catches "failed finding central directory" and other ZIP corruption
        # that zipfile.testzip() misses
        logger.info(f"Validating checkpoint integrity (may take 10-30s): {checkpoint_path.name}")
        torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info(f"Checkpoint validation passed: {checkpoint_path.name}")
        return True
    except (RuntimeError, zipfile.BadZipFile, OSError, EOFError) as e:
        # RuntimeError: catches "PytorchStreamReader failed reading zip archive"
        # zipfile.BadZipFile: catches basic ZIP corruption
        # OSError: catches file system issues
        # EOFError: catches truncated files
        logger.warning(f"Checkpoint validation failed for {checkpoint_path}: {e}")
        return False


def _validate_single_checkpoint(checkpoint_path: Path) -> None:
    """Validate a single checkpoint file, removing it if corrupted.

    Uses a `.validated` marker file to cache validation results and avoid
    re-validating unchanged checkpoints on every inference call.

    Args:
        checkpoint_path: Path to a .pt checkpoint file
    """
    marker_path = checkpoint_path.parent / f"{checkpoint_path.name}.validated"

    # Check if validation marker exists and checkpoint hasn't changed
    needs_validation = True
    if marker_path.exists():
        try:
            marker_data = marker_path.read_text().strip().split(',')
            if len(marker_data) == 2:
                cached_size = int(marker_data[0])
                cached_mtime = float(marker_data[1])

                current_stat = checkpoint_path.stat()
                if current_stat.st_size == cached_size and current_stat.st_mtime == cached_mtime:
                    logger.debug(f"Checkpoint validation cached (skipping): {checkpoint_path.name}")
                    needs_validation = False
        except (ValueError, OSError) as e:
            logger.debug(f"Invalid validation marker, will re-validate: {e}")

    if needs_validation:
        logger.info(f"Validating checkpoint: {checkpoint_path.name}")
        if not validate_checkpoint(checkpoint_path):
            logger.warning(f"Removing corrupted checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()
            marker_path.unlink(missing_ok=True)
            logger.info(f"Corrupted checkpoint removed. It will be re-downloaded on next use.")
        else:
            stat = checkpoint_path.stat()
            marker_path.write_text(f"{stat.st_size},{stat.st_mtime}")
            logger.debug(f"Validation marker created: {marker_path.name}")


def cleanup_corrupted_checkpoints(checkpoint_dir: Path, model_name: str) -> None:
    """Check and remove corrupted checkpoint files in the checkpoint directory.

    Validates ALL .pt files in the directory (not just the named model),
    because tools like protenix also download dependency checkpoints
    (e.g., ESM2 embedding models) that can be corrupted by interrupted
    downloads.

    Uses `.validated` marker files to cache validation results and avoid
    re-validating unchanged checkpoints on every inference call.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_name: Name of the primary model (used for logging only)
    """
    if not checkpoint_dir.exists():
        return

    pt_files = list(checkpoint_dir.glob("*.pt"))
    if not pt_files:
        return

    for checkpoint_path in pt_files:
        _validate_single_checkpoint(checkpoint_path)


class ProtenixModel:
    """Protenix model for biomolecular structure prediction."""

    def __init__(self):
        """Initialize Protenix model wrapper."""
        self._loaded = False
        self.protenix_executable = None

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        device: str = "cuda",
        model_name: str = "protenix_base_default_v1.0.0",
        seeds: str = "0",
        num_diffusion_samples: int = 5,
        num_diffusion_steps: int = 200,
        num_pairformer_cycles: int = 10,
        use_msa: bool = True,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run Protenix structure prediction on one or more complexes.

        Args:
            input_json_path: Path to input JSON file (list of job dicts)
            output_dir: Directory to write output to
            device: Device for subprocess environment
            model_name: Protenix model variant to use
            seeds: Comma-separated seed values
            num_diffusion_samples: Number of diffusion samples per seed
            num_diffusion_steps: Number of denoising steps
            num_pairformer_cycles: Number of Pairformer recycling iterations
            use_msa: Whether to use MSA features
            verbose: Whether to print status messages

        Returns:
            List of dicts, each containing structure_cif_output and metrics,
            in the same order as the input jobs.
        """
        if not self._loaded:
            self.load()

        # Validate and cleanup corrupted checkpoints before running inference
        checkpoint_dir = Path(os.environ.get("PROTENIX_ROOT_DIR", Path.home())) / "checkpoint"
        cleanup_corrupted_checkpoints(checkpoint_dir, model_name)

        logger.debug("\n=== Protenix Prediction ===")
        logger.debug(f"Input JSON path: {input_json_path}")
        logger.debug(f"Output directory: {output_dir}")

        # Build the command
        cmd = [
            self.protenix_executable,
            "pred",
            "-i", input_json_path,
            "-o", output_dir,
            "-n", model_name,
            "-s", seeds,
            "-e", str(num_diffusion_samples),  # --sample
            "-p", str(num_diffusion_steps),    # --step
            "-c", str(num_pairformer_cycles),  # --cycle
            "-d", "bf16",                       # --dtype
        ]

        kernel = "cuequivariance" if _cuequivariance_available() else "torch"
        cmd.extend(["--trimul_kernel", kernel, "--triatt_kernel", kernel])

        if not use_msa:
            cmd.append("--use_msa=false")

        logger.debug(f"Running Protenix command: {' '.join(cmd)}")

        # Get subprocess environment with correct CUDA_VISIBLE_DEVICES
        from standalone_helpers import get_subprocess_device_env

        env = get_subprocess_device_env(device)

        # Run Protenix CLI with working directory set to the parent of output_dir
        # This ensures ./esm_embeddings/ is created in the temp dir, not the repo root
        working_dir = str(Path(output_dir).parent)

        subprocess.run(
            cmd,
            check=True,
            text=True,
            env=env,
            encoding="utf-8",
            stdout=sys.stdout if verbose else subprocess.DEVNULL,
            stderr=sys.stderr if verbose else subprocess.DEVNULL,
            cwd=working_dir,
        )

        logger.debug("Protenix prediction completed")

        # Check for protenix error directory (protenix exits 0 even on failure)
        err_dir = Path(output_dir) / "ERR"
        if err_dir.is_dir():
            err_files = list(err_dir.iterdir())
            if err_files:
                messages = []
                for ef in err_files:
                    messages.append(f"{ef.name}: {ef.read_text()[:500]}")
                raise RuntimeError(
                    "Protenix reported errors:\n" + "\n".join(messages)
                )

        # Read job names from input JSON to preserve ordering
        with open(input_json_path, "r") as f:
            input_data = json.load(f)

        job_names = [job["name"] for job in input_data]

        return [
            self._extract_job_output(output_dir, job_name, seeds)
            for job_name in job_names
        ]

    def _extract_job_output(
        self, output_dir: str, job_name: str, seeds: str
    ) -> Dict[str, Any]:
        """Extract structure and metrics for a single job from Protenix output.

        Protenix output structure:
            <output_dir>/<job_name>/<seed>/<job_name>_<seed>_sample_N.cif
            <output_dir>/<job_name>/<seed>/<job_name>_<seed>_summary_confidence_sample_N.json

        Selects the best sample across all seeds by ranking_score.

        Args:
            output_dir: Directory containing Protenix prediction outputs
            job_name: Name of the prediction job
            seeds: Comma-separated seed string

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        # Protenix saves outputs to output_dir/{job_name}/seed_{seed}/predictions/
        job_dir = Path(output_dir) / job_name

        # Parse seeds to find the first seed directory
        seed_list = [s.strip() for s in seeds.split(",")]
        first_seed = seed_list[0]

        # Construct the path to predictions directory
        predictions_dir = job_dir / f"seed_{first_seed}" / "predictions"

        logger.debug(f"Looking for predictions in: {predictions_dir}")

        if not predictions_dir.is_dir():
            raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

        best_cif = None
        best_metrics = None
        best_ranking_score = -1.0

        # Search for all CIF files matching the pattern {job_name}_sample_*.cif
        for sample_idx in range(100):  # search up to 100 samples
            cif_file = predictions_dir / f"{job_name}_sample_{sample_idx}.cif"
            confidence_file = predictions_dir / f"{job_name}_summary_confidence_sample_{sample_idx}.json"

            if not cif_file.exists():
                break

            if confidence_file.exists():
                with open(confidence_file, "r") as f:
                    metrics = json.load(f)

                ranking_score = float(metrics.get("ranking_score", 0.0))
                if ranking_score > best_ranking_score:
                    best_ranking_score = ranking_score
                    best_cif = cif_file
                    best_metrics = metrics
            elif best_cif is None:
                best_cif = cif_file

        if best_cif is None:
            raise FileNotFoundError(
                f"No structure output found for job '{job_name}' in {predictions_dir}"
            )

        return {
            "structure_cif_output": best_cif.read_text(),
            "metrics": best_metrics or {},
        }

    def load(self):
        """Find and validate the Protenix executable."""
        logger.debug("Initializing Protenix")

        # First try to find protenix in the current venv's bin directory
        venv_protenix = Path(sys.executable).parent / "protenix"
        self.protenix_executable = (
            str(venv_protenix)
            if venv_protenix.exists()
            else shutil.which("protenix")
        )
        if not self.protenix_executable:
            raise ImportError(
                "Could not find the 'protenix' executable. "
                "Please make sure Protenix is installed in the current environment."
            )

        self._loaded = True
        logger.debug(
            f"Protenix initialized. Using executable: {self.protenix_executable}"
        )


# ============================================================================
# Dispatch
# ============================================================================
_model: ProtenixModel | None = None


def dispatch(input_dict: dict) -> dict:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    # Resolve checkpoint directory via BPT_MODEL_CACHE
    from standalone_helpers import resolve_weights_dir

    bpt_dir = resolve_weights_dir("protenix")
    if bpt_dir:
        os.environ["PROTENIX_ROOT_DIR"] = bpt_dir
    elif not os.environ.get("PROTENIX_ROOT_DIR"):
        # Fallback: store checkpoints in the tool env directory
        env_path = os.environ.get("TOOL_VENV_PATH")
        if env_path:
            os.environ["PROTENIX_ROOT_DIR"] = env_path

    if _model is None:
        _model = ProtenixModel()

    operation = input_dict.get("operation", "predict")
    if operation == "predict":
        return _model(
            input_json_path=input_dict["input_json_path"],
            output_dir=input_dict["output_dir"],
            device=input_dict.get("device", "cuda"),
            model_name=input_dict.get("model_name", "protenix_base_default_v1.0.0"),
            seeds=input_dict.get("seeds", "0"),
            num_diffusion_samples=input_dict.get("num_diffusion_samples", 5),
            num_diffusion_steps=input_dict.get("num_diffusion_steps", 200),
            num_pairformer_cycles=input_dict.get("num_pairformer_cycles", 10),
            use_msa=input_dict.get("use_msa", True),
            verbose=input_dict.get("verbose", False),
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")



def to_device(device: str) -> dict:
    """Passthrough for CLI tool - Protenix naturally unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict:
    """CLI tool — no persistent GPU state to report."""
    return {"available": False, "framework": "cli", "reason": "CLI tool, no persistent GPU state"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
