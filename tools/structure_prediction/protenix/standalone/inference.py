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


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """Validate that a PyTorch checkpoint file is not corrupted.

    PyTorch checkpoint files are ZIP archives. This performs a lightweight
    validation by checking the ZIP file structure without loading the full model.

    Args:
        checkpoint_path: Path to the checkpoint file to validate

    Returns:
        True if the checkpoint is valid, False if corrupted
    """
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as z:
            # Test the ZIP file integrity - returns name of first bad file or None
            bad_file = z.testzip()
            if bad_file:
                logger.warning(f"Corrupted file in checkpoint: {bad_file}")
                return False
        return True
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning(f"Checkpoint validation failed for {checkpoint_path}: {e}")
        return False


def cleanup_corrupted_checkpoints(checkpoint_dir: Path, model_name: str) -> None:
    """Check and remove corrupted checkpoint files for a specific model.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_name: Name of the model to check (e.g., 'protenix_mini_ism_v0.5.0')
    """
    if not checkpoint_dir.exists():
        return

    checkpoint_path = checkpoint_dir / f"{model_name}.pt"

    if checkpoint_path.exists():
        logger.info(f"Validating checkpoint: {checkpoint_path}")
        if not validate_checkpoint(checkpoint_path):
            logger.warning(f"Removing corrupted checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()
            logger.info(f"Corrupted checkpoint removed. It will be re-downloaded on next use.")


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

        if not use_msa:
            cmd.append("--use_msa=false")

        logger.debug(f"Running Protenix command: {' '.join(cmd)}")

        # Run Protenix CLI with working directory set to the parent of output_dir
        # This ensures ./esm_embeddings/ is created in the temp dir, not the repo root
        working_dir = str(Path(output_dir).parent)

        subprocess.run(
            cmd,
            check=True,
            text=True,
            env=os.environ,
            encoding="utf-8",
            stdout=sys.stdout if verbose else subprocess.DEVNULL,
            stderr=sys.stderr if verbose else subprocess.DEVNULL,
            cwd=working_dir,
        )

        logger.debug("Protenix prediction completed")

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


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python inference.py <input_json_path> <output_json_path>"
        )

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    model = ProtenixModel()

    model_kwargs = {
        "input_json_path": input_data["input_json_path"],
        "output_dir": input_data["output_dir"],
        "model_name": input_data["model_name"],
        "seeds": input_data["seeds"],
        "num_diffusion_samples": input_data["num_diffusion_samples"],
        "num_diffusion_steps": input_data["num_diffusion_steps"],
        "num_pairformer_cycles": input_data["num_pairformer_cycles"],
        "use_msa": input_data["use_msa"],
        "verbose": True,
    }

    results = model(**model_kwargs)

    with open(output_json_path, "w") as f:
        json.dump(results, f)
