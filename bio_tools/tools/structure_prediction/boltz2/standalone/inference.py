"""
Boltz2 inference implementation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def _prepare_output_values(value: any) -> any:
    """Prepare Boltz output dictionaries by unpacking them into lists."""
    if isinstance(value, dict):
        list_value = []
        for k in range(len(value)):
            list_value.append(_prepare_output_values(value[str(k)]))
        return list_value
    else:
        return value


class Boltz2Model:
    """Boltz2 model for multi-modal structure prediction."""

    def __init__(self):
        """Initialize Boltz2 model wrapper."""
        self._loaded = False
        # Use HF_HOME if set (the cloud runtime), otherwise use home directory
        hf_home = os.environ.get("HF_HOME")
        self.cache_dir = (
            Path(hf_home) / "boltz"
            if hf_home
            else Path.home() / ".model_cache" / "checkpoints" / "boltz"
        )
        self.boltz_executable = None

    def __call__(
        self,
        input_yaml_path: str,
        output_dir: str,
        recycling_steps: int = 10,
        sampling_steps: int = 200,
        diffusion_samples: int = 25,
        num_workers: int = 4,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Boltz2 structure prediction.

        Args:
            input_yaml_path: Path to input YAML file
            output_dir: Directory to write output to
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            diffusion_samples: Number of diffusion samples
            num_workers: Number of workers for prediction
            verbose: Whether to print status messages

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        # Lazy load on first call
        if not self._loaded:
            self.load(verbose)

        logger.debug(f"\n=== Boltz2 Prediction ===")
        logger.debug(f"Input YAML path: {input_yaml_path}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Reading input YAML content...")
        with open(input_yaml_path, "r") as f:
            yaml_content = f.read()
        logger.debug(f"\n--- Input YAML ---\n{yaml_content}\n------------------\n")
        sys.stdout.flush()

        # Build the command
        # Determine the number of devices
        num_devices = 1 if torch.cuda.is_available() else 0

        # Base command
        cmd = [
            self.boltz_executable,
            "predict",
            input_yaml_path,
            f"--out_dir={output_dir}",
            f"--recycling_steps={recycling_steps}",
            f"--diffusion_samples={diffusion_samples}",
            f"--sampling_steps={sampling_steps}",
            "--output_format=mmcif",
            f"--devices={num_devices}",
            f"--cache={str(self.cache_dir)}",
            f"--num_workers={num_workers}",
        ]

        logger.debug(f"Running Boltz command: {' '.join(cmd)}")
        sys.stdout.flush()

        # Run the command with stdout/stderr visible
        subprocess.run(
            cmd,
            check=True,
            text=True,
            env=os.environ,
            encoding="utf-8",
            stdout=sys.stdout if verbose else subprocess.DEVNULL,
            stderr=sys.stderr if verbose else subprocess.DEVNULL,
        )

        logger.debug(f"Boltz prediction completed")
        sys.stdout.flush()

        # Extract the output
        return self._extract_boltz_output(output_dir, input_yaml_path)

    def _extract_boltz_output(self, output_dir: str, input_path: str) -> Dict[str, Any]:
        """Extract structure and metrics from Boltz prediction outputs.

        Args:
            output_dir: Directory containing Boltz prediction outputs
            input_path: Path to input YAML file

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        input_name = Path(input_path).stem
        prediction_dir = (
            Path(output_dir)
            / f"boltz_results_{input_name}"
            / "predictions"
            / input_name
        )

        if not prediction_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

        # Read confidence metrics
        confidence_file = prediction_dir / f"confidence_{input_name}_model_0.json"
        if not confidence_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {confidence_file}")

        with open(confidence_file, "r") as f:
            confidence_data = json.load(f)
        metrics = {
            key: _prepare_output_values(value) for key, value in confidence_data.items()
        }

        # Read structure
        cif_file = prediction_dir / f"{input_name}_model_0.cif"
        if not cif_file.exists():
            raise FileNotFoundError(f"Structure file not found: {cif_file}")

        return {
            "structure_cif_output": cif_file.read_text(),
            "metrics": metrics,
        }

    def load(self, verbose: bool = False):
        """Load Boltz2 model components."""
        logger.debug("Initializing Boltz2")

        # First try to find boltz in the current venv's bin directory
        venv_boltz = Path(sys.executable).parent / "boltz"
        self.boltz_executable = (
            str(venv_boltz) if venv_boltz.exists() else shutil.which("boltz")
        )
        if not self.boltz_executable:
            raise ImportError(
                "Could not find the 'boltz' executable. Please make sure Boltz2 is installed in the current environment."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True

        logger.debug(
            f"Boltz2 initialized successfully. Using executable: {self.boltz_executable}"
        )


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python inference.py <input_json_path> <output_json_path>"
        )

    # Get the input and output json paths
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Create model and run inference
    model = Boltz2Model()

    # Build kwargs for model call
    model_kwargs = {
        "input_yaml_path": input_data["input_yaml_path"],
        "output_dir": input_data["output_dir"],
        "recycling_steps": input_data["recycling_steps"],
        "sampling_steps": input_data["sampling_steps"],
        "diffusion_samples": input_data["diffusion_samples"],
        "num_workers": input_data["num_workers"],
        "verbose": True,
    }

    output_data = model(**model_kwargs)

    # Write the output to a json file
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
