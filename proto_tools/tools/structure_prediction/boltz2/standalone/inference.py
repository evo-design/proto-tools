"""Boltz2 inference implementation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _prepare_output_values(value: Any) -> Any:
    """Prepare Boltz output dictionaries by unpacking them into lists."""
    if isinstance(value, dict):
        return [_prepare_output_values(value[str(k)]) for k in range(len(value))]
    return value


class Boltz2Model:
    """Boltz2 model for multi-modal structure prediction."""

    def __init__(self) -> None:
        """Initialize Boltz2 model wrapper."""
        self._loaded = False
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("boltz2")
        if weights_dir:
            self.cache_dir = Path(weights_dir)
        else:
            # NONE mode fallback: HF_HOME for passthrough
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                self.cache_dir = Path(hf_home) / "boltz"
            else:
                raise RuntimeError(
                    "Cannot determine Boltz2 cache directory. Set PROTO_HOME or PROTO_MODEL_CACHE to configure storage."
                )
        self.boltz_executable: str | None = None

    def __call__(
        self,
        input_yaml_path: str,
        output_dir: str,
        device: str = "cuda",
        recycling_steps: int = 10,
        sampling_steps: int = 200,
        diffusion_samples: int = 25,
        num_workers: int = 4,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run Boltz2 structure prediction.

        Args:
            input_yaml_path: Path to input YAML file
            output_dir: Directory to write output to
            device: Device for subprocess environment
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

        logger.debug("\n=== Boltz2 Prediction ===")
        logger.debug(f"Input YAML path: {input_yaml_path}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug("Reading input YAML content...")
        with open(input_yaml_path) as f:
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
            f"--cache={self.cache_dir!s}",
            f"--num_workers={num_workers}",
        ]

        logger.debug(f"Running Boltz command: {' '.join(cmd)}")  # type: ignore[arg-type]
        sys.stdout.flush()

        # Get subprocess environment with correct CUDA_VISIBLE_DEVICES
        from standalone_helpers import get_subprocess_device_env

        env = get_subprocess_device_env(device)

        # Run the command with stdout/stderr visible
        subprocess.run(
            cmd,  # type: ignore[arg-type]
            check=True,
            text=True,
            env=env,
            encoding="utf-8",
            stdout=sys.stdout if verbose else subprocess.DEVNULL,
            stderr=sys.stderr if verbose else subprocess.DEVNULL,
        )

        logger.debug("Boltz prediction completed")
        sys.stdout.flush()

        # Extract the output
        return self._extract_boltz_output(output_dir, input_yaml_path)

    def _extract_boltz_output(self, output_dir: str, input_path: str) -> dict[str, Any]:
        """Extract structure and metrics from Boltz prediction outputs.

        Args:
            output_dir: Directory containing Boltz prediction outputs
            input_path: Path to input YAML file

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        input_name = Path(input_path).stem
        prediction_dir = Path(output_dir) / f"boltz_results_{input_name}" / "predictions" / input_name

        if not prediction_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

        # Read confidence metrics
        confidence_file = prediction_dir / f"confidence_{input_name}_model_0.json"
        if not confidence_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {confidence_file}")

        with open(confidence_file) as f:
            confidence_data = json.load(f)
        metrics = {key: _prepare_output_values(value) for key, value in confidence_data.items()}

        # Read structure
        cif_file = prediction_dir / f"{input_name}_model_0.cif"
        if not cif_file.exists():
            raise FileNotFoundError(f"Structure file not found: {cif_file}")

        return {
            "structure_cif_output": cif_file.read_text(),
            "metrics": metrics,
        }

    def load(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load Boltz2 model components."""
        logger.debug("Initializing Boltz2")

        # First try to find boltz in the current venv's bin directory
        venv_boltz = Path(sys.executable).parent / "boltz"
        exe = str(venv_boltz) if venv_boltz.exists() else shutil.which("boltz")
        if not exe:
            raise ImportError(
                "Could not find the 'boltz' executable. Please make sure Boltz2 is installed in the current environment."
            )
        self.boltz_executable = exe
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True

        logger.debug(f"Boltz2 initialized successfully. Using executable: {self.boltz_executable}")


# ============================================================================
# Dispatch
# ============================================================================
_model: Boltz2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = Boltz2Model()

    operation = input_dict.get("operation", "predict")
    if operation == "predict":
        return _model(
            input_yaml_path=input_dict["input_yaml_path"],
            output_dir=input_dict["output_dir"],
            device=input_dict.get("device", "cuda"),
            recycling_steps=input_dict.get("recycling_steps", 10),
            sampling_steps=input_dict.get("sampling_steps", 200),
            diffusion_samples=input_dict.get("diffusion_samples", 25),
            num_workers=input_dict.get("num_workers", 4),
            verbose=input_dict.get("verbose", False),
        )
    raise ValueError(f"Unknown operation: {operation}")


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool - Boltz2 naturally unloads after each call."""
    # Boltz2 is a CLI tool that spawns subprocesses and naturally unloads
    # after each call, so explicit device management is not needed.
    # This is a passthrough for standardization with other tools.
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
