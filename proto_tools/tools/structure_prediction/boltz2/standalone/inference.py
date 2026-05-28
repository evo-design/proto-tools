"""Boltz2 inference implementation."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from standalone_helpers import get_logger

logger = get_logger(__name__)


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
                raise RuntimeError("boltz2: cannot determine cache directory; set PROTO_HOME or PROTO_MODEL_CACHE")
        self.boltz_executable: str | None = None

    def __call__(
        self,
        input_yaml_path: str,
        output_dir: str,
        device: str = "cuda",
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        step_scale: float = 1.5,
        max_msa_seqs: int = 8192,
        subsample_msa: bool = False,
        num_workers: int = 4,
        seed: int | None = None,
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run Boltz2 structure prediction.

        Args:
            input_yaml_path: Path to input YAML file
            output_dir: Directory to write output to
            device: Device for subprocess environment
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            diffusion_samples: Number of diffusion samples
            step_scale: Diffusion step size; lower = more sample diversity
            max_msa_seqs: Cap on MSA depth fed into the model
            subsample_msa: Stochastically subsample MSA each call
            num_workers: Number of workers for prediction
            seed: Random seed forwarded to the boltz CLI as ``--seed``. None
                leaves the boltz default (unseeded) in place.
            verbose: Whether to print status messages
            include_pae_matrix: Attach the full per-residue PAE matrix.

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        captured = self._run_boltz_predict(
            input_yaml_path=input_yaml_path,
            output_dir=output_dir,
            device=device,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            step_scale=step_scale,
            max_msa_seqs=max_msa_seqs,
            subsample_msa=subsample_msa,
            num_workers=num_workers,
            seed=seed,
            verbose=verbose,
        )
        return self._extract_boltz_output(output_dir, input_yaml_path, include_pae_matrix, boltz_output=captured)

    def predict_affinity(
        self,
        input_yaml_path: str,
        output_dir: str,
        device: str = "cuda",
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        step_scale: float = 1.5,
        max_msa_seqs: int = 8192,
        subsample_msa: bool = False,
        num_workers: int = 4,
        seed: int | None = None,
        verbose: bool = False,
        sampling_steps_affinity: int = 200,
        diffusion_samples_affinity: int = 5,
        affinity_mw_correction: bool = False,
    ) -> dict[str, Any]:
        """Run Boltz2 structure + binding-affinity prediction.

        Structure args mirror ``__call__``; the affinity pass adds its own knobs.
        Returns ``__call__``'s output plus ``affinity_metrics`` parsed from
        ``affinity_<input>.json``.
        """
        extra_flags = [
            f"--sampling_steps_affinity={sampling_steps_affinity}",
            f"--diffusion_samples_affinity={diffusion_samples_affinity}",
        ]
        if affinity_mw_correction:
            extra_flags.append("--affinity_mw_correction")

        captured = self._run_boltz_predict(
            input_yaml_path=input_yaml_path,
            output_dir=output_dir,
            device=device,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            step_scale=step_scale,
            max_msa_seqs=max_msa_seqs,
            subsample_msa=subsample_msa,
            num_workers=num_workers,
            seed=seed,
            verbose=verbose,
            extra_args=extra_flags,
        )
        result = self._extract_boltz_output(
            output_dir, input_yaml_path, include_pae_matrix=False, boltz_output=captured
        )
        result["affinity_metrics"] = self._extract_affinity_json(output_dir, input_yaml_path, boltz_output=captured)
        return result

    def _run_boltz_predict(
        self,
        input_yaml_path: str,
        output_dir: str,
        device: str,
        recycling_steps: int,
        sampling_steps: int,
        diffusion_samples: int,
        step_scale: float,
        max_msa_seqs: int,
        subsample_msa: bool,
        num_workers: int,
        seed: int | None,
        verbose: bool,
        extra_args: list[str] | None = None,
    ) -> str | None:
        """Invoke the ``boltz predict`` CLI; ``extra_args`` appends affinity flags.

        Returns boltz's captured stdout/stderr (or None), surfaced by callers if
        no predictions are written.
        """
        if not self._loaded:
            self.load(verbose)

        logger.debug("\n=== Boltz2 Prediction ===")
        logger.debug(f"Input YAML path: {input_yaml_path}")
        logger.debug(f"Output directory: {output_dir}")
        with open(input_yaml_path) as f:
            yaml_content = f.read()
        logger.debug(f"\n--- Input YAML ---\n{yaml_content}\n------------------\n")
        sys.stdout.flush()

        num_devices = 1 if torch.cuda.is_available() else 0
        cmd = [
            self.boltz_executable,
            "predict",
            input_yaml_path,
            f"--out_dir={output_dir}",
            f"--recycling_steps={recycling_steps}",
            f"--diffusion_samples={diffusion_samples}",
            f"--sampling_steps={sampling_steps}",
            f"--step_scale={step_scale}",
            f"--max_msa_seqs={max_msa_seqs}",
            "--output_format=mmcif",
            f"--devices={num_devices}",
            f"--cache={self.cache_dir!s}",
            f"--num_workers={num_workers}",
        ]
        if subsample_msa:
            cmd.append("--subsample_msa")
        if seed is not None:
            cmd.append(f"--seed={seed}")
        if extra_args:
            cmd.extend(extra_args)

        logger.debug(f"Running Boltz command: {' '.join(cmd)}")  # type: ignore[arg-type]
        sys.stdout.flush()

        from standalone_helpers import get_subprocess_device_env

        env = get_subprocess_device_env(device)

        try:
            result = subprocess.run(
                cmd,  # type: ignore[arg-type]
                check=True,
                text=True,
                env=env,
                encoding="utf-8",
                stdout=sys.stdout if verbose else subprocess.PIPE,
                stderr=sys.stderr if verbose else subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = " | ".join((e.stderr or "").strip().splitlines()[-10:]) or "<no stderr>"
            raise RuntimeError(f"boltz2: failed (exit {e.returncode}): {stderr_tail}") from e

        logger.debug("Boltz prediction completed")
        sys.stdout.flush()
        # boltz can exit 0 yet skip the input; pass its output so the missing-predictions error explains why.
        return "\n".join(s for s in (result.stdout, result.stderr) if s) or None

    def _extract_boltz_output(
        self, output_dir: str, input_path: str, include_pae_matrix: bool, boltz_output: str | None = None
    ) -> dict[str, Any]:
        """Extract structure and metrics from Boltz prediction outputs.

        Args:
            output_dir: Directory containing Boltz prediction outputs
            input_path: Path to input YAML file
            include_pae_matrix: Attach the full per-residue PAE matrix.
            boltz_output: boltz's captured stdout/stderr, surfaced in the error if no predictions were written.

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        import numpy as np

        input_name = Path(input_path).stem
        prediction_dir = Path(output_dir) / f"boltz_results_{input_name}" / "predictions" / input_name

        if not prediction_dir.is_dir():
            hint = f" boltz exited 0 but wrote no predictions; output: {boltz_output}" if boltz_output else ""
            raise FileNotFoundError(f"boltz2: prediction directory not found: {prediction_dir}.{hint}")

        # Read confidence metrics
        confidence_file = prediction_dir / f"confidence_{input_name}_model_0.json"
        if not confidence_file.exists():
            raise FileNotFoundError(f"boltz2: confidence file not found: {confidence_file}")

        with open(confidence_file) as f:
            confidence_data = json.load(f)
        metrics = {key: _prepare_output_values(value) for key, value in confidence_data.items()}

        # Read structure
        cif_file = prediction_dir / f"{input_name}_model_0.cif"
        if not cif_file.exists():
            raise FileNotFoundError(f"boltz2: structure file not found: {cif_file}")

        # pae_*.npz: (N_token, N_token) float in [0, 32) Å, always written by upstream.
        pae_file = prediction_dir / f"pae_{input_name}_model_0.npz"
        if not pae_file.exists():
            raise FileNotFoundError(f"boltz2: PAE file not found: {pae_file}")
        with np.load(pae_file) as npz:
            pae_array = npz["pae"]
        metrics["avg_pae"] = float(pae_array.mean())
        metrics["pae"] = pae_array.astype(float).tolist() if include_pae_matrix else None

        return {
            "structure_cif_output": cif_file.read_text(),
            "metrics": metrics,
        }

    def _extract_affinity_json(
        self, output_dir: str, input_path: str, boltz_output: str | None = None
    ) -> dict[str, float]:
        """Read ``affinity_<input>.json`` (log10 IC50 μM + binder probability, plus per-model variants)."""
        input_name = Path(input_path).stem
        prediction_dir = Path(output_dir) / f"boltz_results_{input_name}" / "predictions" / input_name
        affinity_file = prediction_dir / f"affinity_{input_name}.json"
        if not affinity_file.exists():
            hint = f" boltz exited 0 but wrote no affinity output; output: {boltz_output}" if boltz_output else ""
            raise FileNotFoundError(f"boltz2: affinity file not found: {affinity_file}.{hint}")
        with open(affinity_file) as f:
            return {k: float(v) for k, v in json.load(f).items()}

    def load(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load Boltz2 model components."""
        logger.debug("Initializing Boltz2")

        # First try to find boltz in the current venv's bin directory
        venv_boltz = Path(sys.executable).parent / "boltz"
        exe = str(venv_boltz) if venv_boltz.exists() else shutil.which("boltz")
        if not exe:
            raise ImportError("boltz2: 'boltz' executable not found in current environment")
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

    operation = input_dict["operation"]
    if operation == "predict":
        return _model(
            input_yaml_path=input_dict["input_yaml_path"],
            output_dir=input_dict["output_dir"],
            device=input_dict["device"],
            recycling_steps=input_dict["recycling_steps"],
            sampling_steps=input_dict["sampling_steps"],
            diffusion_samples=input_dict["diffusion_samples"],
            step_scale=input_dict["step_scale"],
            max_msa_seqs=input_dict["max_msa_seqs"],
            subsample_msa=input_dict["subsample_msa"],
            num_workers=input_dict["num_workers"],
            seed=input_dict["seed"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
        )
    if operation == "predict_affinity":
        return _model.predict_affinity(
            input_yaml_path=input_dict["input_yaml_path"],
            output_dir=input_dict["output_dir"],
            device=input_dict["device"],
            recycling_steps=input_dict["recycling_steps"],
            sampling_steps=input_dict["sampling_steps"],
            diffusion_samples=input_dict["diffusion_samples"],
            step_scale=input_dict["step_scale"],
            max_msa_seqs=input_dict["max_msa_seqs"],
            subsample_msa=input_dict["subsample_msa"],
            num_workers=input_dict["num_workers"],
            seed=input_dict["seed"],
            verbose=input_dict["verbose"],
            sampling_steps_affinity=input_dict["sampling_steps_affinity"],
            diffusion_samples_affinity=input_dict["diffusion_samples_affinity"],
            affinity_mw_correction=input_dict["affinity_mw_correction"],
        )
    raise ValueError(f"boltz2: unknown operation {operation!r}; valid: ['predict', 'predict_affinity']")


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
        raise ValueError("boltz2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
