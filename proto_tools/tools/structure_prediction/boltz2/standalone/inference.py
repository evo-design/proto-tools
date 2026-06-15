"""Boltz2 inference implementation."""

import json
import os
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


_checkpoint_cache_installed = False
# Resident Boltz2 models keyed by checkpoint path; lives at module scope so to_device() can relocate them.
_checkpoint_cache: dict[str, Any] = {}


def _install_checkpoint_cache() -> None:
    """Memoize ``Boltz2.load_from_checkpoint`` so a resident worker loads each checkpoint once.

    ``boltz.main.predict`` reloads the checkpoint on every call; this memoization collapses those
    repeated loads to one per checkpoint. Config fields baked into the load (sampling/recycling/
    diffusion steps, step_scale, subsample_msa, and the affinity knobs) are marked
    ``reload_on_change=True``, so a config change restarts the worker and load args stay fixed for a
    worker's lifetime. The only distinct checkpoints are the structure (``boltz2_conf``) and affinity
    (``boltz2_aff``) files, so keying by checkpoint path bounds the cache to two entries.

    The path-only key does not self-correct: correctness relies on the ``reload_on_change`` set
    covering every field boltz threads into ``Boltz2.load_from_checkpoint``. An unmarked load-affecting
    field would let a same-path entry serve a model loaded with stale args, wrong output, no error.
    """
    global _checkpoint_cache_installed
    if _checkpoint_cache_installed:
        return
    from boltz.model.models.boltz2 import Boltz2

    original_load = Boltz2.load_from_checkpoint

    def cached_load_from_checkpoint(checkpoint: Any, **kwargs: Any) -> Any:
        key = str(checkpoint)
        model = _checkpoint_cache.get(key)
        if model is None:
            logger.debug("boltz2: checkpoint cache miss, loading %s", checkpoint)
            model = original_load(checkpoint, **kwargs)
            _checkpoint_cache[key] = model
        else:
            logger.debug("boltz2: reusing resident checkpoint %s", checkpoint)
        return model

    Boltz2.load_from_checkpoint = cached_load_from_checkpoint
    _checkpoint_cache_installed = True


class Boltz2Model:
    """Boltz2 model for multi-modal structure prediction."""

    def __init__(self) -> None:
        """Initialize Boltz2 model wrapper."""
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
            device: GPU device the worker runs on
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            diffusion_samples: Number of diffusion samples
            step_scale: Diffusion step size; lower = more sample diversity
            max_msa_seqs: Cap on MSA depth fed into the model
            subsample_msa: Stochastically subsample MSA each call
            num_workers: Number of workers for prediction
            seed: Random seed forwarded to boltz. None leaves the boltz default
                (unseeded) in place.
            verbose: Whether to print status messages
            include_pae_matrix: Attach the full per-residue PAE matrix.

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        self._run_boltz_predict(
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
        return self._extract_boltz_output(output_dir, input_yaml_path, include_pae_matrix)

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
        self._run_boltz_predict(
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
            affinity={
                "sampling_steps_affinity": sampling_steps_affinity,
                "diffusion_samples_affinity": diffusion_samples_affinity,
                "affinity_mw_correction": affinity_mw_correction,
            },
        )
        result = self._extract_boltz_output(output_dir, input_yaml_path, include_pae_matrix=False)
        result["affinity_metrics"] = self._extract_affinity_json(output_dir, input_yaml_path)
        return result

    def _run_boltz_predict(
        self,
        input_yaml_path: str,
        output_dir: str,
        device: str,  # noqa: ARG002 -- worker already runs on its assigned GPU
        recycling_steps: int,
        sampling_steps: int,
        diffusion_samples: int,
        step_scale: float,
        max_msa_seqs: int,
        subsample_msa: bool,
        num_workers: int,
        seed: int | None,
        verbose: bool,  # noqa: ARG002 -- boltz logs through its own logger
        affinity: dict[str, Any] | None = None,
    ) -> None:
        """Drive ``boltz.main.predict`` in-process against the memoized resident checkpoint.

        ``affinity`` (when set) enables the binding-affinity pass. The checkpoint loads once per
        worker and stays resident; ``to_device()`` relocates it for offload.
        """
        from boltz.main import predict as predict_cmd
        from standalone_helpers import is_cuda_oom, raise_oom

        logger.debug("\n=== Boltz2 Prediction ===")
        logger.debug(f"Input YAML path: {input_yaml_path}")
        logger.debug(f"Output directory: {output_dir}")
        with open(input_yaml_path) as f:
            logger.debug(f"\n--- Input YAML ---\n{f.read()}\n------------------\n")

        _install_checkpoint_cache()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        num_devices = 1 if torch.cuda.is_available() else 0
        kwargs: dict[str, Any] = {
            "data": input_yaml_path,
            "out_dir": output_dir,
            "cache": str(self.cache_dir),
            "devices": num_devices,
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "step_scale": step_scale,
            "max_msa_seqs": max_msa_seqs,
            "subsample_msa": subsample_msa,
            "num_workers": num_workers,
            "output_format": "mmcif",
            "seed": seed,
        }
        if affinity is not None:
            kwargs["sampling_steps_affinity"] = affinity["sampling_steps_affinity"]
            kwargs["diffusion_samples_affinity"] = affinity["diffusion_samples_affinity"]
            kwargs["affinity_mw_correction"] = affinity["affinity_mw_correction"]

        try:
            predict_cmd.callback(**kwargs)
        except RuntimeError as e:
            if is_cuda_oom(str(e)):
                raise_oom("boltz2", hint="Reduce the complex size or use a GPU with more memory.")
            raise
        logger.debug("Boltz prediction completed")

    def _extract_boltz_output(self, output_dir: str, input_path: str, include_pae_matrix: bool) -> dict[str, Any]:
        """Extract structure and metrics from Boltz prediction outputs.

        Args:
            output_dir: Directory containing Boltz prediction outputs
            input_path: Path to input YAML file
            include_pae_matrix: Attach the full per-residue PAE matrix.

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        import numpy as np

        input_name = Path(input_path).stem
        prediction_dir = Path(output_dir) / f"boltz_results_{input_name}" / "predictions" / input_name

        if not prediction_dir.is_dir():
            raise FileNotFoundError(
                f"boltz2: prediction directory not found: {prediction_dir} (boltz wrote no predictions)."
            )

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

    def _extract_affinity_json(self, output_dir: str, input_path: str) -> dict[str, float]:
        """Read ``affinity_<input>.json`` (log10 IC50 μM + binder probability, plus per-model variants)."""
        input_name = Path(input_path).stem
        prediction_dir = Path(output_dir) / f"boltz_results_{input_name}" / "predictions" / input_name
        affinity_file = prediction_dir / f"affinity_{input_name}.json"
        if not affinity_file.exists():
            raise FileNotFoundError(
                f"boltz2: affinity file not found: {affinity_file} (boltz wrote no affinity output)."
            )
        with open(affinity_file) as f:
            return {k: float(v) for k, v in json.load(f).items()}


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
    """Relocate resident in-process checkpoints to ``device``; frees the GPU on CPU offload.

    The resident model stays on the GPU between calls, so DeviceManager offload must move it
    (and empty the CUDA cache) to actually reclaim memory.
    """
    from standalone_helpers import move_model_to_device

    moved = 0
    for key, model in list(_checkpoint_cache.items()):
        old_device = str(next(model.parameters()).device)
        _checkpoint_cache[key] = move_model_to_device(model, old_device, device)
        moved += 1
    return {"success": True, "device": device, "note": f"moved {moved} resident model(s)"}


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
