"""Protenix inference implementation."""

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


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
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
            marker_data = marker_path.read_text().strip().split(",")
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
            logger.info("Corrupted checkpoint removed. It will be re-downloaded on next use.")
        else:
            stat = checkpoint_path.stat()
            marker_path.write_text(f"{stat.st_size},{stat.st_mtime}")
            logger.debug(f"Validation marker created: {marker_path.name}")


def cleanup_corrupted_checkpoints(checkpoint_dir: Path, model_name: str) -> None:  # noqa: ARG001 — required by tool interface
    """Check and remove corrupted checkpoint files in the checkpoint directory.

    Validates ALL .pt files in the directory (not just the named model),
    because tools like protenix also download dependency checkpoints
    (e.g., ESM2 embedding models) that can be corrupted by interrupted
    downloads.

    Uses `.validated` marker files to cache validation results and avoid
    re-validating unchanged checkpoints on every inference call.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_name: Name of the primary model (reserved by tool interface, currently unused)
    """
    if not checkpoint_dir.exists():
        return

    pt_files = list(checkpoint_dir.glob("*.pt"))
    if not pt_files:
        return

    for checkpoint_path in pt_files:
        _validate_single_checkpoint(checkpoint_path)


class ProtenixModel:
    """Protenix model for biomolecular structure prediction.

    Holds the upstream ``InferenceRunner`` across calls; rebuilds only when a
    weight-affecting config field changes (``model_name``, kernel, ``n_cycle``,
    ``n_step``, ``n_sample``, ``use_msa``). Per-call inputs (seeds, JSON paths,
    output dirs) are mutated on ``runner.configs`` directly.
    """

    def __init__(self) -> None:
        """Initialize Protenix model wrapper."""
        self._loaded = False
        self.runner: Any = None
        self.device: str | None = None
        self._cache_key: tuple[Any, ...] | None = None

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
    ) -> list[dict[str, Any]]:
        """Run Protenix structure prediction on one or more complexes.

        Args:
            input_json_path: Path to input JSON file (list of job dicts)
            output_dir: Directory to write output to
            device: Device string (e.g. ``"cuda:0"``); protenix's ``init_env`` binds to
                ``cuda:0`` of whatever the parent worker made visible via
                ``CUDA_VISIBLE_DEVICES``.
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
        protenix_root = os.environ.get("PROTENIX_ROOT_DIR")
        if not protenix_root:
            raise RuntimeError(
                "protenix: PROTENIX_ROOT_DIR not set — set PROTO_HOME or PROTO_MODEL_CACHE to configure storage"
            )
        checkpoint_dir = Path(protenix_root) / "checkpoint"
        cleanup_corrupted_checkpoints(checkpoint_dir, model_name)

        log = logger.info if verbose else logger.debug
        log(f"Protenix prediction: model={model_name}, input={input_json_path}, output={output_dir}, device={device}")

        kernel = "cuequivariance" if _cuequivariance_available() else "torch"
        new_key = (model_name, kernel, num_pairformer_cycles, num_diffusion_steps, num_diffusion_samples, use_msa)

        from configs.configs_inference import inference_configs  # type: ignore[import-not-found]
        from runner.batch_inference import preprocess_input  # type: ignore[import-not-found]
        from runner.inference import infer_predict  # type: ignore[import-not-found]

        os.makedirs(output_dir, exist_ok=True)

        if not self._loaded or self._cache_key != new_key:
            # init_basics() reads inference_configs["dump_dir"] at runner construction
            # and os.makedirs() it; subsequent calls re-point via runner.dump_dir below.
            inference_configs["dump_dir"] = output_dir
            self.load(
                model_name=model_name,
                num_pairformer_cycles=num_pairformer_cycles,
                num_diffusion_steps=num_diffusion_steps,
                num_diffusion_samples=num_diffusion_samples,
                use_msa=use_msa,
                kernel=kernel,
            )
        # Catches both cache-hit device changes and "cuda" vs "cuda:0" label mismatch post-load.
        if self.device != device:
            self.to_device(device)

        # Re-point the held runner at this call's output_dir; dumper.base_dir
        # was baked in at construction and won't follow configs mutations.
        self.runner.dump_dir = output_dir
        self.runner.error_dir = os.path.join(output_dir, "ERR")
        self.runner.dumper.base_dir = output_dir

        seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

        # protenix writes ./esm_embeddings/ relative to cwd.
        working_dir = str(Path(output_dir).parent)
        prev_cwd = os.getcwd()
        os.chdir(working_dir)
        try:
            processed_json = preprocess_input(
                input_json=input_json_path,
                out_dir=output_dir,
                use_msa=use_msa,
            )
            configs = self.runner.configs
            configs["input_json_path"] = processed_json
            configs["seeds"] = seeds_list
            configs["dump_dir"] = output_dir
            infer_predict(self.runner, configs)
        finally:
            os.chdir(prev_cwd)

        # Check for protenix error directory (protenix exits 0 even on failure)
        err_dir = Path(output_dir) / "ERR"
        if err_dir.is_dir():
            err_files = list(err_dir.iterdir())
            if err_files:
                messages = [f"{ef.name}: {ef.read_text()[:500]}" for ef in err_files]
                raise RuntimeError("protenix: errors reported:\n" + "\n".join(messages))

        # Read job names from input JSON to preserve ordering
        with open(input_json_path) as f:
            input_data = json.load(f)

        job_names = [job["name"] for job in input_data]

        return [self._extract_job_output(output_dir, job_name, seeds) for job_name in job_names]

    def _extract_job_output(self, output_dir: str, job_name: str, seeds: str) -> dict[str, Any]:
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
            raise FileNotFoundError(f"protenix: predictions directory not found: {predictions_dir}")

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
                with open(confidence_file) as f:
                    metrics = json.load(f)

                ranking_score = float(metrics.get("ranking_score", 0.0))
                if ranking_score > best_ranking_score:
                    best_ranking_score = ranking_score
                    best_cif = cif_file
                    best_metrics = metrics
            elif best_cif is None:
                best_cif = cif_file

        if best_cif is None:
            raise FileNotFoundError(f"protenix: no structure output found for job {job_name!r} in {predictions_dir}")

        return {
            "structure_cif_output": best_cif.read_text(),
            "metrics": best_metrics or {},
        }

    def load(
        self,
        *,
        model_name: str,
        num_pairformer_cycles: int,
        num_diffusion_steps: int,
        num_diffusion_samples: int,
        use_msa: bool,
        kernel: str,
    ) -> None:
        """Build and hold an ``InferenceRunner`` for the given config combo."""
        from runner.batch_inference import get_default_runner

        logger.debug(f"Loading Protenix runner: model={model_name}, kernel={kernel}")

        if self.runner is not None:
            self.unload()

        # InferenceRunner.__init__ runs init_env + init_model + load_checkpoint.
        self.runner = get_default_runner(
            seeds=[0],  # placeholder; real seeds are injected per-call into runner.configs
            n_cycle=num_pairformer_cycles,
            n_step=num_diffusion_steps,
            n_sample=num_diffusion_samples,
            dtype="bf16",
            model_name=model_name,
            use_msa=use_msa,
            trimul_kernel=kernel,
            triatt_kernel=kernel,
        )
        self.device = str(self.runner.device)
        self._cache_key = (
            model_name,
            kernel,
            num_pairformer_cycles,
            num_diffusion_steps,
            num_diffusion_samples,
            use_msa,
        )
        self._loaded = True

    def to_device(self, device: str) -> None:
        """Move the held runner's model between devices."""
        import torch
        from standalone_helpers import move_model_to_device

        if self.device == device:
            return

        self.runner.model = move_model_to_device(self.runner.model, self.device, device)
        self.runner.device = torch.device(device)  # match upstream init_env's torch.device type
        self.device = device

    def unload(self) -> None:
        """Move the held runner's model back to CPU. Does not reset ``_loaded``."""
        import torch
        from standalone_helpers import move_model_to_device

        if self.device == "cpu" or self.runner is None:
            return
        self.runner.model = move_model_to_device(self.runner.model, self.device, "cpu")
        self.runner.device = torch.device("cpu")
        self.device = "cpu"


# ============================================================================
# Dispatch
# ============================================================================
_model: ProtenixModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    # Resolve checkpoint directory via PROTO_MODEL_CACHE
    from standalone_helpers import resolve_weights_dir

    weights_dir = resolve_weights_dir("protenix")
    if weights_dir:
        os.environ["PROTENIX_ROOT_DIR"] = weights_dir
    elif not os.environ.get("PROTENIX_ROOT_DIR"):
        # Fallback: store checkpoints in the tool env directory
        env_path = os.environ.get("TOOL_VENV_PATH")
        if env_path:
            os.environ["PROTENIX_ROOT_DIR"] = env_path

    if _model is None:
        _model = ProtenixModel()

    operation = input_dict["operation"]
    if operation == "predict":
        results = _model(
            input_json_path=input_dict["input_json_path"],
            output_dir=input_dict["output_dir"],
            device=input_dict["device"],
            model_name=input_dict["model_name"],
            seeds=input_dict["seeds"],
            num_diffusion_samples=input_dict["num_diffusion_samples"],
            num_diffusion_steps=input_dict["num_diffusion_steps"],
            num_pairformer_cycles=input_dict["num_pairformer_cycles"],
            use_msa=input_dict["use_msa"],
            verbose=input_dict["verbose"],
        )
        return {"results": results}
    if operation == "introspect_loaded":
        return {
            "loaded": _model._loaded,
            "device": _model.device,
            "cache_key": list(_model._cache_key) if _model._cache_key is not None else None,
            "runner_id": id(_model.runner) if _model.runner is not None else None,
        }
    raise ValueError(f"protenix: unknown operation {operation!r}; valid: ['predict', 'introspect_loaded']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("protenix: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
