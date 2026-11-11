"""Protenix inference implementation."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from standalone_helpers import count_a3m_rows, get_logger, report_msa_rows_loaded, retry_on_checkpoint_corruption

logger = get_logger(__name__)

# Retry ProtenixModel.load() when ByteDance TOS drops the checkpoint download.
LOAD_MAX_ATTEMPTS = 3
LOAD_BACKOFF_BASE_S = 2.0


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
    except MemoryError:
        # Resource constraint, not corruption — let the caller decide; we MUST NOT
        # let _validate_single_checkpoint delete a valid file because we OOM'd here.
        raise
    except Exception as e:  # any other torch.load failure means "not a valid checkpoint"
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


def _msa_rows_sent_by_job(
    input_json_path: str, chain_ids: dict[str, list[str]] | None = None
) -> dict[str, dict[str, int]]:
    """MSA rows written per protein chain (keyed by position), per job; warn for chains without a path.

    ``chain_ids`` maps a job name to its chain IDs in sequence order, naming chains in the warning.

    A protein chain without an MSA path runs without an MSA: proto-tools disables Protenix's own
    server search, which would otherwise replace every chain's supplied MSA.
    """
    jobs = json.loads(Path(input_json_path).read_text())
    base = Path(input_json_path).parent
    sent_by_job: dict[str, dict[str, int]] = {}
    for job in jobs if isinstance(jobs, list) else [jobs]:
        sent: dict[str, int] = {}
        for position, entry in enumerate(job["sequences"]):
            chain = entry.get("proteinChain")
            if chain is None:
                continue
            paths = [base / chain[key] for key in ("unpairedMsaPath", "pairedMsaPath") if chain.get(key)]
            sent[str(position)] = max((count_a3m_rows(path.read_text()) for path in paths if path.exists()), default=0)
        labels = (chain_ids or {}).get(job["name"], [])
        missing = [labels[int(p)] if int(p) < len(labels) else p for p, rows in sent.items() if rows == 0]
        if missing:
            logger.warning("protenix: protein chain(s) %s have no MSA and will run query-only", ", ".join(missing))
        sent_by_job[job["name"]] = sent
    return sent_by_job


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
        # Job name -> {protein chain position: MSA rows proto-tools wrote}, for the loaded-rows log.
        self._msa_rows_sent: dict[str, dict[str, int]] = {}
        self._msa_chain_ids: dict[str, list[str]] = {}
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
        chain_ids: dict[str, list[str]] | None = None,
        verbose: bool = False,
        include_pae_matrix: bool = False,
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
            chain_ids: Job name to its chain IDs in sequence order, labelling the MSA-rows log
            verbose: Whether to print status messages
            include_pae_matrix: Attach the full per-token PAE matrix.

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
        if model_name == "protenix-v2":
            checkpoint_path = checkpoint_dir / f"{model_name}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"protenix-v2 weights not found at {checkpoint_path}. "
                    "v2 weights are gated by ByteDance and not auto-downloaded."
                )
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
            # Retry on truncated-checkpoint errors from a dropped TOS download:
            # cleanup deletes the corrupt .pt; upstream re-downloads on retry.
            retry_on_checkpoint_corruption(
                lambda: self.load(
                    model_name=model_name,
                    num_pairformer_cycles=num_pairformer_cycles,
                    num_diffusion_steps=num_diffusion_steps,
                    num_diffusion_samples=num_diffusion_samples,
                    use_msa=use_msa,
                    kernel=kernel,
                ),
                tool_name="protenix",
                max_attempts=LOAD_MAX_ATTEMPTS,
                recover=lambda: cleanup_corrupted_checkpoints(checkpoint_dir, model_name),
                backoff_s=LOAD_BACKOFF_BASE_S,
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
            self._msa_rows_sent = _msa_rows_sent_by_job(input_json_path, chain_ids) if use_msa else {}
            self._msa_chain_ids = chain_ids or {}
            # proto-tools writes every chain's MSA path; use_msa=False here only stops Protenix
            # from searching its own server (which would overwrite them). The runner, built with
            # use_msa above, still featurizes the supplied MSAs.
            processed_json = preprocess_input(
                input_json=input_json_path,
                out_dir=output_dir,
                use_msa=False,
            )
            configs = self.runner.configs
            configs["input_json_path"] = processed_json
            configs["seeds"] = seeds_list
            configs["dump_dir"] = output_dir
            from standalone_helpers import oom_guard

            with oom_guard("protenix", hint="Lower num_diffusion_samples or use a larger GPU."):
                infer_predict(self.runner, configs)
        finally:
            os.chdir(prev_cwd)

        # Check for protenix error directory (protenix exits 0 even on failure)
        err_dir = Path(output_dir) / "ERR"
        if err_dir.is_dir():
            err_files = list(err_dir.iterdir())
            if err_files:
                messages = [f"{ef.name}: {ef.read_text()}" for ef in err_files]
                raise RuntimeError("protenix: errors reported:\n" + "\n".join(messages))

        # Read job names from input JSON to preserve ordering
        with open(input_json_path) as f:
            input_data = json.load(f)

        job_names = [job["name"] for job in input_data]

        return [self._extract_job_output(output_dir, job_name, seeds, include_pae_matrix) for job_name in job_names]

    def _extract_job_output(
        self, output_dir: str, job_name: str, seeds: str, include_pae_matrix: bool
    ) -> dict[str, Any]:
        """Extract structure and metrics for a single job from Protenix output.

        Selects the best sample by ranking_score, then attaches PAE pulled from
        ``<job>_full_data_sample_N.json`` (always written; ``load()`` sets
        ``need_atom_confidence=True``).

        Args:
            output_dir: Directory containing Protenix prediction outputs
            job_name: Name of the prediction job
            seeds: Comma-separated seed string
            include_pae_matrix: Attach the full per-token PAE matrix.

        Returns:
            Dictionary containing structure_cif_output and metrics
        """
        import numpy as np

        # Protenix saves outputs to output_dir/{job_name}/seed_{seed}/predictions/
        job_dir = Path(output_dir) / job_name

        # Pick the best sample by ranking score across ALL seeds (not just seed 0).
        # Samples without a confidence file score -1.0 so they are only a fallback.
        samples: list[tuple[float, Path, int, Path, dict[str, Any] | None]] = []
        for seed in (s.strip() for s in seeds.split(",")):
            predictions_dir = job_dir / f"seed_{seed}" / "predictions"
            for cif_file in sorted(predictions_dir.glob(f"{job_name}_sample_*.cif")):
                sample_idx = int(cif_file.stem.rsplit("_sample_", 1)[1])
                conf_file = predictions_dir / f"{job_name}_summary_confidence_sample_{sample_idx}.json"
                metrics = json.loads(conf_file.read_text()) if conf_file.exists() else None
                score = float(metrics.get("ranking_score", 0.0)) if metrics is not None else -1.0
                samples.append((score, predictions_dir, sample_idx, cif_file, metrics))

        if not samples:
            raise FileNotFoundError(f"protenix: no structure output found for job {job_name!r} under {job_dir}")
        _, predictions_dir, best_rank, best_cif, best_metrics = max(samples, key=lambda s: s[0])

        # token_pair_pae: already serialized as list[list[float]] by upstream save_json.
        full_data_file = predictions_dir / f"{job_name}_full_data_sample_{best_rank}.json"
        if not full_data_file.exists():
            raise FileNotFoundError(f"protenix: full_data file not found: {full_data_file}")
        with open(full_data_file) as f:
            full_data = json.load(f)
        pae = full_data["token_pair_pae"]
        pae_array = np.asarray(pae, dtype=float)

        metrics_out = best_metrics or {}
        metrics_out["avg_pae"] = float(pae_array.mean())
        metrics_out["pae"] = pae if include_pae_matrix else None

        return {
            "structure_cif_output": best_cif.read_text(),
            "metrics": metrics_out,
        }

    def _log_loaded_msa_rows(self, runner: Any) -> None:
        """Wrap ``runner.predict`` to log each sample's featurized MSA rows per protein chain once."""
        from protenix.data.msa.msa_utils import MSA_GAP_IDX  # type: ignore[import-not-found]

        predict = runner.predict

        def predict_logging_msa_rows(data: Any) -> Any:
            # Popped so each call logs a job once; job names repeat across calls to a cached worker.
            sent = self._msa_rows_sent.pop(data["sample_name"], None)
            if sent is not None:
                features = data["input_feature_dict"]
                msa, asym = features["msa"], features["asym_id"]
                loaded = {chain: int((msa[:, asym == int(chain)] != MSA_GAP_IDX).any(dim=1).sum()) for chain in sent}
                labels = self._msa_chain_ids.get(data["sample_name"], [])
                label = {position: labels[int(position)] for position in sent if int(position) < len(labels)}
                report_msa_rows_loaded(
                    "protenix",
                    {label.get(k, k): v for k, v in loaded.items()},
                    {label.get(k, k): v for k, v in sent.items()},
                )
            return predict(data)

        runner.predict = predict_logging_msa_rows

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
        # need_atom_confidence=True makes the dumper write the full_data JSON we read for PAE.
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
            need_atom_confidence=True,
        )
        self._log_loaded_msa_rows(self.runner)
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

    # Protenix downloads checkpoints (incl. the ESM2-3B used by mini_esm) and CCD data under PROTENIX_ROOT_DIR
    from standalone_helpers import export_weights_dir

    export_weights_dir("PROTENIX_ROOT_DIR", "protenix")

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
            chain_ids=input_dict.get("chain_ids"),
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
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
