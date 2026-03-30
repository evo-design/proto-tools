"""
AlphaGenome standalone inference implementation for venv execution.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import huggingface_hub
import jax
from alphagenome.data import genome
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import variant_scorers as variant_scorers_lib
from alphagenome.protos import dna_model_pb2
from alphagenome_research.model import dna_model

logger = logging.getLogger(__name__)


def _safe_tidy_scores(scores) -> list:
    """Wrap variant_scorers_lib.tidy_scores to handle upstream empty-result bug."""
    try:
        df = variant_scorers_lib.tidy_scores(scores)
    except KeyError:
        logger.warning(
            "variant_scorers_lib.tidy_scores raised KeyError (likely empty "
            "gene-based results); returning empty score list."
        )
        return []
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))

_SUPPORTED_CONTEXT_LENGTHS = [1_048_576, 524_288, 131_072, 16_384]


def _ensure_jax_memory_compat() -> None:
    """Patch a minimal jax.memory shim for alphagenome_research on older JAX."""
    if hasattr(jax, "memory"):
        return

    class _MemorySpace:
        # jax.device_put(x, device=None) is valid and keeps behavior compatible.
        Host = None

    class _MemoryCompat:
        Space = _MemorySpace

    setattr(jax, "memory", _MemoryCompat())
    logger.info("Applied JAX memory compatibility shim (jax.memory missing).")


_ensure_jax_memory_compat()

# Minimum center-mask widths required by the recommended scorers.
# Interval scorers: RNA_SEQ GeneMaskScorer has width=200,001.
# Variant scorers: CHIP_HISTONE CenterMaskScorer has width=2,001 (the largest).
_MIN_INTERVAL_SCORER_WIDTH = 200_001
_MIN_VARIANT_SCORER_WIDTH = 2_001

_ORGANISM_PROTOS = {
    "human": dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS,
    "mouse": dna_model_pb2.Organism.ORGANISM_MUS_MUSCULUS,
}

_ORGANISM_ENUMS = {
    "human": dna_model.Organism.HOMO_SAPIENS,
    "mouse": dna_model.Organism.MUS_MUSCULUS,
}




def _resolve_checkpoint_path(model_version: str) -> Optional[Path]:
    """Resolve AlphaGenome checkpoint path for offline-friendly loading."""
    env_checkpoint = os.environ.get("ALPHAGENOME_CHECKPOINT_PATH", "").strip()
    if env_checkpoint:
        checkpoint_path = Path(env_checkpoint).expanduser()
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(
            f"ALPHAGENOME_CHECKPOINT_PATH does not exist: {checkpoint_path}"
        )

    repo_id = f"google/alphagenome-{model_version.replace('_', '-').lower()}"
    try:
        checkpoint = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
        )
    except Exception:
        return None
    return Path(checkpoint)


class AlphaGenomeModel:
    """AlphaGenome model for genomic sequence predictions and variant scoring.

    Supports lazy loading and device management (load/to_device/unload).
    """

    def __init__(self, model_version: str = "all_folds"):
        """Initialize AlphaGenome model wrapper.

        Args:
            model_version: Model version/fold to use (e.g., "all_folds")
        """
        self._loaded = False
        self.model_version = model_version
        self.device = None
        self.model = None

    def _ensure_loaded(self, device: str) -> None:
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

    def predict_intervals(
        self,
        intervals: List[dict[str, Any]],
        requested_outputs: List[str],
        ontology_terms: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run interval predictions through DnaClient.predict_intervals."""
        parsed = [_resize_interval(i["chromosome"], i["interval_start"], i["interval_end"]) for i in intervals]
        self._ensure_loaded(device)

        predictions = self.model.predict_intervals(
            intervals=parsed,
            requested_outputs=[dna_model.OutputType[n] for n in requested_outputs],
            ontology_terms=ontology_terms,
            organism=_ORGANISM_ENUMS[organism],
            progress_bar=False,
        )
        return {"predictions": [_serialize_data(p) for p in predictions]}

    def predict_variants(
        self,
        intervals: List[dict[str, Any]],
        variants: List[dict[str, Any]],
        requested_outputs: List[str],
        ontology_terms: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run variant-effect predictions through DnaClient.predict_variants."""
        parsed_intervals = [_resize_interval(i["chromosome"], i["interval_start"], i["interval_end"]) for i in intervals]
        parsed_variants = [
            genome.Variant(
                chromosome=v["chromosome"],
                position=v["variant_position"],
                reference_bases=v["reference_bases"],
                alternate_bases=v["alternate_bases"],
            )
            for v in variants
        ]
        self._ensure_loaded(device)

        predictions = self.model.predict_variants(
            intervals=parsed_intervals,
            variants=parsed_variants,
            requested_outputs=[dna_model.OutputType[n] for n in requested_outputs],
            ontology_terms=ontology_terms,
            organism=_ORGANISM_ENUMS[organism],
            progress_bar=False,
        )
        return {"predictions": [_serialize_data(p) for p in predictions]}

    def predict_sequences(
        self,
        sequences: List[str],
        requested_outputs: List[str],
        ontology_terms: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run predictions from raw DNA sequence strings."""
        for seq in sequences:
            _validate_sequence_length(len(seq), operation="predict_sequences")
        self._ensure_loaded(device)

        predictions = self.model.predict_sequences(
            sequences=sequences,
            requested_outputs=[dna_model.OutputType[n] for n in requested_outputs],
            ontology_terms=ontology_terms,
            organism=_ORGANISM_ENUMS[organism],
            progress_bar=False,
        )
        return {"predictions": [_serialize_data(p) for p in predictions]}

    def score_variants(
        self,
        intervals: List[dict[str, Any]],
        variants: List[dict[str, Any]],
        variant_scorers: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run variant scoring through DnaClient.score_variants."""
        parsed_intervals = []
        parsed_variants = []
        for i, v in zip(intervals, variants, strict=True):
            width = i["interval_end"] - i["interval_start"]
            _validate_min_scorer_width(width, _MIN_VARIANT_SCORER_WIDTH, "score_variants")
            parsed_intervals.append(_resize_interval(i["chromosome"], i["interval_start"], i["interval_end"]))
            parsed_variants.append(genome.Variant(
                chromosome=v["chromosome"],
                position=v["variant_position"],
                reference_bases=v["reference_bases"],
                alternate_bases=v["alternate_bases"],
            ))
        self._ensure_loaded(device)

        scores_per_variant = self.model.score_variants(
            intervals=parsed_intervals,
            variants=parsed_variants,
            variant_scorers=_resolve_variant_scorers(variant_scorers, organism),
            organism=_ORGANISM_ENUMS[organism],
            progress_bar=False,
        )
        return {"scores": [_safe_tidy_scores(s) for s in scores_per_variant]}

    def score_intervals(
        self,
        intervals: List[dict[str, Any]],
        interval_scorers: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run interval scoring through DnaClient.score_intervals."""
        parsed = []
        for i in intervals:
            width = i["interval_end"] - i["interval_start"]
            _validate_min_scorer_width(width, _MIN_INTERVAL_SCORER_WIDTH, "score_intervals")
            parsed.append(_resize_interval(i["chromosome"], i["interval_start"], i["interval_end"]))
        self._ensure_loaded(device)

        scores_per_interval = self.model.score_intervals(
            intervals=parsed,
            interval_scorers=_resolve_interval_scorers(interval_scorers),
            organism=_ORGANISM_ENUMS[organism],
            progress_bar=False,
        )
        return {"scores": [_safe_tidy_scores(s) for s in scores_per_interval]}

    def score_ism_variants_batch(
        self,
        requests: List[dict[str, Any]],
        variant_scorers: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run batched ISM scoring (sequential — upstream has no batched ISM)."""
        self._ensure_loaded(device)
        scorers = _resolve_variant_scorers(variant_scorers, organism)

        all_scores = []
        for req in requests:
            width = req["interval_end"] - req["interval_start"]
            _validate_min_scorer_width(width, _MIN_VARIANT_SCORER_WIDTH, "score_ism_variants_batch")

            interval = _resize_interval(req["chromosome"], req["interval_start"], req["interval_end"])
            ism_interval = genome.Interval(
                chromosome=interval.chromosome,
                start=req["ism_interval_start"],
                end=req["ism_interval_end"],
            )

            interval_variant = None
            if req.get("variant_position") is not None:
                interval_variant = genome.Variant(
                    chromosome=req["chromosome"],
                    position=req["variant_position"],
                    reference_bases=req["reference_bases"],
                    alternate_bases=req["alternate_bases"],
                )

            scores = self.model.score_ism_variants(
                interval=interval,
                ism_interval=ism_interval,
                variant_scorers=scorers,
                organism=_ORGANISM_ENUMS[organism],
                interval_variant=interval_variant,
            )
            all_scores.append(_safe_tidy_scores(scores))

        return {"scores": all_scores}

    # ============================================================================
    # Device Management
    # ============================================================================
    def load(self, device: str = "cuda", verbose: bool = False):
        """Load AlphaGenome model to device."""
        if verbose:
            logger.info(f"Loading AlphaGenome model: {self.model_version} on {device}")

        from standalone_helpers import resolve_jax_device

        jax_device = resolve_jax_device(device)
        checkpoint_path = _resolve_checkpoint_path(self.model_version)
        if checkpoint_path is not None:
            if verbose:
                logger.info("Loading AlphaGenome checkpoint from %s", checkpoint_path)
            self.model = dna_model.create(checkpoint_path, device=jax_device)
        else:
            if verbose:
                logger.info(
                    "No local checkpoint found for model '%s'; falling back to Hugging Face download.",
                    self.model_version,
                )
            self.model = dna_model.create_from_huggingface(self.model_version, device=jax_device)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("AlphaGenome model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        AlphaGenome is GPU-only — CPU loading is not supported because the
        model requires XLA compilation (10+ minutes) that would be wasted
        on a device where inference is impractical. When asked to move to
        CPU (e.g., LRU eviction), we unload the model to free GPU memory
        but preserve the XLA compilation cache so reloading on GPU is fast.

        When called on an unloaded model (after CPU eviction), reloads on
        the target GPU device.
        """
        if device == "cpu":
            logger.warning(
                "AlphaGenome is GPU-only; unloading model instead of moving "
                "to CPU. Model will reload on next GPU dispatch."
            )
            self.unload(clear_jax_caches=False)
            return

        if not self._loaded:
            self.load(device)
            return

        if self.device != device:
            self.unload()
            self.load(device)

    def unload(self, verbose: bool = False, clear_jax_caches: bool = True) -> None:
        """Unload model and free memory.

        Parameters
        ----------
        verbose : bool
            Log the unload operation.
        clear_jax_caches : bool
            If True (default), also clear JAX's XLA compilation caches.
            Set to False when unloading for GPU eviction to preserve
            compiled kernels — reloading on the same GPU can then skip
            the expensive recompilation step.
        """
        if not self._loaded:
            return

        if verbose:
            logger.info(f"Unloading {self.__class__.__name__} from {self.device}")

        self.model = None
        self.device = None
        self._loaded = False

        import gc
        if clear_jax_caches:
            jax.clear_caches()
        gc.collect()


# ============================================================================
# Helpers
# ============================================================================


def _serialize_data(value: Any) -> Any:
    """Recursively convert nested prediction objects into JSON-safe structures.

    Used for predict_* operations whose outputs contain numpy arrays,
    dataclasses, TrackData, JunctionData, and other complex types.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if type(value).__name__ in ("DataFrame", "Series"):
        return _serialize_data(
            value.to_dict(orient="records")
            if type(value).__name__ == "DataFrame"
            else value.to_dict()
        )
    if isinstance(value, dict):
        return {str(key): _serialize_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serialize_data(value.model_dump())
    if hasattr(value, "tolist"):
        try:
            return _serialize_data(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _serialize_data(vars(value))
    return str(value)

def _resolve_variant_scorers(
    scorer_names: list[str] | None,
    organism: str,
) -> list:
    """Resolve variant scorer names to scorer objects."""
    if scorer_names is None:
        return variant_scorers_lib.get_recommended_scorers(_ORGANISM_PROTOS[organism])
    return [
        variant_scorers_lib.RECOMMENDED_VARIANT_SCORERS[name]
        for name in scorer_names
    ]

def _resolve_interval_scorers(scorer_names: list[str] | None) -> list:
    """Resolve interval scorer names to scorer objects."""
    recommended = interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS
    if scorer_names is None:
        return list(recommended.values())
    return [recommended[name] for name in scorer_names]

def _validate_min_scorer_width(
    interval_width: int,
    min_scorer_width: int,
    operation: str,
) -> None:
    """Validate that the effective context length is large enough for the scorer.

    Computes what ``_resize_interval`` will produce and checks it against the
    minimum scorer width.  Raises early with a clear message instead of letting
    the upstream library crash after model loading.
    """
    if interval_width in _SUPPORTED_CONTEXT_LENGTHS:
        effective = interval_width
    else:
        candidates = [l for l in _SUPPORTED_CONTEXT_LENGTHS if l >= interval_width]
        effective = min(candidates) if candidates else interval_width

    if effective < min_scorer_width:
        min_context = min(
            (l for l in _SUPPORTED_CONTEXT_LENGTHS if l >= min_scorer_width),
            default=max(_SUPPORTED_CONTEXT_LENGTHS),
        )
        raise ValueError(
            f"{operation}: effective context ({effective:,} bp) is smaller than "
            f"the minimum required by the scorer ({min_scorer_width:,} bp). "
            f"Please provide an interval of at least {min_context:,} bp."
        )


def _validate_sequence_length(sequence_length: int, operation: str) -> None:
    """Validate that raw sequence inputs match a supported context length."""
    if sequence_length in _SUPPORTED_CONTEXT_LENGTHS:
        return

    supported = ", ".join(f"{length:,}" for length in sorted(_SUPPORTED_CONTEXT_LENGTHS))
    raise ValueError(
        f"{operation}: sequence length ({sequence_length:,} bp) is unsupported. "
        f"Supported lengths: {supported} bp."
    )


def _resize_interval(
    chromosome: str, interval_start: int, interval_end: int,
) -> genome.Interval:
    """Build a ``genome.Interval``, auto-resizing to a supported context length."""
    interval = genome.Interval(chromosome=chromosome, start=interval_start, end=interval_end)
    if interval.width in _SUPPORTED_CONTEXT_LENGTHS:
        return interval
    target = min((l for l in _SUPPORTED_CONTEXT_LENGTHS if l >= interval.width), default=None)
    if target is None:
        raise ValueError(
            f"Context interval ({interval.width:,} bp) exceeds the largest "
            f"supported context length ({max(_SUPPORTED_CONTEXT_LENGTHS):,} bp)."
        )
    logger.warning(
        "Context length %s bp not supported; auto-resizing to %s bp.",
        f"{interval.width:,}", f"{target:,}",
    )
    return interval.resize(target)



# ============================================================================
# Dispatch
# ============================================================================
_OPERATIONS = {
    "predict_intervals",
    "predict_variants",
    "predict_sequences",
    "score_variants",
    "score_intervals",
    "score_ism_variants_batch",
}

_model: AlphaGenomeModel | None = None


def dispatch(input_dict: dict) -> dict:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model

    kwargs = dict(input_dict)
    operation = kwargs.pop("operation")
    model_version = kwargs.pop("model_version", "all_folds")
    # Tool orchestration metadata (not part of model call signature).
    kwargs.pop("timeout", None)

    if _model is None:
        _model = AlphaGenomeModel(model_version=model_version)

    if operation not in _OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation!r}")

    method = getattr(_model, operation)
    return method(**kwargs)


# ============================================================================
# Standalone script entry point for venv execution
# ============================================================================


def to_device(device: str) -> dict:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and hasattr(_model, "to_device"):
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_jax_memory_stats

    return get_jax_memory_stats(device_index=0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
