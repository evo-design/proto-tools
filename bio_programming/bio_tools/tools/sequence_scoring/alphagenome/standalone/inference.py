"""
AlphaGenome standalone inference implementation for venv execution.
"""
from __future__ import annotations

import functools
import json
import logging
import sys
from typing import Any, List, Optional

import jax
import numpy as np
from alphagenome.data import genome
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import variant_scorers as variant_scorers_lib
from alphagenome.protos import dna_model_pb2
from alphagenome_research.model import dna_model

logger = logging.getLogger(__name__)

_SUPPORTED_CONTEXT_LENGTHS = [1_048_576, 524_288, 131_072, 16_384, 2_048]

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

    def predict_interval(
        self,
        chromosome: str,
        interval_start: int,
        interval_end: int,
        requested_outputs: List[str],
        ontology_terms: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run interval predictions."""
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        interval = _resize_interval(chromosome, interval_start, interval_end)

        prediction = self.model.predict_interval(
            interval=interval,
            requested_outputs=[
                dna_model.OutputType[name] for name in requested_outputs
            ],
            ontology_terms=ontology_terms,
            organism=_ORGANISM_ENUMS[organism],
        )

        return {"predictions": _serialize_data(prediction)}

    def predict_variant(
        self,
        chromosome: str,
        interval_start: int,
        interval_end: int,
        variant_position: int,
        reference_bases: str,
        alternate_bases: str,
        requested_outputs: List[str],
        ontology_terms: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run variant-effect predictions."""
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        interval = _resize_interval(chromosome, interval_start, interval_end)

        prediction = self.model.predict_variant(
            interval=interval,
            variant=genome.Variant(
                chromosome=chromosome,
                position=variant_position,
                reference_bases=reference_bases,
                alternate_bases=alternate_bases,
            ),
            requested_outputs=[dna_model.OutputType[name] for name in requested_outputs],
            ontology_terms=ontology_terms,
            organism=_ORGANISM_ENUMS[organism],
        )

        return {"predictions": _serialize_data(prediction)}

    def predict_sequence(
        self,
        sequence: str,
        requested_outputs: List[str],
        ontology_terms: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run predictions from a raw DNA sequence string."""
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        prediction = self.model.predict_sequence(
            sequence=sequence,
            requested_outputs=[
                dna_model.OutputType[name] for name in requested_outputs
            ],
            ontology_terms=ontology_terms,
            organism=_ORGANISM_ENUMS[organism],
        )

        return {"predictions": _serialize_data(prediction)}

    def score_variant(
        self,
        chromosome: str,
        interval_start: int,
        interval_end: int,
        variant_position: int,
        reference_bases: str,
        alternate_bases: str,
        variant_scorers: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run variant scoring with recommended variant scorers."""
        _validate_min_scorer_width(
            interval_end - interval_start,
            _MIN_VARIANT_SCORER_WIDTH,
            "score_variant",
        )
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        scorers = _resolve_variant_scorers(variant_scorers, organism)

        interval = _resize_interval(chromosome, interval_start, interval_end)

        scores = self.model.score_variant(
            interval=interval,
            variant=genome.Variant(
                chromosome=chromosome,
                position=variant_position,
                reference_bases=reference_bases,
                alternate_bases=alternate_bases,
            ),
            variant_scorers=scorers,
            organism=_ORGANISM_ENUMS[organism],
        )

        _fixup_score_uns_keys(scores)
        df = variant_scorers_lib.tidy_scores(scores)
        return json.loads(df.to_json(orient="records"))

    def score_interval(
        self,
        chromosome: str,
        interval_start: int,
        interval_end: int,
        interval_scorers: Optional[List[str]] = None,
        organism: str = "human",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run interval scoring with recommended interval scorers."""
        _validate_min_scorer_width(
            interval_end - interval_start,
            _MIN_INTERVAL_SCORER_WIDTH,
            "score_interval",
        )
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        scorers = _resolve_interval_scorers(interval_scorers)

        interval = _resize_interval(chromosome, interval_start, interval_end)

        scores = self.model.score_interval(
            interval=interval,
            interval_scorers=scorers,
            organism=_ORGANISM_ENUMS[organism],
        )

        _fixup_score_uns_keys(scores)
        df = variant_scorers_lib.tidy_scores(scores)
        return json.loads(df.to_json(orient="records"))

    def score_ism_variants(
        self,
        chromosome: str,
        interval_start: int,
        interval_end: int,
        ism_interval_start: int,
        ism_interval_end: int,
        variant_scorers: Optional[List[str]] = None,
        organism: str = "human",
        variant_position: Optional[int] = None,
        reference_bases: Optional[str] = None,
        alternate_bases: Optional[str] = None,
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run in-silico mutagenesis scoring."""
        _validate_min_scorer_width(
            interval_end - interval_start,
            _MIN_VARIANT_SCORER_WIDTH,
            "score_ism_variants",
        )
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        scorers = _resolve_variant_scorers(variant_scorers, organism)

        interval = _resize_interval(chromosome, interval_start, interval_end)
        ism_interval = genome.Interval(
            chromosome=chromosome,
            start=ism_interval_start,
            end=ism_interval_end,
        )

        interval_variant = None
        if variant_position is not None:
            interval_variant = genome.Variant(
                chromosome=chromosome,
                position=variant_position,
                reference_bases=reference_bases,
                alternate_bases=alternate_bases,
            )

        scores = self.model.score_ism_variants(
            interval=interval,
            ism_interval=ism_interval,
            variant_scorers=scorers,
            organism=_ORGANISM_ENUMS[organism],
            interval_variant=interval_variant,
        )

        _fixup_score_uns_keys(scores)
        df = variant_scorers_lib.tidy_scores(scores)
        return json.loads(df.to_json(orient="records"))

    # ============================================================================
    # Device Management
    # ============================================================================
    def load(self, device: str = "cuda", verbose: bool = False):
        """Load AlphaGenome model to device."""
        if verbose:
            logger.info(f"Loading AlphaGenome model: {self.model_version} on {device}")

        jax_device = jax.devices("gpu" if device == "cuda" else device)[0]
        self.model = dna_model.create_from_huggingface(self.model_version, device=jax_device)
        _patch_predict_device_put(self.model, jax_device)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("AlphaGenome model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        JAX does not support moving models between devices, so this
        reloads the model on the target device.
        """
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.load(device)

    def unload(self, verbose: bool = False) -> None:
        """Unload model to free GPU memory.

        JAX does not support moving models to CPU like PyTorch. Instead,
        the model reference is deleted and the model will be reloaded on
        the next call.
        """
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = None
            self.device = None
            self._loaded = False


# ============================================================================
# Helpers
# ============================================================================

def _patch_predict_device_put(model, device) -> None:
    """Monkey-patch ``model._predict`` to explicitly ``device_put`` numpy args.

    Upstream bug: ``DnaModel.score_interval`` passes raw numpy arrays for
    ``sequence`` and ``organism_indices`` into a
    ``jax.transfer_guard('disallow')`` block without wrapping them in
    ``jax.device_put(...)`` first — unlike every other method
    (``predict_interval``, ``predict_variant``, ``predict_sequence``,
    ``score_variant``).

    This causes a ``JaxRuntimeError: INVALID_ARGUMENT: Disallowed
    host-to-device transfer`` at inference time.

    Rather than patching the installed package (not reproducible), we wrap
    ``_predict`` to explicitly ``device_put`` any numpy arrays before they
    reach the jit boundary.  ``jax.device_put`` is an *explicit* transfer
    and is always allowed, even inside a ``transfer_guard('disallow')``
    context.  For call sites that already pass JAX arrays (all methods
    except ``score_interval``), this is a no-op.

    TODO: Remove once the upstream bug is fixed.
    Ref: alphagenome_research/model/dna_model.py  score_interval lines 674-696
    """
    original_predict = model._predict

    @functools.wraps(original_predict)
    def _patched_predict(*args, **kwargs):
        args = tuple(
            jax.device_put(a, device) if isinstance(a, np.ndarray) else a
            for a in args
        )
        kwargs = {
            k: jax.device_put(v, device) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        return original_predict(*args, **kwargs)

    model._predict = _patched_predict


def _serialize_data(value: Any) -> Any:
    """Recursively convert nested prediction objects into JSON-safe structures.

    Used for predict_* operations whose outputs contain numpy arrays,
    dataclasses, TrackData, JunctionData, and other complex types.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
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

def _fixup_score_uns_keys(scores):
    """Fix key mismatches in AnnData between alphagenome_research and alphagenome.

    alphagenome_research's ``score_variant``/``score_interval`` store
    ``adata.uns['scored_interval']``, but the ``alphagenome`` dependency's
    ``tidy_scores`` reads ``adata.uns['interval']``.

    Similarly, gene-centric scorers in alphagenome_research put ``Strand``
    (capital S) in ``adata.obs``, while ``alphagenome``'s ``tidy_anndata``
    expects the lowercase ``strand`` column.

    TODO: Remove this workaround once the upstream key mismatch is resolved.
    """
    for item in scores:
        if isinstance(item, (list, tuple)):
            _fixup_score_uns_keys(item)
        elif hasattr(item, "uns"):
            if "scored_interval" in item.uns and "interval" not in item.uns:
                item.uns["interval"] = item.uns.pop("scored_interval")
            # Normalize obs column: Strand -> strand
            if hasattr(item, "obs") and "Strand" in item.obs.columns and "strand" not in item.obs.columns:
                item.obs = item.obs.rename(columns={"Strand": "strand"})
            # Fix gene_id column issues that crash upstream tidy_anndata.
            if hasattr(item, "obs") and "gene_id" in item.obs.columns:
                if len(item.obs) == 0:
                    # Zero genes in interval: drop gene columns so tidy_anndata
                    # falls through to its X.shape[0]==0 empty-result branch
                    # instead of crashing on str.split(..., expand=True)[0].
                    item.obs = item.obs.drop(
                        columns=["gene_id", "strand", "Strand"], errors="ignore"
                    )
                else:
                    # Non-empty obs with NaN gene_ids: coerce to string so
                    # upstream .str accessor doesn't raise AttributeError.
                    item.obs["gene_id"] = item.obs["gene_id"].fillna("").astype(str)


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
# Standalone script entry point for venv execution
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    operation = input_data.pop("operation")
    model_version = input_data.pop("model_version", "all_folds")
    model = AlphaGenomeModel(model_version=model_version)

    if operation == "predict_interval":
        output_data = model.predict_interval(**input_data)
    elif operation == "predict_variant":
        output_data = model.predict_variant(**input_data)
    elif operation == "predict_sequence":
        output_data = model.predict_sequence(**input_data)
    elif operation == "score_variant":
        output_data = model.score_variant(**input_data)
    elif operation == "score_interval":
        output_data = model.score_interval(**input_data)
    elif operation == "score_ism_variants":
        output_data = model.score_ism_variants(**input_data)
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
