"""Standalone inference entry point for SpliceAI scoring and prediction."""

import json
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from standalone_helpers import get_logger, serialize_output

logger = get_logger(__name__)

# SpliceAI's 10k-context models see 5000 bp of flanking sequence per side.
_CONTEXT_PAD = 5000

# One-hot rows indexed N/A/C/G/T = 0..4. _one_hot_encode maps U->T and any
# non-ACGTN base to N (identical to SpliceAI on the documented ACGTN alphabet).
_ONEHOT = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

# ── Lazy global state (kept across persistent-worker calls) ──────────────────
_tf_configured = False
_models: list[Any] | None = None  # 5 Keras models for the predict operation
_models_device: str | None = None
_annotator: Any | None = None  # spliceai Annotator for the score operation
_annotator_key: tuple[str, str, str] | None = None  # (reference_fasta, annotation, device)


def _configure_tf_once() -> None:
    """Enable GPU memory growth once, before any GPU is initialized (for co-tenancy).

    GPU visibility is controlled by CUDA_VISIBLE_DEVICES, which the worker framework
    sets per allocated device; the standalone does not hide/unhide GPUs — TF's
    set_visible_devices is process-sticky and cannot be undone after init.
    """
    global _tf_configured
    if _tf_configured:
        return
    import tensorflow as tf

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning("spliceai: could not set GPU memory growth: %s", e)
    _tf_configured = True


def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence to shape (len, 4); unknown bases map to N."""
    seq = (
        seq.upper()
        .replace("A", "1")
        .replace("C", "2")
        .replace("G", "3")
        .replace("T", "4")
        .replace("U", "4")
        .replace("N", "0")
    )
    idx = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("0")
    idx[(idx < 0) | (idx > 4)] = 0
    return _ONEHOT[idx]


# ── Model loading ────────────────────────────────────────────────────────────
def _spliceai_models_dir() -> str:
    """Return the bundled SpliceAI models directory inside the installed package."""
    import spliceai

    return os.path.join(os.path.dirname(spliceai.__file__), "models")


def _ensure_models(device: str) -> list[Any]:
    """Lazily load the 5-model SpliceAI ensemble for raw prediction."""
    global _models, _models_device
    if _models is not None and _models_device == device:
        return _models

    _configure_tf_once()
    from keras.models import load_model

    models_dir = _spliceai_models_dir()
    logger.update_status("Loading SpliceAI ensemble (5 models)")
    _models = [load_model(os.path.join(models_dir, f"spliceai{i}.h5")) for i in range(1, 6)]
    _models_device = device
    logger.debug("spliceai: loaded 5 models on %s", device)
    return _models


def _ensure_annotator(reference_fasta: str, annotation: str, device: str) -> Any:
    """Lazily build the SpliceAI Annotator (reference genome + annotation + 5 models)."""
    global _annotator, _annotator_key
    key = (reference_fasta, annotation, device)
    if _annotator is not None and _annotator_key == key:
        return _annotator

    _configure_tf_once()
    from spliceai.utils import Annotator

    logger.update_status("Building SpliceAI annotator")
    _annotator = Annotator(reference_fasta, annotation)
    _annotator_key = key
    logger.debug("spliceai: built annotator for %s / %s on %s", reference_fasta, annotation, device)
    return _annotator


# ── Operations ───────────────────────────────────────────────────────────────
def _parse_delta_score(record: str) -> dict[str, Any]:
    """Parse a SpliceAI ``ALLELE|SYMBOL|DS_*x4|DP_*x4`` string; ``.`` (complex MNV) maps to None."""
    allele, symbol, *fields = record.split("|")
    return {
        "allele": allele,
        "symbol": symbol,
        "ds_ag": None if fields[0] == "." else float(fields[0]),
        "ds_al": None if fields[1] == "." else float(fields[1]),
        "ds_dg": None if fields[2] == "." else float(fields[2]),
        "ds_dl": None if fields[3] == "." else float(fields[3]),
        "dp_ag": None if fields[4] == "." else int(fields[4]),
        "dp_al": None if fields[5] == "." else int(fields[5]),
        "dp_dg": None if fields[6] == "." else int(fields[6]),
        "dp_dl": None if fields[7] == "." else int(fields[7]),
    }


def score(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Score variants for splice-altering effects; returns per-variant gene records."""
    from spliceai.utils import get_delta_scores

    annotator = _ensure_annotator(input_dict["reference_fasta"], input_dict["annotation"], input_dict["device"])
    max_distance = input_dict["max_distance"]
    mask = input_dict["mask"]

    results: list[list[dict[str, Any]]] = []
    for variant in input_dict["variants"]:
        record = SimpleNamespace(
            chrom=variant["chromosome"],
            pos=variant["position"],
            ref=variant["ref"],
            alts=(variant["alt"],),
        )
        raw_scores = get_delta_scores(record, annotator, max_distance, mask)
        results.append([_parse_delta_score(s) for s in raw_scores])

    return {"results": results}


def predict(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Predict per-position [neither, acceptor, donor] probabilities for each sequence."""
    models = _ensure_models(input_dict["device"])

    predictions: list[list[list[float]]] = []
    for seq in input_dict["sequences"]:
        encoded = _one_hot_encode("N" * _CONTEXT_PAD + seq + "N" * _CONTEXT_PAD)[None, :]
        averaged = np.mean([model.predict(encoded, verbose=0) for model in models], axis=0)
        predictions.append(averaged[0].tolist())

    return {"predictions": predictions}


# ── Dispatch ─────────────────────────────────────────────────────────────────
_OPERATIONS = {"score": score, "predict": predict}


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    operation = input_dict["operation"]
    handler = _OPERATIONS.get(operation)
    if handler is None:
        raise ValueError(f"spliceai: unknown operation {operation!r}; valid: {sorted(_OPERATIONS)}")
    return handler(input_dict)


# ── Device management protocol ───────────────────────────────────────────────
def to_device(device: str) -> dict[str, Any]:
    """Move SpliceAI to a device (called by DeviceManager).

    Keras models can't be relocated in place, so a CPU move drops the loaded
    models and clears the TF session; the next dispatch reloads on ``device``.
    """
    global _models, _models_device, _annotator, _annotator_key
    if device == "cpu" and (_models is not None or _annotator is not None):
        _models = _models_device = _annotator = _annotator_key = None
        import keras

        keras.backend.clear_session()
        return {"success": True, "device": device, "note": "models unloaded for CPU"}
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report memory usage (called by DeviceManager). TF stats are not tracked."""
    return {"available": False, "framework": "tensorflow", "reason": "TensorFlow memory stats not tracked"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("spliceai: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
