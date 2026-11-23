"""PARADE checkpoint loading, featurization, and inference over the vendored modules."""

import itertools
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import stability_data
import torch
import utrdata_cl as utrdata
from pl_regressor import RNARegressor
from standalone_helpers import get_logger
from standalone_helpers.relaxation import relaxed_sequence
from standalone_helpers.vocabulary import DNA_NUCLEOTIDES
from torch.utils.data import DataLoader
from utrdata_cl import CELLTYPE_CODES_UTR3, CELLTYPE_CODES_UTR5

logger = get_logger(__name__)

DNA_VOCAB = list(DNA_NUCLEOTIDES)

# PARADE regression heads emit several columns; these indices select the scalar the
# published tutorials read back (see the upstream ``predict.py``).
_ACTIVITY_MASS_CENTER_INDEX = 1  # activity model: column 1 is the activity mass-center
_STABILITY_LOG_RATIO_INDEX = 0  # stability model: column 0 is the RNA/gDNA log-ratio

_CELL_CODES = {"utr5": CELLTYPE_CODES_UTR5, "utr3": CELLTYPE_CODES_UTR3}

# Featurization matches the upstream ``predict.py`` exactly: no reporter flanks and no
# test-time augmentation, so the model sees the bare insert plus positional/condition
# channels.
_ACTIVITY_AUGMENT_KWS = {
    "extend_left": 0,
    "extend_right": 0,
    "shift_left": 0,
    "shift_right": 0,
    "revcomp": False,
}


def load_legnet(checkpoint_path: str) -> Any:
    """Load a PARADE ``RNARegressor`` checkpoint and return its LegNet module.

    ``weights_only=False`` is required because the Lightning hyperparameters pickle the LegNet
    model class, which is resolved via the vendored ``legnet_classifier`` module.
    """
    regressor = RNARegressor.load_from_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
    regressor.eval()
    regressor.requires_grad_(False)
    model = regressor.model
    model.eval()
    return model


def _predict(model: Any, dataset: Any, batch_size: int, device: str) -> np.ndarray:
    """Run the model over a dataset and return stacked predictions (N x C)."""
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, drop_last=False)
    chunks = []
    with torch.inference_mode():
        for seqs, _ in loader:
            preds = model(seqs.to(device))
            chunks.append(preds.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def _group_by_length(sequences: list[str]) -> dict[int, list[int]]:
    """Map sequence length -> original indices, so each batch tensor stacks uniformly.

    Grouping by length keeps mixed-length inputs correct and makes the tools safe under the
    framework's per-item iterable cache, which can hand us any subset of the caller's items.
    """
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        by_length[len(sequence)].append(index)
    return by_length


def predict_activity(
    model: Any,
    *,
    sequences: list[str],
    construct_type: str,
    cell_types: list[str],
    batch_size: int,
    device: str,
) -> list[dict[str, float]]:
    """Predict per-cell-type activity for each sequence, batching per length group."""
    rows: list[dict[str, float] | None] = [None] * len(sequences)
    for indices in _group_by_length(sequences).values():
        group = _predict_activity_group(
            model,
            sequences=[sequences[i] for i in indices],
            construct_type=construct_type,
            cell_types=cell_types,
            batch_size=batch_size,
            device=device,
        )
        for local_index, original_index in enumerate(indices):
            rows[original_index] = group[local_index]
    return [row for row in rows if row is not None]


def _predict_activity_group(
    model: Any,
    *,
    sequences: list[str],
    construct_type: str,
    cell_types: list[str],
    batch_size: int,
    device: str,
) -> list[dict[str, float]]:
    """Score one equal-length group.

    Builds one row per (sequence, cell type) pair exactly like the upstream ``predict.py``,
    runs the conditioned model, and pivots back to one dict per sequence keyed by cell code.
    """
    # PARADE conditions the model on the cell type through input channels, so we score
    # one row per (sequence, cell type) pair and pivot the predictions back per sequence.
    col_idseq, col_cell_type = zip(*itertools.product(enumerate(sequences), cell_types), strict=True)
    col_id, col_seq = zip(*col_idseq, strict=True)
    frame = pd.DataFrame({"id": col_id, "seq": col_seq, "cell_type": col_cell_type})

    dataset = utrdata.UTRData(
        df=frame,
        predict_cols=[],
        construct_type=construct_type,
        features=("sequence", "positional", "conditions"),
        augment=False,
        augment_test_time=False,
        augment_kws=dict(_ACTIVITY_AUGMENT_KWS),
    )
    # `batch_size` counts sequences (per the public schema); each expands to one conditioned
    # row per cell type, so scale the row-level loader batch by the number of cell types.
    row_batch_size = batch_size * len(cell_types)
    frame["activity"] = _predict(model, dataset, row_batch_size, device)[:, _ACTIVITY_MASS_CENTER_INDEX]

    # Gather each sequence's conditioned predictions using the requested cell-code order.
    rows: list[dict[str, float]] = []
    for seq_id in range(len(sequences)):
        sub = frame[frame["id"] == seq_id]
        score_map = {ct: float(value) for ct, value in zip(sub["cell_type"], sub["activity"], strict=True)}
        rows.append({cell_type: score_map[cell_type] for cell_type in cell_types})
    return rows


def predict_stability(model: Any, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
    """Predict the RNA/gDNA log-ratio for each 3' UTR sequence, batching per length group."""
    # Batch equal-length UTRs together, then restore predictions to the original order.
    values: list[float | None] = [None] * len(sequences)
    for indices in _group_by_length(sequences).values():
        group = _predict_stability_group(
            model, sequences=[sequences[i] for i in indices], batch_size=batch_size, device=device
        )
        for local_index, original_index in enumerate(indices):
            values[original_index] = group[local_index]
    return [value for value in values if value is not None]


def _predict_stability_group(model: Any, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
    """Score one equal-length group of 3' UTR sequences."""
    frame = pd.DataFrame({"id": np.arange(len(sequences)), "seq": list(sequences)})
    dataset = stability_data.StabilityData(df=frame, features=("sequence",), predict_cols=[])
    predictions = _predict(model, dataset, batch_size, device)[:, _STABILITY_LOG_RATIO_INDEX]
    return [float(value) for value in predictions]


def _featurize_logits(relaxed: Any, construct_type: str, cell: str) -> Any:
    """Build the model input for one cell condition from relaxed (B, L, 4) logits.

    Mirrors the ``utrdata_cl.UTRData`` featurization used for scoring — sequence
    channels, a reading-frame positional channel, and broadcast one-hot condition
    channels — but keeps the graph differentiable w.r.t. the relaxed logits.
    """
    codes = _CELL_CODES[construct_type]
    num_conditions = len(codes)
    batch, length, _ = relaxed.shape
    seq_ch = relaxed.transpose(1, 2)  # (B, 4, L)
    positional = (torch.arange(length, device=relaxed.device) % 3 == 0).to(relaxed.dtype)
    positional_ch = positional.view(1, 1, length).expand(batch, 1, length)
    condition = torch.zeros(num_conditions, device=relaxed.device, dtype=relaxed.dtype)
    condition[codes[cell]] = 1.0
    condition_ch = condition.view(1, num_conditions, 1).expand(batch, num_conditions, length)
    return torch.cat([seq_ch, positional_ch, condition_ch], dim=1)


def compute_stability_gradient(
    model: Any,
    *,
    logits_list: list[list[list[float]]],
    temperature: float | None,
    soft: float,
    hard: float,
    compute_gradient: bool,
    device: str,
) -> dict[str, Any]:
    """Differentiate the PARADE stability log-ratio with respect to a relaxed state.

    The stability checkpoint predicts one RNA/gDNA log-ratio per sequence, higher being more
    stable, and that raw scalar is what is differentiated. No sign is applied, so the gradient is
    ``d(log_ratio)/d(input)``. Its featurization is simpler than the activity model's.
    ``StabilityData`` is built with ``features=("sequence",)``, meaning four sequence channels and
    no positional or condition channels, so the relaxed matrix only needs transposing to
    ``(B, 4, L)``.

    Args:
        model: Loaded stability checkpoint.
        logits_list: Relaxed states, shape ``(B, L, 4)`` in ``A,C,G,T`` order.
        temperature: Optional softmax temperature; ``None`` uses the rows as probabilities.
        soft: Blend from the raw input matrix (0) to softmax probabilities (1).
        hard: Straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient: If False, skip backward and return ``gradients=None``.
        device: Device to run on.

    Returns:
        dict[str, Any]: Per-sample log-ratios, their gradients, and sequence lengths.

    Raises:
        ValueError: If the logits are not rank 3.
        RuntimeError: If an expected gradient is missing after the backward pass.
    """
    with torch.set_grad_enabled(compute_gradient):
        logits = torch.tensor(logits_list, device=device, dtype=torch.float32, requires_grad=compute_gradient)
        if logits.ndim != 3:
            raise ValueError(f"parade: logits must have rank 3 (B, L, 4), got rank {logits.ndim}")

        relaxed = relaxed_sequence(logits, temperature=temperature, soft=soft, hard=hard)
        batch_size = int(relaxed.shape[0])

        # Sequence channels only, matching StabilityData(features=("sequence",)).
        log_ratio = model(relaxed.transpose(1, 2))[:, _STABILITY_LOG_RATIO_INDEX]

        gradients: list[list[list[float]]] | None = None
        if compute_gradient:
            # Sum independent samples to compute their gradients in one backward pass.
            log_ratio.sum().backward()
            if logits.grad is None:
                raise RuntimeError("parade: missing relaxed-state gradient after backward")
            gradients = logits.grad.detach().cpu().tolist()

        return {
            "log_ratios": [float(value) for value in log_ratio.detach().cpu().tolist()],
            "gradients": gradients,
            "lengths": [int(relaxed.shape[1])] * batch_size,
        }


def compute_gradient(
    model: Any,
    *,
    logits_list: list[list[list[float]]],
    temperature: float | None,
    construct_type: str,
    gradient_terms: list[dict[str, Any]],
    soft: float,
    hard: float,
    compute_gradient: bool,
    device: str,
) -> dict[str, Any]:
    """Differentiate each requested PARADE cell activity with respect to a relaxed state.

    Every term names one cell, and its value is that cell's raw predicted activity. No direction,
    weight or sigmoid is applied, so each gradient is ``d(activity)/d(input)``. PARADE conditions
    on a single cell per forward pass via its condition channels, so one forward runs per cell and
    is differentiated immediately, keeping at most one gradient on the device at a time. The
    relaxation is shared by every cell, so its graph is retained until the final backward frees it.

    Args:
        model: Loaded activity checkpoint.
        logits_list: Relaxed states, shape ``(B, L, 4)`` in ``A,C,G,T`` order.
        temperature: Optional softmax temperature; ``None`` uses the rows as probabilities.
        construct_type: ``"utr5"`` or ``"utr3"``, selecting the condition-channel panel.
        gradient_terms: Resolved terms, each naming one ``cell_type``.
        soft: Blend from the raw input matrix (0) to softmax probabilities (1).
        hard: Straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient: If False, skip every backward and return ``gradients=None`` per term.
        device: Device to run on.

    Returns:
        dict[str, Any]: One entry per term with its per-sample values and gradients, plus
            call-level descriptors and the vocabulary.

    Raises:
        ValueError: If the logits are not rank 3.
        RuntimeError: If an expected gradient is missing after a backward pass.
    """
    with torch.set_grad_enabled(compute_gradient):
        logits = torch.tensor(logits_list, device=device, dtype=torch.float32, requires_grad=compute_gradient)
        if logits.ndim != 3:
            raise ValueError(f"parade: logits must have rank 3 (B, L, 4), got rank {logits.ndim}")

        relaxed = relaxed_sequence(logits, temperature=temperature, soft=soft, hard=hard)
        batch_size = int(relaxed.shape[0])

        # One forward per distinct cell, in first-requested order, carrying the caller's term id.
        cells: list[tuple[str, str | None]] = []
        seen: set[str] = set()
        for term in gradient_terms:
            if term["cell_type"] not in seen:
                seen.add(term["cell_type"])
                cells.append((term["cell_type"], term.get("id")))

        terms: list[dict[str, Any]] = []
        for position, (cell, term_id) in enumerate(cells):
            # (B, L, 4) -> featurized -> (B,) raw activity for this cell
            activity = model(_featurize_logits(relaxed, construct_type, cell))[:, _ACTIVITY_MASS_CENTER_INDEX]
            entry: dict[str, Any] = {
                "cell_type": cell,
                "id": term_id,
                "values": [float(value) for value in activity.detach().cpu().tolist()],
                "gradients": None,
            }

            if compute_gradient:
                # Sum independent samples to compute their gradients in one backward pass.
                (gradient,) = torch.autograd.grad(activity.sum(), logits, retain_graph=(position < len(cells) - 1))
                if gradient is None:
                    raise RuntimeError(f"parade: missing gradient for cell {cell!r} after backward")
                # Move this gradient to CPU before evaluating the next cell to limit GPU memory use.
                entry["gradients"] = gradient.detach().cpu().tolist()
                del gradient

            terms.append(entry)
            del activity

    return {
        "terms": terms,
        "metrics": {
            "batch_size": batch_size,
            "temperature": temperature,
            "soft": soft,
            "hard": hard,
            "construct_type": construct_type,
            "objective": "parade_utr_activity",
        },
        "vocab": DNA_VOCAB,
    }
