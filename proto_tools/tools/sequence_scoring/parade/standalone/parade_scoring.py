"""PARADE checkpoint loading, featurization, and inference over the vendored modules."""

import itertools
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import stability_data
import torch
import torch.nn.functional as F
import utrdata_cl as utrdata
from pl_regressor import RNARegressor
from standalone_helpers import get_logger
from torch.utils.data import DataLoader
from utrdata_cl import CELLTYPE_CODES_UTR3, CELLTYPE_CODES_UTR5

logger = get_logger(__name__)

DNA_VOCAB = ["A", "C", "G", "T"]

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

    ``weights_only=False`` is required: the Lightning hyperparameters pickle the LegNet
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

    rows: list[dict[str, float]] = []
    for seq_id in range(len(sequences)):
        sub = frame[frame["id"] == seq_id]
        score_map = {ct: float(value) for ct, value in zip(sub["cell_type"], sub["activity"], strict=True)}
        rows.append({cell_type: score_map[cell_type] for cell_type in cell_types})
    return rows


def predict_stability(model: Any, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
    """Predict the RNA/gDNA log-ratio for each 3' UTR sequence, batching per length group."""
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


def _relaxed_dna_probs(logits: Any, *, temperature: float, soft: float, hard: float) -> Any:
    """Convert ``... x L x 4`` DNA logits to relaxed probabilities with optional STE."""
    probs = F.softmax(logits / temperature, dim=-1)
    hard_probs = F.one_hot(probs.argmax(dim=-1), num_classes=len(DNA_VOCAB)).to(probs.dtype)
    relaxed = (1.0 - soft) * hard_probs + soft * probs
    if hard > 0.0:
        straight_through = hard_probs + probs - probs.detach()
        relaxed = hard * straight_through + (1.0 - hard) * relaxed
    return relaxed


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


def compute_gradient(
    model: Any,
    *,
    logits_list: list[list[list[float]]],
    temperature: float,
    construct_type: str,
    loss_terms: list[dict[str, Any]],
    soft: float,
    hard: float,
    compute_gradient: bool,
    device: str,
) -> dict[str, Any]:
    """Compute a differentiable PARADE activity loss and optional gradient.

    Each loss term maps a cell's raw activity through a max/min sigmoid; the weighted
    terms are summed into one scalar loss and backpropagated to the relaxed input logits.
    """
    with torch.set_grad_enabled(compute_gradient):
        logits = torch.tensor(logits_list, device=device, dtype=torch.float32, requires_grad=compute_gradient)
        if logits.ndim != 3:
            raise ValueError(f"parade: logits must have rank 3 (B, L, 4), got rank {logits.ndim}")

        relaxed = _relaxed_dna_probs(logits, temperature=temperature, soft=soft, hard=hard)
        batch_size = int(relaxed.shape[0])

        # PARADE conditions on a single cell type per forward pass (via the condition
        # channels), so run one forward per requested cell type and reuse its raw
        # activity across every loss term that targets that cell.
        needed_cells = {term["cell_type"] for term in loss_terms}
        raw_by_cell = {
            cell: model(_featurize_logits(relaxed, construct_type, cell))[:, _ACTIVITY_MASS_CENTER_INDEX]
            for cell in needed_cells
        }

        sample_losses = torch.zeros(batch_size, device=device, dtype=relaxed.dtype)
        term_metrics_by_sample: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
        raw_scores_by_sample: list[dict[str, float]] = [{} for _ in range(batch_size)]
        for term in loss_terms:
            cell = term["cell_type"]
            direction = term["direction"]
            weight = float(term.get("weight", 1.0))
            center = float(term.get("sigmoid_center", 2.0))
            scale = float(term.get("sigmoid_scale", 1.0))
            raw_score = raw_by_cell[cell]
            scaled_score = (raw_score - center) / scale
            sigmoid_value = torch.sigmoid(scaled_score)
            if direction == "max":
                score = 1.0 - sigmoid_value
            elif direction == "min":
                score = sigmoid_value
            else:
                raise ValueError(f"parade: invalid direction {direction!r}; expected 'max' or 'min'")
            weighted_score = score * weight
            sample_losses = sample_losses + weighted_score

            raw_score_values = raw_score.detach().cpu().tolist()
            scaled_score_values = scaled_score.detach().cpu().tolist()
            sigmoid_values = sigmoid_value.detach().cpu().tolist()
            score_values = score.detach().cpu().tolist()
            weighted_score_values = weighted_score.detach().cpu().tolist()
            for sample_idx in range(batch_size):
                raw_scores_by_sample[sample_idx][cell] = float(raw_score_values[sample_idx])
                term_metrics_by_sample[sample_idx].append(
                    {
                        "cell_type": cell,
                        "direction": direction,
                        "weight": weight,
                        "raw_score": float(raw_score_values[sample_idx]),
                        "scaled_score": float(scaled_score_values[sample_idx]),
                        "sigmoid_value": float(sigmoid_values[sample_idx]),
                        "score": float(score_values[sample_idx]),
                        "weighted_score": float(weighted_score_values[sample_idx]),
                        "sigmoid_center": center,
                        "sigmoid_scale": scale,
                    }
                )

        total_loss = sample_losses.sum()
        sample_loss_values = [float(value) for value in sample_losses.detach().cpu().tolist()]

        gradient_value: list[list[list[float]]] | None = None
        if compute_gradient:
            total_loss.backward()
            if logits.grad is None:
                raise RuntimeError("parade: missing logits gradient")
            gradient_value = logits.grad.detach().cpu().tolist()

    return {
        "gradient": gradient_value,
        "loss": float(sum(sample_loss_values)),
        "metrics": {
            "raw_scores": raw_scores_by_sample,
            "loss_terms": term_metrics_by_sample,
            "losses": sample_loss_values,
            "batch_size": batch_size,
            "temperature": temperature,
            "soft": soft,
            "hard": hard,
            "construct_type": construct_type,
            "objective": "parade_utr_activity",
        },
        "vocab": DNA_VOCAB,
    }
