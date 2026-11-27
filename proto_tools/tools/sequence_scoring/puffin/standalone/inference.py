"""Puffin standalone inference. Both tools use the Puffin (FFT-conv) model.

`predict` calls Puffin.predict() (forward pass only, returns 10-channel DataFrame).
`interpret` calls Puffin.interpret() (gradient-based motif decomposition for one
target signal/strand).
"""

import contextlib
import json
import os
import sys
from typing import Any

from standalone_helpers import (
    StatusCounter,
    get_logger,
    get_pytorch_memory_stats,
    move_model_to_device,
    resolve_weights_dir,
    serialize_output,
    set_torch_seed,
)
from standalone_helpers.relaxation import relaxed_sequence, resolve_temperature

logger = get_logger(__name__)

# Upstream trims this many bases from each end of its per-base output.
PUFFIN_PADDING = 325

# Upstream predict()/interpret() encode strings with these arguments; any other base (``N``)
# becomes 0.25 in every column.
PUFFIN_BASE_TO_INDEX = {"A": 0, "a": 0, "C": 1, "c": 1, "G": 2, "g": 2, "T": 3, "t": 3}
PUFFIN_BASES = "ACGT"

TARGETI_FORWARD = {
    "FANTOM_CAGE": 0,
    "ENCODE_CAGE": 1,
    "ENCODE_RAMPAGE": 2,
    "GRO_CAP": 3,
    "PRO_CAP": 4,
}

# Row labels emitted by Puffin.predict(); upstream typo "Prediciton" preserved.
PREDICT_ROW_NAMES = [
    "Prediciton FANTOM_CAGE",
    "Prediciton ENCODE_CAGE",
    "Prediciton ENCODE_RAMPAGE",
    "Prediciton GRO_CAP",
    "Prediciton PRO_CAP",
    "Prediciton rev strand PRO_CAP",
    "Prediciton rev strand GRO_CAP",
    "Prediciton rev strand ENCODE_RAMPAGE",
    "Prediciton rev strand ENCODE_CAGE",
    "Prediciton rev strand FANTOM_CAGE",
]

# 9 motifs x 2 strands; Long Inr is dropped by upstream's reindex (folded into the initiator-effect sum).
UPSTREAM_TO_NORMALIZED_MOTIF = {
    "CREB+": "CREB+",
    "CREB-": "CREB-",
    "ETS+": "ETS+",
    "ETS-": "ETS-",
    "NFY+": "NFY+",
    "NFY-": "NFY-",
    "NRF1+": "NRF1+",
    "NRF1-": "NRF1-",
    "SP+": "SP+",
    "SP-": "SP-",
    "TATA+": "TATA+",
    "TATA-": "TATA-",
    "U1 snRNP+": "U1_snRNP+",
    "U1 snRNP-": "U1_snRNP-",
    "YY1+": "YY1+",
    "YY1-": "YY1-",
    "ZNF143+": "ZNF143+",
    "ZNF143-": "ZNF143-",
}


@contextlib.contextmanager
def _chdir(target: str) -> Any:
    """Temporarily switch the process CWD so Puffin can load ./resources/ weights."""
    previous = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


class PuffinModel:
    """Lazy-loaded Puffin model with device tracking."""

    def __init__(self) -> None:
        """Initialize empty model slot."""
        self.model: Any = None
        self.device: str = "cpu"
        self._repo_dir: str | None = None

    def _ensure_repo(self) -> str:
        """Resolve the cloned Puffin repo path and put it on sys.path."""
        if self._repo_dir is None:
            repo_dir = os.path.join(resolve_weights_dir("puffin"), "puffin_repo")
            if not os.path.isdir(repo_dir):
                raise RuntimeError(f"puffin: cloned repo not found at {repo_dir!r}; re-run setup.sh")
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            self._repo_dir = repo_dir
        return self._repo_dir

    def load(self, device: str, verbose: bool = False) -> Any:
        """Load the Puffin model onto the requested device."""
        if self.model is not None and self.device == device:
            return self.model
        repo_dir = self._ensure_repo()
        logger.update_status("Loading Puffin weights")
        if verbose:
            logger.info(f"Loading Puffin on {device}")
        from puffin import Puffin  # type: ignore[import-not-found]
        from standalone_helpers import set_current_cuda_device

        # Puffin's predict/interpret call ``self.cuda()`` and ``.cuda()`` on inputs, which use the
        # current device. Select the leased GPU before anything touches CUDA.
        set_current_cuda_device(device)
        with _chdir(repo_dir):
            model = Puffin(use_cuda=device.startswith("cuda"))
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        return model

    def to_device(self, device: str) -> None:
        """Move the loaded model to a different device."""
        if self.model is not None:
            self.model = move_model_to_device(self.model, self.device, device)
            self.model.use_cuda = device.startswith("cuda")
        self.device = device


def _predict(state: PuffinModel, input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run Puffin.predict() on each input sequence and return per-base 10-channel arrays."""
    import numpy as np

    device = input_dict["device"]
    verbose = bool(input_dict.get("verbose", False))
    sequences = input_dict["sequences"]

    model = state.load(device, verbose=verbose)

    predictions: list[list[list[float]]] = []
    for sequence in StatusCounter(logger, "Predicting sequence %d/%d", sequences):
        df = model.predict(sequence)
        # Stack the 10 prediction rows in TRACK_NAMES order, then transpose to (L-650, 10).
        stacked = np.vstack([np.asarray(df.loc[row].to_numpy(), dtype=float) for row in PREDICT_ROW_NAMES])
        per_base = stacked.T.tolist()
        predictions.append(per_base)

    return {"predictions": predictions}


def _row(df: Any, name: str) -> list[float]:
    """Extract a per-base row from the Puffin interpret DataFrame as a list of floats."""
    import numpy as np

    return [float(v) for v in np.asarray(df.loc[name].to_numpy(), dtype=float)]


def _interpret(state: PuffinModel, input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run Puffin.interpret() on each input sequence and return motif-decomposed signals."""
    device = input_dict["device"]
    verbose = bool(input_dict.get("verbose", False))
    sequences = input_dict["sequences"]
    target_signal = input_dict["target_signal"]
    reverse_strand = bool(input_dict.get("reverse_strand", False))

    if target_signal not in TARGETI_FORWARD:
        raise ValueError(f"puffin: target_signal {target_signal!r} not in {sorted(TARGETI_FORWARD)}")

    model = state.load(device, verbose=verbose)

    results: list[dict[str, Any]] = []
    for sequence in StatusCounter(logger, "Interpreting sequence %d/%d", sequences):
        df = model.interpret(sequence, targeti=target_signal, reverse_strand=reverse_strand)

        motif_activations: dict[str, list[float]] = {}
        motif_effects: dict[str, list[float]] = {}
        bp_contribution_per_motif: dict[str, list[float]] = {}
        bp_contribution_to_motif_activation: dict[str, list[float]] = {}

        for upstream_name, normalized in UPSTREAM_TO_NORMALIZED_MOTIF.items():
            motif_activations[normalized] = _row(df, f"{upstream_name} motif activation")
            motif_effects[normalized] = _row(df, f"{upstream_name} motif effect")
            bp_contribution_per_motif[normalized] = _row(
                df, f"{upstream_name} Basepair contribution score to transcription initiation"
            )
            bp_contribution_to_motif_activation[normalized] = _row(
                df, f"{upstream_name} Basepair contribution score to motif activation"
            )

        results.append(
            {
                "prediction": _row(df, "Prediction"),
                "motif_activations": motif_activations,
                "motif_effects": motif_effects,
                "sum_motif_effects": _row(df, "Sum of motif effect"),
                "sum_initiator_effects": _row(df, "Sum of initiator effect"),
                "sum_trinucleotide_effects": _row(df, "Sum of trinucleotide effect"),
                "sum_total_effects": _row(df, "Sum of total effect"),
                "bp_contribution": _row(df, "Basepair contribution score to transcription initiation"),
                "bp_contribution_per_motif": bp_contribution_per_motif,
                "bp_contribution_to_motif_activation": bp_contribution_to_motif_activation,
            }
        )

    return {"results": results}


def _channel_indices(model: Any, track_names: list[str]) -> list[int]:
    """Map Puffin's named tracks onto model output channels using upstream's own maps.

    A trailing ``+``/``-`` selects the forward or reverse-strand head, which upstream keeps in
    separate dictionaries keyed by the bare signal name.

    Args:
        model: The loaded upstream ``Puffin`` module.
        track_names: Track names in the order the tool reports channels.

    Returns:
        list[int]: One output-channel index per name, in the same order.

    Raises:
        ValueError: If a name has no strand suffix or is absent from upstream's maps.
    """
    indices: list[int] = []
    for name in track_names:
        if name.endswith("+"):
            table, key = model.targeti_map, name[:-1]
        elif name.endswith("-"):
            table, key = model.targeti_rev_map, name[:-1]
        else:
            raise ValueError(f"puffin: track {name!r} must end with '+' or '-' to select a strand")
        if key not in table:
            raise ValueError(f"puffin: track {name!r} is not in upstream's map; have {sorted(table)}")
        indices.append(int(table[key]))
    return indices


def _encode_fixed_dna(sequence: str, device: str) -> Any:
    """One-hot encode fixed DNA exactly as upstream ``Puffin.predict`` encodes its input string.

    Args:
        sequence: DNA string; bases other than ``A``/``C``/``G``/``T`` encode as 0.25 per column.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor: ``(len(sequence), 4)`` float32 encoding in ``A,C,G,T`` order.
    """
    import torch
    from selene_sdk import sequences  # type: ignore[import-not-found]

    encoding = sequences.sequence_to_encoding(sequence, base_to_index=PUFFIN_BASE_TO_INDEX, bases_arr=PUFFIN_BASES)
    return torch.as_tensor(encoding, dtype=torch.float32, device=device)


def _model_input(relaxed: Any, left: Any, right: Any) -> Any:
    """Place fixed context around one relaxed candidate in upstream's channels-first layout.

    Args:
        relaxed: ``(L, 4)`` relaxed candidate, the only part that carries gradient.
        left: ``(L_left, 4)`` encoded left context, or ``None`` for none.
        right: ``(L_right, 4)`` encoded right context, or ``None`` for none.

    Returns:
        torch.Tensor: ``(1, 4, L_left + L + L_right)`` model input.
    """
    import torch

    parts = [part for part in (left, relaxed, right) if part is not None]
    stacked = torch.cat(parts, dim=0) if len(parts) > 1 else relaxed
    return stacked.t().unsqueeze(0)


def _compute_gradient(state: PuffinModel, input_dict: dict[str, Any]) -> dict[str, Any]:
    """Weighted track-and-position objective gradients for relaxed Puffin inputs.

    Each item is relaxed and placed between the fixed ``context_left``/``context_right``, which
    are encoded as upstream encodes a string and never relaxed, so gradients cover the item's
    rows only. The model's ``predict`` is not reusable here because it takes a string, detaches,
    and returns a DataFrame, so this runs the module's tensor forward directly and trims the
    same padding. Term positions index the output window of the full model input.

    Args:
        state: The lazy-loaded model holder.
        input_dict: Dispatch payload with items, loss terms, track names, and relaxation settings.

    Returns:
        dict[str, Any]: One result per item plus the shared vocabulary.

    Raises:
        ValueError: If a matrix is malformed or a term's position span leaves the output window.
        RuntimeError: If an expected gradient is missing after the backward pass.
    """
    import torch

    # Read the relaxation settings, loss terms, and fixed flanking context from the payload
    device = input_dict["device"]
    verbose = bool(input_dict.get("verbose", False))
    backprop = bool(input_dict.get("compute_gradient", True))
    temperature = resolve_temperature(input_dict["temperature"])
    soft = float(input_dict.get("soft", 1.0))
    hard = float(input_dict.get("hard", 0.0))
    gradient_terms = input_dict["gradient_terms"]
    context_left = input_dict["context_left"]
    context_right = input_dict["context_right"]

    # Load the model, map each track name to its output channel, and one-hot encode the context
    model = state.load(device, verbose=verbose)
    channels = _channel_indices(model, list(input_dict["track_names"]))
    track_channel = dict(zip(input_dict["track_names"], channels, strict=True))
    left = _encode_fixed_dna(context_left, state.device) if context_left else None
    right = _encode_fixed_dna(context_right, state.device) if context_right else None

    results: list[dict[str, Any]] = []
    for item in input_dict["items"]:
        # Check the item is L x 4 logits and that context + item leaves a non-empty output window
        if not item or any(len(row) != 4 for row in item):
            raise ValueError("puffin: compute_gradient expects L x 4 logits in A,C,G,T order")
        length = len(context_left) + len(item) + len(context_right)
        output_length = length - 2 * PUFFIN_PADDING
        if output_length < 1:
            raise ValueError(
                f"puffin: compute_gradient needs at least {2 * PUFFIN_PADDING + 1} bp of context + item, got {length}"
            )

        with torch.set_grad_enabled(backprop):
            # Relax the logits, run the forward pass, and trim the padding off the predicted tracks
            seq_dist = torch.tensor(item, device=state.device, dtype=torch.float32, requires_grad=backprop)
            relaxed = relaxed_sequence(seq_dist, temperature=temperature, soft=soft, hard=hard)
            prediction = model(_model_input(relaxed, left, right))
            scored = prediction[0, :, PUFFIN_PADDING:-PUFFIN_PADDING]

            # One backward per track, sharing one forward graph retained until the last.
            terms: list[dict[str, Any]] = []
            for position, term in enumerate(gradient_terms):
                # Resolve open span endpoints to the full window and average the track over the span
                name = term["track"]
                start = 1 if term["position_start"] is None else int(term["position_start"])
                end = output_length if term["position_end"] is None else int(term["position_end"])
                if start > output_length or end > output_length:
                    raise ValueError(
                        f"puffin: term span {start}..{end} leaves the {output_length}-position output window"
                    )
                signal = scored[track_channel[name], start - 1 : end].float().mean()

                gradient_value: list[list[float]] | None = None
                # Keep the graph alive for every term except the last
                if backprop:
                    retain = position < len(gradient_terms) - 1
                    grad = torch.autograd.grad(signal, seq_dist, retain_graph=retain, allow_unused=False)[0]
                    gradient_value = grad.detach().cpu().tolist()

                # Report the span actually averaged, resolving open endpoints for each candidate.
                terms.append(
                    {
                        "term": {"track": name, "position_start": start, "position_end": end},
                        "value": float(signal.item()),
                        "gradient": gradient_value,
                    }
                )

        results.append({"terms": terms, "metrics": {"output_length": output_length}})

    return {"results": results, "vocab": ["A", "C", "G", "T"]}


# ============================================================================
# Dispatch
# ============================================================================
_state: PuffinModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _state
    set_torch_seed(input_dict.get("seed"))
    if _state is None:
        _state = PuffinModel()

    operation = input_dict["operation"]
    if operation == "predict":
        return _predict(_state, input_dict)
    if operation == "interpret":
        return _interpret(_state, input_dict)
    if operation == "compute_gradient":
        return _compute_gradient(_state, input_dict)
    raise ValueError(f"puffin: unknown operation {operation!r}; valid: ['predict', 'interpret', 'compute_gradient']")


def to_device(device: str) -> dict[str, Any]:
    """Move loaded model to a specified device (called by DeviceManager)."""
    global _state
    if _state is not None and _state.model is not None:
        _state.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    global _state
    device = _state.device if _state is not None else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("puffin: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
