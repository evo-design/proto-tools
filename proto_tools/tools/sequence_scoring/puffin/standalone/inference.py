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
    get_logger,
    get_pytorch_memory_stats,
    move_model_to_device,
    resolve_weights_dir,
    serialize_output,
    set_torch_seed,
)

logger = get_logger(__name__)

PUFFIN_PADDING = 325
PUFFIN_OUTPUT_CHANNELS = 10

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
        if verbose:
            logger.info(f"Loading Puffin on {device}")
        from puffin import Puffin  # type: ignore[import-not-found]

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
    for sequence in sequences:
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
    for sequence in sequences:
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
    raise ValueError(f"puffin: unknown operation {operation!r}; valid: ['predict', 'interpret']")


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
