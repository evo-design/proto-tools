"""Standalone inference entry point for Pangolin tissue-specific splice prediction."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger, move_model_to_device, serialize_output

logger = get_logger(__name__)

_STANDALONE_DIR = Path(__file__).resolve().parent
if str(_STANDALONE_DIR) not in sys.path:
    # Needed for persistent-worker imports where script dir is not auto-added.
    sys.path.insert(0, str(_STANDALONE_DIR))

# Pangolin crops 5000 bp from each side; in sync with shared_data_models.PANGOLIN_FLANK.
FLANK = 5000

# Tissue -> (checkpoint index, P(splice) output channel), per the upstream CLI.
# In sync with shared_data_models.TISSUE_SPEC (standalones cannot import proto_tools).
TISSUE_SPEC: dict[str, tuple[int, int]] = {
    "HEART": (0, 1),
    "LIVER": (2, 4),
    "BRAIN": (4, 7),
    "TESTIS": (6, 10),
}


class PangolinModel:
    """Manages Pangolin ensemble loading and inference per tissue."""

    def __init__(self) -> None:
        """Initialize PangolinModel."""
        self.device: str | None = None
        # tissue -> (list of 3 loaded models, output channel)
        self.models: dict[str, tuple[list[Any], int]] = {}
        self._loaded = False

    # ============================================================================
    # Model Loading & Device Management
    # ============================================================================
    def _models_dir(self) -> str:
        """Locate the bundled Pangolin model weights directory."""
        import pangolin

        return os.path.join(os.path.dirname(pangolin.__file__), "models")

    def load(self, device: str, tissues: list[str]) -> None:
        """Load the 3-model ensemble for each requested tissue onto device."""
        import torch
        from pangolin.model import AR, L, Pangolin, W

        logger.debug("Loading Pangolin ensembles for tissues=%s on %s", tissues, device)

        models_dir = self._models_dir()
        loaded: dict[str, tuple[list[Any], int]] = {}
        for tissue in tissues:
            i, channel = TISSUE_SPEC[tissue]
            ensemble: list[Any] = []
            for j in (1, 2, 3):
                model = Pangolin(L, W, AR).to(device)
                # weights_only=False matches the other standalones (state_dict pickles).
                checkpoint = os.path.join(models_dir, f"final.{j}.{i}.3.v2")
                weights = torch.load(checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(weights)
                ensemble.append(model.eval())
            loaded[tissue] = (ensemble, channel)

        self.models = loaded
        self.device = device
        self._loaded = True
        logger.debug("Pangolin models loaded successfully")

    def _ensure_loaded(self, device: str, tissues: list[str]) -> None:
        """Load or reload the ensemble if device or tissue set changed."""
        if not self._loaded or self.device != device or set(self.models) != set(tissues):
            self.load(device, tissues)

    def _one_hot(self, seq: str, strand: str) -> Any:
        """One-hot encode a DNA sequence, reverse-complementing on the '-' strand."""
        import numpy as np

        IN_MAP = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        seq = seq.upper().replace("A", "1").replace("C", "2").replace("G", "3").replace("T", "4").replace("N", "0")
        if strand == "+":
            arr = np.asarray(list(map(int, list(seq))))
        else:
            arr = np.asarray(list(map(int, list(seq[::-1]))))
            arr = (5 - arr) % 5
        return IN_MAP[arr.astype("int8")]  # (len, 4)

    def _run_model(self, model: Any, onehot: Any, channel: int) -> Any:
        """Run a single model and return one output channel (the model crops 5000 bp/side internally)."""
        import numpy as np
        import torch

        tensor = torch.from_numpy(np.expand_dims(onehot.T, 0)).float().to(self.device)
        with torch.no_grad():
            out = model(tensor)
        return out[0][channel, :].cpu().numpy()

    # ============================================================================
    # Operations
    # ============================================================================
    def predict(self, sequences: list[str], tissues: list[str], device: str) -> list[dict[str, Any]]:
        """Predict per-position splice-site probability for each sequence across tissues."""
        import numpy as np

        self._ensure_loaded(device, tissues)

        results: list[dict[str, Any]] = []
        for seq in sequences:
            onehot = self._one_hot(seq, "+")
            per_tissue: list[Any] = []
            for tissue in tissues:
                ensemble, channel = self.models[tissue]
                stacked = np.stack([self._run_model(m, onehot, channel) for m in ensemble])
                per_tissue.append(stacked.mean(axis=0))  # (len - 10000,)
            # (n_tissues, L') -> (L', n_tissues), column order matches `tissues`.
            scores = np.stack(per_tissue, axis=1)
            results.append({"scores": scores.tolist()})
        return results

    def _compute_score(self, ref_seq: str, alt_seq: str, strand: str, d: int, tissues: list[str]) -> tuple[Any, Any]:
        """Port of upstream compute_score: per-position loss/gain across tissues."""
        import numpy as np

        ref_onehot = self._one_hot(ref_seq, strand)
        alt_onehot = self._one_hot(alt_seq, strand)
        win_len = 2 * d + 1

        per_tissue: list[Any] = []
        for tissue in tissues:
            ensemble, channel = self.models[tissue]
            deltas: list[Any] = []
            for model in ensemble:
                ref = self._run_model(model, ref_onehot, channel)
                alt = self._run_model(model, alt_onehot, channel)
                if strand == "-":
                    ref = ref[::-1]
                    alt = alt[::-1]
                ndiff = abs(len(ref) - len(alt))
                if len(ref) > len(alt):
                    alt = np.concatenate([alt[0 : win_len // 2 + 1], np.zeros(ndiff), alt[win_len // 2 + 1 :]])
                elif len(ref) < len(alt):
                    alt = np.concatenate(
                        [
                            alt[0 : win_len // 2],
                            np.max(alt[win_len // 2 : win_len // 2 + ndiff + 1], keepdims=True),
                            alt[win_len // 2 + ndiff + 1 :],
                        ]
                    )
                deltas.append(alt - ref)
            per_tissue.append(np.mean(deltas, axis=0))

        stacked = np.stack(per_tissue)  # (n_tissues, L')
        loss = stacked.min(axis=0)
        gain = stacked.max(axis=0)
        return loss, gain

    def score_variants(
        self, variants: list[dict[str, Any]], tissues: list[str], distance: int, device: str
    ) -> list[dict[str, Any]]:
        """Score the splice gain/loss of each variant over its reporting window."""
        import numpy as np

        self._ensure_loaded(device, tissues)

        results: list[dict[str, Any]] = []
        for variant in variants:
            vp = variant["variant_position"]
            ref = variant["reference_bases"]
            alt = variant["alternate_bases"]
            strand = variant["strand"]
            seq = variant["sequence"]
            ref_len = len(ref)

            # Clip the reporting window to the available flank (validator guarantees >= FLANK).
            eff_d = min(distance, vp - FLANK, len(seq) - (vp + ref_len) - FLANK)
            if eff_d < distance:
                logger.debug("variant at %d: window clipped to +/-%d (requested +/-%d) by flank", vp, eff_d, distance)

            lo = vp - FLANK - eff_d
            hi = vp + ref_len + FLANK + eff_d
            window = seq[lo:hi]
            off = FLANK + eff_d
            alt_window = window[:off] + alt + window[off + ref_len :]

            loss, gain = self._compute_score(window, alt_window, strand, eff_d, tissues)
            gain_idx = int(np.argmax(gain))
            loss_idx = int(np.argmin(loss))
            results.append(
                {
                    "loss_scores": loss.tolist(),
                    "gain_scores": gain.tolist(),
                    "increase_position": gain_idx - eff_d,
                    "increase_score": float(gain[gain_idx]),
                    "decrease_position": loss_idx - eff_d,
                    "decrease_score": float(loss[loss_idx]),
                }
            )
        return results

    def to_device(self, device: str) -> None:
        """Move all loaded models to a different device."""
        if not self._loaded:
            raise ValueError("pangolin: cannot move unloaded model to device — call load() first")
        if self.device == device:
            return
        for tissue, (ensemble, channel) in self.models.items():
            moved = [move_model_to_device(model, self.device, device) for model in ensemble]
            self.models[tissue] = (moved, channel)
        self.device = device


# ============================================================================
# Dispatch
# ============================================================================
_model: PangolinModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = PangolinModel()

    operation = input_dict["operation"]
    if operation == "predict":
        results = _model.predict(
            sequences=input_dict["sequences"],
            tissues=input_dict["tissues"],
            device=input_dict["device"],
        )
        return {"results": results}
    if operation == "score_variants":
        results = _model.score_variants(
            variants=input_dict["variants"],
            tissues=input_dict["tissues"],
            distance=input_dict["distance"],
            device=input_dict["device"],
        )
        return {"results": results}
    raise ValueError(f"pangolin: unknown operation {operation!r}; valid: ['predict', 'score_variants']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    # Model not loaded yet - will use device on next call
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("pangolin: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
