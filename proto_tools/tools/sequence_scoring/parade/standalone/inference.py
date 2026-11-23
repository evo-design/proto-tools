"""PARADE standalone inference implementation for venv execution."""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import parade_scoring
from standalone_helpers import (
    download_weights,
    get_logger,
    move_model_to_device,
    require_weights_dir,
    serialize_output,
    set_torch_seed,
)
from standalone_helpers.relaxation import resolve_temperature

logger = get_logger(__name__)


# Bundled checkpoint files, which setup.sh downloads (pinned and checksummed) into the weights dir,
# by what each operation scores. Keys match the host-side ``PARADE_BUNDLED_MODELS``.
_BUNDLED_CHECKPOINTS: dict[str, dict[str, str]] = {
    "parade": {
        "utr5": "parade-model-utr5-deltas-epoch9-step840.ckpt",
        "utr3": "parade-model-utr3-deltas-epoch9-step1330.ckpt",
        "stability": "parade-stability-epoch24-step725.ckpt",
    },
}


def _resolve_checkpoint(input_dict: dict[str, Any]) -> str:
    """Resolve the checkpoint to load: a bundled one for the operation, an https link, or a local path.

    The host validates ``model_checkpoint`` and normalizes a link to ``https://``, so anything that
    is neither a bundled name nor a link is a path.
    """
    stability = input_dict["operation"] in ("stability", "stability_gradient")
    target = "stability" if stability else input_dict.get("construct_type")
    if target not in _BUNDLED_CHECKPOINTS["parade"]:
        raise ValueError("parade: construct_type must be one of ['utr3', 'utr5']")
    requested = input_dict["model_checkpoint"]
    weights_dir = Path(require_weights_dir("parade"))
    if requested in _BUNDLED_CHECKPOINTS:
        bundled = weights_dir / _BUNDLED_CHECKPOINTS[requested][target]
        if not bundled.is_file():
            raise FileNotFoundError(f"parade: bundled checkpoint missing at {bundled}; re-run setup.sh")
        return str(bundled)
    if requested.startswith("https://"):
        # ``download_weights`` refuses non-HTTPS URLs and redirects (the checkpoint is unpickled),
        # sends ``HF_TOKEN`` only to Hugging Face, and redacts the URL in logs and errors
        dest = weights_dir / f"parade-{target}-custom-{hashlib.sha256(requested.encode()).hexdigest()[:12]}.ckpt"
        if not dest.is_file():
            logger.update_status("Downloading PARADE weights")
        return str(download_weights(requested, dest))
    local = Path(requested).expanduser()
    if not local.is_file():
        raise FileNotFoundError(f"parade: checkpoint path does not exist: {requested}")
    return str(local)


class ParadeModel:
    """Caches a loaded PARADE LegNet module and delegates inference to ``parade_scoring``."""

    def __init__(self) -> None:
        """Initialize an unloaded PARADE model wrapper."""
        self.model: Any = None
        self.device: str | None = None
        self.checkpoint_key: str | None = None

    def load(self, *, checkpoint_path: str, device: str, verbose: bool = False) -> None:
        """Load a PARADE checkpoint (once per checkpoint) and move it to the device."""
        if self.model is not None and self.checkpoint_key == checkpoint_path:
            self.to_device(device)
            return
        logger.update_status("Loading PARADE weights")
        if verbose:
            logger.info("Loading PARADE checkpoint from %s", checkpoint_path)
        self.model = parade_scoring.load_legnet(checkpoint_path)
        self.checkpoint_key = checkpoint_path
        self.device = "cpu"
        self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move the cached LegNet module to a device."""
        if self.model is None or self.device == device:
            return
        previous_device = self.device or "cpu"
        self.model = move_model_to_device(self.model, previous_device, device)
        self.device = device

    def score_activity(
        self, *, sequences: list[str], construct_type: str, cell_types: list[str], batch_size: int, device: str
    ) -> list[dict[str, float]]:
        """Predict per-cell-type activity for each sequence."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(device)
        return parade_scoring.predict_activity(  # type: ignore[no-any-return]
            self.model,
            sequences=sequences,
            construct_type=construct_type,
            cell_types=cell_types,
            batch_size=batch_size,
            device=device,
        )

    def score_stability(self, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
        """Predict the RNA/gDNA log-ratio for each 3' UTR sequence."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(device)
        return parade_scoring.predict_stability(  # type: ignore[no-any-return]
            self.model, sequences=sequences, batch_size=batch_size, device=device
        )

    def compute_gradient(self, **kwargs: Any) -> dict[str, Any]:
        """Compute a differentiable PARADE activity loss and optional gradient."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(kwargs["device"])
        return parade_scoring.compute_gradient(self.model, **kwargs)  # type: ignore[no-any-return]

    def compute_stability_gradient(self, **kwargs: Any) -> dict[str, Any]:
        """Compute a differentiable PARADE stability loss and optional gradient."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(kwargs["device"])
        return parade_scoring.compute_stability_gradient(self.model, **kwargs)  # type: ignore[no-any-return]


_model = ParadeModel()


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    set_torch_seed(input_dict.get("seed"))
    operation = input_dict["operation"]
    valid_operations = {"activity", "stability", "gradient", "stability_gradient"}
    if operation not in valid_operations:
        raise ValueError(f"parade: unknown operation {operation!r}; valid: {sorted(valid_operations)}")

    checkpoint_path = _resolve_checkpoint(input_dict)
    _model.load(
        checkpoint_path=checkpoint_path,
        device=input_dict["device"],
        verbose=bool(input_dict.get("verbose", False)),
    )

    if operation == "stability_gradient":
        return _model.compute_stability_gradient(
            logits_list=input_dict["logits"],
            temperature=resolve_temperature(input_dict["temperature"]),
            soft=float(input_dict.get("soft", 1.0)),
            hard=float(input_dict.get("hard", 0.0)),
            compute_gradient=bool(input_dict.get("compute_gradient", True)),
            device=input_dict["device"],
        )

    if operation == "gradient":
        return _model.compute_gradient(
            logits_list=input_dict["logits"],
            temperature=resolve_temperature(input_dict["temperature"]),
            construct_type=input_dict["construct_type"],
            gradient_terms=input_dict["gradient_terms"],
            soft=float(input_dict.get("soft", 1.0)),
            hard=float(input_dict.get("hard", 0.0)),
            compute_gradient=bool(input_dict.get("compute_gradient", True)),
            device=input_dict["device"],
        )

    batch_size = max(1, int(input_dict.get("batch_size", 1)))
    count = len(input_dict["sequences"])
    logger.update_status(f"Scoring {count} sequence{'s' if count != 1 else ''}", level=logging.DEBUG)
    if operation == "activity":
        return {
            "scores": _model.score_activity(
                sequences=input_dict["sequences"],
                construct_type=input_dict["construct_type"],
                cell_types=input_dict["cell_types"],
                batch_size=batch_size,
                device=input_dict["device"],
            )
        }
    return {
        "log_ratios": _model.score_stability(
            sequences=input_dict["sequences"],
            batch_size=batch_size,
            device=input_dict["device"],
        )
    }


def to_device(device: str) -> dict[str, Any]:
    """Move the model to the specified device."""
    _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(_model.device or 0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("parade: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
