"""Enformer standalone inference implementation for venv execution."""

import json
import logging
import sys
from typing import Any, cast

import torch
from standalone_helpers import StatusCounter, get_logger, serialize_output, set_torch_seed
from standalone_helpers.relaxation import relaxed_sequence, resolve_temperature

logger = get_logger(__name__)

ENFORMER_CONTEXT = 196_608
ENFORMER_OUTPUT = 896


class EnformerModel:
    """Enformer model for regulatory activity and gene expression prediction.

    Uses the EleutherAI/enformer-official-rough model for human and mouse predictions.
    """

    def __init__(self) -> None:
        """Initialize Enformer model wrapper."""
        self._loaded = False
        self.model: Any = None
        self.device: str | None = None

    def load(self, device: str, verbose: bool = False) -> None:
        """Load Enformer model to device."""
        from enformer_pytorch import from_pretrained

        logger.update_status("Loading Enformer weights")
        if verbose:
            logger.info(f"Loading Enformer model on {device}")

        repo = "EleutherAI/enformer-official-rough"
        try:
            self.model = from_pretrained(repo).to(device)
        except OSError as e:
            raise RuntimeError(f"enformer: HF weight load from {repo!r} failed: {e}") from e
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Enformer model loaded successfully")

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU and clear CUDA cache."""
        if self._loaded:
            if verbose:
                logger.info(f"Moving model from {self.device} to CPU")
            self.to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("enformer: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def _predict_sequences(
        self,
        sequences: list[str],
        output_tracks: list[int],
        species: str,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run Enformer inference on a batch of DNA sequences.

        Args:
            sequences: DNA sequences to score. Each sequence must be
                ENFORMER_CONTEXT=196,608 bp.
            output_tracks: Track indices to extract from the selected species
                output head.
            species: Species output head to use, either ``"human"`` or
                ``"mouse"``.
            device: Device to run inference on.
            verbose: Whether to log status messages.

        Returns:
            Prediction tensor with shape ``[batch, ENFORMER_OUTPUT, num_tracks]``.
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        # Prepare input (Enformer handles 'N' as index 4)
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        input_ids = torch.tensor([[mapping.get(char, 4) for char in sequence] for sequence in sequences]).to(device)

        # Run prediction
        with torch.inference_mode():
            output = self.model(input_ids)

        # Extract prediction for specified tracks and species
        return cast(torch.Tensor, output[species][:, :, output_tracks])

    def compute_gradient(
        self,
        sequence: str,
        positions: list[int],
        logits_list: list[list[float]],
        *,
        gradient_terms: list[dict[str, Any]],
        species: str,
        temperature: float | None = None,
        soft: float = 1.0,
        hard: float = 0.0,
        backprop: bool = True,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Per-track activity and gradient for relaxed positions in an Enformer context.

        Enformer's forward accepts a float one-hot of shape ``(batch, length, 4)`` directly, so the
        relaxed distribution is written into the designed rows and differentiated through without
        an embedding hook. Only those positions carry gradients. The rest of the context is fixed.

        Args:
            sequence: Full ``ENFORMER_CONTEXT`` bp context window.
            positions: 1-indexed inclusive context positions the relaxed rows map to.
            logits_list: Relaxed state, shape ``(len(positions), 4)`` in ``A,C,G,T`` order.
            gradient_terms: Quantities to differentiate, each with track, bin_start, bin_end.
            species: Species output head to read tracks from.
            temperature: Optional softmax temperature; ``None`` uses the rows as probabilities.
            soft: Blend from the raw input matrix (0) to softmax probabilities (1).
            hard: Straight-through coefficient; 1 forwards a hard one-hot.
            backprop: If False, skip backward and return ``gradient=None``.
            device: Device to run on.
            verbose: Whether to log status messages.

        Returns:
            dict[str, Any]: One entry per requested quantity with its raw mean activity and
                gradient (or ``None``), plus per-candidate metrics and the vocab.

        Raises:
            ValueError: If the context length, positions, rows, or a track index do not line up.
            RuntimeError: If an expected gradient is missing after the backward pass.
        """
        if len(sequence) != ENFORMER_CONTEXT:
            raise ValueError(f"enformer: compute_gradient expects a {ENFORMER_CONTEXT} bp context, got {len(sequence)}")
        if len(positions) != len(logits_list):
            raise ValueError("enformer: compute_gradient needs one relaxed row per designed position")
        if any(len(row) != 4 for row in logits_list):
            raise ValueError("enformer: compute_gradient expects L x 4 logits in A,C,G,T order")

        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        with torch.set_grad_enabled(backprop):
            seq_dist = torch.tensor(logits_list, device=self.device, dtype=torch.float32, requires_grad=backprop)
            relaxed = relaxed_sequence(seq_dist, temperature=temperature, soft=soft, hard=hard)

            # Channels-last (batch, length, 4), which is the layout Enformer's forward expects.
            onehot = torch.zeros(1, ENFORMER_CONTEXT, 4, device=self.device)
            fixed = [index for index, char in enumerate(sequence) if char in mapping]
            if fixed:
                onehot[
                    0,
                    torch.tensor(fixed, dtype=torch.long, device=self.device),
                    torch.tensor([mapping[sequence[i]] for i in fixed], dtype=torch.long, device=self.device),
                ] = 1.0

            # Designed rows are replaced wholesale, so any placeholder base there is discarded.
            designed = torch.tensor([p - 1 for p in positions], dtype=torch.long, device=self.device)
            onehot = onehot.index_copy(1, designed, relaxed.unsqueeze(0))

            # Keep autograd enabled here so the model can differentiate the relaxed sequence.
            output = self.model(onehot)
            if species not in output:
                raise ValueError(f"enformer: unknown species head {species!r}; have {sorted(output)}")
            tracks = output[species]

            # One backward per track, sharing one forward graph retained until the last.
            terms: list[dict[str, Any]] = []
            for position, term in enumerate(gradient_terms):
                track = int(term["track"])
                start = int(term["bin_start"]) - 1
                end = int(term["bin_end"])
                if track >= tracks.shape[-1]:
                    raise ValueError(f"enformer: track {track} is out of range for {tracks.shape[-1]} tracks")
                activity = tracks[0, start:end, track].float().mean()

                gradient_value: list[list[float]] | None = None
                if backprop:
                    retain = position < len(gradient_terms) - 1
                    grad = torch.autograd.grad(activity, seq_dist, retain_graph=retain, allow_unused=False)[0]
                    gradient_value = grad.detach().cpu().tolist()

                terms.append(
                    {
                        "term": {"track": track, "bin_start": int(term["bin_start"]), "bin_end": int(term["bin_end"])},
                        "value": float(activity.item()),
                        "gradient": gradient_value,
                    }
                )

        return {
            "terms": terms,
            "metrics": {"designed_positions": len(positions)},
            "vocab": ["A", "C", "G", "T"],
        }


# ============================================================================
# Dispatch
# ============================================================================
_model: EnformerModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    set_torch_seed(input_dict["seed"])
    if _model is None:
        _model = EnformerModel()

    operation = input_dict["operation"]
    if operation == "predict" and input_dict["sequences"] and not _model._loaded:
        # Load before the progress statuses below, so the loading status doesn't replace them mid-run.
        _model.load(input_dict["device"], verbose=input_dict["verbose"])
    if operation == "predict":
        # Split the sequences into forward-pass chunks.
        sequences = input_dict["sequences"]
        batch_size = max(1, int(input_dict.get("batch_size", len(sequences))))
        predictions = []
        chunks = [sequences[start : start + batch_size] for start in range(0, len(sequences), batch_size)]
        if len(chunks) == 1:
            logger.update_status(
                f"Predicting {len(sequences)} sequence{'s' if len(sequences) != 1 else ''}", level=logging.DEBUG
            )
        # Predict each chunk and move the tracks off the GPU before the next one.
        counter = StatusCounter(logger, "Predicting batch %d/%d", total=len(chunks))
        for number, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                counter(number)
            predictions.append(
                _model._predict_sequences(
                    sequences=chunk,
                    output_tracks=input_dict["output_tracks"],
                    species=input_dict["species"],
                    device=input_dict["device"],
                    verbose=input_dict["verbose"],
                )
                .detach()
                .cpu()
            )
        # Stack the chunk predictions back into one tensor in input order.
        return {
            "predictions": torch.cat(predictions, dim=0),
            "applied_species": input_dict["species"],
        }
    if operation == "compute_gradient":
        # Load the weights once, then evaluate each independent candidate in this chunk.
        results = [
            _model.compute_gradient(
                sequence=item["sequence"],
                positions=item["positions"],
                logits_list=item["logits"],
                gradient_terms=input_dict["gradient_terms"],
                species=input_dict["species"],
                temperature=resolve_temperature(input_dict["temperature"]),
                soft=float(input_dict.get("soft", 1.0)),
                hard=float(input_dict.get("hard", 0.0)),
                backprop=input_dict.get("compute_gradient", True),
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
            for item in input_dict["items"]
        ]
        # Return per-candidate terms and metrics with the shared vocabulary listed once.
        return {
            "results": [{key: result[key] for key in ("terms", "metrics")} for result in results],
            "vocab": results[0]["vocab"],
        }

    raise ValueError(f"enformer: unknown operation {operation!r}; valid: ['predict', 'compute_gradient']")


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
        raise ValueError("enformer: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
