"""Borzoi standalone inference implementation for venv execution."""

import json
import sys
from typing import Any, cast

import torch
from standalone_helpers import get_logger, serialize_output, set_torch_seed
from standalone_helpers.relaxation import relaxed_sequence, resolve_temperature

logger = get_logger(__name__)

BORZOI_CONTEXT = 524_288
BORZOI_OUTPUT = 6_144


class BorzoiModel:
    """Borzoi model for regulatory activity prediction.

    Supports human and mouse models with optional FlashAttention.
    """

    def __init__(
        self,
        species: str = "human",
        replicate: str = "0",
        use_flash_attn: bool = True,
    ):
        """Initialize Borzoi model wrapper.

        Args:
            species: Species to predict for ('human' or 'mouse')
            replicate: Replicate to use (e.g., '0', '1', '2', '3')
            use_flash_attn: Whether to use FlashAttention
        """
        if species == "mouse" and use_flash_attn:
            raise ValueError(
                "borzoi: FlashAttention (flashzoi) is not available for mouse checkpoints — set use_flash_attn=False"
            )

        self.species = species
        self.replicate = replicate
        self.use_flash_attn = use_flash_attn
        self._loaded = False
        self.model: Any = None
        self.device: str | None = None

        # Determine model name
        prefix = "flashzoi" if use_flash_attn else "borzoi"
        self.model_name = f"johahi/{prefix}-replicate-{replicate}"
        if species == "mouse":
            self.model_name += "-mouse"

    def load(self, device: str, verbose: bool = False) -> None:
        """Load Borzoi model to device."""
        from borzoi_pytorch import Borzoi

        if verbose:
            logger.info(f"Loading Borzoi model: {self.model_name} on {device}")

        try:
            self.model = Borzoi.from_pretrained(self.model_name).to(device)
        except OSError as e:
            raise RuntimeError(f"borzoi: HF weight load from {self.model_name!r} failed: {e}") from e
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Borzoi model loaded successfully")

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU and clear CUDA cache."""
        if self._loaded:
            if verbose:
                logger.info(f"Moving model from {self.device} to CPU")
            self.to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reload_to_device(self, model: Any, old_device: str, new_device: str) -> Any:  # noqa: ARG002 — required by device transition callback signature
        """Custom move function: reload the model onto the target GPU.

        Borzoi uses flash-attn/Triton kernels that cannot run on CPU, so
        CPU→GPU transitions require a full reload rather than a simple .to().
        """
        self.load(new_device)
        return self.model

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        GPU→GPU and GPU→CPU use standard .to(). CPU→GPU requires a full
        reload because flash-attn/Triton kernels need fresh GPU initialization.
        """
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("borzoi: cannot move unloaded model to device — call load() first")

        if self.device == device:
            return

        if self.device == "cpu" and device.startswith("cuda"):
            # CPU→GPU: reload needed for flash-attn/Triton
            self.model = move_model_to_device(
                self.model,
                self.device,
                device,
                custom_move_fn=self._reload_to_device,
            )
        else:
            # GPU→GPU or GPU→CPU: standard .to() works
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def _predict_sequences(
        self,
        sequences: list[str],
        output_tracks: list[int],
        avg_output_tracks: bool = True,
        device: str = "cuda",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run Borzoi inference on a batch of DNA sequences.

        Args:
            sequences: DNA sequences to score. Each sequence must be
                BORZOI_CONTEXT=524,288 bp.
            output_tracks: Track indices to extract from the model output.
            avg_output_tracks: Whether to average selected tracks into one
                output track per sequence.
            device: Device to run inference on.
            verbose: Whether to log status messages.

        Returns:
            Prediction tensor with shape ``[batch, num_tracks, BORZOI_OUTPUT]``.
            If ``avg_output_tracks`` is true, ``num_tracks`` is 1.
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        # Prepare input
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq_len = len(sequences[0])
        onehot = torch.zeros(len(sequences), 4, seq_len, device=device)
        for batch_idx, sequence in enumerate(sequences):
            if len(sequence) != seq_len:
                raise ValueError(
                    f"borzoi: all batch sequences must have the same length; "
                    f"sequence {batch_idx} has length {len(sequence)} != {seq_len}"
                )
            positions: list[int] = []
            bases: list[int] = []
            for pos, char in enumerate(sequence):
                base_idx = mapping.get(char)
                if base_idx is None:
                    continue
                positions.append(pos)
                bases.append(base_idx)
            if positions:
                onehot[
                    batch_idx,
                    torch.tensor(bases, dtype=torch.long, device=device),
                    torch.tensor(positions, dtype=torch.long, device=device),
                ] = 1

        # Run prediction with autocast
        with torch.autocast(device_type=torch.device(device).type), torch.inference_mode():
            output = self.model(onehot, is_human=(self.species == "human"))

        # Process output - always return 2D (num_tracks, positions)
        if avg_output_tracks:
            prediction = output[:, output_tracks, :].mean(1, keepdim=True)
        else:
            prediction = output[:, output_tracks, :]

        return cast(torch.Tensor, prediction)

    def compute_gradient(
        self,
        sequence: str,
        positions: list[int],
        logits_list: list[list[float]],
        *,
        gradient_terms: list[dict[str, Any]],
        temperature: float | None = None,
        soft: float = 1.0,
        hard: float = 0.0,
        backprop: bool = True,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Weighted track-and-bin objective gradient for relaxed positions in a Borzoi context.

        The context is one-hot encoded, the relaxed distribution is written into the designed
        positions, and the objective is a weighted sum of mean track activity over each term's
        bin span. Only the designed positions carry gradient; the rest of the context is fixed.

        Args:
            sequence: Full ``BORZOI_CONTEXT`` bp context window.
            positions: 1-indexed inclusive context positions the relaxed rows map to.
            logits_list: Relaxed state, shape ``(len(positions), 4)`` in ``A,C,G,T`` order.
            gradient_terms: Quantities to differentiate, each with track, bin_start, bin_end.
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
            ValueError: If the context length, positions, or rows do not line up.
            RuntimeError: If an expected gradient is missing after the backward pass.
        """
        if len(sequence) != BORZOI_CONTEXT:
            raise ValueError(f"borzoi: compute_gradient expects a {BORZOI_CONTEXT} bp context, got {len(sequence)}")
        if len(positions) != len(logits_list):
            raise ValueError("borzoi: compute_gradient needs one relaxed row per designed position")
        if any(len(row) != 4 for row in logits_list):
            raise ValueError("borzoi: compute_gradient expects L x 4 logits in A,C,G,T order")

        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        with torch.set_grad_enabled(backprop):
            seq_dist = torch.tensor(logits_list, device=self.device, dtype=torch.float32, requires_grad=backprop)
            relaxed = relaxed_sequence(seq_dist, temperature=temperature, soft=soft, hard=hard)

            onehot = torch.zeros(1, 4, BORZOI_CONTEXT, device=self.device)
            fixed_positions = [index for index, char in enumerate(sequence) if char in mapping]
            if fixed_positions:
                onehot[
                    0,
                    torch.tensor([mapping[sequence[i]] for i in fixed_positions], dtype=torch.long, device=self.device),
                    torch.tensor(fixed_positions, dtype=torch.long, device=self.device),
                ] = 1.0

            # Designed columns are replaced wholesale, so any placeholder base there is discarded.
            designed = torch.tensor([p - 1 for p in positions], dtype=torch.long, device=self.device)
            onehot = onehot.index_copy(2, designed, relaxed.t().unsqueeze(0))

            # No inference_mode here: the prediction path uses it, which would block autograd.
            with torch.autocast(device_type=torch.device(self.device).type):
                output = self.model(onehot, is_human=(self.species == "human"))

            # One backward per track, sharing one forward graph retained until the last.
            terms: list[dict[str, Any]] = []
            for position, term in enumerate(gradient_terms):
                track = int(term["track"])
                start = int(term["bin_start"]) - 1
                end = int(term["bin_end"])
                if track >= output.shape[1]:
                    raise ValueError(f"borzoi: track {track} is out of range for {output.shape[1]} output tracks")
                activity = output[0, track, start:end].float().mean()

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
_model: BorzoiModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    set_torch_seed(input_dict["seed"])

    species = input_dict["species"]
    replicate = input_dict["replicate"]
    use_flash_attn = input_dict["use_flash_attn"]

    # Auto-detect flash-attn availability
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning("flash-attn not installed, falling back to use_flash_attn=False (standard borzoi model)")
            use_flash_attn = False

    # Hold one checkpoint at a time and swap weights in-process when the requested replicate/species
    # changes (the four borzoi replicate models don't co-reside in GPU memory). This avoids
    # restarting the whole worker process per replicate — the churn that triggered the GPU-teardown
    # wedge and dominated runtime for borzoi-ensemble.
    if (
        _model is None
        or _model.species != species
        or _model.replicate != replicate
        or _model.use_flash_attn != use_flash_attn
    ):
        if _model is not None and _model._loaded:
            _model.unload()  # free the old checkpoint's GPU memory before loading the next
        _model = BorzoiModel(
            species=species,
            replicate=replicate,
            use_flash_attn=use_flash_attn,
        )

    operation = input_dict["operation"]
    if operation == "predict":
        sequences = input_dict["sequences"]
        batch_size = max(1, int(input_dict.get("batch_size", len(sequences))))
        predictions = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            predictions.append(
                _model._predict_sequences(
                    sequences=chunk,
                    output_tracks=input_dict["output_tracks"],
                    avg_output_tracks=input_dict["avg_output_tracks"],
                    device=input_dict["device"],
                    verbose=input_dict["verbose"],
                )
                .detach()
                .cpu()
            )
        return {
            "predictions": torch.cat(predictions, dim=0),
            "applied_species": _model.species,
            "applied_replicate": _model.replicate,
        }
    if operation == "compute_gradient":
        # One chunk of independent candidates: the weights load once and each is evaluated in turn.
        results = [
            _model.compute_gradient(
                sequence=item["sequence"],
                positions=item["positions"],
                logits_list=item["logits"],
                gradient_terms=input_dict["gradient_terms"],
                temperature=resolve_temperature(input_dict["temperature"]),
                soft=float(input_dict.get("soft", 1.0)),
                hard=float(input_dict.get("hard", 0.0)),
                backprop=input_dict.get("compute_gradient", True),
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
            for item in input_dict["items"]
        ]
        return {
            "results": [{key: result[key] for key in ("terms", "metrics")} for result in results],
            "vocab": results[0]["vocab"],
        }

    raise ValueError(f"borzoi: unknown operation {operation!r}; valid: ['predict', 'compute_gradient']")


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
        raise ValueError("borzoi: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
