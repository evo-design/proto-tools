"""Local Evo1 inference implementation.

Uses the ``evo-model`` pip package (``from evo import Evo``) and delegates
generation to ``evo.generation.generate()``.

Usage (called by ToolInstance, not directly):
    python inference.py <input.json> <output.json>
"""

import json
import logging
import sys
from typing import Any, Literal

import torch
from standalone_helpers import move_model_to_device, set_torch_seed
from tqdm import tqdm

logger = logging.getLogger(__name__)

EVO1_MODEL_CHECKPOINTS = Literal[
    "evo-1-8k-base",
    "evo-1-131k-base",
    "evo-1-8k-crispr",
    "evo-1-8k-transposon",
]


class Evo1Model:
    """Evo1 model implementation using the evo-model pip package.

    Example:
        >>> model = Evo1Model("evo-1-8k-base")
        >>> out = model.sample(prompts=["ATCG"], num_tokens=100)
        >>> out["sequences"]
    """

    def __init__(
        self,
        model_name: EVO1_MODEL_CHECKPOINTS = "evo-1-8k-base",
        device: str = "cuda",
    ):
        """Initialize Evo1Model."""
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self.model: Any = None
        self.tokenizer: Any = None

    def sample(
        self,
        prompts: list[str],
        num_tokens: int = 100,
        top_k: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        batch_size: int = 1,
        verbose: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample DNA sequences autoregressively from prompts.

        Delegates to ``evo.generation.generate()`` which supports batched
        generation with cached recurrent state.

        Args:
            prompts: DNA prompt sequences.
            num_tokens: Number of tokens to generate per prompt.
            top_k: Top-k sampling parameter.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            verbose: Whether to print progress.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with keys "sequences" (List[str]) and "scores" (List[float]).
        """
        set_torch_seed(seed)

        if not self._loaded:
            self.load(self.device, verbose=verbose)

        from evo.generation import generate as evo_generate

        all_sequences: list[str] = []
        all_scores: list[float] = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            sequences, scores = evo_generate(
                prompt_seqs=batch,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=num_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cached_generation=True,
                batched=True,
                prepend_bos=False,
                device=self.device,
                verbose=1 if verbose else 0,
            )
            all_sequences.extend(sequences)
            all_scores.extend(scores)

        assert len(all_sequences) == len(prompts)
        # Convert numpy floats to native Python floats for JSON serialization
        return {
            "sequences": all_sequences,
            "scores": [float(s) for s in all_scores],
        }

    def score(
        self,
        sequences: list[str],
        batch_size: int = 1,
        return_logits: bool = False,
        verbose: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Score DNA sequences by computing autoregressive log-likelihood.

        Uses ``evo.scoring.prepare_batch`` and ``evo.scoring.logits_to_logprobs``
        which handle BOS prepending/trimming internally.

        Args:
            sequences: DNA sequences to score.
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_logits: Whether to include per-position logits in the output.
            verbose: Whether to print progress.
            seed: Random seed. Scoring is deterministic given the model state,
                but we still seed RNGs/cudnn flags for consistency with sampling
                so consecutive calls in a persistent worker behave identically
                regardless of call order.

        Returns:
            Dictionary with keys:
                - ``"logits"``: List of per-sequence tensors (seq_len, vocab_size)
                  if return_logits=True, else None.
                - ``"metrics"``: List of dicts with ``log_likelihood``,
                  ``avg_log_likelihood``, ``perplexity`` per sequence.
                - ``"vocab"``: List of 512 byte-level token characters.
        """
        if not self._loaded:
            self.load(self.device, verbose=verbose)

        set_torch_seed(seed)

        if not sequences:
            raise ValueError("Cannot score empty sequence list")

        from evo.scoring import logits_to_logprobs, prepare_batch

        # Evo1 uses same byte-level vocab as Evo2 (CharLevelTokenizer, 512 tokens)
        vocab_size = 512
        vocab = [chr(j) for j in range(vocab_size)]

        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]

        all_logits = []
        all_metrics = []

        with torch.inference_mode():
            for batch_idx, batch_seqs in enumerate(tqdm(batches, desc="Evo1 Sequence Scoring", unit="batch")):
                if verbose:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_seqs)} sequences)")

                seq_lengths = [len(s) for s in batch_seqs]

                # prepare_batch prepends BOS and pads to max length
                input_ids, _ = prepare_batch(
                    batch_seqs,
                    self.tokenizer,
                    prepend_bos=True,
                    device=self.device,
                )

                # Forward pass: (batch, length, vocab_size)
                logits, _ = self.model(input_ids)

                if verbose:
                    logger.info(f"Scored batch of {input_ids.shape[0]}, logits shape: {logits.shape}")

                # logits_to_logprobs handles BOS trim and shift-gather
                # Returns (batch, seq_len) per-position log-probs
                logprobs = logits_to_logprobs(logits, input_ids, trim_bos=True)

                for i, length in enumerate(seq_lengths):
                    seq_logprobs = logprobs[i, :length].float()
                    all_metrics.append(
                        {
                            "log_likelihood": seq_logprobs.sum().item(),
                            "avg_log_likelihood": seq_logprobs.mean().item(),
                            "perplexity": torch.exp(-seq_logprobs.mean()).item(),
                        }
                    )
                    # Store full logits (unshifted) for the sequence positions
                    # logits has BOS-prepended length, so positions 0..length-1
                    # correspond to predictions for positions 1..length
                    # We return logits for the original sequence length
                    all_logits.append(logits[i, :length, :].cpu())

        return {
            "logits": all_logits if return_logits else None,
            "metrics": all_metrics,
            "vocab": vocab,
        }

    def load(self, device: str = "cuda", verbose: bool = False) -> Any:
        """Load Evo1 model and tokenizer."""
        if verbose:
            logger.info(f"Loading Evo1: {self.model_name} on {device}")

        # The evo library's load_checkpoint() calls snapshot_download() without
        # any file filters, which downloads ALL formats in the HF repo: safetensors
        # shards (~14 GB), pytorch .bin shards (~10 GB), and a monolithic
        # pytorch_model.pt (~16 GB), totaling ~40 GB for a 7B model. Only the
        # safetensors files are actually used (load_checkpoint reads them via
        # safetensors.torch.load_file). The redundant .bin/.pt files waste disk
        # space and, on slow filesystems like Oak/Lustre, can cause warm-up
        # timeouts during the download phase. We monkey-patch snapshot_download
        # to ignore these formats.
        import huggingface_hub

        _orig_snapshot_download = huggingface_hub.snapshot_download

        def _filtered_snapshot_download(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("ignore_patterns", ["*.bin", "*.pt"])
            return _orig_snapshot_download(*args, **kwargs)

        huggingface_hub.snapshot_download = _filtered_snapshot_download

        from evo import Evo

        evo_obj = Evo(self.model_name)
        huggingface_hub.snapshot_download = _orig_snapshot_download
        self.model = evo_obj.model
        self.tokenizer = evo_obj.tokenizer

        self.model = self.model.to(device).eval()
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Evo1 model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model. Call load() first.")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info("Unloading Evo1 from GPU")
            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# Dispatch entry point
# =============================================================================

_model: Evo1Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = Evo1Model(
            model_name=input_dict["model_name"],
            device=input_dict["device"],
        )

    operation = input_dict["operation"]

    if operation == "sample":
        return _model.sample(
            prompts=input_dict["prompts"],
            num_tokens=input_dict["num_tokens"],
            top_k=input_dict["top_k"],
            temperature=input_dict["temperature"],
            top_p=input_dict["top_p"],
            batch_size=input_dict["batch_size"],
            verbose=input_dict["verbose"],
            seed=input_dict["seed"],
        )
    if operation == "score":
        result = _model.score(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            return_logits=input_dict["return_logits"],
            verbose=input_dict["verbose"],
            seed=input_dict["seed"],
        )
        if result["logits"] is not None:
            result["logits"] = [t.tolist() for t in result["logits"]]
        return result
    raise ValueError(f"Unknown operation: {operation}")


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
        raise ValueError(f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
