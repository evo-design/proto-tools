"""
Local Evo2 inference implementation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from standalone_helpers import move_model_to_device
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress vortex library INFO logs
logging.getLogger('vortex').setLevel(logging.ERROR)
logging.getLogger('StripedHyena').setLevel(logging.ERROR)


# Evo2 uses 1 as padding token: https://github.com/Zymrael/vortex/blob/3e2511427794d02f46e464bc34a8895c9b911e76/vortex/model/tokenizer.py#L140
EVO2_PAD_TOKEN_ID = 1

EVO2_MODEL_CHECKPOINTS = Literal[
    "evo2_7b",
    "evo2_20b",
    "evo2_40b",
    "evo2_7b_base",
    "evo2_40b_base",
    "evo2_1b_base",
    "evo2_7b_262k",
    "evo2_7b_microviridae",
]

# Evo2 uses a byte-level vocabulary (CharLevelTokenizer from Vortex). Vocab size: 512.
# Token id j maps to chr(j) for interpretation. Special tokens: 0=eos, 1=pad. DNA nucleotides (ASCII): 'A'=65, 'C'=67, 'G'=71, 'T'=84, 'N'=78.
# logits[t, j] = score for token j; character is EVO2_VOCAB[j].
EVO2_VOCAB_SIZE = 512
EVO2_VOCAB: List[str] = [chr(j) for j in range(EVO2_VOCAB_SIZE)]


class Evo2Model:
    """
    Evo2 model implementation.
    Example:
        >>> model = Evo2Model("evo2_7b")
        >>>
        >>> # Basic generation
        >>> out = model.sample(prompts=["ATCG"], num_tokens=100)
        >>> out["sequences"]
        >>>
        >>> # With caching for beam search
        >>> out1 = model.sample(
        ...     prompts=["ATCG"],
        ...     num_tokens=100,
        ...     cached_generation=True,  # vortex internal caching
        ... )
        >>> out2 = model.sample(
        ...     prompts=out1["sequences"],
        ...     num_tokens=100,
        ...     old_kv_cache=out1["kv_caches"][0],  # continue from cache
        ... )
    """

    def __init__(self, model_checkpoint: EVO2_MODEL_CHECKPOINTS = "evo2_7b", local_path: Optional[str] = None):
        """
        Initialize Evo2 model wrapper.

        Args:
            model_checkpoint: Evo2 checkpoint to use
            local_path: Local path to the Evo2 model
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.local_path = local_path
        self.tokenizer = None
        self.device = None
        self.model = None

    def sample(
        self,
        # input arguments
        prompts: List[str],

        # vortex model arguments
        top_k: int = 4,
        top_p: float = 1,
        temperature: float = 1.0,

        # vortex generation arguments
        device: str = "cuda",
        num_tokens: int = 32,
        cached_generation: bool = True,
        force_prompt_threshold: Optional[int] = None,
        max_seqlen: Optional[int] = None,
        print_generation: bool = True,
        verbose: bool = False,
        stop_at_eos: bool = True,
        old_kv_cache: Optional[Dict] = None,
        batch_size: int = 1,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample DNA sequences using vortex generation.

        Args:
            prompts: DNA prompt sequences (same length if using cached_inference_params_dict)
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Sampling temperature
            device: Device to run on
            num_tokens: Number of tokens to generate
            cached_generation: Whether to use vortex KV caching for generation
            force_prompt_threshold: Number of tokens to prefill in parallel before
                switching to prompt forcing. Used to reduce peak memory usage and
                support longer prompts.
            max_seqlen: Maximum sequence length to generate. Determines the max size
                of the cache if larger. Otherwise automatically determined using
                prompt length + max_tokens.
            print_generation: Whether to print generation tokens
            verbose: Whether to print verbose output
            stop_at_eos: Whether to stop at end-of-sequence token
            old_kv_cache: Dictionary of inference parameters to use for replaying cached sampling (KV cache)
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_logits: Whether to include logits in the output
        Returns:
            Dictionary with keys: "sequences" (List[str]), optionally "logits" (List[torch.Tensor]), "kv_caches" (Optional[List[Dict]])
        """
        from vortex.model.generation import Generator as VortexGenerator

        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        if isinstance(prompts, str):
            prompts = [prompts]

        # Create vortex inference generator
        gen = VortexGenerator(
            self.model.model,
            self.tokenizer,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Batch processing logic
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        all_sequences, all_logits, all_inference_params_dicts = [], [], []

        for batch_idx in tqdm(range(num_batches), desc="Evo2 Sequence Generation", unit="batch", total=num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            if verbose:
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} prompts)")

            # Prepare batch - always use batched generation when uniform lengths
            # Note: Evo2/vortex doesn't use attention_mask for generation, so we only need input_ids
            uniform_lengths = all(len(s) == len(batch_prompts[0]) for s in batch_prompts)
            if uniform_lengths:
                input_ids, _ = self._prepare_batch(batch_prompts)
                input_ids_list = [input_ids]
            else:
                if verbose:
                    logger.warning("Prompts have different lengths, processing individually.")
                input_ids_list = [self._prepare_batch([prompt])[0] for prompt in batch_prompts]

            # Generate - loop over input_ids_list
            for input_ids in input_ids_list:
                current_batch_size = input_ids.shape[0]

                # Vortex generator internally calls torch.inference_mode()
                output_ids, logits, new_kv_cache = gen.generate(
                    device=self.device,
                    input_ids=input_ids,
                    num_tokens=num_tokens,
                    cached_generation=cached_generation,
                    force_prompt_threshold=force_prompt_threshold,
                    max_seqlen=max_seqlen,
                    print_generation=print_generation,
                    verbose=verbose,
                    stop_at_eos=stop_at_eos,
                    inference_params_dict=old_kv_cache,
                )

                if verbose:
                    logger.info(f'input_ids.shape: {input_ids.shape}')
                    logger.info(f'output_ids.shape: {output_ids.shape}')
                    logger.info(f'logits.shape: {logits.shape}')

                # Detokenize batch
                batch_sequences = list(self.tokenizer.detokenize_batch(output_ids))
                assert len(batch_sequences) == current_batch_size

                # Collect sequences, logits, and inference params dicts
                all_sequences.extend(batch_sequences)
                all_logits.extend([logits[i] for i in range(logits.shape[0])])
                if new_kv_cache:
                    all_inference_params_dicts.extend(_split_cache(new_kv_cache))
                else:
                    all_inference_params_dicts.extend([None] * current_batch_size) # no cache generated

        assert len(prompts) == len(all_sequences) == len(all_logits)
        return {
            "sequences": all_sequences,
            "logits": all_logits if return_logits else None,
            "kv_caches": all_inference_params_dicts,
        }

    def _prepare_batch(
        self,
        sequences: List[str],
        pad_left: bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Tokenize and pad sequences into a batch."""
        if not sequences:
            raise ValueError("Cannot prepare empty batch")

        # Tokenize sequences and convert to tensors
        tokens = [self.tokenizer.tokenize(seq) for seq in sequences]
        if isinstance(tokens[0], list):
            tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

        lengths = [len(t) for t in tokens]
        max_len = max(lengths)

        # Create padded input_ids
        input_ids = torch.full((len(sequences), max_len), EVO2_PAD_TOKEN_ID, dtype=torch.long)
        for i, (t, length) in enumerate(zip(tokens, lengths)):
            if pad_left:
                input_ids[i, max_len - length:] = t
            else:
                input_ids[i, :length] = t

        return input_ids.to(self.device), lengths

    def score(
        self,
        sequences: List[str],
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Score DNA sequences by computing logits and metrics via forward pass.

        Sequences are batched with padding and attention masking. Metrics are
        computed only over non-padded tokens.

        Args:
            sequences: DNA sequences to score
            device: Device to run on
            verbose: Whether to print status messages
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_logits: Whether to include logits in the output

        Returns a dict with optional logits (per-sequence tensors), metrics, and vocab tokens."""
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        if not sequences:
            raise ValueError("Cannot score empty sequence list")

        # Batch processing logic
        all_logits = []
        all_metrics = []
        batches = [
            sequences[i:i + batch_size]
            for i in range(0, len(sequences), batch_size)
        ]

        with torch.inference_mode():
            for batch_idx, batch_sequences in enumerate(tqdm(batches, desc="Evo2 Sequence Scoring", unit="batch", total=len(batches))):
                if verbose:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_sequences)} sequences)")

                input_ids, lengths = self._prepare_batch(batch_sequences, pad_left=False)

                output = self.model.model(input_ids)
                # Model returns tuple (logits,)
                logits = output[0] if isinstance(output, tuple) else output

                if verbose:
                    logger.info(f"Scored batch of {input_ids.shape[0]}, logits shape: {logits.shape}")

                # Shift for autoregressive scoring: predict position i from positions 0..i-1
                shifted_logits = logits[:, :-1, :].float() # [batch, seq_len - 1, vocab_size]
                shifted_targets = input_ids[:, 1:] # [batch, seq_len - 1]

                # Compute log probabilities
                log_probs = torch.log_softmax(shifted_logits, dim=-1) # [batch, seq_len - 1, vocab_size]
                batch_log_probs = log_probs.gather(2, shifted_targets.unsqueeze(2)).squeeze(2) # [batch, seq_len - 1]

                for i, length in enumerate(lengths):
                    seq_log_probs = batch_log_probs[i, :length - 1] # [seq_len - 1]
                    all_metrics.append({
                        "log_likelihood": seq_log_probs.sum().item(),
                        "avg_log_likelihood": seq_log_probs.mean().item(),
                        "perplexity": torch.exp(-seq_log_probs.mean()).item(),
                    })
                    all_logits.append(logits[i, :length, :].cpu())

        return {
            "logits": all_logits if return_logits else None,
            "metrics": all_metrics,
            "vocab": EVO2_VOCAB,
        }

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False):
        """Load Evo2 model and tokenizer to device.

        Restricts CUDA_VISIBLE_DEVICES before loading so that vortex's
        auto-sharding (``torch.cuda.device_count()``) only sees the
        allocated GPU(s).
        """
        if verbose:
            logger.info(f"Loading Evo2: {self.model_checkpoint} on {device}")

        # Vortex auto-shards across all visible GPUs via torch.cuda.device_count().
        # Restrict visibility to the allocated device(s) before model construction.
        from standalone_helpers import get_subprocess_device_env
        restricted_env = get_subprocess_device_env(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = restricted_env["CUDA_VISIBLE_DEVICES"]
        logger.info(
            f"Restricted CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']} "
            f"for device {device}"
        )

        from evo2 import Evo2
        _cleanup_vortex_debug_log()
        self.model = Evo2(model_name=self.model_checkpoint, local_path=self.local_path)
        self.tokenizer = self.model.tokenizer
        self.model.model = self.model.model.eval()

        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Evo2 model loaded successfully")

    def _reload_to_device(self, model, old_device: str, new_device: str):
        """Custom move function: reload the model onto the target GPU.

        Vortex auto-shards based on CUDA_VISIBLE_DEVICES at construction
        time, so moving back to GPU requires a full reload rather than
        a simple ``.to()`` call.  ``load()`` updates CUDA_VISIBLE_DEVICES
        before constructing the model.
        """
        self.load(new_device)
        return self.model.model

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        GPU → CPU uses standard PyTorch ``.to("cpu")``.
        CPU → GPU (or GPU → GPU) fully reloads the model so that vortex
        auto-shards onto the correct CUDA_VISIBLE_DEVICES.
        """
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device == device:
            return

        if device == "cpu":
            # Standard offload — no reload needed
            self.model.model = move_model_to_device(
                self.model.model, self.device, device,
            )
            self.device = device
        else:
            # Moving to GPU requires full reload (vortex auto-sharding)
            self.model.model = move_model_to_device(
                self.model.model, self.device, device,
                custom_move_fn=self._reload_to_device,
            )

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")
            self.model.model = self.model.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def _slice_cache(cache: Dict, start: int, end: int) -> Dict:
    """Slice a batched cache from index start to end.

    Args:
        cache: Batched cache dictionary with 'mha', 'hcl', 'hcm', 'hcs' components
        start: Start index (inclusive)
        end: End index (exclusive)

    Returns:
        Sliced cache dictionary for samples [start:end]
    """
    from vortex.model.cache import (
        HyenaCascadeFIRInferenceParams,
        HyenaCascadeIIRInferenceParams,
        InferenceParams,
    )

    mha, hcl, hcm, hcs = cache['mha'], cache['hcl'], cache['hcm'], cache['hcs']

    return {
        'mha': InferenceParams(
            max_seqlen=mha.max_seqlen,
            max_batch_size=mha.max_batch_size,
            seqlen_offset=mha.seqlen_offset,
            batch_size_offset=mha.batch_size_offset,
            key_value_memory_dict={
                key: data[start:end]
                for key, data in mha.key_value_memory_dict.items()
            },
        ),
        'hcl': HyenaCascadeIIRInferenceParams(
            fir_filter_length=hcl.fir_filter_length,
            state_dim=hcl.state_dim,
            seqlen_offset=hcl.seqlen_offset,
            fir_state_dict={
                key: data[start:end]
                for key, data in hcl.fir_state_dict.items()
            },
            state_dict={
                key: data[start:end]
                for key, data in hcl.state_dict.items()
            },
        ),
        'hcm': HyenaCascadeFIRInferenceParams(
            fir_filter_length=hcm.fir_filter_length,
            seqlen_offset=hcm.seqlen_offset,
            fir_inner_filter_length=hcm.fir_inner_filter_length,
            fir_state_dict={
                key: data[start:end]
                for key, data in hcm.fir_state_dict.items()
            },
            fir_inner_state_dict={
                key: data[start:end]
                for key, data in hcm.fir_inner_state_dict.items()
            },
            state_dict={
                key: data[start:end]
                for key, data in hcm.state_dict.items()
            },
        ),
        'hcs': HyenaCascadeFIRInferenceParams(
            fir_filter_length=hcs.fir_filter_length,
            seqlen_offset=hcs.seqlen_offset,
            fir_inner_filter_length=hcs.fir_inner_filter_length,
            fir_state_dict={
                key: data[start:end]
                for key, data in hcs.fir_state_dict.items()
            },
            fir_inner_state_dict={
                key: data[start:end]
                for key, data in hcs.fir_inner_state_dict.items()
            },
            state_dict={
                key: data[start:end]
                for key, data in hcs.state_dict.items()
            },
        ),
    }


def _split_cache(cache: Dict) -> List[Dict]:
    """Split batched cache into per-sample caches.

    Args:
        cache: Batched cache dictionary with 'mha', 'hcl', 'hcm', 'hcs' components

    Returns:
        List of individual cache dictionaries, one per sample in the batch
    """
    kv = next(iter(cache['mha'].key_value_memory_dict.values()))
    n_samples = kv.shape[0]
    return [_slice_cache(cache, i, i + 1) for i in range(n_samples)]


def _serialize_output(value: Any) -> Any:
    """Recursively serialize torch tensors and other non-JSON types to Python primitives."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _serialize_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_output(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# ============================================================================
# Dispatch
# ============================================================================
_model: Evo2Model | None = None


def dispatch(input_dict: dict) -> dict:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = Evo2Model(
            model_checkpoint=input_dict.get("model_checkpoint", "evo2_7b"),
            local_path=input_dict.get("local_path"),
        )

    operation = input_dict.get("operation", "sample")
    if operation == "sample":
        result = _model.sample(
            prompts=input_dict.get("prompts", []),
            top_k=input_dict.get("top_k", 4),
            top_p=input_dict.get("top_p", 1.0),
            temperature=input_dict.get("temperature", 1.0),
            device=input_dict.get("device", "cuda"),
            num_tokens=input_dict.get("num_tokens", 32),
            cached_generation=input_dict.get("cached_generation", True),
            force_prompt_threshold=input_dict.get("force_prompt_threshold"),
            max_seqlen=input_dict.get("max_seqlen"),
            print_generation=input_dict.get("print_generation", True),
            verbose=input_dict.get("verbose", False),
            stop_at_eos=input_dict.get("stop_at_eos", True),
            old_kv_cache=None,  # KV caching not supported in venv mode
            batch_size=input_dict.get("batch_size"),
            return_logits=input_dict.get("return_logits", False),
        )
        # KV caches are vortex GPU objects, not JSON-serializable
        result["kv_caches"] = None
        return result
    elif operation == "score":
        return _model.score(
            sequences=input_dict.get("sequences", []),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
            batch_size=input_dict.get("batch_size"),
            return_logits=input_dict.get("return_logits", False),
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")


def _cleanup_vortex_debug_log():
    """Remove activations_debug.log created by vortex.logging at import time.

    The vortex library unconditionally creates a FileHandler for
    'activations_debug.log' in the CWD when vortex.logging is imported.
    Remove the handler and delete the empty file.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith("activations_debug.log"):
            root.removeHandler(handler)
            handler.close()
    debug_log = Path("activations_debug.log")
    if debug_log.exists():
        debug_log.unlink()


def to_device(device: str) -> dict:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    else:
        # Model not loaded yet - will use device on next call
        return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(_serialize_output(result), f)
