"""Local Evo2 inference implementation."""

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Literal

import torch
from standalone_helpers import (
    get_logger,
    log_likelihood_metrics,
    move_model_to_device,
    serialize_output,
    set_torch_seed,
)
from tqdm import tqdm

logger = get_logger(__name__)

# Suppress vortex library INFO logs
logging.getLogger("vortex").setLevel(logging.ERROR)
logging.getLogger("StripedHyena").setLevel(logging.ERROR)


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
EVO2_VOCAB: list[str] = [chr(j) for j in range(EVO2_VOCAB_SIZE)]


class Evo2Model:
    """Evo2 model implementation.

    Example:
        >>> model = Evo2Model("evo2_7b")
        >>>
        >>> # Basic generation
        >>> out = model.sample(prompts=["ATCG"], max_new_tokens=100)
        >>> out["sequences"]
        >>>
        >>> # With caching for beam search
        >>> out1 = model.sample(
        ...     prompts=["ATCG"],
        ...     max_new_tokens=100,
        ...     cached_generation=True,  # vortex internal caching
        ... )
        >>> out2 = model.sample(
        ...     prompts=out1["sequences"],
        ...     max_new_tokens=100,
        ...     old_kv_cache=out1["kv_caches"][0],  # continue from cache
        ... )
    """

    def __init__(self, model_checkpoint: EVO2_MODEL_CHECKPOINTS = "evo2_7b", local_path: str | None = None):
        """Initialize Evo2 model wrapper.

        Args:
            model_checkpoint: Evo2 checkpoint to use
            local_path: Local path to the Evo2 model
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.local_path = local_path
        self.tokenizer: Any = None
        self.device: str | None = None
        self.model: Any = None

    @property
    def _local_device(self) -> str:
        """Device to place tensors on, accounting for CUDA_VISIBLE_DEVICES isolation.

        ``load()`` pins the worker to one physical GPU via CUDA_VISIBLE_DEVICES,
        which torch re-indexes to ``cuda:0``. Tensors must target that, not the
        physical ``self.device`` (e.g. ``cuda:3``), which would be out of range.

        Returns:
            str: ``"cpu"`` when running on CPU, else ``"cuda:0"``.
        """
        return "cpu" if self.device == "cpu" else "cuda:0"

    def sample(
        self,
        # input arguments
        prompts: list[str],
        # vortex model arguments
        top_k: int = 4,
        top_p: float = 1,
        temperature: float = 1.0,
        # vortex generation arguments
        device: str = "cuda",
        max_new_tokens: int = 32,
        cached_generation: bool = True,
        force_prompt_threshold: int | None = None,
        max_seqlen: int | None = None,
        verbose: bool = False,
        skip_special_tokens: bool = False,
        stop_at_eos: bool = True,
        old_kv_cache: dict[str, Any] | None = None,
        batch_size: int = 1,
        return_kv_cache: bool = False,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample DNA sequences using vortex generation.

        Args:
            prompts: DNA prompt sequences (same length if using cached_inference_params_dict)
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Sampling temperature
            device: CUDA device to run inference on
            max_new_tokens: Maximum number of new tokens to generate
            cached_generation: Whether to use vortex KV caching for generation
            force_prompt_threshold: Number of tokens to prefill in parallel before
                switching to prompt forcing. Used to reduce peak memory usage and
                support longer prompts.
            max_seqlen: Maximum sequence length to generate. Determines the max size
                of the cache if larger. Otherwise automatically determined using
                prompt length + max_tokens.
            verbose: Whether to print verbose output
            skip_special_tokens: Whether to filter EOS/PAD bytes from the detokenized output
            stop_at_eos: Whether to stop at end-of-sequence token
            old_kv_cache: Resolved KV cache state to continue generation from
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_kv_cache: Whether to include KV caches in the output
            return_logits: Whether to include logits in the output
            seed: Random seed for reproducible generation.

        Returns:
            Dictionary with generated sequences, optional logits, and optional KV caches.
        """
        from vortex.model.generation import Generator as VortexGenerator

        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        # Seed after load so each dispatch enters sampling with the same RNG state.
        set_torch_seed(seed)

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
        old_cache_batch_size = _cache_batch_size(old_kv_cache) if old_kv_cache is not None else None
        if old_cache_batch_size is not None and old_cache_batch_size != len(prompts):
            raise ValueError(
                f"evo2: old_kv_cache batch size must match the number of prompts "
                f"({old_cache_batch_size} != {len(prompts)})"
            )

        for batch_idx in tqdm(range(num_batches), desc="Evo2 Sequence Generation", unit="batch", total=num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_old_kv_cache = (
                _slice_cache(old_kv_cache, batch_start, batch_end) if old_kv_cache is not None else None
            )

            if verbose:
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} prompts)")

            # Prepare batch - always use batched generation when uniform lengths
            # Note: Evo2/vortex doesn't use attention_mask for generation, so we only need input_ids
            uniform_lengths = all(len(s) == len(batch_prompts[0]) for s in batch_prompts)
            if uniform_lengths:
                input_ids, _ = self._prepare_batch(batch_prompts)
                input_batches = [(input_ids, batch_old_kv_cache)]
            else:
                if verbose:
                    logger.warning("Prompts have different lengths, processing individually.")
                input_batches = [
                    (
                        self._prepare_batch([prompt])[0],
                        _slice_cache(old_kv_cache, batch_start + offset, batch_start + offset + 1)
                        if old_kv_cache is not None
                        else None,
                    )
                    for offset, prompt in enumerate(batch_prompts)
                ]

            # Generate - loop over input_ids_list
            for input_ids, current_old_kv_cache in input_batches:
                current_batch_size = input_ids.shape[0]

                # Vortex generator internally calls torch.inference_mode()
                output_ids, logits, new_kv_cache = gen.generate(
                    device=self._local_device,
                    input_ids=input_ids,
                    num_tokens=max_new_tokens,
                    cached_generation=cached_generation,
                    force_prompt_threshold=force_prompt_threshold,
                    max_seqlen=max_seqlen,
                    print_generation=False,
                    verbose=verbose,
                    skip_special_tokens=skip_special_tokens,
                    stop_at_eos=stop_at_eos,
                    inference_params_dict=current_old_kv_cache,
                )

                if verbose:
                    logger.info(f"input_ids.shape: {input_ids.shape}")
                    logger.info(f"output_ids.shape: {output_ids.shape}")
                    logger.info(f"logits.shape: {logits.shape}")

                # Detokenize batch
                batch_sequences = list(self.tokenizer.detokenize_batch(output_ids))
                assert len(batch_sequences) == current_batch_size

                # Collect sequences, logits, and inference params dicts
                all_sequences.extend(batch_sequences)
                all_logits.extend([logits[i] for i in range(logits.shape[0])])
                if return_kv_cache and new_kv_cache:
                    all_inference_params_dicts.extend(_split_cache(new_kv_cache))
                elif return_kv_cache:
                    all_inference_params_dicts.extend([None] * current_batch_size)  # type: ignore[list-item]  # no cache generated

        assert len(prompts) == len(all_sequences) == len(all_logits)
        if return_kv_cache:
            assert len(prompts) == len(all_inference_params_dicts)
        return {
            "sequences": all_sequences,
            "logits": all_logits if return_logits else None,
            "kv_caches": all_inference_params_dicts if return_kv_cache else None,
        }

    def _prepare_batch(
        self,
        sequences: list[str],
        pad_left: bool = False,
    ) -> tuple[torch.Tensor, list[int]]:
        """Tokenize and pad sequences into a batch."""
        if not sequences:
            raise ValueError("evo2: cannot prepare empty batch")

        # Tokenize sequences and convert to tensors
        tokens = [self.tokenizer.tokenize(seq) for seq in sequences]
        if isinstance(tokens[0], list):
            tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

        lengths = [len(t) for t in tokens]
        max_len = max(lengths)

        # Create padded input_ids
        input_ids = torch.full((len(sequences), max_len), EVO2_PAD_TOKEN_ID, dtype=torch.long)
        for i, (t, length) in enumerate(zip(tokens, lengths, strict=False)):
            if pad_left:
                input_ids[i, max_len - length :] = t
            else:
                input_ids[i, :length] = t

        return input_ids.to(self._local_device), lengths

    def score(
        self,
        sequences: list[str],
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,
        return_logits: bool = False,
        prepend_bos: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Score DNA sequences by computing logits and metrics via forward pass.

        Sequences are batched with padding and attention masking. Metrics are
        computed only over non-padded tokens.

        Args:
            sequences: DNA sequences to score
            device: CUDA device to run inference on
            verbose: Whether to print status messages
            batch_size: Number of sequences per GPU forward pass. Larger batches
                are faster but use more memory.
            return_logits: Whether to include logits in the output
            prepend_bos: Prepend a beginning-of-sequence token before scoring.
            seed: Random seed. Scoring is deterministic given the model state,
                but we still seed RNGs/cudnn flags so consecutive calls in a
                persistent worker behave identically regardless of call order.

        Returns a dict with optional logits (per-sequence tensors), metrics, and vocab tokens.
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        set_torch_seed(seed)

        if not sequences:
            raise ValueError("evo2: cannot score empty sequence list")

        # chr(0) tokenizes to vortex's eod_id (BOS).
        scoring_sequences = [chr(0) + seq for seq in sequences] if prepend_bos else sequences

        # Batch processing logic
        all_logits = []
        all_metrics = []
        batches = [scoring_sequences[i : i + batch_size] for i in range(0, len(scoring_sequences), batch_size)]

        with torch.inference_mode():
            for batch_idx, batch_sequences in enumerate(
                tqdm(batches, desc="Evo2 Sequence Scoring", unit="batch", total=len(batches))
            ):
                if verbose:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_sequences)} sequences)")

                input_ids, lengths = self._prepare_batch(batch_sequences, pad_left=False)

                output = self.model.model(input_ids)
                # Model returns tuple (logits,)
                logits = output[0] if isinstance(output, tuple) else output

                if verbose:
                    logger.info(f"Scored batch of {input_ids.shape[0]}, logits shape: {logits.shape}")

                # Shift for autoregressive scoring: predict position i from positions 0..i-1
                shifted_logits = logits[:, :-1, :].float()  # [batch, seq_len - 1, vocab_size]
                shifted_targets = input_ids[:, 1:]  # [batch, seq_len - 1]

                # Compute log probabilities
                log_probs = torch.log_softmax(shifted_logits, dim=-1)  # [batch, seq_len - 1, vocab_size]
                batch_log_probs = log_probs.gather(2, shifted_targets.unsqueeze(2)).squeeze(2)  # [batch, seq_len - 1]

                for i, length in enumerate(lengths):
                    seq_log_probs = batch_log_probs[i, : length - 1]  # [seq_len - 1]
                    all_metrics.append(log_likelihood_metrics(seq_log_probs.mean().item(), seq_log_probs.shape[0]))
                    all_logits.append(logits[i, :length, :].cpu())

        return {
            "logits": all_logits if return_logits else None,
            "metrics": all_metrics,
            "vocab": EVO2_VOCAB,
        }

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
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
        logger.debug(f"Restricted CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']} for device {device}")

        from evo2 import Evo2

        _cleanup_vortex_debug_log()
        self.model = Evo2(model_name=self.model_checkpoint, local_path=self.local_path)
        self.tokenizer = self.model.tokenizer
        self.model.model = self.model.model.eval()

        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Evo2 model loaded successfully")

    def _reload_to_device(self, model: Any, old_device: str, new_device: str) -> Any:  # noqa: ARG002 — required by device transition callback signature
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
            raise ValueError("evo2: cannot move unloaded model to device — call load() first")

        if self.device == device:
            return

        if device == "cpu":
            # Standard offload, no reload needed
            self.model.model = move_model_to_device(
                self.model.model,
                self.device,
                device,
            )
            self.device = device
        else:
            # Moving to GPU requires full reload (vortex auto-sharding)
            self.model.model = move_model_to_device(
                self.model.model,
                self.device,
                device,
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


def _slice_cache(cache: dict[str, Any], start: int, end: int) -> dict[str, Any]:
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

    mha, hcl, hcm, hcs = cache["mha"], cache["hcl"], cache["hcm"], cache["hcs"]

    return {
        "mha": InferenceParams(
            max_seqlen=mha.max_seqlen,
            max_batch_size=mha.max_batch_size,
            seqlen_offset=mha.seqlen_offset,
            batch_size_offset=mha.batch_size_offset,
            key_value_memory_dict={key: data[start:end] for key, data in mha.key_value_memory_dict.items()},
        ),
        "hcl": HyenaCascadeIIRInferenceParams(
            fir_filter_length=hcl.fir_filter_length,
            state_dim=hcl.state_dim,
            seqlen_offset=hcl.seqlen_offset,
            fir_state_dict={key: data[start:end] for key, data in hcl.fir_state_dict.items()},
            state_dict={key: data[start:end] for key, data in hcl.state_dict.items()},
        ),
        "hcm": HyenaCascadeFIRInferenceParams(
            fir_filter_length=hcm.fir_filter_length,
            seqlen_offset=hcm.seqlen_offset,
            fir_inner_filter_length=hcm.fir_inner_filter_length,
            fir_state_dict={key: data[start:end] for key, data in hcm.fir_state_dict.items()},
            fir_inner_state_dict={key: data[start:end] for key, data in hcm.fir_inner_state_dict.items()},
            state_dict={key: data[start:end] for key, data in hcm.state_dict.items()},
        ),
        "hcs": HyenaCascadeFIRInferenceParams(
            fir_filter_length=hcs.fir_filter_length,
            seqlen_offset=hcs.seqlen_offset,
            fir_inner_filter_length=hcs.fir_inner_filter_length,
            fir_state_dict={key: data[start:end] for key, data in hcs.fir_state_dict.items()},
            fir_inner_state_dict={key: data[start:end] for key, data in hcs.fir_inner_state_dict.items()},
            state_dict={key: data[start:end] for key, data in hcs.state_dict.items()},
        ),
    }


def _split_cache(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Split batched cache into per-sample caches.

    Args:
        cache: Batched cache dictionary with 'mha', 'hcl', 'hcm', 'hcs' components

    Returns:
        List of individual cache dictionaries, one per sample in the batch
    """
    kv = next(iter(cache["mha"].key_value_memory_dict.values()))
    n_samples = kv.shape[0]
    return [_slice_cache(cache, i, i + 1) for i in range(n_samples)]


def _cache_batch_size(cache: dict[str, Any]) -> int:
    kv = next(iter(cache["mha"].key_value_memory_dict.values()))
    return int(kv.shape[0])


def _repeat_cache_dict(cache_dict: dict[Any, torch.Tensor], n_replicates: int) -> dict[Any, torch.Tensor]:
    return {key: data.repeat((n_replicates,) + (1,) * (data.ndim - 1)) for key, data in cache_dict.items()}


def _replicate_cache(cache: dict[str, Any], n_replicates: int) -> dict[str, Any]:
    """Replicate a single-sequence cache for beam branching."""
    from vortex.model.cache import (
        HyenaCascadeFIRInferenceParams,
        HyenaCascadeIIRInferenceParams,
        InferenceParams,
    )

    if n_replicates < 1:
        raise ValueError(f"evo2: n_replicates must be at least 1 (got {n_replicates})")
    batch_size = _cache_batch_size(cache)
    if batch_size != 1:
        raise ValueError(f"evo2: can only replicate a single-sequence cache (got batch size {batch_size})")

    mha, hcl, hcm, hcs = cache["mha"], cache["hcl"], cache["hcm"], cache["hcs"]
    return {
        "mha": InferenceParams(
            max_seqlen=mha.max_seqlen,
            max_batch_size=max(mha.max_batch_size, n_replicates),
            seqlen_offset=mha.seqlen_offset,
            batch_size_offset=mha.batch_size_offset,
            key_value_memory_dict=_repeat_cache_dict(mha.key_value_memory_dict, n_replicates),
        ),
        "hcl": HyenaCascadeIIRInferenceParams(
            fir_filter_length=hcl.fir_filter_length,
            state_dim=hcl.state_dim,
            seqlen_offset=hcl.seqlen_offset,
            fir_state_dict=_repeat_cache_dict(hcl.fir_state_dict, n_replicates),
            state_dict=_repeat_cache_dict(hcl.state_dict, n_replicates),
        ),
        "hcm": HyenaCascadeFIRInferenceParams(
            fir_filter_length=hcm.fir_filter_length,
            seqlen_offset=hcm.seqlen_offset,
            fir_inner_filter_length=hcm.fir_inner_filter_length,
            fir_state_dict=_repeat_cache_dict(hcm.fir_state_dict, n_replicates),
            fir_inner_state_dict=_repeat_cache_dict(hcm.fir_inner_state_dict, n_replicates),
            state_dict=_repeat_cache_dict(hcm.state_dict, n_replicates),
        ),
        "hcs": HyenaCascadeFIRInferenceParams(
            fir_filter_length=hcs.fir_filter_length,
            seqlen_offset=hcs.seqlen_offset,
            fir_inner_filter_length=hcs.fir_inner_filter_length,
            fir_state_dict=_repeat_cache_dict(hcs.fir_state_dict, n_replicates),
            fir_inner_state_dict=_repeat_cache_dict(hcs.fir_inner_state_dict, n_replicates),
            state_dict=_repeat_cache_dict(hcs.state_dict, n_replicates),
        ),
    }


def _move_cache_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        for key, inner in value.items():
            value[key] = _move_cache_to_device(inner, device)
        return value
    if isinstance(value, list):
        for idx, inner in enumerate(value):
            value[idx] = _move_cache_to_device(inner, device)
        return value
    if isinstance(value, tuple):
        return tuple(_move_cache_to_device(inner, device) for inner in value)
    if hasattr(value, "__dict__"):
        for attr_name, attr_value in vars(value).items():
            setattr(value, attr_name, _move_cache_to_device(attr_value, device))
    return value


KV_CACHE_REF_TYPE = "evo2_kv_cache"
_kv_cache_store: dict[str, dict[str, Any]] = {}


def _cache_id_from_handle(cache_handle: dict[str, Any]) -> str:
    cache_id = cache_handle.get("cache_id")
    if cache_handle.get("type") != KV_CACHE_REF_TYPE or not isinstance(cache_id, str):
        raise ValueError("evo2: expected a KV-cache handle returned by a persistent Evo2 worker")
    return cache_id


def _resolve_cache_handle(cache_handle: dict[str, Any]) -> dict[str, Any]:
    cache_id = _cache_id_from_handle(cache_handle)
    cache = _kv_cache_store.get(cache_id)
    if cache is None:
        raise ValueError("evo2: KV-cache handle no longer available — released, or the persistent worker restarted")
    return cache


def _store_cache_handle(cache: dict[str, Any]) -> dict[str, str]:
    cache_id = uuid.uuid4().hex
    _kv_cache_store[cache_id] = cache
    return {"type": KV_CACHE_REF_TYPE, "cache_id": cache_id}


def _release_cache_handles(cache_handles: dict[str, Any] | list[dict[str, Any] | None] | None) -> None:
    if cache_handles is None:
        return
    handles = [cache_handles] if isinstance(cache_handles, dict) else cache_handles
    for cache_handle in handles:
        if cache_handle is None:
            continue
        _kv_cache_store.pop(_cache_id_from_handle(cache_handle), None)


def _uses_worker_local_cache_handles(input_dict: dict[str, Any]) -> bool:
    operation = input_dict["operation"]
    return operation == "release_kv_caches" or (
        operation == "sample"
        and (input_dict.get("old_kv_cache") is not None or bool(input_dict.get("return_kv_cache")))
    )


def _ensure_cache_handles_use_persistent_worker(input_dict: dict[str, Any]) -> None:
    if __name__ == "__main__" and _uses_worker_local_cache_handles(input_dict):
        raise RuntimeError(
            "evo2: KV-cache handles require a persistent ToolInstance worker — "
            "wrap calls in ToolInstance.persist() or pass a persistent ToolInstance"
        )


# ============================================================================
# Dispatch
# ============================================================================
_model: Evo2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    operation = input_dict["operation"]
    _ensure_cache_handles_use_persistent_worker(input_dict)

    if operation == "release_kv_caches":
        _release_cache_handles(input_dict.get("kv_caches"))
        return {"released": True}

    if _model is None:
        _model = Evo2Model(
            model_checkpoint=input_dict["model_checkpoint"],
            local_path=input_dict.get("local_path"),
        )

    if operation == "sample":
        old_kv_cache = None
        max_seqlen = input_dict.get("max_seqlen")
        old_kv_cache_handle = input_dict.get("old_kv_cache")
        if old_kv_cache_handle is not None:
            resolved_cache = _resolve_cache_handle(old_kv_cache_handle)
            required_seqlen = max(len(prompt) for prompt in input_dict["prompts"]) + input_dict["max_new_tokens"]
            if resolved_cache["mha"].max_seqlen < required_seqlen:
                logger.warning(
                    "KV cache max_seqlen (%s) is insufficient for continued generation (need %s). Discarding cache.",
                    resolved_cache["mha"].max_seqlen,
                    required_seqlen,
                )
            else:
                old_kv_cache = _replicate_cache(resolved_cache, len(input_dict["prompts"]))
            if max_seqlen is None:
                max_seqlen = required_seqlen

        return_kv_cache = bool(input_dict.get("return_kv_cache"))
        result = _model.sample(
            prompts=input_dict["prompts"],
            top_k=input_dict["top_k"],
            top_p=input_dict["top_p"],
            temperature=input_dict["temperature"],
            device=input_dict["device"],
            max_new_tokens=input_dict["max_new_tokens"],
            cached_generation=input_dict["cached_generation"],
            force_prompt_threshold=input_dict["force_prompt_threshold"],
            max_seqlen=max_seqlen,
            verbose=input_dict["verbose"],
            skip_special_tokens=input_dict["skip_special_tokens"],
            stop_at_eos=input_dict["stop_at_eos"],
            old_kv_cache=old_kv_cache,
            batch_size=input_dict["batch_size"],
            return_kv_cache=return_kv_cache,
            return_logits=input_dict["return_logits"],
            seed=input_dict["seed"],
        )
        kv_caches = result["kv_caches"] if return_kv_cache else None
        result["kv_caches"] = (
            [_store_cache_handle(cache) for cache in kv_caches if cache is not None]
            if kv_caches and all(cache is not None for cache in kv_caches)
            else None
        )
        return result
    if operation == "score":
        return _model.score(
            sequences=input_dict["sequences"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            batch_size=input_dict["batch_size"],
            return_logits=input_dict["return_logits"],
            prepend_bos=input_dict["prepend_bos"],
            seed=input_dict["seed"],
        )
    raise ValueError(f"evo2: unknown operation {operation!r}; valid: ['release_kv_caches', 'sample', 'score']")


def _cleanup_vortex_debug_log() -> None:
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


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        for cache in _kv_cache_store.values():
            _move_cache_to_device(cache, device)
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
        raise ValueError("evo2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
