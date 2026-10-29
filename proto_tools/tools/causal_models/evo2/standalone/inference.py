"""Evo 2 inference through evo2_flash and vLLM."""

import json
import os
import secrets
import sys
import uuid
from typing import Any

import torch
from standalone_helpers import get_logger, log_likelihood_metrics, serialize_output

logger = get_logger(__name__)
EVO2_VOCAB = [chr(j) for j in range(512)]


class Evo2Model:
    """Persistent Evo 2 engine with continuous batching and automatic prefix caching."""

    def __init__(self, model_checkpoint: str = "evo2_7b", local_path: str | None = None):
        """Store checkpoint selection; create the engine on first use."""
        self.model_checkpoint = model_checkpoint
        self.local_path = local_path
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str | None = None
        self._loaded = False

    def load(self, device: str, verbose: bool = False) -> None:
        """Load the engine on one GPU; keep checkpoint conversion in the model cache."""
        if device == "cpu":
            raise ValueError("Evo 2 Flash inference requires an NVIDIA GPU")
        from standalone_helpers import get_subprocess_device_env

        os.environ["CUDA_VISIBLE_DEVICES"] = get_subprocess_device_env(device)["CUDA_VISIBLE_DEVICES"]
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from evo2_flash import Evo2

        if verbose:
            logger.info("Loading Evo 2 Flash: %s on %s", self.model_checkpoint, device)
        self.model = Evo2(self.model_checkpoint, local_path=self.local_path)
        self.tokenizer = self.model.tokenizer
        self.device = device
        self._loaded = True

    def _ready(self, device: str, verbose: bool) -> None:
        """Load the engine before inference on its assigned GPU."""
        if device == "cpu":
            raise ValueError("Evo 2 Flash inference requires an NVIDIA GPU")
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

    def sample(
        self,
        prompts: list[str],
        top_k: int = 4,
        top_p: float = 1.0,
        temperature: float = 1.0,
        device: str = "cuda",
        max_new_tokens: int = 32,
        cached_generation: bool = True,  # noqa: ARG002 — vLLM manages batching and prefix state
        force_prompt_threshold: int | None = None,  # noqa: ARG002 — vLLM manages batching and prefix state
        max_seqlen: int | None = None,
        verbose: bool = False,
        skip_special_tokens: bool = False,
        stop_at_eos: bool = True,
        old_kv_cache: dict[str, Any] | None = None,
        batch_size: int = 1,  # noqa: ARG002 — vLLM manages batching and prefix state
        return_kv_cache: bool = False,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate continuations in one continuously batched vLLM submission.

        Legacy batch_size and force_prompt_threshold are compatibility arguments;
        vLLM schedules ragged requests and chunked prefill automatically. Cache
        handles retain exact token prefixes; vLLM owns and may evict their GPU state.
        """
        self._ready(device, verbose)
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if not prompts:
            raise ValueError("evo2: cannot generate from an empty prompt list")
        ids = []
        for prompt in prompts:
            if old_kv_cache is None:
                tokens = self.tokenizer.tokenize(prompt)
            else:
                prefix = old_kv_cache["text"]
                if not prompt.startswith(prefix):
                    raise ValueError("evo2: continued prompts must include the full cached sequence prefix")
                tokens = old_kv_cache["tokens"] + self.tokenizer.tokenize(prompt[len(prefix) :])
            if not tokens:
                raise ValueError("evo2: cannot generate from an empty prompt")
            needed = len(tokens) + max_new_tokens
            if max_seqlen is not None and needed > max_seqlen:
                raise ValueError(f"evo2: prompt + generation needs {needed} positions, above max_seqlen={max_seqlen}")
            if needed > self.model.max_model_len:
                raise ValueError(
                    f"evo2: prompt + generation needs {needed} positions, above engine context {self.model.max_model_len}"
                )
            ids.append(tokens)
        # An explicit request seed is required by vLLM. A missing user seed gets
        # a fresh base, rather than restarting the engine's default seeded stream.
        base_seed = seed if seed is not None else secrets.randbits(32)
        sampling = self.model.sampling_kwargs(temperature, top_k, top_p)
        params = [
            SamplingParams(
                max_tokens=max_new_tokens,
                min_tokens=0 if stop_at_eos else max_new_tokens,
                ignore_eos=not stop_at_eos,
                stop_token_ids=[0] if stop_at_eos else None,
                detokenize=False,
                seed=base_seed + i,
                **sampling,
            )
            for i in range(len(ids))
        ]
        requests = [TokensPrompt(prompt_token_ids=tokens) for tokens in ids]
        outs = self.model.llm.generate(requests, params, use_tqdm=bool(verbose))
        if len(outs) != len(prompts):
            raise RuntimeError("evo2: vLLM returned an unexpected number of generations")
        sequences, caches, all_logits = [], [], []
        for prompt, prefix_ids, out in zip(prompts, ids, outs, strict=True):
            tokens = list(out.outputs[0].token_ids)
            if not stop_at_eos and len(tokens) != max_new_tokens:
                raise RuntimeError("evo2: vLLM did not return the requested fixed generation length")
            visible = [t for t in tokens if t not in (0, 1)] if skip_special_tokens else tokens
            text = self.tokenizer.detokenize(visible)
            sequences.append(text)
            if return_kv_cache:
                caches.append({"tokens": prefix_ids + tokens, "text": prompt + text})
            if return_logits:
                # Full-vocabulary logits are deliberately opt-in: reconstruct
                # them by prefill only for this request, without padding neighbors.
                if tokens:
                    tensor = torch.tensor([prefix_ids + tokens[:-1]], dtype=torch.long, device="cuda:0")
                    logits, _ = self.model(tensor)
                    all_logits.append(logits[0, len(prefix_ids) - 1 :].float().cpu())
                else:
                    all_logits.append(torch.empty((0, 512)))
        return {
            "sequences": sequences,
            "logits": all_logits if return_logits else None,
            "kv_caches": caches if return_kv_cache else None,
        }

    def score(
        self,
        sequences: list[str],
        device: str = "cuda",
        verbose: bool = False,
        batch_size: int = 1,  # noqa: ARG002 — vLLM manages batching and prefix state
        return_logits: bool = False,
        prepend_bos: bool = False,
        seed: int | None = None,  # noqa: ARG002 — deterministic scoring keeps the shared call signature
    ) -> dict[str, Any]:
        """Score unpadded sequences with prompt logprobs; capture logits only on request."""
        self._ready(device, verbose)
        if not sequences or any(not seq for seq in sequences):
            raise ValueError("evo2: cannot score empty sequences")
        means = self.model.score_sequences(sequences, prepend_bos=prepend_bos)
        counts = [len(self.tokenizer.tokenize(s)) + int(prepend_bos) - 1 for s in sequences]
        metrics = [log_likelihood_metrics(float(mean), count) for mean, count in zip(means, counts, strict=True)]
        logits = None
        if return_logits:
            logits = []
            for seq in sequences:
                ids = ([0] if prepend_bos else []) + self.tokenizer.tokenize(seq)
                tensor = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                values, _ = self.model(tensor)
                logits.append(values[0].cpu())
        return {"metrics": metrics, "logits": logits, "vocab": EVO2_VOCAB}

    def to_device(self, device: str) -> None:
        """Require worker restart for device changes; vLLM owns its GPU allocations."""
        if self._loaded and self.device != device:
            raise ValueError("Evo 2 Flash engines are GPU-bound; use a new worker to change devices")


KV_CACHE_REF_TYPE = "evo2_kv_cache"
_kv_cache_store: dict[str, dict[str, Any]] = {}


def _cache_id_from_handle(cache_handle: dict[str, Any]) -> str:
    cache_id = cache_handle.get("cache_id")
    if cache_handle.get("type") != KV_CACHE_REF_TYPE or not isinstance(cache_id, str):
        raise ValueError("evo2: expected a KV-cache handle returned by a persistent Evo 2 worker")
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


_model: Evo2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool operation in the managed standalone worker."""
    global _model
    _ensure_cache_handles_use_persistent_worker(input_dict)
    operation = input_dict["operation"]
    if operation == "release_kv_caches":
        _release_cache_handles(input_dict.get("kv_caches"))
        return {"released": True}
    if operation not in ("sample", "score"):
        raise ValueError(f"evo2: unknown operation {operation!r}")
    if _model is None:
        _model = Evo2Model(input_dict["model_checkpoint"], input_dict.get("local_path"))
    if operation == "sample":
        keys: tuple[str, ...] = (
            "prompts",
            "top_k",
            "top_p",
            "temperature",
            "device",
            "max_new_tokens",
            "cached_generation",
            "force_prompt_threshold",
            "max_seqlen",
            "verbose",
            "skip_special_tokens",
            "stop_at_eos",
            "batch_size",
            "return_kv_cache",
            "return_logits",
            "seed",
        )
        kwargs = {k: input_dict[k] for k in keys if k in input_dict}
        handle = input_dict.get("old_kv_cache")
        kwargs["old_kv_cache"] = _resolve_cache_handle(handle) if handle is not None else None
        result = _model.sample(**kwargs)
        if result["kv_caches"] is not None:
            result["kv_caches"] = [_store_cache_handle(c) for c in result["kv_caches"]]
        return result
    keys = ("sequences", "device", "verbose", "batch_size", "return_logits", "prepend_bos", "seed")
    return _model.score(**{k: input_dict[k] for k in keys if k in input_dict})


def to_device(device: str) -> dict[str, Any]:
    """Reject in-process moves; Proto restarts GPU-bound workers on device changes."""
    if _model is not None and _model._loaded:
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report the process-local GPU's allocations."""
    from standalone_helpers import get_pytorch_memory_stats

    stats: dict[str, Any] = get_pytorch_memory_stats("cuda:0")
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("evo2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
