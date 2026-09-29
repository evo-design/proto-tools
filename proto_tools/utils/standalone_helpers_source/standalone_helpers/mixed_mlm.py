"""Shared inference operations for case-sensitive protein/DNA masked models."""

import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, ClassVar

from .compression import compress_array
from .device import move_model_to_device
from .mixed_sequence import DNA_TOKENS, MIXED_VOCAB, PROTEIN_TOKENS, tokenize_mixed_sequence
from .proto_logging import get_logger
from .scoring import log_likelihood_metrics
from .seeding import set_torch_seed
from .serialization import serialize_output

logger = get_logger(__name__)


class MixedMLMAdapter:
    """Shared checkpoint lifecycle for the two upstream Hugging Face model classes."""

    toolkit: ClassVar[str]
    checkpoints: ClassVar[dict[str, str]]
    default_checkpoint: ClassVar[str]

    def __init__(self) -> None:
        """Initialize an unloaded model adapter."""
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str | None = None
        self.checkpoint: str | None = None

    def _load_checkpoint(self, checkpoint: str) -> tuple[Any, Any]:
        raise NotImplementedError

    @property
    def embedding_layer(self) -> Any:
        """Return the upstream embedding module used by differentiable context hooks."""
        return getattr(self.model, self.toolkit).tok_embeddings

    def load(self, checkpoint: str, device: str) -> None:
        """Load once per checkpoint and freeze model parameters for input gradients."""
        if self.checkpoint != checkpoint:
            logger.info("Loading %s on %s", checkpoint, device)
            model, self.tokenizer = self._load_checkpoint(checkpoint)
            self.model = model.eval().requires_grad_(False).to(device)
            self.checkpoint = checkpoint
            self.device = device
        elif self.device != device:
            self.to_device(device)

    def forward(self, ids: Any, *, output_hidden_states: bool = False, **kwargs: Any) -> Any:
        """Run an unpadded forward, requesting extra model outputs only when needed."""
        import torch

        return self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

    def reset_rotary_cache(self) -> None:
        """Drop unregistered RoPE tensors before device moves and backward passes."""
        if self.model is None:
            return
        for module in self.model.modules():
            if hasattr(module, "_seq_len_cached"):
                module._seq_len_cached = 0
                for name in ("_cos_cached", "_sin_cached", "_cos_k_cached", "_sin_k_cached"):
                    if hasattr(module, name):
                        setattr(module, name, None)

    def to_device(self, device: str) -> None:
        """Move the model using the shared cleanup helper and reset rotary caches."""
        if self.model is not None and self.device != device:
            self.reset_rotary_cache()
            self.model = move_model_to_device(self.model, self.device or "cpu", device)
            self.device = device


def _array(value: Any) -> dict[str, Any]:
    """Serialize dense tensors without materializing nested Python floats."""
    return compress_array(value.detach().float().cpu().numpy())


def _batches(tokens: list[list[str]], batch_size: int) -> Iterator[list[int]]:
    """Group equal token lengths, retaining indices for restoring input order."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(tokens):
        groups[len(row)].append(index)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]


class MixedMLMRuntime:
    """Run shared mixed-modality operations through a small upstream model adapter."""

    def __init__(self, adapter: Any) -> None:
        """Bind the upstream model adapter without loading its weights."""
        self.adapter = adapter

    def _ids(self, tokens: list[list[str]]) -> Any:
        """Map validated atomic tokens to model IDs without adding special tokens."""
        import torch

        vocab = self.adapter.tokenizer.get_vocab()
        return torch.tensor(
            [[vocab["<mask>" if symbol == "_" else symbol] for symbol in row] for row in tokens],
            dtype=torch.long,
            device=self.adapter.device,
        )

    def _vocab_ids(self) -> list[int]:
        vocab = self.adapter.tokenizer.get_vocab()
        return [vocab[token] for token in MIXED_VOCAB]

    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate token axes, load the requested checkpoint, and run one operation."""
        operation = payload.get("operation")
        if operation not in {"embeddings", "sample", "score", "gradient", "interactions"}:
            raise ValueError(f"{self.adapter.toolkit}: unknown operation {operation!r}")
        if operation == "interactions" and self.adapter.toolkit != "minerva":
            raise ValueError("Only Minerva provides trained interaction heads")
        checkpoint = payload.get("model_checkpoint", self.adapter.default_checkpoint)
        if checkpoint not in self.adapter.checkpoints:
            raise ValueError(f"Unknown {self.adapter.toolkit} checkpoint: {checkpoint!r}")
        batch_size = payload.get("batch_size", 1)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        sequences = [payload["sequence"]] if operation == "gradient" else payload["sequences"]
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("At least one prepared sequence is required")
        tokens = [tokenize_mixed_sequence(sequence, allow_masks=operation == "sample") for sequence in sequences]
        limit = 8192 if checkpoint.endswith("-8k") else 4096
        if any(len(row) > limit for row in tokens):
            raise ValueError(f"{checkpoint} supports at most {limit} model tokens per sequence")
        if operation in {"score", "gradient"} and any(not any(t in MIXED_VOCAB for t in row) for row in tokens):
            raise ValueError("Masked PLL requires at least one canonical protein or DNA token")
        if operation == "sample":
            mappings = payload.get("mask_modalities") or [{} for _ in tokens]
            if len(mappings) != len(tokens):
                raise ValueError("mask_modalities must have one mapping per sequence")
            mappings = [{int(position): modality for position, modality in mapping.items()} for mapping in mappings]
            for row, mapping in zip(tokens, mappings, strict=True):
                if set(mapping) != {i + 1 for i, symbol in enumerate(row) if symbol == "_"}:
                    raise ValueError("mask_modalities must label exactly the masked token positions")
                if any(modality not in {"protein", "dna"} for modality in mapping.values()):
                    raise ValueError("Masked positions must have protein or dna modality")
        self.adapter.load(checkpoint, payload.get("device", "cuda"))
        if operation == "embeddings":
            return self.embeddings(
                tokens, batch_size, payload.get("repr_layer", -1), payload.get("return_logits", False)
            )
        if operation == "score":
            return self.score(tokens, batch_size, payload.get("return_logits", False))
        if operation == "sample":
            return self.sample(tokens, mappings, payload)
        if operation == "gradient":
            return self.gradient(payload)
        return self.interactions(tokens, payload)

    def embeddings(
        self, tokens: list[list[str]], batch_size: int, repr_layer: int, return_logits: bool
    ) -> dict[str, Any]:
        """Mean-pool all unpadded tokens, including fixed strand markers."""
        import torch

        depth = self.adapter.model.config.depth
        if repr_layer < -1 or repr_layer > depth:
            raise ValueError(f"repr_layer must be -1 or between 0 and {depth}")
        results: list[Any] = [None] * len(tokens)
        for indices in _batches(tokens, batch_size):
            ids = self._ids([tokens[index] for index in indices])
            with torch.no_grad():
                output: Any = (
                    self.adapter.forward(ids, output_hidden_states=repr_layer != 0)
                    if repr_layer != 0 or return_logits
                    else None
                )
                if repr_layer == 0:
                    hidden = self.adapter.embedding_layer(ids)
                else:
                    layer = -1 if repr_layer == -1 else repr_layer - 1
                    hidden = output.hidden_states[layer]
                pooled = hidden.float().mean(dim=1)
            for row, index in enumerate(indices):
                results[index] = {
                    "mean_embedding": _array(pooled[row]),
                    "attention_mask": [1] * len(tokens[index]),
                    "tokens": tokens[index],
                    "vocab": MIXED_VOCAB,
                    "logits": _array(output.logits[row, :, self._vocab_ids()]) if return_logits else None,
                }
        return {"results": results}

    def score(self, tokens: list[list[str]], batch_size: int, return_logits: bool) -> dict[str, Any]:
        """Compute full-vocabulary masked PLL at canonical biological positions."""
        import torch
        import torch.nn.functional as F

        scores = []
        mask_id = self.adapter.tokenizer.mask_token_id
        for row in tokens:
            ids = self._ids([row])
            positions = [index for index, token in enumerate(row) if token in MIXED_VOCAB]
            if not positions:
                raise ValueError("Masked PLL requires at least one canonical protein or DNA token")
            returned_logits = (
                torch.zeros((len(row), len(MIXED_VOCAB)), device=self.adapter.device) if return_logits else None
            )
            total_nll = 0.0
            for start in range(0, len(positions), batch_size):
                selected = torch.tensor(positions[start : start + batch_size], device=self.adapter.device)
                batch_ids = ids.expand(len(selected), -1).clone()
                batch_rows = torch.arange(len(selected), device=self.adapter.device)
                batch_ids[batch_rows, selected] = mask_id
                with torch.no_grad():
                    prediction = self.adapter.forward(batch_ids).logits[batch_rows, selected].float()
                    total_nll += F.cross_entropy(prediction, ids[0, selected], reduction="sum").item()
                if returned_logits is not None:
                    returned_logits[selected] = prediction[:, self._vocab_ids()]
            scores.append(
                {
                    **log_likelihood_metrics(-total_nll / len(positions), len(positions)),
                    "tokens": row,
                    "scored_positions": [position + 1 for position in positions],
                    "vocab": MIXED_VOCAB,
                    "logits": _array(returned_logits) if return_logits else None,
                }
            )
        return {"scores": scores}

    def sample(
        self, tokens: list[list[str]], mask_modalities: list[dict[int, str]], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Fill editable positions in exact-length batches with modality restrictions."""
        import torch

        from .iterative_sampling import iterative_sample

        set_torch_seed(payload.get("seed"))
        vocab = self.adapter.tokenizer.get_vocab()
        reverse_vocab = {value: key for key, value in vocab.items()}
        vocab_ids = self._vocab_ids()
        mask_id = self.adapter.tokenizer.mask_token_id
        results: list[Any] = [None] * len(tokens)
        for indices in _batches(tokens, payload.get("batch_size", 1)):
            ids = self._ids([tokens[index] for index in indices])
            allowed = torch.zeros((*ids.shape, len(vocab)), dtype=torch.bool, device=self.adapter.device)
            allowed[:, :, vocab_ids] = True
            for row, index in enumerate(indices):
                for position, modality in mask_modalities[index].items():
                    allowed[row, position - 1] = False
                    candidates = vocab_ids[:20] if modality == "protein" else vocab_ids[20:]
                    allowed[row, position - 1, candidates] = True

            def forward(masked_ids: Any, permitted: Any = allowed) -> Any:
                # iterative_sample enters inference_mode; RoPE caches must remain usable in backward.
                with torch.inference_mode(False), torch.no_grad():
                    predictions = self.adapter.forward(masked_ids.clone()).logits.float()
                    return predictions.masked_fill(~permitted, -torch.inf)

            if any(mask_modalities[index] for index in indices):
                if payload.get("sampling_method", "single_pass") == "iterative_refinement":
                    ids = iterative_sample(
                        forward,
                        ids,
                        mask_id,
                        vocab_ids,
                        num_steps=payload.get("num_steps", 20),
                        schedule=payload.get("schedule", "cosine"),
                        strategy=payload.get("strategy", "random"),
                        temperature=payload.get("temperature", 1.0),
                        top_p=payload.get("top_p", 1.0),
                        temperature_annealing=payload.get("temperature_annealing", True),
                    )
                else:
                    predictions = forward(ids) / payload.get("temperature", 1.0)
                    for row, index in enumerate(indices):
                        if not mask_modalities[index]:
                            continue
                        positions = torch.tensor(
                            [position - 1 for position in sorted(mask_modalities[index])], device=self.adapter.device
                        )
                        probs = torch.softmax(predictions[row, positions], dim=-1)
                        ids[row, positions] = torch.multinomial(probs, 1).squeeze(-1)
            final_logits = None
            if payload.get("return_logits", False):
                with torch.no_grad():
                    final_logits = self.adapter.forward(ids).logits[:, :, vocab_ids]
            for row, index in enumerate(indices):
                completed = [reverse_vocab[symbol] for symbol in ids[row].tolist()]
                results[index] = {
                    "sequence": "".join(completed),
                    "tokens": completed,
                    "vocab": MIXED_VOCAB,
                    "logits": _array(final_logits[row]) if final_logits is not None else None,
                }
        return {"results": results}

    def gradient(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Differentiate mean masked NLL through relaxed context-token embeddings."""
        import torch
        import torch.nn.functional as F

        tokens = tokenize_mixed_sequence(payload["sequence"])
        backprop = payload.get("compute_gradient", True)
        temperature = payload.get("temperature", 1.0)
        batch_size = payload.get("batch_size", 1)
        self.adapter.reset_rotary_cache()
        embed_layer = self.adapter.embedding_layer
        vocab_ids = torch.tensor(self._vocab_ids(), device=self.adapter.device)
        ids = self._ids([tokens])
        positions = torch.tensor(
            [i for i, token in enumerate(tokens) if token in MIXED_VOCAB], device=self.adapter.device
        )
        if not len(positions):
            raise ValueError("Masked PLL requires at least one canonical protein or DNA token")
        allowed = torch.zeros((len(tokens), len(MIXED_VOCAB)), dtype=torch.bool, device=self.adapter.device)
        for index, token in enumerate(tokens):
            if token in PROTEIN_TOKENS:
                allowed[index, :20] = True
            elif token in DNA_TOKENS:
                allowed[index, 20:] = True
        with torch.set_grad_enabled(backprop):
            state = torch.tensor(
                payload["logits"], dtype=torch.float32, device=self.adapter.device, requires_grad=backprop
            )
            if state.shape != allowed.shape or not torch.isfinite(state).all():
                raise ValueError("Gradient input must be a finite token-aligned L x 24 matrix")
            if temperature is None:
                x = state * allowed
            else:
                # Fixed context rows need a finite softmax denominator, then contribute zero.
                softmax_allowed = allowed | ~allowed.any(dim=1, keepdim=True)
                x = F.softmax((state / temperature).masked_fill(~softmax_allowed, -torch.inf), dim=-1) * allowed
            hard_indices = x.argmax(dim=-1).detach()
            if payload.get("use_ste", False):
                hard = F.one_hot(hard_indices, num_classes=len(MIXED_VOCAB)).float() * allowed
                x = hard + x - x.detach()
            ids[0, positions] = vocab_ids[hard_indices[positions]]
            fixed_embeddings = embed_layer(ids).detach()
            relaxed = x.to(embed_layer.weight.dtype) @ embed_layer.weight[vocab_ids]
            input_embeddings = torch.where(allowed.any(dim=1).view(1, -1, 1), relaxed.unsqueeze(0), fixed_embeddings)
            mask_id = self.adapter.tokenizer.mask_token_id
            mask_embedding = embed_layer.weight[mask_id].detach()
            embedding_grad = torch.zeros_like(input_embeddings) if backprop else None
            all_positions = torch.arange(len(tokens), device=self.adapter.device)
            total_nll = 0.0
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                batch_rows = torch.arange(len(selected), device=self.adapter.device)
                chunk_leaf = input_embeddings.detach().requires_grad_(backprop)
                masked_embeddings = torch.where(
                    (all_positions[None, :] == selected[:, None]).unsqueeze(-1),
                    mask_embedding.view(1, 1, -1),
                    chunk_leaf.expand(len(selected), -1, -1),
                )
                masked_ids = ids.expand(len(selected), -1).clone()
                masked_ids[batch_rows, selected] = mask_id
                handle = embed_layer.register_forward_hook(
                    lambda _module, _inputs, _output, value=masked_embeddings: value
                )
                try:
                    predictions = self.adapter.forward(masked_ids).logits[batch_rows, selected].float()
                finally:
                    handle.remove()
                loss = F.cross_entropy(predictions, ids[0, selected], reduction="sum")
                total_nll += loss.item()
                if backprop:
                    (chunk_gradient,) = torch.autograd.grad(loss / len(positions), chunk_leaf)
                    if embedding_grad is None:
                        raise RuntimeError("Missing input-embedding gradient accumulator")
                    embedding_grad = embedding_grad + chunk_gradient
            gradient = None
            if backprop:
                (state_grad,) = torch.autograd.grad(input_embeddings, state, grad_outputs=embedding_grad)
                gradient = _array(state_grad)
        mean_nll = total_nll / len(positions)
        return {
            "gradient": gradient,
            "loss": mean_nll,
            "vocab": MIXED_VOCAB,
            "tokens": tokens,
            "metrics": {
                **log_likelihood_metrics(-mean_nll, len(positions)),
                "sequence_length": len(positions),
                "model_checkpoint": self.adapter.checkpoint,
                "objective": "masked_pll",
            },
        }

    def interactions(self, tokens: list[list[str]], payload: dict[str, Any]) -> dict[str, Any]:
        """Return selected upstream sigmoid-probability maps on the full token axis."""
        import torch

        results: list[Any] = [None] * len(tokens)
        heads = payload.get("heads", ["base_pairing", "protein", "repeat"])
        for indices in _batches(tokens, payload.get("batch_size", 1)):
            ids = self._ids([tokens[index] for index in indices])
            with torch.no_grad():
                output = self.adapter.forward(
                    ids, output_interactions=True, interaction_layers=payload.get("interaction_layers", 2)
                )
            for row, index in enumerate(indices):
                results[index] = {
                    "maps": {
                        head: {"tokens": tokens[index], "values": _array(output.interactions[head][row])}
                        for head in heads
                    }
                }
        return {"results": results}


def run_json_main(dispatch: Any) -> None:
    """Implement the standard standalone JSON input/output entry point."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: inference.py <input.json> <output.json>")
    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    result = dispatch(payload)
    with open(sys.argv[2], "w") as handle:
        json.dump(serialize_output(result), handle)
