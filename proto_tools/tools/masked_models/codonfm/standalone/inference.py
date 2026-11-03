"""CodonFM (Encodon) standalone inference for venv execution.

Drives the vendored ``EncodonInference`` (under ``src/``) for all five CodonFM tools. Reads a
JSON input file and writes a JSON output file, matching the proto one-shot dispatch contract
(``python inference.py input.json output.json``).
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from standalone_helpers import (
    download_weights,
    file_lock,
    get_logger,
    log_likelihood_metrics,
    move_model_to_device,
    require_weights_dir,
    serialize_output,
    set_torch_seed,
)
from standalone_helpers.relaxation import relaxed_sequence, resolve_temperature
from standalone_helpers.vocabulary import DNA_CODONS

# The vendored CodonFM source lives under ``src/`` next to this file.
sys.path.insert(0, str(Path(__file__).parent))

logger = get_logger(__name__)

# The tokenizer's codon vocabulary is all 64 codons (``product("ACGT", 3)``), which includes the
# three stop codons. Masked-codon resampling must never draw one of these into an interior codon,
# so sampling is restricted to the 61 sense codons.
_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# A codon marked for resampling. The caller masks whole codons, so a mask is three characters
# wide and the reading frame is unchanged by masking.
_MASK_CODON = "___"


# ---------------------------------------------------------------------------
# HuggingFace checkpoint provisioning (public NVIDIA Open Model License weights)
# ---------------------------------------------------------------------------
def _resolve_checkpoint(input_dict: dict[str, Any]) -> str:
    """Provision ``<name>.safetensors`` + ``config.json`` into one cache dir; return the safetensors path.

    ``EncodonInference`` requires both files side by side. The repositories are public under the
    NVIDIA Open Model License; an available ``HF_TOKEN`` is used but is not required.
    """
    safetensors_url = input_dict["safetensors_url"]
    config_url = input_dict["config_url"]
    filename = input_dict["safetensors_filename"]
    cache_dir = Path(require_weights_dir("codonfm")) / input_dict["cache_subdir"]
    safetensors_path = cache_dir / filename
    config_path = cache_dir / "config.json"

    with file_lock(cache_dir.parent / f".{input_dict['cache_subdir']}.download.lock", progress=cache_dir):
        download_weights(safetensors_url, safetensors_path, timeout=300)
        download_weights(config_url, config_path, timeout=300)
    return str(safetensors_path)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
def special_token_layout(tokenizer: Any) -> tuple[int, int]:
    """Return the ``(cls, sep)`` ids Encodon wraps a codon row with.

    Args:
        tokenizer (Any): The Encodon tokenizer.

    Returns:
        tuple[int, int]: The leading and trailing special-token ids.
    """
    return int(tokenizer.cls_token_id), int(tokenizer.sep_token_id)


def build_model_input_row(tokenizer: Any, codon_ids: list[int]) -> tuple[list[int], list[int]]:
    """Assemble the ``(input_ids, attention_mask)`` rows Encodon expects for a codon sequence.

    Wraps the codons in CLS/SEP, pads to the smallest multiple-of-eight context that holds them,
    and masks the padding out — the same layout upstream ``process_item`` produces when it
    tokenizes a CDS. The gradient builds its tensors here instead of tokenizing, because its
    input is a relaxed distribution rather than a sequence.

    Args:
        tokenizer (Any): The Encodon tokenizer.
        codon_ids (list[int]): Codon vocabulary ids, in sequence order.

    Returns:
        tuple[list[int], list[int]]: Padded input ids and their attention mask.
    """
    cls_id, sep_id = special_token_layout(tokenizer)
    wrapped = [cls_id, *codon_ids, sep_id]
    padding = min(2048, ((len(wrapped) + 7) // 8) * 8) - len(wrapped)
    return wrapped + [int(tokenizer.pad_token_id)] * padding, [1] * len(wrapped) + [0] * padding


class CodonFMModel:
    """Caches a loaded ``EncodonInference`` and drives the codon-scoring tasks."""

    def __init__(self) -> None:
        """Initialize an unloaded CodonFM model wrapper."""
        self.inference: Any = None
        self.tokenizer: Any = None
        self.device: str | None = None
        self.checkpoint_key: str | None = None

    def load(self, *, checkpoint_path: str, device: str, verbose: bool = False) -> None:
        """Load an Encodon checkpoint (once per checkpoint) and move it to the device."""
        from src.inference.encodon import EncodonInference
        from src.inference.task_types import TaskTypes

        if self.inference is not None and self.checkpoint_key == checkpoint_path:
            self.to_device(device)
            return
        if verbose:
            logger.info("Loading CodonFM checkpoint from %s", checkpoint_path)
        inference = EncodonInference(model_path=checkpoint_path, task_type=TaskTypes.FITNESS_PREDICTION.value)
        inference.configure_model()  # builds on CPU (LightningModule default device)
        inference.eval()
        inference.requires_grad_(False)
        self.inference = inference
        self.tokenizer = inference.tokenizer
        self.checkpoint_key = checkpoint_path
        self.device = "cpu"
        self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move the loaded Encodon model to another device."""
        if self.inference is None:
            raise ValueError("codonfm: cannot move an unloaded model; call load() first")
        if self.device != device:
            self.inference = move_model_to_device(self.inference, self.device or "cpu", device)
            self.device = device

    def _batch(self, items: list[dict[str, np.ndarray]]) -> dict[str, Any]:
        """Stack per-item numpy fields (all padded to a shared length) into device tensors."""
        import torch

        batch: dict[str, Any] = {}
        for key in items[0]:
            stacked = np.stack([item[key] for item in items])
            batch[key] = torch.from_numpy(stacked).to(self.device)
        return batch

    def _context_length(self, sequences: list[str], extra_codons: int = 2) -> int:
        """Smallest multiple-of-eight context that fits the batch, capped at the model maximum."""
        max_codons = max((len(seq) // 3 for seq in sequences), default=1)
        required = max_codons + extra_codons
        return min(2048, ((required + 7) // 8) * 8)

    def score_fitness(self, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
        """Mean per-token log-likelihood for each coding sequence (higher = more model-typical)."""
        from src.data.preprocess.codon_sequence import process_item

        self.to_device(device)
        fitness: list[float] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            ctx = self._context_length(chunk)
            items = [process_item(seq, ctx, self.tokenizer) for seq in chunk]
            out = self.inference.predict_fitness(self._batch(items))
            fitness.extend(float(v) for v in np.asarray(out.fitness).reshape(-1))
        return fitness

    def extract_embeddings(self, *, sequences: list[str], batch_size: int, device: str) -> list[list[float]]:
        """CLS-token embedding vector for each coding sequence."""
        from src.data.preprocess.codon_sequence import process_item

        self.to_device(device)
        embeddings: list[list[float]] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            ctx = self._context_length(chunk)
            items = [process_item(seq, ctx, self.tokenizer) for seq in chunk]
            out = self.inference.extract_embeddings(self._batch(items))
            embeddings.extend(np.asarray(row, dtype=float).tolist() for row in out.embeddings)
        return embeddings

    def score_mutations(
        self, *, mutations: list[dict[str, Any]], batch_size: int, device: str
    ) -> list[dict[str, float]]:
        """Ref-vs-alt codon log-likelihood ratio per mutation (ref_ll - alt_ll; higher = ref favored)."""
        from src.data.preprocess.mutation_pred import mlm_process_item

        self.to_device(device)
        results: list[dict[str, float]] = []
        for start in range(0, len(mutations), batch_size):
            chunk = mutations[start : start + batch_size]
            ctx = self._context_length([m["sequence"] for m in chunk])
            items = [
                mlm_process_item(
                    m["sequence"], int(m["codon_position"]), m["ref_codon"], m["alt_codon"], ctx, self.tokenizer
                )
                for m in chunk
            ]
            out = self.inference.predict_mutation(self._batch(items))
            for ref_ll, alt_ll, ratio in zip(
                np.asarray(out.ref_likelihoods).reshape(-1),
                np.asarray(out.alt_likelihoods).reshape(-1),
                np.asarray(out.likelihood_ratios).reshape(-1),
                strict=True,
            ):
                results.append(
                    {"ref_log_likelihood": float(ref_ll), "alt_log_likelihood": float(alt_ll), "llr": float(ratio)}
                )
        return results

    def _embedding_layer(self) -> Any:
        """Locate the model's full ``CodonEmbedding`` module, robust to model wrapping."""
        from src.models.components.codon_embedding import CodonEmbedding

        for module in self.inference.model.modules():
            if isinstance(module, CodonEmbedding):
                return module
        raise RuntimeError("codonfm: could not locate the CodonEmbedding layer")

    def compute_gradient(
        self,
        logits_list: list[list[float]],
        *,
        temperature: float | None = None,
        soft: float = 1.0,
        hard: float = 0.0,
        backprop: bool = True,
        batch_size: int,
        device: str,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Chunked masked pseudo-log-likelihood gradient for a relaxed (soft) coding sequence.

        The differentiable state is a ``(L, 64)`` distribution over codons (lexicographic DNA
        order). Each codon position is masked in turn and predicted from the rest of the
        sequence; the objective is the mean masked negative log-likelihood of the current
        (argmax) codons, and the gradient flows back to the input codon logits through the
        soft codon embedding.

        Args:
            logits_list: Relaxed codon logits, shape ``(L, 64)``.
            temperature: Optional softmax temperature; ``None`` uses the logits as-is.
            soft: Blend from the raw input matrix (0) to softmax probabilities (1).
            hard: Straight-through coefficient; 1 forwards a hard one-hot.
            backprop: If ``False``, skip the backward pass and return ``gradient=None``.
            batch_size: Codon positions per forward pass.
            device: Execution device.
            verbose: Log progress.

        Returns:
            Dictionary with ``gradient``, mean-NLL ``loss``, metrics, and the codon ``vocab``.
        """
        import torch
        import torch.nn.functional as F
        from src.data.metadata import MetadataFields
        from tqdm import tqdm

        # Proto's column order, not the tokenizer's: codon_token_ids below carries the mapping,
        # so every gather keyed off it is proto-ordered on both the input and the gradient side.
        codons = list(DNA_CODONS)
        n_codons_vocab = len(codons)
        missing = [codon for codon in codons if codon not in self.tokenizer.encoder]
        if missing:
            raise ValueError(f"codonfm: tokenizer is missing codons {missing}")
        if not logits_list:
            raise ValueError("codonfm: compute_gradient requires at least one codon position")
        if any(len(row) != n_codons_vocab for row in logits_list):
            raise ValueError(f"codonfm: compute_gradient expects L x {n_codons_vocab} logits")

        self.to_device(device)
        dev = torch.device(device)

        embed_layer = self._embedding_layer()
        weight = embed_layer.word_embeddings.weight
        codon_token_ids = torch.tensor([self.tokenizer.encoder[c] for c in codons], device=dev)
        cls_id, sep_id = special_token_layout(self.tokenizer)
        mask_id = int(self.tokenizer.mask_token_id)
        pad_id = int(self.tokenizer.pad_token_id)

        with torch.set_grad_enabled(backprop):
            seq_dist = torch.tensor(logits_list, device=dev, dtype=torch.float32, requires_grad=backprop)
            sequence_length = int(seq_dist.shape[0])

            # Targets come from P, not the blend, so they do not move with soft/hard.
            probs = relaxed_sequence(seq_dist, temperature=temperature)
            codon_idx = probs.argmax(dim=-1).detach()
            residue_token_ids = codon_token_ids[codon_idx].detach()

            x = relaxed_sequence(seq_dist, temperature=temperature, soft=soft, hard=hard)

            codon_weight = weight[codon_token_ids]
            residue_embeddings = x @ codon_weight
            special_embeddings = weight[torch.tensor([cls_id, sep_id], device=dev)]

            raw_embeddings = torch.cat(
                (special_embeddings[0:1], residue_embeddings, special_embeddings[1:2]),
                dim=0,
            )
            # Ids and mask come from the same assembly the scoring paths tokenize into, so the
            # embeddings below are padded to a layout that cannot drift away from it.
            id_row, mask_row = build_model_input_row(self.tokenizer, residue_token_ids.tolist())
            padding = len(id_row) - raw_embeddings.shape[0]
            if padding:
                raw_embeddings = torch.cat((raw_embeddings, weight[pad_id].expand(padding, -1)), dim=0)
            input_embeddings = embed_layer.dropout(embed_layer.post_ln(raw_embeddings)).unsqueeze(0)
            token_ids = torch.tensor(id_row, device=dev, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor(mask_row, device=dev, dtype=torch.long).unsqueeze(0)

            codon_positions = torch.arange(1, sequence_length + 1, device=dev)
            seq_len = input_embeddings.shape[1]
            pos_idx = torch.arange(seq_len, device=dev)
            mask_emb = embed_layer.dropout(embed_layer.post_ln(weight[mask_id])).detach()

            ie_grad = torch.zeros_like(input_embeddings) if backprop else None
            total_loss_val = 0.0
            for start in tqdm(
                range(0, sequence_length, batch_size),
                desc="CodonFM gradient",
                unit="batch",
                disable=not verbose,
            ):
                end = min(start + batch_size, sequence_length)
                chunk_pos = codon_positions[start:end]
                chunk_len = end - start

                chunk_masked = pos_idx.unsqueeze(0) == chunk_pos.unsqueeze(1)
                ie_chunk = input_embeddings.detach().requires_grad_(True) if backprop else input_embeddings
                chunk_input = torch.where(
                    chunk_masked.unsqueeze(-1),
                    mask_emb.view(1, 1, -1).expand(chunk_len, seq_len, -1),
                    ie_chunk.expand(chunk_len, -1, -1),
                )
                chunk_idx = torch.arange(chunk_len, device=dev)
                chunk_token_ids = token_ids.expand(chunk_len, -1).clone()
                chunk_token_ids[chunk_idx, chunk_pos] = mask_id
                chunk_attention = attention_mask.expand(chunk_len, -1)

                handle = embed_layer.register_forward_hook(lambda _m, _i, _o, ci=chunk_input: ci)
                try:
                    outputs = self.inference.model(
                        {
                            MetadataFields.INPUT_IDS: chunk_token_ids,
                            MetadataFields.ATTENTION_MASK: chunk_attention,
                        }
                    )
                finally:
                    handle.remove()

                pred = outputs.logits[chunk_idx, chunk_pos, :]
                labels = token_ids[0, chunk_pos]
                loss_sum = F.cross_entropy(pred, labels, reduction="sum")
                total_loss_val += loss_sum.item()

                if backprop:
                    (loss_sum / sequence_length).backward()  # type: ignore[no-untyped-call, unused-ignore]
                    ie_chunk_grad = ie_chunk.grad
                    if ie_chunk_grad is None:
                        raise RuntimeError("codonfm: missing input-embedding gradient")
                    if ie_grad is None:
                        raise RuntimeError("codonfm: missing accumulated input-embedding gradient")
                    ie_grad = ie_grad + ie_chunk_grad

        mean_nll = total_loss_val / sequence_length
        gradient_value: list[list[float]] | None = None
        if backprop:
            if ie_grad is None:
                raise RuntimeError("codonfm: missing accumulated input-embedding gradient")
            (x_grad,) = torch.autograd.grad(input_embeddings, seq_dist, grad_outputs=ie_grad)
            gradient_value = x_grad.detach().cpu().tolist()

        return {
            "gradient": gradient_value,
            "loss": mean_nll,
            "metrics": {
                **log_likelihood_metrics(-mean_nll, sequence_length),
                "sequence_length": sequence_length,
                "objective": "masked_pll",
            },
            "vocab": codons,
        }

    def sample_sequences(
        self,
        *,
        sequences: list[str],
        temperature: float,
        batch_size: int,
        device: str,
        excluded_codons: list[str] | None = None,
    ) -> list[str]:
        """Refill the codons marked ``___`` by drawing new ones from Encodon.

        Which codons to resample is decided before the call and carried in the sequences
        themselves, the same contract the ESM samplers use. Each masked codon is refilled from
        the model's per-codon distribution (temperature-scaled softmax over the 61 sense codons;
        the three stop codons are excluded so resampling never inserts a premature stop, as are
        any ``excluded_codons``) in a single forward pass. Sequence length is preserved.

        Args:
            sequences: Coding sequences (codon-aligned DNA) with resample positions as ``___``.
            temperature: Softmax temperature for codon sampling (higher = more diverse).
            batch_size: Sequences per forward pass (grouped by length).
            device: Execution device. Codon draws follow the process-wide torch seed, set by
                the dispatcher before this runs.
            excluded_codons: Sense codons removed from the draw, as resolved by the caller from
                its ``excluded_amino_acids`` setting.

        Returns:
            list[str]: One mutated coding sequence per input, in input order.
        """
        import torch
        from src.data.metadata import MetadataFields
        from src.data.preprocess.codon_sequence import process_item

        self.to_device(device)
        dev = torch.device(device)
        # Restrict resampling to the 61 sense codons so a masked interior codon can never be
        # replaced with a premature stop (TAA/TAG/TGA); ``codons``/``codon_token_ids`` stay aligned.
        # Codons for an excluded amino acid drop out here, so the draw never reaches them.
        banned = _STOP_CODONS | frozenset(excluded_codons or ())
        codons = [c for c in self.tokenizer.codons if c not in banned]
        if not codons:
            raise ValueError("codonfm: excluded_codons leaves no sense codon to sample")
        codon_token_ids = torch.tensor([self.tokenizer.encoder[c] for c in codons], device=dev)
        mask_id = int(self.tokenizer.mask_token_id)

        # Read the resample positions off the sequences, and fill each masked codon with a
        # placeholder so the tokenizer sees a real codon. The placeholder never reaches the
        # model: every masked position is overwritten with mask_id before the forward pass.
        placeholder = codons[0]
        plans: list[tuple[int, list[int]]] = []
        unmasked: list[str] = []
        for seq in sequences:
            seq_codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
            positions = [i for i, codon in enumerate(seq_codons) if codon == _MASK_CODON]
            plans.append((len(seq_codons), positions))
            unmasked.append("".join(placeholder if codon == _MASK_CODON else codon for codon in seq_codons))
        sequences = unmasked

        mutated = list(sequences)
        # Group by codon count so a batch shares one context length.
        order = sorted(range(len(sequences)), key=lambda i: plans[i][0])
        grouped: dict[int, list[int]] = {}
        for i in order:
            grouped.setdefault(plans[i][0], []).append(i)

        for indices in grouped.values():
            ctx = self._context_length([sequences[indices[0]]])
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                items = [process_item(sequences[i], ctx, self.tokenizer) for i in chunk]
                batch: dict[str, Any] = {}
                for key in items[0]:
                    batch[key] = torch.from_numpy(np.stack([item[key] for item in items])).to(dev)
                input_ids = batch[MetadataFields.INPUT_IDS]
                for row, i in enumerate(chunk):
                    for pos in plans[i][1]:
                        input_ids[row, pos + 1] = mask_id  # +1 for the leading CLS token
                with torch.no_grad():
                    logits = self.inference.model(batch).logits.float()
                for row, i in enumerate(chunk):
                    positions = plans[i][1]
                    if not positions:
                        continue
                    seq_chars = list(sequences[i])
                    for pos in positions:
                        codon_logits = logits[row, pos + 1, codon_token_ids] / temperature
                        probs = torch.softmax(codon_logits, dim=-1)
                        sampled = int(torch.multinomial(probs, num_samples=1).item())
                        seq_chars[pos * 3 : pos * 3 + 3] = codons[sampled]
                    mutated[i] = "".join(seq_chars)
        return mutated


_MODEL = CodonFMModel()


def _validate_request(input_dict: dict[str, Any], operation: str) -> None:
    """Reject malformed raw-worker payloads before resolving or loading a checkpoint."""
    # Sampling rewrites codons, so its sequences carry ``___`` marking which ones to redraw.
    # No other operation does, and a mask reaching one of those is a caller error worth naming.
    alphabet = "ACGT_" if operation == "sample" else "ACGT"

    def valid_sequence(sequence: Any) -> bool:
        return (
            isinstance(sequence, str)
            and 0 < len(sequence) <= 6138
            and len(sequence) % 3 == 0
            and all(base in alphabet for base in sequence)
        )

    if operation in {"fitness", "embeddings", "sample"}:
        sequences = input_dict.get("sequences")
        if not isinstance(sequences, list) or not sequences or not all(valid_sequence(seq) for seq in sequences):
            raise ValueError(f"codonfm: {operation} requires non-empty, codon-aligned {alphabet} sequences")
    elif operation == "score":
        mutations = input_dict.get("mutations")
        if not isinstance(mutations, list) or not mutations or not all(isinstance(item, dict) for item in mutations):
            raise ValueError("codonfm: score requires a non-empty list of mutations")
        for mutation in mutations:
            position = mutation.get("codon_position")
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("codonfm: worker codon_position must be a non-negative integer")
            if not valid_sequence(mutation.get("sequence")):
                raise ValueError("codonfm: score requires codon-aligned ACGT sequences")
            if position >= len(mutation["sequence"]) // 3:
                raise ValueError("codonfm: worker codon_position is out of range")
    elif operation == "gradient":
        items = input_dict.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("codonfm: gradient requires at least one item")
        for item in items:
            if not isinstance(item, list) or not item or len(item) > 2046:
                raise ValueError("codonfm: gradient requires 1 to 2046 logits rows per item")
            if any(
                not isinstance(row, list)
                or len(row) != 64
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in row
                )
                for row in item
            ):
                raise ValueError("codonfm: gradient logits must be a finite L x 64 numeric matrix")

    if operation == "sample":
        value = input_dict.get("temperature", 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("codonfm: temperature must be a positive finite number")
        if not any(_MASK_CODON in sequence for sequence in input_dict["sequences"]):
            raise ValueError("codonfm: sample needs at least one masked codon ('___') to resample")
    elif operation == "gradient":
        temperature = input_dict.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("codonfm: temperature must be a positive finite number or null")


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Route a decoded input dict to the requested CodonFM operation."""
    operation = input_dict.get("operation")
    allowed_operations = {"fitness", "embeddings", "score", "sample", "gradient"}
    if operation not in allowed_operations:
        raise ValueError(f"codonfm: unknown operation {operation!r}")
    device = input_dict.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("codonfm: device must be a non-empty string")
    verbose_raw = input_dict.get("verbose", False)
    if isinstance(verbose_raw, bool):
        verbose = verbose_raw
    elif isinstance(verbose_raw, int) and verbose_raw >= 0:
        verbose = verbose_raw > 0
    else:
        raise ValueError("codonfm: verbose must be a boolean or non-negative integer")
    batch_size = input_dict.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("codonfm: batch_size must be a positive integer")
    for name in ("compute_gradient",):
        if name in input_dict and not isinstance(input_dict[name], bool):
            raise ValueError(f"codonfm: {name} must be a boolean")
    _validate_request(input_dict, operation)
    set_torch_seed(input_dict.get("seed"))

    _MODEL.load(checkpoint_path=_resolve_checkpoint(input_dict), device=device, verbose=verbose)

    if operation == "fitness":
        return {
            "fitness": _MODEL.score_fitness(sequences=input_dict["sequences"], batch_size=batch_size, device=device)
        }
    if operation == "embeddings":
        return {
            "embeddings": _MODEL.extract_embeddings(
                sequences=input_dict["sequences"], batch_size=batch_size, device=device
            )
        }
    if operation == "score":
        return {
            "mutations": _MODEL.score_mutations(mutations=input_dict["mutations"], batch_size=batch_size, device=device)
        }
    if operation == "sample":
        return {
            "sequences": _MODEL.sample_sequences(
                sequences=input_dict["sequences"],
                temperature=float(input_dict.get("temperature", 1.0)),
                batch_size=batch_size,
                device=device,
                excluded_codons=input_dict.get("excluded_codons"),
            )
        }
    if operation == "gradient":
        # One chunk of independent candidates: the checkpoint loads once and items are evaluated in turn.
        results = [
            _MODEL.compute_gradient(
                item,
                temperature=resolve_temperature(input_dict.get("temperature")),
                soft=float(input_dict.get("soft", 1.0)),
                hard=float(input_dict.get("hard", 0.0)),
                backprop=input_dict.get("compute_gradient", True),
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
            for item in input_dict["items"]
        ]
        return {
            "results": [{key: result[key] for key in ("gradient", "loss", "metrics")} for result in results],
            "vocab": results[0]["vocab"],
        }
    raise AssertionError("unreachable")


def to_device(device: str) -> dict[str, Any]:
    """Move the loaded Encodon model to a DeviceManager-selected device."""
    if _MODEL.inference is not None:
        _MODEL.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report PyTorch device memory usage for DeviceManager monitoring."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(_MODEL.device or "cpu")  # type: ignore[no-any-return]


def main() -> None:
    """One-shot entry point: ``python inference.py input.json output.json``."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: inference.py <input.json> <output.json>")
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    result = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)


if __name__ == "__main__":
    main()
