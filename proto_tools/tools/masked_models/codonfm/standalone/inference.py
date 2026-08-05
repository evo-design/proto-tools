"""CodonFM (Encodon) standalone inference for venv execution.

Drives the vendored ``EncodonInference`` (under ``src/``) for all five CodonFM tools. Reads a
JSON input file and writes a JSON output file, matching the proto one-shot dispatch contract
(``python inference.py input.json output.json``).
"""

import contextlib
import fcntl
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from standalone_helpers import (
    get_logger,
    log_likelihood_metrics,
    move_model_to_device,
    resolve_weights_dir,
    serialize_output,
    set_torch_seed,
)

# The vendored CodonFM source lives under ``src/`` next to this file.
sys.path.insert(0, str(Path(__file__).parent))

logger = get_logger(__name__)

LOCK_TIMEOUT_SECONDS = 600

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
def _is_trusted_hf_url(url: str) -> bool:
    """True only for an HTTPS URL on ``huggingface.co`` or a real subdomain."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co"))


def _redact_url(url: str) -> str:
    """Strip userinfo/query so signed URLs and creds never reach logs/errors."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.hostname or ""
        if ":" in netloc:
            netloc = f"[{netloc}]"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        return "<unparseable-url>"
    cleaned = parsed._replace(netloc=netloc, path=("/<redacted-path>" if parsed.path else ""), query="", fragment="")
    return urllib.parse.urlunparse(cleaned)


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Abort insecure redirects and drop ``Authorization`` when a hop leaves trusted HF."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "codonfm: refusing insecure (non-HTTPS) redirect", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _is_trusted_hf_url(newurl):
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-process advisory lock via ``fcntl.flock`` (kernel-released on process exit)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as err:
                if time.monotonic() - started > LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(f"codonfm: timed out waiting for lock {lock_path}") from err
                time.sleep(1)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _resolve_hf_token() -> str | None:
    """Resolve a HuggingFace token from every source the main-process gate accepts.

    Mirrors ``proto_tools.utils.auth.resolve_hf_token`` (which isn't importable in the isolated
    standalone env) so a token provided via ``hf auth login`` works for the download too, not just
    the ``HF_TOKEN`` env var: env ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``, then the cached token
    file, then git credentials.
    """
    token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if token:
        return token
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_file):
        with contextlib.suppress(OSError):
            token = Path(token_file).read_text().strip()
            if token:
                return token
    git_creds = os.path.expanduser("~/.git-credentials")
    if os.path.isfile(git_creds):
        with contextlib.suppress(OSError):
            for line in Path(git_creds).read_text().splitlines():
                match = re.search(r"https?://[^:]+:(hf_[^@]+)@huggingface\.co", line)
                if match:
                    return match.group(1)
    return None


def _download(url: str, dest: Path) -> None:
    """Download a public HuggingFace file over HTTPS, using an optional token when available."""
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"codonfm: checkpoint URL must use https, got {_redact_url(url)!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    logger.info("Downloading CodonFM checkpoint file from %s to %s", _redact_url(url), dest)
    request = urllib.request.Request(url)
    hf_token = _resolve_hf_token()
    if _is_trusted_hf_url(url) and hf_token:
        request.add_header("Authorization", f"Bearer {hf_token}")
    opener = urllib.request.build_opener(_SecureRedirectHandler())
    try:
        with opener.open(request, timeout=300) as response, open(tmp, "wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _weights_dir() -> Path:
    resolved = resolve_weights_dir("codonfm")
    if resolved:
        return Path(resolved)
    fallback = Path(tempfile.gettempdir()) / "proto_codonfm_weights"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _resolve_checkpoint(input_dict: dict[str, Any]) -> str:
    """Provision ``<name>.safetensors`` + ``config.json`` into one cache dir; return the safetensors path.

    ``EncodonInference`` requires both files side by side. The repositories are public under the
    NVIDIA Open Model License; an available ``HF_TOKEN`` is used but is not required.
    """
    safetensors_url = input_dict["safetensors_url"]
    config_url = input_dict["config_url"]
    filename = input_dict["safetensors_filename"]
    cache_dir = _weights_dir() / input_dict["cache_subdir"]
    safetensors_path = cache_dir / filename
    config_path = cache_dir / "config.json"

    with _file_lock(cache_dir.parent / f".{input_dict['cache_subdir']}.download.lock"):
        if not safetensors_path.is_file():
            _download(safetensors_url, safetensors_path)
        if not config_path.is_file():
            _download(config_url, config_path)
    return str(safetensors_path)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
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
        use_ste: bool = False,
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
            use_ste: Straight-Through Estimator — hard one-hot codons in the forward pass with
                gradients flowing through the soft probabilities.
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

        codons = list(self.tokenizer.codons)
        n_codons_vocab = len(codons)
        if not logits_list:
            raise ValueError("codonfm: compute_gradient requires at least one codon position")
        if any(len(row) != n_codons_vocab for row in logits_list):
            raise ValueError(f"codonfm: compute_gradient expects L x {n_codons_vocab} logits")

        self.to_device(device)
        dev = torch.device(device)

        embed_layer = self._embedding_layer()
        weight = embed_layer.word_embeddings.weight
        codon_token_ids = torch.tensor([self.tokenizer.encoder[c] for c in codons], device=dev)
        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        mask_id = int(self.tokenizer.mask_token_id)
        pad_id = int(self.tokenizer.pad_token_id)

        with torch.set_grad_enabled(backprop):
            seq_dist = torch.tensor(logits_list, device=dev, dtype=torch.float32, requires_grad=backprop)
            sequence_length = int(seq_dist.shape[0])

            x = F.softmax(seq_dist / temperature, dim=-1) if temperature is not None else seq_dist
            codon_idx = x.argmax(dim=-1).detach()
            residue_token_ids = codon_token_ids[codon_idx].detach()

            if use_ste:
                hard = F.one_hot(codon_idx, num_classes=n_codons_vocab).float()
                x = hard + (x - x.detach())

            codon_weight = weight[codon_token_ids]
            residue_embeddings = x @ codon_weight
            special_embeddings = weight[torch.tensor([cls_id, sep_id], device=dev)]

            raw_embeddings = torch.cat(
                (special_embeddings[0:1], residue_embeddings, special_embeddings[1:2]),
                dim=0,
            )
            raw_token_ids = torch.cat(
                (
                    torch.tensor([cls_id], device=dev),
                    residue_token_ids,
                    torch.tensor([sep_id], device=dev),
                ),
                dim=0,
            )
            context_length = min(2048, ((sequence_length + 2 + 7) // 8) * 8)
            padding = context_length - raw_token_ids.shape[0]
            if padding:
                raw_embeddings = torch.cat((raw_embeddings, weight[pad_id].expand(padding, -1)), dim=0)
                raw_token_ids = F.pad(raw_token_ids, (0, padding), value=pad_id)
            input_embeddings = embed_layer.dropout(embed_layer.post_ln(raw_embeddings)).unsqueeze(0)
            token_ids = raw_token_ids.unsqueeze(0)
            attention_mask = torch.zeros_like(token_ids)
            attention_mask[:, : sequence_length + 2] = 1

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
                    (loss_sum / sequence_length).backward()  # type: ignore[no-untyped-call]
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
    ) -> list[str]:
        """Refill the codons marked ``___`` by drawing new ones from Encodon.

        Which codons to resample is decided before the call and carried in the sequences
        themselves, the same contract the ESM samplers use. Each masked codon is refilled from
        the model's per-codon distribution (temperature-scaled softmax over the 61 sense codons;
        the three stop codons are excluded so resampling never inserts a premature stop) in a
        single forward pass. Sequence length is preserved.

        Args:
            sequences: Coding sequences (codon-aligned DNA) with resample positions as ``___``.
            temperature: Softmax temperature for codon sampling (higher = more diverse).
            batch_size: Sequences per forward pass (grouped by length).
            device: Execution device. Codon draws follow the process-wide torch seed, set by
                the dispatcher before this runs.

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
        codons = [c for c in self.tokenizer.codons if c not in _STOP_CODONS]
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
        logits = input_dict.get("logits")
        if not isinstance(logits, list) or not logits or len(logits) > 2046:
            raise ValueError("codonfm: gradient requires 1 to 2046 logits rows")
        if any(
            not isinstance(row, list)
            or len(row) != 64
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in row
            )
            for row in logits
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
    for name in ("use_ste", "compute_gradient"):
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
            )
        }
    if operation == "gradient":
        return _MODEL.compute_gradient(
            input_dict["logits"],
            temperature=input_dict.get("temperature"),
            use_ste=input_dict.get("use_ste", False),
            backprop=input_dict.get("compute_gradient", True),
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
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
