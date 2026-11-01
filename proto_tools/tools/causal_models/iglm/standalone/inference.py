"""Local IgLM inference implementation.

Weights ship inside the ``iglm`` wheel (``iglm/trained_models/{IgLM,IgLM-S}``),
so there is nothing to download and no model cache to resolve.
Based on the IgLM repository: https://github.com/Graylab/IgLM

Upstream's ``iglm.IgLM.generate``/``infill``/``log_likelihood`` are deliberately
not called: they hardcode ``max_length=150``, draw one candidate per forward
pass, and return ``list(set(...))``, whose order varies between processes even
under a fixed seed. This module reuses upstream's checkpoint paths, bad-word
ids, and format helpers while batching the draws and keeping the dedup ordered.
"""

import json
import sys
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from standalone_helpers import (
    get_logger,
    log_likelihood_metrics,
    move_model_to_device,
    serialize_output,
    set_torch_seed,
)
from tqdm import tqdm

logger = get_logger(__name__)

IGLM_MODEL_CHECKPOINTS = Literal[
    "IgLM",  # 13M parameters, the model from the paper
    "IgLM-S",  # 1.4M parameters
]

# Positional-embedding cap of both released checkpoints (GPT-2 n_positions=512).
IGLM_MAX_CONTEXT = 512

# Generation cap used by upstream's iglm.IgLM._generate.
IGLM_DEFAULT_MAX_LENGTH = 150

# Rejection-sampling budget per item: candidates drawn before giving up on the requested count.
IGLM_CANDIDATES_PER_SEQUENCE = 20
IGLM_MIN_CANDIDATE_BUDGET = 100


class IgLMModel:
    """IgLM model wrapper for antibody generation, infilling, and scoring.

    Handles model loading, device management, batched rejection sampling, and
    batched likelihood scoring.
    """

    def __init__(self, model_checkpoint: IGLM_MODEL_CHECKPOINTS = "IgLM"):
        """Initialize IgLM model wrapper.

        Args:
            model_checkpoint (IGLM_MODEL_CHECKPOINTS): IgLM weights variant.
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.tokenizer: Any = None
        self.device: str | None = None
        self.model: Any = None
        self.vocab: list[str] = []

    def load(self, device: str, verbose: bool = False) -> None:
        """Load IgLM weights and tokenizer from the installed package to device."""
        if verbose:
            logger.info(f"Loading IgLM model: {self.model_checkpoint} on {device}")
        logger.update_status(f"Loading IgLM {self.model_checkpoint}")

        import transformers
        from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE

        if self.model_checkpoint not in CHECKPOINT_DICT:
            raise ValueError(
                f"iglm: unknown model_checkpoint {self.model_checkpoint!r}; valid: {sorted(CHECKPOINT_DICT)}"
            )

        checkpoint_dir = CHECKPOINT_DICT[self.model_checkpoint]
        try:
            self.model = transformers.GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device).eval()
        except OSError as e:
            raise RuntimeError(f"iglm: weight load from {checkpoint_dir!r} failed: {e}") from e

        self.tokenizer = transformers.BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)
        self.vocab = self.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("IgLM model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise ValueError("iglm: cannot move unloaded model to device — call load() first")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Unload model to free GPU memory."""
        if self._loaded and self.device != "cpu":
            import torch

            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")
            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_ready(self, device: str, verbose: bool, seed: int | None) -> None:
        """Lazy-load (or relocate) the model, then seed RNGs for this dispatch."""
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)
        # Seed after load so each dispatch enters sampling with the same RNG state.
        set_torch_seed(seed)

    def _token_ids(self, tokens: list[str], item_index: int) -> list[int]:
        """Convert IgLM tokens to ids, rejecting anything outside IgLM's vocabulary."""
        token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(tokens)
        unknown = sorted({t for t, i in zip(tokens, token_ids, strict=True) if i == self.tokenizer.unk_token_id})
        if unknown:
            raise ValueError(
                f"iglm: item {item_index} uses token(s) {unknown} that IgLM's vocabulary does not cover; "
                f"sequences must be the 20 standard amino acids and conditioning must be an IgLM chain/species token."
            )
        return token_ids

    def _ran_out_of_budget(self, candidate: Any, max_length: int) -> bool:
        """Report whether ``max_length`` ended this candidate instead of the model.

        ``forced_eos_token_id`` writes the terminator at the last budgeted position, so a
        generation that ran out of room still satisfies upstream's well-formedness check
        while missing part of what it was writing — for infilling, where the mask and
        separator come from the prompt, that silently returns a clipped span. The
        terminator landing on the final position is the exact signature, and discarding
        those candidates is what keeps a short budget from passing as a finished design.
        """
        terminator_positions = (candidate == self.tokenizer.cls_token_id).nonzero()[0]
        return bool(terminator_positions.size and terminator_positions.min() == max_length - 1)

    def _reassemble(self, candidate: Any) -> str:
        """Turn one generated id row into IgLM's infilled residue sequence."""
        from iglm.model.utils import iglm_to_infilled

        decoded = self.tokenizer.decode(iglm_to_infilled(candidate, self.tokenizer))
        return "".join(decoded).replace(" ", "")

    def _sample_unique(
        self,
        start_tokens: list[str],
        item_index: int,
        num_sequences: int,
        temperature: float,
        top_p: float,
        max_length: int,
        batch_size: int,
        accept: Callable[[str], Any],
    ) -> list[Any]:
        """Draw candidates in batches until ``num_sequences`` distinct ones are accepted.

        All candidates for one item share the same prompt, so the batch needs no
        padding: a single ``num_return_sequences`` expansion covers it.

        Args:
            start_tokens (list[str]): Conditioning plus prompt tokens shared by every candidate.
            item_index (int): Index of the input item, used in error messages.
            num_sequences (int): Distinct accepted candidates to return.
            temperature (float): Softmax temperature.
            top_p (float): Nucleus sampling cutoff.
            max_length (int): Token budget per generation, including ``start_tokens``.
            batch_size (int): Candidates drawn per forward pass.
            accept (Callable[[str], Any]): Maps a reassembled sequence to the value to
                keep, or returns ``None`` to reject the candidate as malformed.

        Returns:
            list[Any]: The accepted values, in generation order.
        """
        import torch
        from iglm.model.tokens import BAD_WORD_IDS
        from iglm.model.utils import validate_generated_seq

        if max_length > IGLM_MAX_CONTEXT:
            raise ValueError(
                f"iglm: max_length is {max_length} but both checkpoints are limited to {IGLM_MAX_CONTEXT} positions."
            )
        if len(start_tokens) >= max_length:
            raise ValueError(
                f"iglm: item {item_index} conditions on {len(start_tokens)} tokens but max_length is {max_length}, "
                f"leaving no room to generate; raise max_length (up to {IGLM_MAX_CONTEXT})."
            )

        token_ids = self._token_ids(start_tokens, item_index)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        # Ordered dedup: dict keys preserve generation order, unlike upstream's set().
        accepted: dict[str, Any] = {}
        budget = max(IGLM_CANDIDATES_PER_SEQUENCE * num_sequences, IGLM_MIN_CANDIDATE_BUDGET)
        drawn = 0
        truncated = 0

        with torch.no_grad():
            while len(accepted) < num_sequences and drawn < budget:
                # Capped by what is still outstanding (with room for a rejection or two), so a
                # large batch_size speeds up a library without over-drawing a single design.
                outstanding = num_sequences - len(accepted)
                num_to_draw = max(1, min(batch_size, 2 * outstanding, budget - drawn))
                candidates = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.cls_token_id,
                    forced_eos_token_id=self.tokenizer.cls_token_id,
                    bad_words_ids=BAD_WORD_IDS,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=num_to_draw,
                )
                drawn += num_to_draw

                for candidate in candidates.detach().cpu().numpy():
                    # Attributed before the format check: a clipped generation is usually also
                    # malformed, and blaming the distribution for it points at the wrong remedy.
                    if self._ran_out_of_budget(candidate, max_length):
                        truncated += 1
                        continue
                    if not validate_generated_seq(candidate, self.tokenizer):
                        continue
                    sequence = self._reassemble(candidate)
                    if sequence in accepted:
                        continue
                    # A rejected candidate still spends its draw, so the budget bounds total work.
                    value = accept(sequence)
                    if value is not None:
                        accepted[sequence] = value

        if len(accepted) < num_sequences:
            # Truncation dominating the draws means the budget, not the distribution, is the problem.
            remedy = (
                f"raise max_length (currently {max_length}, up to {IGLM_MAX_CONTEXT})"
                if truncated > drawn - truncated
                else "raise temperature or top_p to widen the distribution, or lower num_sequences_per_input"
            )
            raise ValueError(
                f"iglm: item {item_index} yielded only {len(accepted)} of {num_sequences} requested unique "
                f"sequences after {drawn} candidates ({truncated} ran out of max_length), the per-item "
                f"ceiling; {remedy}."
            )

        if truncated:
            logger.info(
                "iglm: item %d discarded %d of %d candidates that ran out of max_length (%d); "
                "raise it if generation is slow.",
                item_index,
                truncated,
                drawn,
                max_length,
            )

        return list(accepted.values())[:num_sequences]

    def sample(
        self,
        prompts: list[str],
        chain_tokens: list[str],
        species_tokens: list[str],
        num_sequences_per_input: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_length: int = IGLM_DEFAULT_MAX_LENGTH,
        prepend_prompt: bool = True,
        batch_size: int = 8,
        device: str = "cuda",
        verbose: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate full-length antibody sequences, one bundle per prompt.

        Args:
            prompts (list[str]): N-terminal prefix per item; an empty string generates
                from the conditioning tokens alone.
            chain_tokens (list[str]): IgLM chain token per item.
            species_tokens (list[str]): IgLM species token per item.
            num_sequences_per_input (int): Distinct sequences per prompt.
            temperature (float): Softmax temperature.
            top_p (float): Nucleus sampling cutoff.
            max_length (int): Token budget per generation, including conditioning and prompt.
            prepend_prompt (bool): Keep the prompt at the start of each returned sequence.
            batch_size (int): Candidates drawn per forward pass.
            device (str): Device to run on.
            verbose (bool): Verbose logging.
            seed (int | None): Random seed for reproducibility.

        Returns:
            dict[str, Any]: ``{"sequences": [[str, ...], ...]}`` with one list per prompt,
                in input order.
        """
        if not prompts:
            raise ValueError("iglm: cannot sample from empty prompt list")

        self._ensure_ready(device, verbose, seed)
        logger.update_status(f"Generating {num_sequences_per_input} sequence(s) for {len(prompts)} prompt(s)")

        items = zip(prompts, chain_tokens, species_tokens, strict=True)
        all_sequences: list[list[str]] = []

        for item_index, (prompt, chain_token, species_token) in enumerate(
            tqdm(list(items), desc="IgLM Sequence Generation", unit="prompt", disable=not verbose)
        ):
            # Upstream conditions on [chain, species] then the prompt's residues, one token each.
            start_tokens = [chain_token, species_token, *prompt]
            all_sequences.append(
                self._sample_unique(
                    start_tokens=start_tokens,
                    item_index=item_index,
                    num_sequences=num_sequences_per_input,
                    temperature=temperature,
                    top_p=top_p,
                    max_length=max_length,
                    batch_size=batch_size,
                    accept=partial(_reported_sample, prompt=prompt, prepend_prompt=prepend_prompt),
                )
            )

        return {"sequences": all_sequences}

    def infill(
        self,
        sequences: list[str],
        infill_ranges: list[list[int]],
        chain_tokens: list[str],
        species_tokens: list[str],
        num_sequences_per_input: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_length: int = IGLM_DEFAULT_MAX_LENGTH,
        batch_size: int = 8,
        device: str = "cuda",
        verbose: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Re-design a masked span of each sequence, one bundle per sequence.

        Candidates that lose a flank, or that write nothing into the span, are
        rejected against the same attempt budget as a malformed generation; see
        :func:`_extract_infilled_residues`.

        Args:
            sequences (list[str]): Parent antibody sequences.
            infill_ranges (list[list[int]]): ``[start, end)`` per sequence, 0-indexed
                and end-exclusive, matching upstream's ``mask_span``.
            chain_tokens (list[str]): IgLM chain token per item.
            species_tokens (list[str]): IgLM species token per item.
            num_sequences_per_input (int): Distinct sequences per parent sequence.
            temperature (float): Softmax temperature.
            top_p (float): Nucleus sampling cutoff.
            max_length (int): Token budget per generation, including conditioning and
                the masked parent sequence.
            batch_size (int): Candidates drawn per forward pass.
            device (str): Device to run on.
            verbose (bool): Verbose logging.
            seed (int | None): Random seed for reproducibility.

        Returns:
            dict[str, Any]: ``{"sequences": [[str, ...], ...], "infilled_residues": [[str, ...], ...]}``
                with aligned inner lists, one pair per parent sequence in input order.
        """
        from iglm.model.utils import mask_span

        if not sequences:
            raise ValueError("iglm: cannot infill an empty sequence list")

        self._ensure_ready(device, verbose, seed)
        logger.update_status(f"Infilling {num_sequences_per_input} span(s) for {len(sequences)} sequence(s)")

        items = zip(sequences, infill_ranges, chain_tokens, species_tokens, strict=True)
        all_sequences: list[list[str]] = []
        all_residues: list[list[str]] = []

        for item_index, (sequence, infill_range, chain_token, species_token) in enumerate(
            tqdm(list(items), desc="IgLM Span Re-design", unit="sequence", disable=not verbose)
        ):
            start, end = infill_range
            prefix, suffix = sequence[:start], sequence[end:]
            start_tokens = [chain_token, species_token, *mask_span(list(sequence), start, end)]

            accepted = self._sample_unique(
                start_tokens=start_tokens,
                item_index=item_index,
                num_sequences=num_sequences_per_input,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length,
                batch_size=batch_size,
                accept=partial(_extract_infilled_residues, prefix=prefix, suffix=suffix),
            )
            all_sequences.append([designed for designed, _ in accepted])
            all_residues.append([residues for _, residues in accepted])

        return {"sequences": all_sequences, "infilled_residues": all_residues}

    def score(
        self,
        sequences: list[str],
        infill_ranges: list[list[int] | None],
        chain_tokens: list[str],
        species_tokens: list[str],
        batch_size: int = 8,
        return_logits: bool = False,
        device: str = "cuda",
        verbose: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Score sequences by IgLM's autoregressive likelihood, batched with right padding.

        Replicates upstream's ``IgLM.log_likelihood``: without a range the whole
        sequence plus its terminal ``[SEP]`` is scored, and with a range the span is
        appended after ``[SEP]`` and only that region is scored. Right padding plus an
        attention mask keeps each sequence's position ids identical to the unbatched
        case, and metrics are computed over each sequence's own eval region.

        Args:
            sequences (list[str]): Antibody sequences to score.
            infill_ranges (list[list[int] | None]): ``[start, end)`` per sequence,
                0-indexed and end-exclusive, or ``None`` to score the whole sequence.
            chain_tokens (list[str]): IgLM chain token per item.
            species_tokens (list[str]): IgLM species token per item.
            batch_size (int): Sequences per forward pass.
            return_logits (bool): Include the logits rows that produced the scored labels.
            device (str): Device to run on.
            verbose (bool): Verbose logging.
            seed (int | None): Random seed. Scoring is deterministic given the model
                state, but seeding keeps consecutive calls in a persistent worker
                behaving identically regardless of call order.

        Returns:
            dict[str, Any]: ``{"metrics": [...], "logits": [...] | None, "vocab": [...]}``
                with one metrics dict per sequence, in input order.
        """
        import torch

        if not sequences:
            raise ValueError("iglm: cannot score an empty sequence list")

        self._ensure_ready(device, verbose, seed)
        logger.update_status(f"Scoring {len(sequences)} sequence(s)")

        items = list(zip(sequences, infill_ranges, chain_tokens, species_tokens, strict=True))
        encoded = [self._encode_for_scoring(item, index) for index, item in enumerate(items)]
        batches = [encoded[i : i + batch_size] for i in range(0, len(encoded), batch_size)]

        all_metrics: list[dict[str, float]] = []
        all_logits: list[Any] = []

        with torch.inference_mode():
            for batch_index, batch in enumerate(
                tqdm(batches, desc="IgLM Sequence Scoring", unit="batch", total=len(batches))
            ):
                if verbose:
                    logger.info(f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} sequences)")

                lengths = [len(token_ids) for token_ids, _ in batch]
                max_len = max(lengths)
                input_ids = torch.full(
                    (len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
                )
                attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=self.device)
                for i, (token_ids, _) in enumerate(batch):
                    input_ids[i, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=self.device)
                    attention_mask[i, : len(token_ids)] = 1

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)

                for i, ((_, eval_start), length) in enumerate(zip(batch, lengths, strict=True)):
                    # Upstream shifts logits[eval_start:-1] against labels[eval_start+1:].
                    scored_log_probs = log_probs[i, eval_start : length - 1]
                    scored_labels = input_ids[i, eval_start + 1 : length]
                    token_log_probs = scored_log_probs.gather(1, scored_labels.unsqueeze(1)).squeeze(1)

                    all_metrics.append(log_likelihood_metrics(token_log_probs.mean().item(), token_log_probs.shape[0]))
                    if return_logits:
                        all_logits.append(logits[i, eval_start : length - 1, : len(self.vocab)].cpu())

        return {
            "metrics": all_metrics,
            "logits": all_logits if return_logits else None,
            "vocab": self.vocab,
        }

    def _encode_for_scoring(
        self, item: tuple[str, list[int] | None, str, str], item_index: int
    ) -> tuple[list[int], int]:
        """Build one scoring token sequence and its eval start, following upstream exactly."""
        from iglm.model.utils import mask_span

        sequence, infill_range, chain_token, species_token = item
        residues = list(sequence)

        if infill_range is None:
            token_seq = [chain_token, species_token, *residues, self.tokenizer.sep_token]
            eval_start = 1
        else:
            masked = mask_span(residues, infill_range[0], infill_range[1], append_span=True)
            token_seq = [chain_token, species_token, *masked, self.tokenizer.cls_token]
            # mask_span's [SEP] separates the masked chain from the appended span; scoring starts there.
            eval_start = token_seq.index(self.tokenizer.sep_token)

        token_ids = self._token_ids(token_seq, item_index)
        if len(token_ids) > IGLM_MAX_CONTEXT:
            raise ValueError(
                f"iglm: item {item_index} needs {len(token_ids)} tokens but IgLM's context is "
                f"{IGLM_MAX_CONTEXT}; score a shorter sequence."
            )
        return token_ids, eval_start


def _reported_sample(candidate: str, prompt: str, prepend_prompt: bool) -> str:
    """Return the sequence to report for one accepted ``sample`` candidate.

    Every candidate for an item is reassembled with the prompt at its start, so
    ``prepend_prompt=False`` just drops that known prefix.
    """
    return candidate if prepend_prompt else candidate[len(prompt) :]


def _extract_infilled_residues(candidate: str, prefix: str, suffix: str) -> tuple[str, str] | None:
    """Split a reassembled candidate into (full sequence, infilled residues), or reject it.

    IgLM rebuilds the candidate as ``prefix + <new residues> + suffix``. Two
    candidates are rejected by returning ``None`` so the caller draws another:

    - One that lost either flank, which means the reassembly is malformed.
    - One that wrote nothing into the span, which deletes the segment rather than
      re-designing it. A zero-length write is not a usable design, and the tool
      layer reports each design's resulting span from the written length, which
      an empty write would make degenerate.
    """
    if len(candidate) <= len(prefix) + len(suffix):
        return None
    if not candidate.startswith(prefix) or not candidate.endswith(suffix):
        return None
    return candidate, candidate[len(prefix) : len(candidate) - len(suffix)]


# ============================================================================
# Dispatch
# ============================================================================
_model: IgLMModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = IgLMModel(model_checkpoint=input_dict["model_checkpoint"])

    operation = input_dict["operation"]
    if operation == "sample":
        return _model.sample(
            prompts=input_dict["prompts"],
            chain_tokens=input_dict["chain_tokens"],
            species_tokens=input_dict["species_tokens"],
            num_sequences_per_input=input_dict["num_sequences_per_input"],
            temperature=input_dict["temperature"],
            top_p=input_dict["top_p"],
            max_length=input_dict["max_length"],
            prepend_prompt=input_dict["prepend_prompt"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            seed=input_dict["seed"],
        )
    if operation == "infill":
        return _model.infill(
            sequences=input_dict["sequences"],
            infill_ranges=input_dict["infill_ranges"],
            chain_tokens=input_dict["chain_tokens"],
            species_tokens=input_dict["species_tokens"],
            num_sequences_per_input=input_dict["num_sequences_per_input"],
            temperature=input_dict["temperature"],
            top_p=input_dict["top_p"],
            max_length=input_dict["max_length"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            seed=input_dict["seed"],
        )
    if operation == "score":
        return _model.score(
            sequences=input_dict["sequences"],
            infill_ranges=input_dict["infill_ranges"],
            chain_tokens=input_dict["chain_tokens"],
            species_tokens=input_dict["species_tokens"],
            batch_size=input_dict["batch_size"],
            return_logits=input_dict["return_logits"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            seed=input_dict["seed"],
        )
    raise ValueError(f"iglm: unknown operation {operation!r}; valid: ['sample', 'infill', 'score']")


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
        raise ValueError("iglm: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
