"""Local AbLang inference: embeddings, scoring, sampling, and masked PLL gradient."""

import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from standalone_helpers import (
    StatusCounter,
    download_archive,
    download_weights,
    get_logger,
    link_to_cache,
    log_likelihood_metrics,
    require_weights_dir,
    serialize_output,
)
from standalone_helpers.relaxation import relaxed_sequence, resolve_temperature
from standalone_helpers.vocabulary import AMINO_ACIDS_LIST

logger = get_logger(__name__)

STANDARD_AMINO_ACIDS = "".join(AMINO_ACIDS_LIST)


# Mirror of setup.sh's URL table — kept in sync by hand (URLs are static).
_ABLANG_MODEL_INFO: dict[str, tuple[str, str]] = {
    "ablang1-heavy": ("https://opig.stats.ox.ac.uk/data/downloads/ablang-heavy.tar.gz", "amodel.pt"),
    "ablang1-light": ("https://opig.stats.ox.ac.uk/data/downloads/ablang-light.tar.gz", "amodel.pt"),
    "ablang2-paired": ("https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz", "model.pt"),
}

# Mirror of the Zenodo archive's two extracted files, tried first because Zenodo answers some
# networks with HTTP 403. Upstream publishes no digest, so each file carries its own SHA-256.
# setup.sh carries the same table.
_ABLANG2_MIRROR_URL = "https://huggingface.co/bviggiano/AbLang2/resolve/main"
_ABLANG2_MIRROR_FILES: dict[str, str] = {
    "model.pt": "56d6f07862a6f824f88c8707bbc03e4026c9db762be2d3041e9767e2e6f86386",
    "hparams.json": "33474b76d78e01beb24b3621cd44d42b0813333d21e5b806168580bd836c5c8d",
}


def _mirror_ablang2_weights(target: Path) -> bool:
    """Fetch AbLang2's weights from the mirror into ``target``, or report that it could not.

    A digest miss counts as a miss, so the caller falls back to Zenodo rather than staging
    unverified weights. A partial set is harmless: ``download_archive`` empties the directory.

    Args:
        target (Path): The ``model-weights-ablang2-paired`` cache directory.

    Returns:
        bool: Whether every mirrored file is now present and verified.
    """
    try:
        for name, digest in _ABLANG2_MIRROR_FILES.items():
            download_weights(f"{_ABLANG2_MIRROR_URL}/{name}", target / name, sha256=digest)
    except (OSError, RuntimeError) as err:
        logger.warning("AbLang2 mirror unavailable (%s); falling back to the Zenodo archive", err)
        return False
    return True


def _stage_ablang_weights(model_choice: str) -> Path:
    """Ensure the weights are in the managed cache and ablang2's package path points at them.

    ablang2 only looks for weights at ``$(dirname ablang2.__file__)/model-weights-<name>``, so the
    files live in ``resolve_weights_dir("ablang")`` and that package path is a symlink to them.
    Repointing on every load also repairs a link left stale by a cache move.

    Args:
        model_choice (str): Key from ``_ABLANG_MODEL_INFO``.

    Returns:
        Path: The cached weights directory.
    """
    import ablang2

    url, weight_file = _ABLANG_MODEL_INFO[model_choice]
    target = Path(require_weights_dir("ablang")) / f"model-weights-{model_choice}"
    if model_choice == "ablang2-paired" and _mirror_ablang2_weights(target):
        cache_dir = target
    else:
        cache_dir = Path(download_archive(url, target, marker=weight_file))
    link_to_cache(Path(os.path.dirname(ablang2.__file__)) / f"model-weights-{model_choice}", cache_dir)
    return cache_dir


def _validate_ablang_weights(model_choice: str) -> None:
    """Stage the cached weights, re-fetching them once if the checkpoint does not load.

    Mirrors protenix's ``cleanup_corrupted_checkpoints`` pattern: a truncated or corrupt file is
    deleted from the cache and downloaded again rather than left for ablang2 to choke on.

    Args:
        model_choice (str): One of the keys in ``ablang2.load_model.list_of_models``
            (e.g. ``"ablang1-heavy"``, ``"ablang2-paired"``).
    """
    import pickle

    import torch

    cache_dir = _stage_ablang_weights(model_choice)
    weights_path = cache_dir / _ABLANG_MODEL_INFO[model_choice][1]
    try:
        torch.load(weights_path, map_location="cpu", weights_only=False)
    except (RuntimeError, OSError, EOFError, pickle.UnpicklingError) as e:
        logger.warning("AbLang %s: weights validation failed (%s); re-downloading", model_choice, e)
        shutil.rmtree(cache_dir)
        _stage_ablang_weights(model_choice)


class AbLangModel:
    """AbLang model for antibody sequence embeddings, scoring, and restoration.

    Supports heavy-only, light-only, and paired (heavy|light) antibody sequences.
    Wraps the ablang2.pretrained API with lazy loading and device management.
    """

    def __init__(self, model_choice: str = "ablang2-paired") -> None:
        """Initialize AbLang model wrapper.

        Args:
            model_choice (str): One of "ablang1-heavy", "ablang1-light", "ablang2-paired".
        """
        self._loaded = False
        self.model_choice = model_choice
        self.device: str | None = None
        self.model: Any = None
        self._ablang_vocab: dict[str, int] | None = None

    @property
    def _is_ablang1(self) -> bool:
        return "ablang1" in self.model_choice

    def load(self, device: str, verbose: bool = False) -> None:
        """Load AbLang model to device."""
        # Validate checkpoint files and recover corrupted downloads before loading.
        _validate_ablang_weights(self.model_choice)

        logger.update_status("Loading AbLang weights")

        import ablang2
        from ablang2.models.ablang2.vocab import ablang_vocab

        self.model = ablang2.pretrained(
            model_to_use=self.model_choice,
            random_init=False,
            ncpu=1,
            device=device,
        )
        if hasattr(self.model, "freeze"):
            self.model.freeze()
        self.device = device
        self._loaded = True
        self._ablang_vocab = dict(ablang_vocab)

        if self._is_ablang1:
            self._patch_ablang1_tokenizer()

        if verbose:
            logger.info("AbLang model loaded successfully")

    def _patch_ablang1_tokenizer(self) -> None:
        """Patch ablang1 tokenizer for compatibility with ablang2's unified API.

        The ablang2 library's encoding/scoring/restoration mixins call
        self.tokenizer(..., w_extra_tkns=False) and reference self.tokenizer.mask_token
        and self.tokenizer.all_special_tokens — none of which exist on the ablang1
        tokenizer. This patch adds them so the library's own methods work.
        """
        import torch

        tokenizer = self.model.tokenizer
        vocab = tokenizer.vocab_to_token
        pad_token_id = tokenizer.pad_token

        # Attributes expected by scores.py / restoration.py
        # Note: "*" is ablang's native mask token in its vocabulary.
        # The user-facing API uses "_", which is converted to "*" in sample().
        tokenizer.mask_token = vocab["*"]
        tokenizer.all_special_tokens = [
            t
            for t in (pad_token_id, vocab["<"], vocab[">"], vocab.get("|"), vocab["*"], vocab.get("X"))
            if t is not None
        ]
        tokenizer.aa_to_token = vocab
        tokenizer.token_to_aa = tokenizer.vocab_to_aa

        # Python resolves __call__ on the class — patch via a one-off subclass + __class__ reassign so
        # the original ABtokenizer isn't mutated (would break ablang2-paired in the same process).
        original_cls = type(tokenizer)
        original_call = original_cls.__call__

        class _PatchedABtokenizer(original_cls):  # type: ignore[misc, valid-type]
            def __call__(
                self_tok: Any,
                sequence_list: Any,
                mode: str = "encode",
                pad: bool = False,
                w_extra_tkns: bool | None = None,
                device: str = "cpu",
                **_kwargs: Any,
            ) -> Any:
                if mode == "decode":
                    return original_call(self_tok, sequence_list, encode=False, pad=pad, device=device)

                if isinstance(sequence_list, str):
                    sequence_list = [sequence_list]

                if w_extra_tkns is False:
                    # Sequences already have <> tokens — tokenize char by char
                    data = [
                        torch.tensor([vocab[c] for c in seq], dtype=torch.long, device=device) for seq in sequence_list
                    ]
                    if pad:
                        return torch.nn.utils.rnn.pad_sequence(
                            data,
                            batch_first=True,
                            padding_value=pad_token_id,
                        )
                    return data

                # Default: original ablang1 behaviour (adds <> tokens)
                return original_call(self_tok, sequence_list, encode=True, pad=pad, device=device)

        tokenizer.__class__ = _PatchedABtokenizer

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise ValueError("ablang: cannot move unloaded model to device — call load() first")

        if self.device != device:
            import torch
            from standalone_helpers import move_model_to_device

            if hasattr(self.model, "AbRep"):
                self.model.AbRep = move_model_to_device(self.model.AbRep, self.device, device)
            if hasattr(self.model, "AbLang"):
                self.model.AbLang = move_model_to_device(self.model.AbLang, self.device, device)

            self.model.device = device
            self.model.used_device = torch.device(device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.to_device("cpu")

    def _ensure_loaded(self, device: str, verbose: bool = False) -> None:
        """Lazy load or move model to the requested device."""
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

    def _format_sequences(self, sequences: list[str]) -> list[Any]:
        """Convert user-facing strings to the format the library expects.

        - ablang1: "SEQ" -> "<SEQ>"  (pre-formatted for direct method calls)
        - ablang2-paired: "HEAVY|LIGHT" -> ("HEAVY", "LIGHT")
        """
        if self._is_ablang1:
            return [f"<{seq}>" for seq in sequences]

        formatted: list[Any] = []
        for seq in sequences:
            if "|" in seq:
                formatted.append(tuple(seq.split("|", 1)))
            else:
                logger.warning(
                    "Single-chain sequence passed to ablang2-paired model; "
                    "pairing with empty light chain. Consider using ablang1-heavy "
                    "or ablang1-light for single-chain sequences."
                )
                formatted.append((seq, ""))
        return formatted

    def _compute_attention_masks(self, sequences: list[str]) -> list[list[int]]:
        """Compute attention masks from sequences.

        For paired sequences (heavy|light format), the mask covers both chains.
        Masks are 1 for real residue positions and 0 for padding.
        """
        lengths = []
        for seq in sequences:
            if "|" in seq:
                parts = seq.split("|")
                lengths.append(len(parts[0]) + len(parts[1]))
            else:
                lengths.append(len(seq))

        max_len = max(lengths) if lengths else 0
        masks = []
        for length in lengths:
            mask = [1] * length + [0] * (max_len - length)
            masks.append(mask)
        return masks

    # ========================================================================
    # Embeddings
    # ========================================================================

    def embeddings(
        self,
        sequences: list[str],
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Extract sequence embeddings (768-dim for ablang1, 480-dim for ablang2).

        When ``return_logits=True``, also runs AbLang's ``likelihood`` mode and
        returns per-residue amino-acid logits restricted to the 20 standard AAs.
        """
        self._ensure_loaded(device, verbose)

        if not sequences:
            raise ValueError("ablang: embeddings requires at least one input sequence")

        if verbose:
            logger.info(f"Computing embeddings for {len(sequences)} sequences (batch_size={batch_size})")

        all_mean_embeddings: list[Any] = []
        all_attention_masks: list[list[int]] = []
        all_logits: list[Any] | None = [] if return_logits else None

        counter = StatusCounter(logger, "Embedding batch %d/%d", total=-(-len(sequences) // batch_size))
        for i in range(0, len(sequences), batch_size):
            counter(i // batch_size + 1)
            batch = sequences[i : i + batch_size]
            formatted = self._format_sequences(batch)

            if self._is_ablang1:
                mean_embeddings = self.model.seqcoding(formatted)
            else:
                mean_embeddings = self.model(formatted, mode="seqcoding")

            attention_masks = self._compute_attention_masks(batch)

            all_mean_embeddings.extend(mean_embeddings)
            all_attention_masks.extend(attention_masks)

            if all_logits is not None:
                all_logits.extend(self._per_position_aa_logits(formatted))

        return {
            "mean_embeddings": all_mean_embeddings,
            "attention_masks": all_attention_masks,
            "logits": all_logits,
        }

    def _per_position_aa_logits(self, formatted: list[Any]) -> list[list[list[float]]]:
        """Run a likelihood-mode forward pass and return per-residue AA-only logits.

        Doubles inference time vs the primary forward pass. Rows include
        format-time special tokens; columns map to ``STANDARD_AMINO_ACIDS``.
        Both ablang1 and ablang2 vocabs share AA IDs 1-20.
        """
        raw_logits = self.model.likelihood(formatted) if self._is_ablang1 else self.model(formatted, mode="likelihood")
        if self._ablang_vocab is None:
            raise RuntimeError("ablang: vocabulary unavailable after load")
        aa_cols = [self._ablang_vocab[aa] for aa in STANDARD_AMINO_ACIDS]
        out: list[list[list[float]]] = []
        for arr in raw_logits:
            arr_list = arr.tolist() if hasattr(arr, "tolist") else list(arr)
            out.append([[row[c] for c in aa_cols] for row in arr_list])
        return out

    # ========================================================================
    # Scoring
    # ========================================================================

    def score(
        self,
        sequences: list[str],
        scoring_mode: str = "pseudo_log_likelihood",
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Score antibody sequences using pseudo-log-likelihood or confidence.

        When ``return_logits=True``, also runs AbLang's ``likelihood`` mode and
        returns per-residue AA-only logits alongside the scalar metrics.
        """
        self._ensure_loaded(device, verbose)

        if not sequences:
            raise ValueError("ablang: score requires at least one input sequence")

        if verbose:
            logger.info(f"Scoring {len(sequences)} sequences with mode={scoring_mode} (batch_size={batch_size})")

        if scoring_mode not in ("pseudo_log_likelihood", "confidence"):
            raise ValueError(
                f"ablang: unknown scoring_mode {scoring_mode!r}; valid: ['pseudo_log_likelihood', 'confidence']"
            )

        all_metrics: list[dict[str, float]] = []
        all_logits: list[Any] | None = [] if return_logits else None

        # Score the sequences batch by batch.
        counter = StatusCounter(logger, "Scoring batch %d/%d", total=-(-len(sequences) // batch_size))
        for i in range(0, len(sequences), batch_size):
            counter(i // batch_size + 1)
            batch = sequences[i : i + batch_size]
            formatted = self._format_sequences(batch)

            # AbLang-1 exposes each mode as a method; AbLang-2 takes the mode as an argument.
            if self._is_ablang1:
                scores = getattr(self.model, scoring_mode)(formatted)
            else:
                scores = self.model(formatted, mode=scoring_mode)

            # Collect one metrics dict per sequence, with log-likelihood stats for PLL mode.
            for j, s in enumerate(scores):
                score_val = float(s)
                m: dict[str, float] = {scoring_mode: score_val}
                if scoring_mode == "pseudo_log_likelihood":
                    # Library returns mean log-prob (reduction="mean" in ablang2's scores.py).
                    m.update(log_likelihood_metrics(score_val, len(batch[j].replace("|", ""))))
                all_metrics.append(m)

            # Run a separate likelihood pass for per-residue logits when requested.
            if all_logits is not None:
                all_logits.extend(self._per_position_aa_logits(formatted))

        return {
            "metrics": all_metrics,
            "logits": all_logits,
            "vocab": list(STANDARD_AMINO_ACIDS),
        }

    # ========================================================================
    # Sampling / Restore
    # ========================================================================

    def sample(
        self,
        sequences: list[str],
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        align: bool = False,
        return_logits: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
        excluded_amino_acids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fill masked (_) positions in antibody sequences.

        The user-facing API uses ``_`` as the standard mask token. This method
        converts ``_`` to ``*`` (ablang's native mask token) before processing.

        With ``temperature > 0`` (the default), this runs the underlying AbLang
        model in ``likelihood`` mode to get per-position amino-acid logits,
        applies temperature-scaled softmax, and draws one amino acid per
        masked position via ``torch.multinomial``. The seed (when set) makes
        the draws reproducible.

        With ``temperature == 0``, this falls back to the library's
        ``restore`` mode (pure argmax decoding), giving deterministic
        most-likely-residue restoration. ``align=True`` also forces this
        path because the ANARCI spread-of-variants logic is incompatible
        with single-pass stochastic sampling.

        When ``return_logits=True``, runs an additional ``likelihood`` mode pass
        on the filled-in sequences and returns per-residue AA-only logits.

        ``excluded_amino_acids`` are never drawn, on the sampling, greedy, or aligned path.
        """
        import torch

        self._ensure_loaded(device, verbose)

        if not sequences:
            raise ValueError("ablang: sample requires at least one input sequence")

        # Convert standard mask token (_) to ablang's native mask token (*)
        sequences = [seq.replace("_", "*") for seq in sequences]

        if seed is not None:
            torch.manual_seed(seed)

        use_sampling = temperature > 0 and not align

        if verbose:
            mode_name = "sampling" if use_sampling else "restore"
            logger.info(
                "Filling masked positions in %d sequences via %s (batch_size=%d, align=%s, temperature=%g)",
                len(sequences),
                mode_name,
                batch_size,
                align,
                temperature,
            )

        restored_sequences: list[str] = []
        all_logits: list[Any] | None = [] if return_logits else None

        counter = StatusCounter(logger, "Sampling batch %d/%d", total=-(-len(sequences) // batch_size))
        for i in range(0, len(sequences), batch_size):
            counter(i // batch_size + 1)
            batch = sequences[i : i + batch_size]
            formatted = self._format_sequences(batch)

            # Scoped to the draw only, so the return_logits pass below reports unmasked logits.
            with self._excluded_amino_acid_logits(excluded_amino_acids):
                if use_sampling:
                    restored = self._sample_at_mask_positions(formatted, temperature, n_seqs=len(batch))
                elif self._is_ablang1:
                    restored = self.model.restore(formatted, align=align)
                else:
                    restored = self.model(formatted, mode="restore", align=align)

            # Strip <> special tokens and padding dashes from library output
            restored_clean = [str(s).replace("<", "").replace(">", "").replace("-", "") for s in restored]
            restored_sequences.extend(restored_clean)

            if all_logits is not None:
                # Re-format the filled sequences and run a likelihood pass on them so
                # logits reflect the final filled-in positions, not the masked input.
                all_logits.extend(self._per_position_aa_logits(self._format_sequences(restored_clean)))

        return {"sequences": restored_sequences, "logits": all_logits}

    @contextmanager
    def _excluded_amino_acid_logits(self, excluded_amino_acids: list[str] | None) -> Iterator[None]:
        """Drive every residue choice away from ``excluded_amino_acids`` for the duration.

        A forward hook on the model drops the excluded amino-acid columns to ``-inf``, which
        covers all three fill paths at once: our own multinomial draw, the library's greedy
        ``restore``, and its ``align=True`` spread. ``restore`` reads the same logits for its
        ``torch.max`` decode, so no upstream code has to be reimplemented to constrain it.

        On the ``align=True`` path the library also ranks its alignment variants from these
        logits, so an exclusion can shift which variant wins; that only happens when the
        model's top residue at the anchor position is one the caller banned.

        Yields nothing and installs no hook when there is nothing to exclude.
        """
        import torch

        if not excluded_amino_acids:
            yield
            return
        if self._ablang_vocab is None:
            raise RuntimeError("ablang: vocabulary unavailable; call load() first")
        # Full-vocab column indices; the library slices [:, :, 1:21] after this hook runs.
        columns = [self._ablang_vocab[aa] for aa in excluded_amino_acids]

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            masked = output.clone()
            masked[..., columns] = -torch.inf
            return masked

        handle = self.model.AbLang.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def _sample_at_mask_positions(self, formatted: list[Any], temperature: float, n_seqs: int) -> list[str]:
        """Sample at masked positions using temperature-scaled multinomial.

        Mirrors the library's ``AbRestore.restore(align=False)`` flow but
        replaces the final ``torch.argmax`` with a ``torch.multinomial`` draw
        from the temperature-scaled softmax over the 20 amino-acid tokens.
        Excluded residues are already ``-inf`` here, masked by the caller's
        ``_excluded_amino_acid_logits`` hook.
        Mask tokens (==23 in both ablang1 and ablang2 vocabs) are the only
        positions whose tokens get rewritten; everything else passes through.
        """
        import torch

        if self._is_ablang1:
            tokens = self.model.tokenizer(formatted, pad=True, device=self.model.used_device)
        else:
            # ablang2 tokenizer with w_extra_tkns=False expects pre-joined "<H>|<L>" strings, not (H, L) tuples.
            seqs = [f"<{heavy}>|<{light}>".replace("<>", "") for heavy, light in formatted]
            tokens = self.model.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.model.used_device)

        with torch.no_grad():
            predictions = self.model.AbLang(tokens)[:, :, 1:21]  # AA columns only

        scaled = predictions / temperature
        probs = torch.softmax(scaled, dim=-1)

        batch_n, seq_len, num_aa = probs.shape
        flat_probs = probs.reshape(-1, num_aa)
        sampled_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(batch_n, seq_len) + 1  # vocab IDs 1-20

        mask_token_id = 23  # ablang1 and ablang2 both use 23
        restored_tokens = torch.where(tokens == mask_token_id, sampled, tokens)

        decoded = self.model.tokenizer(restored_tokens, mode="decode")

        # ablang2 paired returns heavy and light separately; rejoin with '|'
        # to match the library's restore() output convention.
        if not self._is_ablang1 and len(decoded) > n_seqs:
            decoded = [f"{heavy}|{light}" for heavy, light in zip(decoded[:n_seqs], decoded[n_seqs:], strict=True)]

        return list(decoded)

    # ========================================================================
    # Gradient
    # ========================================================================

    def _canonical_amino_acid_to_vocab_matrix(self) -> Any:
        """Map canonical protein logit columns into AbLang's token vocabulary."""
        import torch

        if self._ablang_vocab is None:
            raise ValueError("ablang: vocabulary not initialized — call load() first")

        mapping_matrix = torch.zeros(
            len(STANDARD_AMINO_ACIDS),
            len(self._ablang_vocab),
            dtype=torch.float32,
            device=self.device,
        )
        for idx, amino_acid in enumerate(STANDARD_AMINO_ACIDS):
            mapping_matrix[idx, self._ablang_vocab[amino_acid]] = 1.0
        return mapping_matrix

    def compute_gradient(
        self,
        logits_list: list[list[float]],
        *,
        temperature: float | None = None,
        soft: float = 1.0,
        hard: float = 0.0,
        chain_break_position: int | None = None,
        seed: int | None = None,
        backprop: bool = True,
        batch_size: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Chunked masked PLL gradient. Masks each AA position, predicts from bidirectional context.

        Args:
            logits_list (list[list[float]]): Input logits, shape (L, 20).
            temperature (float | None): Softmax temperature; ``None`` uses logits as-is.
            soft (float): Blend from the raw input matrix (0) to softmax probabilities (1).
            hard (float): Straight-through coefficient; 1 forwards a hard one-hot.
            chain_break_position (int | None): VH/VL split for ``<VH>|<VL>`` paired layout.
            seed (int | None): Random seed.
            backprop (bool): If False, skip backward and return ``gradient=None``.
            batch_size (int | None): AA positions per forward pass (default: 8 paired, 32 single).
            device (str): Execution device.
            verbose (bool): Log progress.

        Returns:
            dict[str, Any]: Keys: ``gradient``, ``loss``, ``metrics``, ``vocab``.
        """
        import torch
        import torch.nn.functional as F

        self._ensure_loaded(device, verbose)
        if seed is not None:
            torch.manual_seed(seed)

        # Disables autograd graph building through model params in forward-only mode.
        with torch.set_grad_enabled(backprop):
            seq_dist = torch.tensor(logits_list, device=self.device, dtype=torch.float32, requires_grad=backprop)
            sequence_length = int(seq_dist.shape[0])

            # Targets come from P, not the blend, so they do not move with soft/hard.
            to_vocab = self._canonical_amino_acid_to_vocab_matrix()
            probs = relaxed_sequence(seq_dist, temperature=temperature)  # (L, 20)
            residue_token_ids = (probs @ to_vocab).argmax(dim=-1).detach()  # (L,)

            x = relaxed_sequence(seq_dist, temperature=temperature, soft=soft, hard=hard)  # (L, 20)
            mapped = x @ to_vocab  # (L, V)

            # Soft embeddings: (L, V) @ (V, D) → (L, D)
            if self._is_ablang1:
                embed_layer = self.model.AbRep.AbEmbeddings.AAEmbeddings
                residue_embeddings = mapped[:, :-2] @ embed_layer.weight  # ablang1 vocab has 2 extra tokens
            else:
                embed_layer = self.model.AbLang.get_aa_embeddings()
                residue_embeddings = mapped @ embed_layer.weight

            assert self._ablang_vocab is not None
            ablang_vocab = self._ablang_vocab

            def special_ids(tokens: list[str]) -> Any:
                return torch.tensor([ablang_vocab[token] for token in tokens], device=self.device)

            def special_embeddings(tokens: list[str]) -> Any:
                return embed_layer.weight[special_ids(tokens)]

            # Wrap with special tokens: ablang1 → <residues>, ablang2 → <VH>|<VL>
            if self._is_ablang1:
                if chain_break_position is not None:
                    raise ValueError("ablang: AbLang1 gradient layout does not support paired chain_break_position")
                input_embeddings = torch.cat(
                    (special_embeddings(["<"]), residue_embeddings, special_embeddings([">"])),
                    dim=0,
                )  # (L+2, D)
                token_ids = torch.cat((special_ids(["<"]), residue_token_ids, special_ids([">"])), dim=0)  # (L+2,)
            else:
                if chain_break_position is None:
                    raise ValueError("ablang: AbLang2 paired gradient requires chain_break_position")
                if chain_break_position <= 0 or chain_break_position >= residue_token_ids.shape[0]:
                    raise ValueError(
                        f"ablang: chain_break_position={chain_break_position} out of range; expected 1..{sequence_length - 1}"
                    )
                input_embeddings = torch.cat(
                    (
                        special_embeddings(["<"]),
                        residue_embeddings[:chain_break_position],
                        special_embeddings([">", "|", "<"]),
                        residue_embeddings[chain_break_position:],
                        special_embeddings([">"]),
                    ),
                    dim=0,
                )  # (L+5, D)
                token_ids = torch.cat(
                    (
                        special_ids(["<"]),
                        residue_token_ids[:chain_break_position],
                        special_ids([">", "|", "<"]),
                        residue_token_ids[chain_break_position:],
                        special_ids([">"]),
                    ),
                    dim=0,
                )  # (L+5,)

            input_embeddings = input_embeddings.unsqueeze(0)  # (1, S, D)
            token_ids = token_ids.unsqueeze(0)  # (1, S)

            # AA positions: indices into token_ids[0] that are amino acids (not BOS/EOS/SEP)
            aa_vocab_ids = torch.tensor([ablang_vocab[aa] for aa in STANDARD_AMINO_ACIDS], device=self.device)
            aa_positions = torch.isin(token_ids[0], aa_vocab_ids).nonzero().squeeze(1)  # (n_aa,)

            n_aa = aa_positions.shape[0]
            if batch_size is None:
                batch_size = 8 if not self._is_ablang1 else 32

            seq_len = input_embeddings.shape[1]
            mask_emb = embed_layer.weight[ablang_vocab["*"]].detach()  # (D,)
            pos_idx = torch.arange(seq_len, device=self.device)  # (S,)

            # Chunked PLL: mask each AA position, predict from bidirectional context.
            # Backpropagate each chunk separately to free its transformer activations.
            ie_grad = torch.zeros_like(input_embeddings) if backprop else None
            total_loss_val = 0.0
            for start in range(0, n_aa, batch_size):
                end = min(start + batch_size, n_aa)
                chunk_aa_pos = aa_positions[start:end]  # (C,)
                chunk_len = end - start

                chunk_masked = pos_idx.unsqueeze(0) == chunk_aa_pos.unsqueeze(1)  # (C, S)
                ie_chunk = input_embeddings.detach().requires_grad_(True) if backprop else input_embeddings
                chunk_input = torch.where(
                    chunk_masked.unsqueeze(-1),
                    mask_emb.view(1, 1, -1).expand(chunk_len, seq_len, -1),
                    ie_chunk.expand(chunk_len, -1, -1),
                )  # (C, S, D)

                handle = embed_layer.register_forward_hook(lambda _m, _i, _o, ci=chunk_input: ci)
                try:
                    chunk_logits = self.model.AbLang(token_ids.expand(chunk_len, -1))  # (C, S, V)
                finally:
                    handle.remove()

                chunk_idx = torch.arange(chunk_len, device=self.device)
                pred = chunk_logits[chunk_idx, chunk_aa_pos, :]  # (C, V)
                labels = token_ids[0, aa_positions[start:end]]  # (C,)
                loss_sum = F.cross_entropy(pred, labels, reduction="sum")
                total_loss_val += loss_sum.item()

                if backprop:
                    (loss_sum / n_aa).backward()
                    ie_grad += ie_chunk.grad

        mean_nll = total_loss_val / n_aa
        gradient_value: list[list[float]] | None = None
        if backprop:
            (x_grad,) = torch.autograd.grad(input_embeddings, seq_dist, grad_outputs=ie_grad)
            gradient_value = x_grad.detach().cpu().tolist()  # (L, 20)

        return {
            "gradient": gradient_value,
            "loss": mean_nll,
            "metrics": {
                **log_likelihood_metrics(-mean_nll, n_aa),
                "sequence_length": sequence_length,
                "model_choice": self.model_choice,
                "objective": "masked_pll",
            },
            "vocab": list(STANDARD_AMINO_ACIDS),
        }


# ============================================================================
# Dispatch
# ============================================================================
_model: AbLangModel | None = None


def _ensure_model(model_choice: str) -> AbLangModel:
    """Return the worker's model for ``model_choice``, swapping checkpoints only when it changes.

    AbLang picks its checkpoint from the chains an antibody carries, so one dispatch may need
    more than one. Swapping unloads the previous checkpoint first, keeping a single one resident.

    Args:
        model_choice (str): One of ``ablang1-heavy``, ``ablang1-light``, ``ablang2-paired``.

    Returns:
        AbLangModel: The wrapper for that checkpoint, loaded on first use by the operation.
    """
    global _model
    if _model is None or _model.model_choice != model_choice:
        if _model is not None and _model._loaded:
            _model.unload()
        _model = AbLangModel(model_choice=model_choice)
    return _model


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    operation = input_dict["operation"]
    if operation != "compute_gradient":
        # Only the gradient carries a checkpoint per item; it resolves them below, grouped.
        model = _ensure_model(input_dict["model_choice"])

    if operation == "embeddings":
        return model.embeddings(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
        )
    if operation == "score":
        return model.score(
            sequences=input_dict["sequences"],
            scoring_mode=input_dict["scoring_mode"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
        )
    if operation == "sample":
        return model.sample(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            align=input_dict["align"],
            return_logits=input_dict["return_logits"],
            temperature=input_dict.get("temperature", 1.0),
            seed=input_dict.get("seed"),
            excluded_amino_acids=input_dict["excluded_amino_acids"],
        )
    if operation == "compute_gradient":
        # Group by checkpoint to load each variant once, then restore the original result order.
        items = input_dict["items"]
        groups: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault(item["model_choice"], []).append(index)

        # Run each item's gradient with its checkpoint, writing results back to the item's slot.
        results: list[dict[str, Any] | None] = [None] * len(items)
        for model_choice, indices in groups.items():
            model = _ensure_model(model_choice)
            for index in indices:
                item = items[index]
                results[index] = model.compute_gradient(
                    logits_list=item["logits"],
                    temperature=resolve_temperature(input_dict["temperature"]),
                    soft=float(input_dict.get("soft", 1.0)),
                    hard=float(input_dict.get("hard", 0.0)),
                    chain_break_position=item["chain_break_position"],
                    seed=input_dict["seed"],
                    backprop=input_dict.get("compute_gradient", True),
                    batch_size=input_dict.get("batch_size"),
                    device=input_dict["device"],
                    verbose=input_dict["verbose"],
                )

        # Ensure every item produced a result.
        completed = [result for result in results if result is not None]
        if len(completed) != len(items):
            raise RuntimeError(f"ablang: produced {len(completed)} results for {len(items)} item(s)")
        # Return per-item results with the shared vocabulary listed once.
        return {
            "results": [{key: result[key] for key in ("gradient", "loss", "metrics")} for result in completed],
            "vocab": completed[0]["vocab"],
        }
    raise ValueError(
        f"ablang: unknown operation {operation!r}; valid: ['embeddings', 'score', 'sample', 'compute_gradient']"
    )


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") and _model.device else "cpu"
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("ablang: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
