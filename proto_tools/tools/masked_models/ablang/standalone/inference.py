"""Local AbLang inference implementation.

The ablang2 library (v0.2.1) loads ablang1 models correctly via fetch_ablang1(),
but the ablang1 tokenizer is missing attributes and kwargs expected by the
unified API (encodings.py, scores.py, restoration.py all pass w_extra_tkns
and use mask_token/all_special_tokens). We monkey-patch the tokenizer after
loading so the library's own methods work for all model variants.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from standalone_helpers import serialize_output

logger = logging.getLogger(__name__)


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

    @property
    def _is_ablang1(self) -> bool:
        return "ablang1" in self.model_choice

    def load(self, device: str, verbose: bool = False) -> None:
        """Load AbLang model to device."""
        if verbose:
            logger.info(f"Loading AbLang model: {self.model_choice} on {device}")

        import ablang2

        self.model = ablang2.pretrained(
            model_to_use=self.model_choice,
            random_init=False,
            ncpu=1,
            device=device,
        )
        self.device = device
        self._loaded = True

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
        tokenizer.all_special_tokens = [vocab["<"], pad_token_id, vocab[">"], vocab["*"]]
        tokenizer.aa_to_token = vocab
        tokenizer.token_to_aa = tokenizer.vocab_to_aa

        # Python resolves __call__ on the class, not the instance, so we must
        # patch the class.  To avoid mutating the original ABtokenizer class
        # (which would break if ablang2-paired were loaded in the same process),
        # we create a one-off subclass and reassign the instance's __class__.
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
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

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
    ) -> dict[str, Any]:
        """Extract sequence embeddings (768-dim for ablang1, 480-dim for ablang2)."""
        self._ensure_loaded(device, verbose)

        if not sequences:
            raise ValueError("AbLangModel.embeddings requires at least one input sequence.")

        if verbose:
            logger.info(f"Computing embeddings for {len(sequences)} sequences (batch_size={batch_size})")

        all_mean_embeddings: list[Any] = []
        all_attention_masks: list[list[int]] = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            formatted = self._format_sequences(batch)

            if self._is_ablang1:
                mean_embeddings = self.model.seqcoding(formatted)
            else:
                mean_embeddings = self.model(formatted, mode="seqcoding")

            attention_masks = self._compute_attention_masks(batch)

            all_mean_embeddings.extend(mean_embeddings)
            all_attention_masks.extend(attention_masks)

        return {
            "mean_embeddings": all_mean_embeddings,
            "attention_masks": all_attention_masks,
        }

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
    ) -> dict[str, Any]:
        """Score antibody sequences using pseudo-log-likelihood or confidence."""
        self._ensure_loaded(device, verbose)

        if not sequences:
            raise ValueError("AbLangModel.score requires at least one input sequence.")

        if verbose:
            logger.info(f"Scoring {len(sequences)} sequences with mode={scoring_mode} (batch_size={batch_size})")

        if scoring_mode not in ("pseudo_log_likelihood", "confidence"):
            raise ValueError(f"Unknown scoring_mode: {scoring_mode}. Must be 'pseudo_log_likelihood' or 'confidence'.")

        all_metrics: list[dict[str, float]] = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            formatted = self._format_sequences(batch)

            if self._is_ablang1:
                scores = getattr(self.model, scoring_mode)(formatted)
            else:
                scores = self.model(formatted, mode=scoring_mode)

            all_metrics.extend([{scoring_mode: float(s)} for s in scores])

        return {"metrics": all_metrics}

    # ========================================================================
    # Sampling / Restore
    # ========================================================================

    def sample(
        self,
        sequences: list[str],
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Restore masked (_) positions in antibody sequences.

        The user-facing API uses ``_`` as the standard mask token. This method
        converts ``_`` to ``*`` (ablang's native mask token) before processing.
        """
        self._ensure_loaded(device, verbose)

        if not sequences:
            raise ValueError("AbLangModel.sample requires at least one input sequence.")

        # Convert standard mask token (_) to ablang's native mask token (*)
        sequences = [seq.replace("_", "*") for seq in sequences]

        if verbose:
            logger.info(f"Restoring masked positions in {len(sequences)} sequences (batch_size={batch_size})")

        restored_sequences: list[str] = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            formatted = self._format_sequences(batch)

            restored = self.model.restore(formatted) if self._is_ablang1 else self.model(formatted, mode="restore")

            # Strip <> special tokens and padding dashes from library output
            restored_sequences.extend(str(s).replace("<", "").replace(">", "").replace("-", "") for s in restored)

        return {"sequences": restored_sequences}


# ============================================================================
# Serialization
# ============================================================================


# ============================================================================
# Dispatch
# ============================================================================
_model: AbLangModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    model_choice = input_dict["model_choice"]
    if _model is None or _model.model_choice != model_choice:
        if _model is not None and _model._loaded:
            _model.unload()
        _model = AbLangModel(model_choice=model_choice)

    operation = input_dict["operation"]
    if operation == "embeddings":
        return _model.embeddings(
            sequences=input_dict["sequences"],
            batch_size=input_dict.get("batch_size", 1),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
        )
    if operation == "score":
        return _model.score(
            sequences=input_dict["sequences"],
            scoring_mode=input_dict.get("scoring_mode", "pseudo_log_likelihood"),
            batch_size=input_dict.get("batch_size", 1),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
        )
    if operation == "sample":
        return _model.sample(
            sequences=input_dict["sequences"],
            batch_size=input_dict.get("batch_size", 1),
            device=input_dict.get("device", "cuda"),
            verbose=input_dict.get("verbose", False),
        )
    raise ValueError(f"Unknown operation: {operation}")


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
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
