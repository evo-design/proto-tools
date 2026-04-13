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

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


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
        if verbose:
            logger.info(f"Loading AbLang model: {self.model_choice} on {device}")

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

    # ========================================================================
    # Germinal Gradient
    # ========================================================================

    def _canonical_amino_acid_to_vocab_matrix(self) -> Any:
        """Map canonical protein logit columns into AbLang's token vocabulary."""
        import torch

        if self._ablang_vocab is None:
            raise RuntimeError("AbLang vocabulary not initialized. Call load() first.")

        mapping_matrix = torch.zeros(
            len(STANDARD_AMINO_ACIDS),
            len(self._ablang_vocab),
            dtype=torch.float32,
            device=self.device,
        )
        for idx, amino_acid in enumerate(STANDARD_AMINO_ACIDS):
            mapping_matrix[idx, self._ablang_vocab[amino_acid]] = 1.0
        return mapping_matrix

    def _one_hot_from_logits(
        self,
        relaxed_logits: Any,
        *,
        temperature: float,
    ) -> tuple[Any, Any]:
        """Build Germinal's straight-through AbLang token relaxation."""
        import torch.nn.functional as F

        probabilities = F.softmax(relaxed_logits / temperature, dim=-1)
        mapped_probabilities = probabilities @ self._canonical_amino_acid_to_vocab_matrix()

        vocab_size = mapped_probabilities.size(-1)
        token_ids = mapped_probabilities.argmax(dim=-1)
        hard = F.one_hot(token_ids, num_classes=vocab_size).float()
        one_hot = hard + (mapped_probabilities - mapped_probabilities.detach())
        return one_hot, token_ids

    def _insert_chain_separator(
        self,
        embeddings: Any,
        token_ids: Any,
        *,
        insert_position: int,
    ) -> tuple[Any, Any]:
        """Insert the paired-chain separator exactly as Germinal does."""
        import torch

        if self._ablang_vocab is None:
            raise RuntimeError("AbLang vocabulary not initialized. Call load() first.")

        separator_embed = self.model.AbLang.get_aa_embeddings().weight[self._ablang_vocab["|"]]
        updated_embeddings = torch.cat(
            (
                embeddings[:insert_position],
                separator_embed.unsqueeze(0),
                embeddings[insert_position:],
            ),
            dim=0,
        )
        updated_token_ids = torch.cat(
            (
                token_ids[:insert_position],
                torch.tensor([self._ablang_vocab["|"]], device=self.device, dtype=torch.long),
                token_ids[insert_position:],
            ),
            dim=0,
        )
        return updated_embeddings, updated_token_ids

    def compute_germinal_gradient(
        self,
        logits_list: list[list[float]],
        temperature: float,
        *,
        use_single_chain_variable_fragment: bool = False,
        heavy_chain_first: bool = True,
        heavy_chain_length: int | None = None,
        light_chain_length: int | None = None,
        seed: int | None = 0,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Differentiate Germinal's existing AbLang objective with respect to relaxed logits."""
        import torch
        import torch.nn.functional as F

        self._ensure_loaded(device, verbose)
        if seed is not None:
            torch.manual_seed(seed)

        logits = torch.tensor(logits_list, device=self.device, dtype=torch.float32, requires_grad=True)
        sequence_length = int(logits.shape[0])
        active_logits = logits
        first_chain_length = 0
        second_chain_length = 0
        linker_length = 0

        if use_single_chain_variable_fragment:
            if heavy_chain_length is None or light_chain_length is None:
                raise ValueError(
                    "heavy_chain_length and light_chain_length are required when "
                    "use_single_chain_variable_fragment=True"
                )
            if heavy_chain_length + light_chain_length > sequence_length:
                raise ValueError(
                    "heavy_chain_length + light_chain_length cannot exceed the full relaxed sequence length"
                )
            linker_length = sequence_length - heavy_chain_length - light_chain_length
            first_chain_length, second_chain_length = (
                (heavy_chain_length, light_chain_length)
                if heavy_chain_first
                else (light_chain_length, heavy_chain_length)
            )
            active_logits = torch.cat([logits[:first_chain_length], logits[-second_chain_length:]], dim=0)

        one_hot, residue_token_ids = self._one_hot_from_logits(
            active_logits,
            temperature=temperature,
        )

        if self._is_ablang1:
            embed_layer = self.model.AbRep.AbEmbeddings.AAEmbeddings
            residue_embeddings = one_hot[:, :-2] @ embed_layer.weight
        else:
            embed_layer = self.model.AbLang.get_aa_embeddings()
            residue_embeddings = one_hot @ embed_layer.weight

        residue_token_ids = residue_token_ids.detach()
        if use_single_chain_variable_fragment:
            residue_embeddings, residue_token_ids = self._insert_chain_separator(
                residue_embeddings,
                residue_token_ids,
                insert_position=first_chain_length,
            )

        token_ids = residue_token_ids.unsqueeze(0).to(self.device)
        input_embeddings = residue_embeddings.unsqueeze(0)

        def _embedding_hook(_module: Any, _input: Any, _output: Any) -> Any:
            return input_embeddings

        hook_handle = embed_layer.register_forward_hook(_embedding_hook)
        try:
            logits_out = self.model.AbLang(token_ids)
        finally:
            hook_handle.remove()

        shift_logits = logits_out[:, :-1, :]
        shift_labels = token_ids[:, 1:]
        position_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)
        position_losses = position_losses[:, 1:-1]
        loss = position_losses.mean()
        (gradient,) = torch.autograd.grad(loss, active_logits)

        full_gradient = gradient
        if use_single_chain_variable_fragment:
            zeros = torch.zeros((linker_length, logits.shape[1]), device=self.device, dtype=gradient.dtype)
            full_gradient = torch.cat(
                [gradient[:first_chain_length], zeros, gradient[-second_chain_length:]],
                dim=0,
            )

        loss_value = loss.item()
        return {
            "gradient": full_gradient.detach().cpu().tolist(),
            "loss": loss_value,
            "metrics": {
                "log_likelihood": -loss_value,
                "sequence_length": sequence_length,
                "effective_sequence_length": int(active_logits.shape[0]),
                "linker_length": linker_length,
                "use_single_chain_variable_fragment": use_single_chain_variable_fragment,
                "model_choice": self.model_choice,
                "objective": "germinal_shifted_cross_entropy",
            },
            "vocab": list(STANDARD_AMINO_ACIDS),
        }


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
    operation = input_dict["operation"]
    if operation == "compute_germinal_gradient":
        model_choice = (
            "ablang2-paired" if input_dict.get("use_single_chain_variable_fragment", False) else "ablang1-heavy"
        )
    else:
        model_choice = input_dict["model_choice"]
    if _model is None or _model.model_choice != model_choice:
        if _model is not None and _model._loaded:
            _model.unload()
        _model = AbLangModel(model_choice=model_choice)

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
    if operation == "compute_germinal_gradient":
        return _model.compute_germinal_gradient(
            logits_list=input_dict.get("logits", []),
            temperature=input_dict.get("temperature", 1.0),
            use_single_chain_variable_fragment=input_dict.get("use_single_chain_variable_fragment", False),
            heavy_chain_first=input_dict.get("heavy_chain_first", True),
            heavy_chain_length=input_dict.get("heavy_chain_length"),
            light_chain_length=input_dict.get("light_chain_length"),
            seed=input_dict.get("seed", 0),
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
