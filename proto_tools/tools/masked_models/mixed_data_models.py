"""Shared contracts for mixed protein/DNA masked language models."""

import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from standalone_helpers.mixed_sequence import (
    AMBIGUOUS_PROTEIN_TOKENS,
    DNA_TOKENS,
    MIXED_VOCAB,
    PROTEIN_TOKENS,
    one_hot_mixed_logits,
    tokenize_mixed_sequence,
)

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelEmbeddingsOutput,
    MaskedModelSample,
    MaskedModelSampleOutput,
    MaskedModelScoringMetrics,
    MaskedModelScoringOutput,
    SequenceEmbedding,
)
from proto_tools.transforms.masking import MASK_TOKEN
from proto_tools.utils import (
    BaseToolInput,
    GradientOutput,
    InputField,
)

MixedModality = Literal["protein", "dna"]
# Biological tokens that fix a neighboring mask's modality, including ambiguous protein symbols.
MIXED_VOCAB_SET = frozenset(PROTEIN_TOKENS + AMBIGUOUS_PROTEIN_TOKENS + DNA_TOKENS)


class MixedSequenceInput(BaseToolInput):
    """Prepared protein/DNA loci in the upstream case-sensitive notation.

    Attributes:
        sequences (list[str]): Uppercase proteins and lowercase DNA, optionally separated
            by ``+``/``-`` strand markers (``<+>``/``<->`` are stored as ``+``/``-``).
            A single string is normalized to a list.
    """

    ALLOW_MASKS: ClassVar[bool] = False
    sequences: list[str] = InputField(
        title="Sequences",
        description="Prepared loci: uppercase protein, lowercase acgt, +/- strand markers (<+>/<-> accepted)",
        min_length=1,
        examples=["+MKTL+acgt-ACDE"],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        """Accept a single locus or a list of loci."""
        return [value] if isinstance(value, str) else value

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate tokens and store one-character strand markers, without changing case."""
        return ["".join(tokenize_mixed_sequence(sequence, allow_masks=cls.ALLOW_MASKS)) for sequence in sequences]

    def __len__(self) -> int:
        """Return the total number of model tokens."""
        return sum(len(sequence) for sequence in self.sequences)


def _implied_mask_modality(sequence: str, index: int) -> MixedModality | None:
    """Return the modality a mask's context requires, or None when the context is ambiguous."""
    start = index
    while start > 0 and sequence[start - 1] == MASK_TOKEN:
        start -= 1
    end = index
    while end < len(sequence) - 1 and sequence[end + 1] == MASK_TOKEN:
        end += 1
    left = sequence[start - 1] if start > 0 else None
    right = sequence[end + 1] if end < len(sequence) - 1 else None
    # A '-' marker only starts protein elements.
    if left == "-":
        return "protein"
    # A mask run between two same-modality tokens within one element shares their modality.
    if left in MIXED_VOCAB_SET and right in MIXED_VOCAB_SET:
        left_modality = _token_modality(left)
        if left_modality == _token_modality(right):
            return left_modality
    return None


def _token_modality(token: str) -> MixedModality:
    """Classify a biological token as protein (uppercase) or DNA (lowercase)."""
    return "dna" if token in DNA_TOKENS else "protein"


class MixedSequenceSampleInput(MixedSequenceInput):
    """Prepared loci and explicit modality metadata for any premasked sites.

    Attributes:
        sequences (list[str]): Prepared loci, optionally carrying ``_`` masks.
        mask_modalities (list[dict[int, MixedModality]] | None): One mapping per locus,
            assigning every mask's 1-indexed token position to protein or DNA. Positions
            index the ``+``/``-`` form. Omit for fully specified sequences.
    """

    ALLOW_MASKS: ClassVar[bool] = True
    sequences: list[str] = InputField(
        title="Sequences",
        description="Prepared mixed loci; '_' marks editable sites with explicit mask_modalities",
        min_length=1,
        examples=["+MKTL+acgt"],
    )
    mask_modalities: list[dict[int, MixedModality]] | None = InputField(
        default=None,
        title="Mask Modalities",
        description="Per-locus maps from 1-indexed masked token positions to protein or dna",
    )

    @model_validator(mode="after")
    def validate_mask_modalities(self) -> "MixedSequenceSampleInput":
        """Require exact mask coverage and keep the parallel input lists aligned."""
        modalities = self.mask_modalities
        if modalities is None:
            modalities = [{} for _ in self.sequences]
        if len(modalities) != len(self.sequences):
            raise ValueError("mask_modalities must have one mapping per sequence")
        for index, (sequence, mapping) in enumerate(zip(self.sequences, modalities, strict=True)):
            masked = {i + 1 for i, token in enumerate(sequence) if token == MASK_TOKEN}
            if set(mapping) != masked:
                raise ValueError(
                    f"mask_modalities[{index}] must label exactly the masked token positions {sorted(masked)}"
                )
            for position, modality in mapping.items():
                expected = _implied_mask_modality(sequence, position - 1)
                if expected is not None and modality != expected:
                    raise ValueError(f"mask_modalities[{index}][{position}] must be {expected!r} from its context")
        object.__setattr__(self, "mask_modalities", modalities)
        return self


class MixedSequenceGradientInput(BaseToolInput):
    """Relaxed biological tokens aligned to a fixed mixed-sequence template.

    Attributes:
        sequence (str): Fully specified template fixing modality and ``+``/``-`` strand markers.
        logits (list[list[float]]): One 24-column row per model token in
            ``ACDEFGHIKLMNPQRSTVWYacgt`` order. Fixed context rows must be zero.
        temperature (float | None): Softmax temperature; ``None`` requires probability
            distributions over the permitted modality and zero elsewhere.
    """

    sequence: str = InputField(
        title="Sequence", description="Prepared template fixing each token's modality and strand markers"
    )
    logits: list[list[float]] = InputField(
        title="Logits",
        description="Token-aligned (L, 24) state in ACDEFGHIKLMNPQRSTVWYacgt order; fixed rows are zero",
    )
    temperature: float | None = InputField(
        default=1.0,
        title="Temperature",
        gt=0.0,
        allow_inf_nan=False,
        description="Apply softmax(input / T); None requires modality-restricted probabilities",
    )

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, sequence: str) -> str:
        """Validate tokens and store one-character strand markers."""
        return "".join(tokenize_mixed_sequence(sequence))

    @model_validator(mode="after")
    def validate_relaxed_sequence(self) -> "MixedSequenceGradientInput":
        """Check token alignment, finite state, fixed context, and probability domains."""
        tokens = list(self.sequence)
        if len(tokens) != len(self.logits):
            raise ValueError("logits must have one row per model token, including strand markers")
        if not any(t in MIXED_VOCAB for t in tokens):
            raise ValueError("Gradient input requires at least one canonical protein or DNA token")
        for index, (token, row) in enumerate(zip(tokens, self.logits, strict=True)):
            if len(row) != len(MIXED_VOCAB) or not all(math.isfinite(v) for v in row):
                raise ValueError(f"logits row {index + 1} must contain 24 finite numbers")
            if token not in MIXED_VOCAB:
                if any(row):
                    raise ValueError(f"Fixed context row {index + 1} must contain zeros")
            elif self.temperature is None:
                permitted = range(20) if token in PROTEIN_TOKENS else range(20, 24)
                if any(v < 0.0 or (j not in permitted and v != 0.0) for j, v in enumerate(row)) or not math.isclose(
                    sum(row), 1.0, abs_tol=1e-6
                ):
                    raise ValueError(f"logits row {index + 1} must be a probability distribution within its modality")
        return self


class MixedSequenceEmbedding(SequenceEmbedding):
    """A pooled embedding with optional per-position logits.

    Attributes:
        vocab (list[str]): Column order of optional biological logits.
        logits (list[list[float]] | None): Optional per-token protein/DNA logits.
    """

    logits: list[list[float]] | None = Field(
        default=None, title="Logits", description="Optional per-token biological logits in vocabulary order"
    )
    vocab: list[str] = Field(
        default_factory=lambda: list(MIXED_VOCAB), title="Vocabulary", description="Biological-logit column order"
    )


class MixedEmbeddingsOutput(MaskedModelEmbeddingsOutput):
    """Embeddings and labeled logits, one bundle per input locus.

    Attributes:
        results (list[MixedSequenceEmbedding]): Per-locus embedding bundles in input order.
    """

    results: list[MixedSequenceEmbedding] = Field(title="Results", description="Embedding bundles in input order")  # type: ignore[assignment]

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Preserve vocabulary and optional logits in JSON; reuse pooled-vector exports."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        else:
            super()._export_output(export_path, file_format)


class MixedSequenceSample(MaskedModelSample):
    """One sampled locus with labeled biological logits.

    Attributes:
        sequence (str): Completed mixed protein/DNA locus with original strand markers.
        logits (list[list[float]] | None): Optional per-token protein/DNA logits.
        vocab (list[str]): Column order of optional biological logits.
    """

    sequence: str = Field(title="Sequence", description="Completed mixed protein/DNA locus with strand markers")
    logits: list[list[float]] | None = Field(
        default=None, title="Logits", description="Optional per-token biological logits with shape (L, 24)"
    )
    vocab: list[str] = Field(
        default_factory=lambda: list(MIXED_VOCAB), title="Vocabulary", description="Biological-logit column order"
    )


class MixedSampleOutput(MaskedModelSampleOutput):
    """Sampled loci and optional logits, retaining case and marker placement.

    Attributes:
        results (list[MixedSequenceSample]): Per-locus sampling bundles in input order.
    """

    results: list[MixedSequenceSample] = Field(title="Results", description="Sampled locus bundles in input order")  # type: ignore[assignment]

    @property
    def output_format_options(self) -> list[str]:
        """Mixed strings are exported as JSON or text, rather than biological FASTA."""
        return ["json", "txt"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Preserve vocabulary and optional logits in JSON exports."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        else:
            super()._export_output(export_path, file_format)


class MixedScoringMetrics(MaskedModelScoringMetrics):
    """Canonical masked-PLL metrics with explicit target-position metadata.

    Attributes:
        scored_positions (list[int]): 1-indexed canonical protein/DNA target token positions.
    """

    scored_positions: list[int] = Field(
        title="Scored Positions", description="1-indexed model-token positions contributing to the masked PLL metrics"
    )


class MixedScoringOutput(MaskedModelScoringOutput):
    """Per-locus masked-PLL metrics and optional labeled logits.

    Attributes:
        scores (list[MixedScoringMetrics]): Per-locus scores in input order.
    """

    scores: list[MixedScoringMetrics] = Field(title="Scores", description="Per-locus scores in input order")  # type: ignore[assignment]

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Preserve scored-position metadata in JSON; reuse the shared metric CSV export."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.scores]))
        else:
            super()._export_output(export_path, file_format)


class MixedGradientOutput(GradientOutput):
    """Token-aligned masked-PLL gradient with fixed-context rows preserved.

    Attributes:
        gradient (GradientValue): Derivative of masked NLL with respect to the input state.
        loss (float): Mean masked negative log-likelihood.
        metrics (dict[str, Any]): Masked PLL metrics and checkpoint metadata.
        vocab (list[str]): Protein/DNA column order shared by input and gradient.
    """

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient, loss, metrics, and vocabulary as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        payload = self.model_dump(include={"gradient", "loss", "metrics", "vocab"}, mode="json")
        Path(str(export_path) + ".json").write_text(json.dumps(payload))


__all__ = [
    "MIXED_VOCAB",
    "MixedEmbeddingsOutput",
    "MixedGradientOutput",
    "MixedSampleOutput",
    "MixedScoringMetrics",
    "MixedScoringOutput",
    "MixedSequenceEmbedding",
    "MixedSequenceGradientInput",
    "MixedSequenceInput",
    "MixedSequenceSample",
    "MixedSequenceSampleInput",
    "one_hot_mixed_logits",
]
