"""Shared contracts for mixed protein/DNA masked language models."""

import json
import logging
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from standalone_helpers.mixed_sequence import (
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
from proto_tools.transforms.masking import MASK_TOKEN, MaskingInput, MaskingStrategy
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    GradientOutput,
    InputField,
    ToolInstance,
)
from proto_tools.utils.compressed_array import decompress_result

logger = logging.getLogger(__name__)

MixedModality = Literal["protein", "dna"]
InteractionHead = Literal["base_pairing", "protein", "repeat"]
CHECKPOINT_CONTEXT = {
    "tattabio/gLM2_150M": 4096,
    "tattabio/gLM2_650M": 4096,
    "gbrixi/minerva-mlm": 4096,
    "gbrixi/minerva-mlm-8k": 8192,
}
CHECKPOINT_DEPTH = {
    "tattabio/gLM2_150M": 30,
    "tattabio/gLM2_650M": 33,
    "gbrixi/minerva-mlm": 33,
    "gbrixi/minerva-mlm-8k": 33,
}


class MixedSequenceInput(BaseToolInput):
    """Prepared protein/DNA loci in the upstream case-sensitive notation.

    Attributes:
        sequences (list[str]): Uppercase proteins and lowercase DNA, optionally separated
            by atomic strand markers. A single string is normalized to a list.
    """

    ALLOW_MASKS: ClassVar[bool] = False
    sequences: list[str] = InputField(
        title="Sequences",
        description="Prepared loci: uppercase protein, lowercase acgt, and <+>/<-> strand markers",
        min_length=1,
        examples=["<+>MKTL<+>acgt<->ACDE"],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        """Accept a single locus or a list of loci."""
        return [value] if isinstance(value, str) else value

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate atomic tokens without changing case or inserting markers."""
        for sequence in sequences:
            tokenize_mixed_sequence(sequence, allow_masks=cls.ALLOW_MASKS)
        return sequences

    def __len__(self) -> int:
        """Return the total number of model tokens, counting each marker once."""
        return sum(len(tokenize_mixed_sequence(s, allow_masks=self.ALLOW_MASKS)) for s in self.sequences)


class MixedSequenceSampleInput(MixedSequenceInput):
    """Prepared loci and explicit modality metadata for any premasked sites.

    Attributes:
        sequences (list[str]): Prepared loci, optionally carrying ``_`` masks.
        mask_modalities (list[dict[int, MixedModality]] | None): One mapping per locus,
            assigning every mask's 1-indexed model-token position to protein or DNA.
            Strand markers count as one token. Omit for fully specified sequences.
    """

    ALLOW_MASKS: ClassVar[bool] = True
    sequences: list[str] = InputField(
        title="Sequences",
        description="Prepared mixed loci; '_' marks editable sites with explicit mask_modalities",
        min_length=1,
        examples=["<+>MKTL<+>acgt"],
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
            tokens = tokenize_mixed_sequence(sequence, allow_masks=True)
            masked = {i + 1 for i, token in enumerate(tokens) if token == MASK_TOKEN}
            if set(mapping) != masked:
                raise ValueError(
                    f"mask_modalities[{index}] must label exactly the masked token positions {sorted(masked)}"
                )
        object.__setattr__(self, "mask_modalities", modalities)
        return self


class MixedSequenceGradientInput(BaseToolInput):
    """Relaxed biological tokens aligned to a fixed mixed-sequence template.

    Attributes:
        sequence (str): Fully specified template fixing modality and strand markers.
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

    @model_validator(mode="after")
    def validate_relaxed_sequence(self) -> "MixedSequenceGradientInput":
        """Check token alignment, finite state, fixed context, and probability domains."""
        tokens = tokenize_mixed_sequence(self.sequence)
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


class MixedModelConfig(BaseConfig):
    """Shared execution configuration; registered tools specialize checkpoint choices.

    Attributes:
        model_checkpoint (str): Public Hugging Face checkpoint.
        batch_size (int): Sequences or masked PLL variants per forward pass.
        device (str): Device used for model inference.
    """

    model_checkpoint: str = ConfigField(
        title="Model Checkpoint", description="Public mixed-modality model checkpoint", reload_on_change=True
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Sequences or masked PLL variants per forward pass; reduce if memory is limited",
    )
    device: str = ConfigField(
        default="cuda", title="Device", description="Device used for model inference", include_in_key=False
    )

    def preprocess(self, inputs: Any) -> Any:
        """Check checkpoint-specific context limits before loading model weights."""
        sequences = inputs.sequences if hasattr(inputs, "sequences") else [inputs.sequence]
        limit = CHECKPOINT_CONTEXT[self.model_checkpoint]
        for index, sequence in enumerate(sequences):
            length = len(tokenize_mixed_sequence(sequence, allow_masks=True))
            if length > limit:
                raise ValueError(f"{self.model_checkpoint}: sequence {index} has {length} tokens; limit is {limit}")
        return inputs


class MixedEmbeddingsConfig(MixedModelConfig):
    """Embedding extraction options.

    Attributes:
        return_logits (bool): Include token-aligned biological logits.
        repr_layer (int): Representation layer: 0 is the embedding table, 1..N are
            transformer outputs, and -1 selects the last transformer output.
    """

    return_logits: bool = ConfigField(
        default=False,
        title="Return Logits",
        description="Include token-aligned logits over the 24 canonical biological tokens",
    )
    repr_layer: int = ConfigField(
        default=-1,
        ge=-1,
        title="Representation Layer",
        description="0=embedding table, 1..N=transformer outputs, -1=last transformer output",
    )

    @model_validator(mode="after")
    def validate_layer(self) -> "MixedEmbeddingsConfig":
        """Reject layers absent from the selected checkpoint."""
        if self.repr_layer > CHECKPOINT_DEPTH[self.model_checkpoint]:
            raise ValueError("repr_layer exceeds the selected model's transformer depth")
        return self


class MixedScoringConfig(MixedModelConfig):
    """Masked pseudo-log-likelihood scoring options.

    Attributes:
        return_logits (bool): Return masked logits; unscored context rows contain zeros.
    """

    return_logits: bool = ConfigField(
        default=False,
        title="Return Logits",
        description="Include masked biological logits; unscored context rows contain zeros",
    )


class MixedSampleConfig(MixedModelConfig):
    """Modality-preserving sampling with the existing masking and refinement strategies.

    Attributes:
        masking_strategy (MaskingStrategy): Select editable token positions when no masks are supplied.
        sampling_method (Literal['single_pass', 'iterative_refinement']): Sampling algorithm.
        temperature (float): Temperature for sampling alternatives within each position's modality.
        top_p (float): Nucleus threshold for iterative refinement.
        num_steps (int): Number of iterative refinement rounds.
        schedule (Literal['cosine', 'linear']): Iterative unmask schedule.
        strategy (Literal['random', 'entropy']): Select commitments randomly or by confidence.
        temperature_annealing (bool): Cool temperature during iterative refinement.
        return_logits (bool): Return biological logits, evaluated on the completed sequence.
    """

    masking_inputs: ClassVar[frozenset[MaskingInput]] = frozenset({MaskingInput.LOGITS})
    masking_strategy: MaskingStrategy = ConfigField(
        default_factory=MaskingStrategy,
        title="Masking Strategy",
        description="Select editable model-token positions; explicit '_' masks bypass selection",
    )
    sampling_method: Literal["single_pass", "iterative_refinement"] = ConfigField(
        default="single_pass",
        title="Sampling Method",
        description="Fill all masks once, or progressively commit predictions across refinement rounds",
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Temperature",
        description="Sampling temperature within each position's protein or DNA alphabet",
    )
    top_p: float = ConfigField(
        default=1.0,
        gt=0.0,
        le=1.0,
        title="Top P",
        description="Iterative nucleus threshold; 1.0 disables nucleus filtering",
    )
    num_steps: int = ConfigField(
        default=20, ge=1, title="Num Steps", description="Number of iterative refinement rounds"
    )
    schedule: Literal["cosine", "linear"] = ConfigField(
        default="cosine", title="Schedule", description="Iterative unmask schedule"
    )
    strategy: Literal["random", "entropy"] = ConfigField(
        default="random", title="Strategy", description="Commit masked sites randomly or by lowest predictive entropy"
    )
    temperature_annealing: bool = ConfigField(
        default=True,
        title="Temperature Annealing",
        description="Cool sampling temperature over iterative refinement rounds",
    )
    return_logits: bool = ConfigField(
        default=False,
        title="Return Logits",
        description="Include biological logits from a final forward pass over each completed sequence",
    )

    def preprocess(self, inputs: Any) -> Any:
        """Select tokens once, preserving original modalities in the normalized input."""
        inputs = super().preprocess(inputs)
        if any("_" in sequence for sequence in inputs.sequences):
            if self.masking_strategy != MaskingStrategy():
                logger.warning(
                    "Sequences already contain mask tokens ('_'); ignoring custom masking_strategy. "
                    "Remove '_' tokens to use the strategy, or omit masking_strategy to silence this warning."
                )
            return inputs
        tokens = [tokenize_mixed_sequence(sequence) for sequence in inputs.sequences]
        eligibility = [[token in MIXED_VOCAB for token in row] for row in tokens]
        position_score_fn = None
        if self.masking_strategy.method != "random":
            # Import the matching embedding tool lazily to avoid registration cycles.
            from importlib import import_module

            toolkit = "glm2" if self.model_checkpoint.startswith("tattabio/") else "minerva"
            module = import_module(f"proto_tools.tools.masked_models.{toolkit}.{toolkit}_embeddings")
            prefix = "GLM2" if toolkit == "glm2" else "Minerva"
            embedding_config = getattr(module, f"{prefix}EmbeddingsConfig")(
                model_checkpoint=self.model_checkpoint,
                batch_size=self.batch_size,
                device=self.device,
                return_logits=True,
            )
            result = getattr(module, f"run_{toolkit}_embeddings")(
                MixedSequenceInput(sequences=inputs.sequences),
                embedding_config,
            )
            restricted_logits = []
            for row_tokens, item in zip(tokens, result.results, strict=True):
                rows = []
                for token, logits in zip(row_tokens, item.logits, strict=True):
                    permitted = range(20, 24) if token in DNA_TOKENS else range(20)
                    rows.append([value if j in permitted else -math.inf for j, value in enumerate(logits)])
                restricted_logits.append(rows)

            def position_score_fn(_sequences: list[str]) -> list[list[list[float]]]:
                return restricted_logits

        masked = self.masking_strategy.mask_tokens(
            tokens,
            eligibility=eligibility,
            position_score_fn=position_score_fn,
            seed=self.seed,
        )
        modalities = [
            {i + 1: ("protein" if before[i] in PROTEIN_TOKENS else "dna") for i, t in enumerate(after) if t == "_"}
            for before, after in zip(tokens, masked, strict=True)
        ]
        return inputs.model_copy(update={"sequences": ["".join(row) for row in masked], "mask_modalities": modalities})


class MixedGradientConfig(MixedModelConfig):
    """Differentiable masked pseudo-log-likelihood options.

    Attributes:
        use_ste (bool): Use hard forward tokens with a soft backward derivative.
        compute_gradient (bool): Compute a backward pass, or return only the objective.
    """

    use_ste: bool = ConfigField(
        default=False,
        title="Straight-Through Estimator",
        description="Use hard one-hot forward tokens with soft-probability gradients",
    )
    compute_gradient: bool = ConfigField(
        default=True,
        title="Compute Gradient",
        description="Run backward and return the gradient; False evaluates only the masked PLL objective",
    )


class MixedSequenceEmbedding(SequenceEmbedding):
    """A pooled embedding with labeled mixed-token axes.

    Attributes:
        tokens (list[str]): Unpadded model tokens, including strand markers.
        vocab (list[str]): Column order of optional biological logits.
        logits (list[list[float]] | None): Optional per-token protein/DNA logits.
    """

    logits: list[list[float]] | None = Field(
        default=None, title="Logits", description="Optional per-token biological logits in vocabulary order"
    )
    tokens: list[str] = Field(title="Tokens", description="Unpadded token axis, including strand markers")
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
        """Preserve token labels and optional logits in JSON; reuse pooled-vector exports."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        else:
            super()._export_output(export_path, file_format)


class MixedSequenceSample(MaskedModelSample):
    """One sampled locus with labeled biological logits.

    Attributes:
        sequence (str): Completed mixed protein/DNA locus with original strand markers.
        logits (list[list[float]] | None): Optional per-token protein/DNA logits.
        tokens (list[str]): Completed token sequence, including fixed strand markers.
        vocab (list[str]): Column order of optional biological logits.
    """

    sequence: str = Field(title="Sequence", description="Completed mixed protein/DNA locus with strand markers")
    logits: list[list[float]] | None = Field(
        default=None, title="Logits", description="Optional per-token biological logits with shape (L, 24)"
    )
    tokens: list[str] = Field(title="Tokens", description="Completed model-token sequence, including strand markers")
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
        """Preserve token labels and optional logits in JSON exports."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        else:
            super()._export_output(export_path, file_format)


class MixedScoringMetrics(MaskedModelScoringMetrics):
    """Canonical masked-PLL metrics with explicit target-position metadata.

    Attributes:
        tokens (list[str]): All unpadded model tokens, including unscored context.
        scored_positions (list[int]): 1-indexed canonical protein/DNA target token positions.
    """

    tokens: list[str] = Field(title="Tokens", description="Unpadded model tokens, including unscored context")
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
        tokens (list[str]): Model-token axis shared by the input and gradient matrices.
    """

    tokens: list[str] = Field(
        title="Tokens", description="Model-token axis shared by input logits and returned gradient"
    )

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Include token labels alongside the gradient, loss, metrics, and vocabulary."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        payload = self.model_dump(include={"gradient", "loss", "metrics", "vocab", "tokens"}, mode="json")
        Path(str(export_path) + ".json").write_text(json.dumps(payload))


class SequenceInteractionMap(BaseModel):
    """A dense probability matrix with explicitly labeled token axes.

    Attributes:
        tokens (list[str]): Ordered labels for both matrix axes, including strand markers.
        values (list[list[float]]): Square matrix of finite interaction probabilities.
    """

    tokens: list[str] = Field(
        title="Tokens", description="Labels for both matrix axes, including strand markers", min_length=1
    )
    values: list[list[float]] = Field(
        title="Values", description="Square token-by-token interaction probability matrix"
    )

    @model_validator(mode="after")
    def validate_matrix(self) -> "SequenceInteractionMap":
        """Validate square shape, axis alignment, and finite probability values."""
        length = len(self.tokens)
        if len(self.values) != length or any(len(row) != length for row in self.values):
            raise ValueError("Interaction matrix must be square and match its token axis")
        if any(not 0.0 <= value <= 1.0 for row in self.values for value in row):
            raise ValueError("Interaction probabilities must be finite values in [0, 1]")
        return self


class SequenceInteractions(BaseModel):
    """Selected interaction channels for one input locus.

    Attributes:
        maps (dict[InteractionHead, SequenceInteractionMap]): Named model interaction heads.
    """

    maps: dict[InteractionHead, SequenceInteractionMap] = Field(
        title="Maps", description="Selected named interaction-head probability maps"
    )


class MixedInteractionsOutput(BaseToolOutput):
    """Dense interaction maps, one bundle per input sequence.

    Attributes:
        results (list[SequenceInteractions]): Interaction bundles in input order.
    """

    results: list[SequenceInteractions] = Field(title="Results", description="Interaction bundles in input order")

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "npz"]

    @property
    def output_format_default(self) -> str:
        """Prefer compact archives for dense interaction matrices."""
        return "npz"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export matrices and token labels without pickle-dependent object arrays."""
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        elif file_format == "npz":
            import numpy as np

            arrays: dict[str, Any] = {}
            for index, result in enumerate(self.results):
                for head, contact_map in result.maps.items():
                    arrays[f"{index}_{head}"] = np.asarray(contact_map.values, dtype=np.float32)
                    arrays[f"{index}_{head}_tokens"] = np.asarray(contact_map.tokens, dtype=str)
            np.savez_compressed(str(export_path) + ".npz", **arrays)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


def dispatch_mixed_model(
    toolkit: str, operation: str, inputs: BaseToolInput, config: MixedModelConfig, instance: Any
) -> dict[str, Any]:
    """Dispatch a shared plain-data contract through the standard worker infrastructure."""
    payload = {**config.model_dump(), **inputs.model_dump(), "operation": operation}
    result = ToolInstance.dispatch(toolkit, payload, instance=instance, config=config)
    result = decompress_result(result, to_list=True)
    result["metadata"] = {"model_checkpoint": config.model_checkpoint}
    return result  # type: ignore[no-any-return]


__all__ = [
    "MIXED_VOCAB",
    "MixedEmbeddingsOutput",
    "MixedGradientOutput",
    "MixedScoringMetrics",
    "MixedScoringOutput",
    "MixedSequenceGradientInput",
    "MixedSequenceInput",
    "MixedSequenceSampleInput",
    "SequenceInteractionMap",
    "SequenceInteractions",
    "one_hot_mixed_logits",
]
