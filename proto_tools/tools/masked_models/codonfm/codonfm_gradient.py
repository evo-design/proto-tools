"""CodonFM (Encodon) masked pseudo-log-likelihood gradient for relaxed coding sequences."""

import math
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CODONFM_CODON_VOCAB,
    CODONFM_MAX_NT,
    CODONFM_NUM_CODONS,
    CodonFMCheckpoint,
    CodonSequenceLogits,
    one_hot_codon_logits,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    DifferentiableTerm,
    GradientInput,
    GradientOutput,
    RelaxationConfig,
    ToolInstance,
    attach_intrinsic_term,
    coerce_to_item_list,
)
from proto_tools.utils.tool_io import Directionality, InputField

# Encodon caps sequences at (2048 - 2) codons (CLS/SEP); the differentiable state is per-codon.
CODONFM_MAX_CODONS = CODONFM_MAX_NT // 3


class CodonFMGradientInput(GradientInput):
    """Input for the CodonFM gradient tool.

    Attributes:
        inputs (list[CodonSequenceLogits]): One relaxed coding-sequence state per candidate, each
            ``(L, 64)`` in lexicographic DNA codon order (``AAA, AAC, AAG, AAT, ...``; see
            :data:`CODONFM_CODON_VOCAB`). Every ``L`` must be ≤ 2046 (Encodon's positional cap
            minus the CLS/SEP tokens); over-length inputs raise ``ValueError``. Each row's argmax
            codon is that position's masked-language-model target.
        temperature (float | None): Softmax temperature. The default ``1.0`` reads ``logits``
            as unnormalized and applies ``softmax(logits / T)``, returning a logit-space
            gradient; ``None`` reads them as probabilities and returns a probability-space
            gradient.
    """

    inputs: list[CodonSequenceLogits] = InputField(
        title="Relaxed Sequences",
        description="One relaxed (L, 64) coding-sequence state per candidate, in lexicographic codon order",
        examples=[[[[0.0] * CODONFM_NUM_CODONS, [0.0] * CODONFM_NUM_CODONS]]],
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one relaxed state as well as a list of them."""
        return coerce_to_item_list(value)

    @model_validator(mode="after")
    def validate_codon_count(self) -> Self:
        """Reject any item beyond Encodon's positional cap."""
        for index, matrix in enumerate(self.inputs):
            if len(matrix) > CODONFM_MAX_CODONS:
                raise ValueError(
                    f"codonfm: supports sequences up to {CODONFM_MAX_CODONS} codons; "
                    f"inputs[{index}] has length {len(matrix)}."
                )
        return self


class CodonFMMaskedPLLTerm(DifferentiableTerm):
    """CodonFM's one differentiable quantity: the codon-level masked pseudo-log-likelihood.

    The tool exposes a single scalar, so the term carries no selectors beyond the identifier
    naming what it is.

    Attributes:
        id (Literal["codon_masked_pll"]): Fixed identifier for this kind of quantity.
    """

    #: A mean negative log-likelihood is better the smaller it is.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"

    #: Nats, averaged over the codon positions the mean is taken across.
    UNITS: ClassVar[str] = "nats per codon"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Mean masked negative log-likelihood over codon positions"

    id: Literal["codon_masked_pll"] = Field(
        default="codon_masked_pll",
        title="Term ID",
        description="Identifies this quantity as CodonFM's codon-level masked pseudo-log-likelihood.",
    )


class CodonFMGradientOutput(GradientOutput):
    """CodonFM masked-PLL output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input item, in input order, each holding that
            item's gradient w.r.t. its codon logits (``None`` in forward-only mode), its mean
            masked negative log-likelihood over codon positions, and its log-likelihood,
            perplexity, sequence length and objective details.
        vocab (list[str]): Codon column ordering for the input logits and returned gradient.
    """


class CodonFMGradientConfig(RelaxationConfig):
    """Configuration for the CodonFM masked-PLL gradient tool.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon weights variant.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only masked-log-likelihood scoring.
        batch_size (int): Codon positions per forward pass for batched masked-PLL.
        device (str): Device to run the model on.
    """

    model_checkpoint: CodonFMCheckpoint = ConfigField(
        title="Model Checkpoint",
        default="encodon_80m",
        description="Encodon checkpoint: encodon_80m | encodon_600m | encodon_1b | encodon_1b_cdwt.",
        reload_on_change=True,
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only masked log-likelihood.",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=32,
        ge=1,
        description="Codon positions per forward pass. Lower if OOM, higher for throughput.",
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="CUDA device to run CodonFM inference on.",
        include_in_key=False,
    )


def example_input() -> CodonFMGradientInput:
    """Minimal valid input for testing and examples."""
    return CodonFMGradientInput(inputs=[CodonSequenceLogits(one_hot_codon_logits("ATGGTGAGCAAG", sharpness=2.0))])


@tool(
    key="codonfm-gradient",
    label="CodonFM Gradient",
    category="masked_models",
    input_class=CodonFMGradientInput,
    config_class=CodonFMGradientConfig,
    output_class=CodonFMGradientOutput,
    description="Compute the CodonFM/Encodon masked pseudo-log-likelihood gradient for relaxed coding sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=8,
)
def run_codonfm_gradient(
    inputs: CodonFMGradientInput,
    config: CodonFMGradientConfig,
    instance: Any = None,
) -> CodonFMGradientOutput:
    """Compute the CodonFM masked-PLL gradient with respect to relaxed codon logits.

    Args:
        inputs (CodonFMGradientInput): Relaxed ``(L, 64)`` codon logits (+ optional temperature).
        config (CodonFMGradientConfig): CodonFM runtime, checkpoint, and gradient configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CodonFMGradientOutput: Gradient (unless forward-only), mean-NLL loss, metrics, and codon vocab.
    """
    safetensors_url, config_url, filename, subdir = resolve_checkpoint_source(config.model_checkpoint)
    result = ToolInstance.dispatch(
        "codonfm",
        {
            "operation": "gradient",
            "items": [matrix.root for matrix in inputs.inputs],
            "temperature": inputs.temperature,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "batch_size": config.batch_size,
            "safetensors_url": safetensors_url,
            "config_url": config_url,
            "safetensors_filename": filename,
            "cache_subdir": subdir,
            "device": config.device,
            "seed": config.seed,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    if result.get("vocab") != CODONFM_CODON_VOCAB:
        raise ValueError("CodonFM returned an unexpected codon vocabulary")
    items = result["results"]
    if len(items) != len(inputs.inputs):
        raise ValueError(f"CodonFM returned {len(items)} results for {len(inputs.inputs)} item(s)")
    for index, (item, matrix) in enumerate(zip(items, inputs.inputs, strict=True)):
        loss = float(item["loss"])
        if not math.isfinite(loss):
            raise ValueError(f"CodonFM returned a non-finite gradient loss for item {index}")
        item["loss"] = loss
        gradient = item.get("gradient")
        if config.compute_gradient:
            if not isinstance(gradient, list) or len(gradient) != len(matrix):
                raise ValueError(f"CodonFM returned a gradient with an unexpected length for item {index}")
            for row_index, row in enumerate(gradient):
                if len(row) != CODONFM_NUM_CODONS or not all(math.isfinite(float(value)) for value in row):
                    raise ValueError(f"CodonFM returned an invalid gradient row at item {index}, row {row_index}")
        elif gradient is not None:
            raise ValueError("CodonFM returned a gradient when compute_gradient=False")
    result["metadata"] = {"model_checkpoint": config.model_checkpoint}
    attach_intrinsic_term(
        result,
        CodonFMMaskedPLLTerm(),
        matrices=[matrix.root for matrix in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
    return CodonFMGradientOutput(**result)
