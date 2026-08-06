"""CodonFM (Encodon) masked pseudo-log-likelihood gradient for relaxed coding sequences."""

import math
from typing import Any

from pydantic import Field, field_validator, model_validator

from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CODONFM_CODON_VOCAB,
    CODONFM_MAX_NT,
    CODONFM_NUM_CODONS,
    CodonFMCheckpoint,
    one_hot_codon_logits,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, GradientOutput, ToolInstance
from proto_tools.utils.tool_io import InputField

# Encodon caps sequences at (2048 - 2) codons (CLS/SEP); the differentiable state is per-codon.
CODONFM_MAX_CODONS = CODONFM_MAX_NT // 3


class CodonFMGradientInput(BaseToolInput):
    """Input for the CodonFM gradient tool.

    Attributes:
        logits (list[list[float]]): Relaxed coding-sequence state with shape ``(L, 64)`` in
            lexicographic DNA codon order (``AAA, AAC, AAG, AAT, ...``; see
            :data:`CODONFM_CODON_VOCAB`). ``L`` must be ≤ 2046 (Encodon's positional cap minus
            the CLS/SEP tokens); over-length inputs raise ``ValueError``. Each row's argmax codon
            is used as that position's masked-language-model target.
        temperature (float | None): Optional softmax temperature. When set, applies
            ``softmax(input / temperature)`` before computing the gradient. When ``None``,
            every input row must already be a probability distribution.
    """

    logits: list[list[float]] = InputField(
        title="Logits",
        description="Relaxed coding-sequence logits with shape (L, 64) in lexicographic DNA codon order.",
        examples=[[[0.0] * CODONFM_NUM_CODONS, [0.0] * CODONFM_NUM_CODONS]],
    )

    temperature: float | None = InputField(
        default=1.0,
        title="Temperature",
        description="Softmax temperature. Applies softmax(input / T) when set.",
        gt=0.0,
        allow_inf_nan=False,
    )

    @field_validator("logits")
    @classmethod
    def validate_logits(cls, logits: list[list[float]]) -> list[list[float]]:
        """Ensure logits are a non-empty rectangular ``L x 64`` matrix within the codon cap."""
        if not logits:
            raise ValueError("logits must contain at least one codon position")
        if len(logits) > CODONFM_MAX_CODONS:
            raise ValueError(
                f"codonfm: supports sequences up to {CODONFM_MAX_CODONS} codons; input has length {len(logits)}."
            )
        for idx, row in enumerate(logits):
            if len(row) != CODONFM_NUM_CODONS:
                raise ValueError(f"logits row {idx} must have {CODONFM_NUM_CODONS} columns, got {len(row)}")
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"logits row {idx} must contain only finite numbers")
        return logits

    @model_validator(mode="after")
    def validate_probabilities_without_temperature(self) -> "CodonFMGradientInput":
        """Require probability-simplex rows when softmax is explicitly disabled."""
        if self.temperature is None:
            for idx, row in enumerate(self.logits):
                if any(value < 0.0 for value in row) or not math.isclose(sum(row), 1.0, abs_tol=1e-6):
                    raise ValueError(f"logits row {idx} must be a probability distribution when temperature is None")
        return self


class CodonFMGradientOutput(GradientOutput):
    """CodonFM masked-PLL output; gradient is optional in forward-only mode.

    Attributes:
        gradient (list[list[float]] | None): Gradient w.r.t. the input codon logits, or ``None``
            when ``compute_gradient=False``.
        loss (float): Mean masked negative log-likelihood over codon positions.
        metrics (dict[str, Any]): Log-likelihood, perplexity, sequence length, and objective details.
        vocab (list[str]): Codon column ordering for the input logits and returned gradient.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. the input codon logits. None when compute_gradient=False.",
    )


class CodonFMGradientConfig(BaseConfig):
    """Configuration for the CodonFM masked-PLL gradient tool.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon weights variant.
        use_ste (bool): Straight-Through Estimator: hard one-hot codons in the forward pass with
            gradients flowing through soft probabilities. When ``False``, uses soft blended
            codon embeddings directly.
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
    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=False,
        description="Hard one-hot forward pass with soft-probability gradients.",
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
    return CodonFMGradientInput(logits=one_hot_codon_logits("ATGGTGAGCAAG", sharpness=2.0))


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
    stochastic=True,
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
            "logits": inputs.logits,
            "temperature": inputs.temperature,
            "use_ste": config.use_ste,
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
    loss = float(result["loss"])
    if not math.isfinite(loss):
        raise ValueError("CodonFM returned a non-finite gradient loss")
    if result.get("vocab") != CODONFM_CODON_VOCAB:
        raise ValueError("CodonFM returned an unexpected codon vocabulary")
    gradient = result.get("gradient")
    if config.compute_gradient:
        if not isinstance(gradient, list) or len(gradient) != len(inputs.logits):
            raise ValueError("CodonFM returned a gradient with an unexpected length")
        for idx, row in enumerate(gradient):
            if len(row) != CODONFM_NUM_CODONS or not all(math.isfinite(float(value)) for value in row):
                raise ValueError(f"CodonFM returned an invalid gradient row at index {idx}")
    elif gradient is not None:
        raise ValueError("CodonFM returned a gradient when compute_gradient=False")
    result["loss"] = loss
    result["metadata"] = {"model_checkpoint": config.model_checkpoint}
    return CodonFMGradientOutput(**result)
