"""ESM2 masked pseudo-log-likelihood gradient tool."""

import logging
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from proto_tools.tools.masked_models.esm2.esm2_sample import ESM2_MODEL_CHECKPOINTS
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    ConfigField,
    GradientOutput,
    ProteinGradientInput,
    ToolInstance,
    one_hot_protein_logits,
)
from proto_tools.utils.device import RemoteDevice

logger = logging.getLogger(__name__)

ESM2_MAX_SEQ_LENGTH = 1022


class ESM2GradientInput(ProteinGradientInput):
    """Input for the ESM2 gradient tool.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed protein sequence state with shape
            ``(L, 20)`` in canonical amino-acid order ``ACDEFGHIKLMNPQRSTVWY``.
            ``L`` must be ≤ 1022 (ESM-2's positional-encoding cap); over-length
            inputs raise ``ValueError``.
        temperature (float | None): Softmax temperature. The default ``1.0`` reads ``logits``
            as unnormalized and applies ``softmax(logits / T)``, returning a logit-space
            gradient; ``None`` reads them as probabilities and returns a probability-space
            gradient.
    """

    @model_validator(mode="after")
    def validate_sequence_length(self) -> Self:
        """Reject inputs beyond ESM-2's positional-encoding cap."""
        if len(self.logits) > ESM2_MAX_SEQ_LENGTH:
            raise ValueError(
                f"esm2: supports sequences up to {ESM2_MAX_SEQ_LENGTH} residues; input has length {len(self.logits)}."
            )
        return self


class ESM2GradientOutput(GradientOutput):
    """ESM2 masked PLL output; gradient is optional in forward-only mode.

    Attributes:
        gradient (list[list[float]] | None): Gradient w.r.t. input logits, or ``None``
            when ``compute_gradient=False``.
        loss (float): Mean negative log-likelihood over AA positions.
        metrics (dict[str, Any]): Log-likelihood, perplexity, sequence length, and objective details.
        vocab (list[str]): Amino-acid column ordering for the input logits.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )


class ESM2GradientConfig(BaseConfig):
    """Configuration for the ESM2 masked PLL gradient tool.

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 weights variant.
        use_ste (bool): Straight-Through Estimator: hard one-hot in the forward pass with
            gradients flowing through soft probabilities. When ``False``, uses soft blended
            embeddings directly.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only log-likelihood scoring.
        batch_size (int | None): AA positions per forward pass for batched PLL. ``None``
            selects the backend default.
        device (str): Device to run the model on.
    """

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="ESM2 weights variant; trade off speed vs scoring fidelity",
        reload_on_change=True,
    )
    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=False,
        description="Hard one-hot forward pass with soft-probability gradients",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only log-likelihood",
    )
    batch_size: int | None = ConfigField(
        title="PLL Batch Size",
        default=None,
        gt=0,
        description="AA positions per forward pass. Lower if OOM, higher for throughput",
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """The 15B variant is too large for Proto to host; other remotes are unaffected."""
        if device == "proto" and self.model_checkpoint == "esm2_t48_15B_UR50D":
            return (
                "The 15B variant (esm2_t48_15B_UR50D) is not hosted on device='proto'. Choose a "
                "smaller variant, run locally, or deploy it yourself with device='modal'."
            )
        return None


def example_input() -> ESM2GradientInput:
    """Minimal valid input for testing and examples."""
    return ESM2GradientInput(logits=one_hot_protein_logits("MKTL", sharpness=2.0))


@tool(
    key="esm2-gradient",
    label="ESM2 Gradient",
    category="masked_models",
    input_class=ESM2GradientInput,
    config_class=ESM2GradientConfig,
    output_class=ESM2GradientOutput,
    description="Compute ESM2 masked pseudo-log-likelihood gradient for relaxed protein sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    stochastic=True,
)
def run_esm2_gradient(
    inputs: ESM2GradientInput,
    config: ESM2GradientConfig,
    instance: Any = None,
) -> ESM2GradientOutput:
    """Compute ESM2 masked PLL gradient with respect to relaxed protein logits."""
    logger.debug("Using local worker for ESM2 gradient: %s", config.model_checkpoint)
    result = ToolInstance.dispatch(
        "esm2",
        {
            "operation": "compute_gradient",
            "logits": inputs.logits.root,
            "temperature": inputs.temperature,
            "use_ste": config.use_ste,
            "compute_gradient": config.compute_gradient,
            "batch_size": config.batch_size,
            "model_checkpoint": config.model_checkpoint,
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    return ESM2GradientOutput(**result)
