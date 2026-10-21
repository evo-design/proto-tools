"""AbLang masked pseudo-log-likelihood gradient tool."""

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import field_validator

from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    GradientInput,
    GradientOutput,
    ProteinSequenceLogits,
    RelaxationConfig,
    ToolInstance,
    coerce_to_item_list,
    one_hot_protein_logits,
)
from proto_tools.utils.tool_io import InputField

logger = logging.getLogger(__name__)


class AbLangGradientInput(GradientInput):
    """Input for the AbLang gradient tool.

    Attributes:
        inputs (list[AntibodyLogits]): One antibody per candidate, each with relaxed sequence
            distributions and one matrix per chain. A single antibody may be passed on its own.
            The model variant is selected from which chains an antibody provides, and its
            returned gradient concatenates them heavy-then-light.
        temperature (float | None): Softmax temperature shared by every candidate. The default
            ``1.0`` reads each chain as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[AntibodyLogits] = InputField(
        title="Antibodies",
        description="One antibody per candidate, each with relaxed per-chain distributions; one is accepted alone",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one antibody as well as a list of them."""
        return coerce_to_item_list(value)

    @classmethod
    def from_probabilities(  # type: ignore[override]  # chains are named, not a single matrix
        cls,
        *,
        heavy_chain: Sequence[Sequence[float]] | None = None,
        light_chain: Sequence[Sequence[float]] | None = None,
        temperature: float = 1.0,
        **fields: Any,
    ) -> "AbLangGradientInput":
        """Build an input whose chain logits reproduce the given distributions.

        At least one chain must be supplied, and both are converted at the same temperature the
        input stores, so the two cannot disagree.

        Args:
            heavy_chain (Sequence[Sequence[float]] | None): ``(Lh, 20)`` strictly positive rows.
            light_chain (Sequence[Sequence[float]] | None): ``(Ll, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature stored on the input.
            fields (Any): Any remaining input fields.

        Returns:
            AbLangGradientInput: Input satisfying ``softmax(chain / temperature) == chain probs``.
        """
        convert = ProteinSequenceLogits.from_probabilities
        return cls(
            inputs=[
                AntibodyLogits(
                    heavy_chain=None if heavy_chain is None else convert(heavy_chain, temperature=temperature),
                    light_chain=None if light_chain is None else convert(light_chain, temperature=temperature),
                )
            ],
            temperature=temperature,
            **fields,
        )


class AbLangGradientOutput(GradientOutput):
    """AbLang masked PLL output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input antibody, each holding its gradient
            w.r.t. the supplied logits (``None`` in forward-only mode), its mean negative
            log-likelihood over AA positions, and its ``log_likelihood``,
            ``avg_log_likelihood``, ``perplexity`` and chain-length metrics.
        vocab (list[str]): Amino-acid column ordering for the input logits.
    """


class AbLangGradientConfig(RelaxationConfig):
    """Configuration for the AbLang masked PLL gradient tool.

    Attributes:
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only log-likelihood scoring (e.g. MCMC proposal ranking).
        batch_size (int | None): AA positions per forward pass for batched PLL. ``None`` auto-
            selects a per-model default (lower if OOM, higher for throughput).
        device (str): Device to run the model on.
    """

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


def example_input() -> AbLangGradientInput:
    """Minimal valid input for testing and examples."""
    heavy = ProteinSequenceLogits(one_hot_protein_logits("EVQLVESG", sharpness=2.0))
    return AbLangGradientInput(inputs=[AntibodyLogits(heavy_chain=heavy)])


@tool(
    key="ablang-gradient",
    label="AbLang Gradient",
    category="masked_models",
    input_class=AbLangGradientInput,
    config_class=AbLangGradientConfig,
    output_class=AbLangGradientOutput,
    description="Compute AbLang masked pseudo-log-likelihood gradient for relaxed antibody sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=8,
    stochastic=True,
)
def run_ablang_gradient(
    inputs: AbLangGradientInput,
    config: AbLangGradientConfig,
    instance: Any = None,
) -> AbLangGradientOutput:
    """Compute AbLang masked PLL gradient with respect to relaxed antibody logits."""
    items: list[dict[str, Any]] = []
    for antibody in inputs.inputs:
        # AbLang's checkpoint follows the chains an antibody carries, so each candidate names its
        # own and the worker groups a mixed chunk by checkpoint. AbLang also scores one matrix per
        # antibody, so a paired one is concatenated heavy-first and the break position tells the
        # worker where to split it again.
        if antibody.heavy_chain is not None and antibody.light_chain is not None:
            logits = antibody.heavy_chain.root + antibody.light_chain.root
            chain_break_position: int | None = len(antibody.heavy_chain)
        elif antibody.heavy_chain is not None:
            logits, chain_break_position = antibody.heavy_chain.root, None
        else:
            logits, chain_break_position = antibody.light_chain.root, None  # type: ignore[union-attr]
        items.append(
            {
                "logits": logits,
                "chain_break_position": chain_break_position,
                "model_choice": _resolve_model_choice([antibody]),
            }
        )

    logger.debug(
        "Using local worker for AbLang gradient (%d item(s), checkpoints=%s)",
        len(items),
        sorted({item["model_choice"] for item in items}),
    )
    payload = {
        "operation": "compute_gradient",
        "items": items,
        "temperature": inputs.temperature,
        "soft": config.soft,
        "hard": config.hard,
        "compute_gradient": config.compute_gradient,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "device": config.device,
        "verbose": config.verbose,
    }
    result = ToolInstance.dispatch(
        "ablang",
        payload,
        instance=instance,
        config=config,
    )
    return AbLangGradientOutput(**result)
