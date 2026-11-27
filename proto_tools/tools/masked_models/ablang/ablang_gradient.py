"""AbLang masked pseudo-log-likelihood gradient tool."""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    DifferentiableTerm,
    GradientInput,
    GradientOutput,
    ProteinSequenceLogits,
    RelaxationConfig,
    ToolInstance,
    attach_intrinsic_term,
    coerce_to_item_list,
    one_hot_protein_logits,
)
from proto_tools.utils.tool_io import Directionality, InputField

logger = logging.getLogger(__name__)


class AbLangGradientInput(GradientInput):
    """Input for the AbLang gradient tool.

    Attributes:
        inputs (list[AntibodyLogits]): One antibody per candidate, each with relaxed sequence
            distributions and one matrix per chain.
            The model variant is selected from which chains an antibody provides, and its
            returned gradient concatenates them heavy-then-light.
        temperature (float | None): Softmax temperature shared by every candidate. The default
            ``1.0`` reads each chain as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[AntibodyLogits] = InputField(
        title="Antibodies",
        description="One antibody per candidate, each with relaxed per-chain distributions",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one antibody as well as a list of them."""
        return coerce_to_item_list(value)

    @classmethod
    def from_probabilities(  # type: ignore[override]  # chains are named, not a single matrix
        cls,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        *,
        heavy_chain: Sequence[Sequence[float]] | None = None,
        light_chain: Sequence[Sequence[float]] | None = None,
        temperature: float = 1.0,
        **fields: Any,
    ) -> "AbLangGradientInput":
        """Build an input whose chain logits reproduce the given distributions.

        An antibody's chains are named rather than stacked into one matrix, so a single candidate
        is given by keyword and several are given as mappings of the same keywords. Every chain of
        every candidate is converted at the one shared temperature the input stores, so no two can
        disagree. At least one chain is required per candidate.

        Args:
            candidates (Sequence[Mapping[str, Any]] | None): One mapping per candidate, each with
                ``heavy_chain`` and/or ``light_chain`` probabilities. ``None`` takes the single
                candidate named by keyword instead.
            heavy_chain (Sequence[Sequence[float]] | None): ``(Lh, 20)`` strictly positive rows.
            light_chain (Sequence[Sequence[float]] | None): ``(Ll, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature stored on the input.
            fields (Any): Any remaining input fields.

        Returns:
            AbLangGradientInput: Input satisfying ``softmax(chain / temperature) == chain probs``.

        Raises:
            ValueError: If the temperature is not positive and finite, if ``candidates`` is empty,
                or if it is combined with a chain given by keyword.
        """
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
        if candidates is None:
            specs: list[Mapping[str, Any]] = [{"heavy_chain": heavy_chain, "light_chain": light_chain}]
        elif heavy_chain is not None or light_chain is not None:
            raise ValueError("pass chains either as candidates or by keyword, not both")
        elif not candidates:
            raise ValueError("candidates must contain at least one antibody")
        else:
            specs = list(candidates)

        convert = ProteinSequenceLogits.from_probabilities
        antibodies = [
            AntibodyLogits(
                heavy_chain=(
                    None if spec.get("heavy_chain") is None else convert(spec["heavy_chain"], temperature=temperature)
                ),
                light_chain=(
                    None if spec.get("light_chain") is None else convert(spec["light_chain"], temperature=temperature)
                ),
            )
            for spec in specs
        ]
        return cls(inputs=antibodies, temperature=temperature, **fields)


class AbLangMaskedPLLTerm(DifferentiableTerm):
    """AbLang's one differentiable quantity: the masked pseudo-log-likelihood.

    The tool exposes a single scalar, so the term carries no selectors beyond the identifier
    naming what it is.

    Attributes:
        id (Literal["masked_pll"]): Fixed identifier for this kind of quantity.
    """

    #: A mean negative log-likelihood is better the smaller it is.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"

    #: Nats, averaged over the amino-acid positions the mean is taken across.
    UNITS: ClassVar[str] = "nats per position"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Mean masked negative log-likelihood over amino-acid positions"

    id: Literal["masked_pll"] = Field(
        default="masked_pll",
        title="Term ID",
        description="Identifies this quantity as AbLang's masked pseudo-log-likelihood.",
    )


class AbLangGradientOutput(GradientOutput):
    """AbLang masked PLL output. Gradients are omitted in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input antibody. Contains a single masked-PLL
            term carrying that item's mean negative log-likelihood over AA positions and its raw
            gradient w.r.t. the supplied logits (``None`` in forward-only mode), plus
            ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` and sequence length in
            ``metrics``.
        vocab (list[str]): Amino-acid column ordering for the input logits.
    """


class AbLangGradientConfig(RelaxationConfig):
    """Configuration for the AbLang masked PLL gradient tool.

    Attributes:
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
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
)
def run_ablang_gradient(
    inputs: AbLangGradientInput,
    config: AbLangGradientConfig,
    instance: Any = None,
) -> AbLangGradientOutput:
    """Compute AbLang masked PLL gradient with respect to relaxed antibody logits."""
    items: list[dict[str, Any]] = []
    for antibody in inputs.inputs:
        # Each antibody names its checkpoint; a paired one concatenates heavy-first at chain_break_position.
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
    # Run inference in the toolkit's isolated environment.
    result = ToolInstance.dispatch(
        "ablang",
        payload,
        instance=instance,
        config=config,
    )
    attach_intrinsic_term(
        result,
        AbLangMaskedPLLTerm(),
        matrices=[item["logits"] for item in items],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
    return AbLangGradientOutput(**result)
