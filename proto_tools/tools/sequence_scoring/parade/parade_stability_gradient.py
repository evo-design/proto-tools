"""PARADE 3' UTR stability gradient tool."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from pydantic import Field

from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    ParadeCheckpointConfig,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BatchedNucleotideGradientInput,
    ConfigField,
    DifferentiableTerm,
    GradientMetadata,
    GradientOutput,
    GradientResult,
    NucleotideSequenceLogits,
    RelaxationConfig,
    TermGradient,
    ToolInstance,
)
from proto_tools.utils.tool_io import Directionality, Metrics, MetricSpec

logger = logging.getLogger(__name__)


class ParadeStabilityGradientInput(BatchedNucleotideGradientInput):
    """Input for the PARADE stability gradient tool: one or many relaxed 3' UTR states.

    Attributes:
        inputs (list[NucleotideSequenceLogits]): One relaxed ``(L, 4)`` DNA state per candidate in
            ``A,C,G,T`` order. The backend evaluates the batch in one vectorized pass, so every
            item must share one length.
        temperature (float | None): Softmax temperature, shared by every candidate. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """


class ParadeStabilityLogRatioTerm(DifferentiableTerm):
    """PARADE stability's one differentiable quantity: the predicted RNA/gDNA log-ratio.

    The tool exposes a single scalar, so the term carries no selectors beyond the identifier
    naming what it is.

    Attributes:
        id (Literal["log_ratio"]): Fixed identifier for this kind of quantity.
    """

    #: A higher log-ratio is a more stable transcript.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Predicted RNA/gDNA log-ratio; higher means a more stable transcript"

    id: Literal["log_ratio"] = Field(
        default="log_ratio",
        title="Term ID",
        description="Identifies this quantity as PARADE's predicted RNA/gDNA log-ratio.",
    )


class ParadeStabilityGradientMetrics(Metrics):
    """Per-candidate differentiable PARADE stability metrics.

    The predicted log-ratio is the differentiated scalar, so it is reported as a term value rather
    than as a metric and carries its direction there.

    Metrics documented in ``metric_spec``:
        sequence_length (int): Length of the scored UTR.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "sequence_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Length of the scored 3' UTR sequence.",
            "better_values_are": "context-dependent",
        },
    }

    primary_metric: str | None = Field(
        default=None,
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class ParadeStabilityGradientOutput(GradientOutput):
    """Differentiable PARADE stability output.

    Attributes:
        results (list[GradientResult]): One per input UTR, in input order. Contains a single
            log-ratio term carrying that candidate's predicted RNA/gDNA log-ratio and its raw
            gradient (``None`` in forward-only mode), plus the scored length in ``metrics``.
        vocab (list[str]): DNA column ordering for logits and gradient.
    """

    @classmethod
    def from_batch(
        cls,
        payload: Mapping[str, Any],
        *,
        matrices: Sequence[Sequence[Sequence[float]]],
        temperature: float | None,
        soft: float = 1.0,
        hard: float = 0.0,
    ) -> "ParadeStabilityGradientOutput":
        """Unbatch one worker payload into per-candidate results.

        The worker evaluates the batch in one vectorized pass and reports a stacked ``(B, L, 4)``
        gradient alongside per-sample log-ratios, so this splits them back into one result per UTR.

        Args:
            payload (Mapping[str, Any]): Worker reply with log ratios, gradients, and lengths.
            matrices (Sequence[Sequence[Sequence[float]]]): The relaxed UTR each candidate was
                differentiated at, in input order.
            temperature (float | None): The input's temperature, which fixes the coordinate space.
            soft (float): Relaxation blend in force.
            hard (float): Straight-through coefficient in force.

        Returns:
            ParadeStabilityGradientOutput: One result per input candidate, in input order.
        """
        log_ratios = list(payload["log_ratios"])
        lengths = list(payload["lengths"])
        gradients = payload.get("gradients") or [None] * len(log_ratios)
        vocab = list(NucleotideSequenceLogits.VOCAB)
        results = [
            GradientResult(
                terms=[
                    TermGradient(
                        term=ParadeStabilityLogRatioTerm(),
                        term_value=float(log_ratios[index]),
                        gradient=gradients[index],
                        metadata=GradientMetadata.from_input(
                            matrices[index],
                            vocab=vocab,
                            temperature=temperature,
                            soft=soft,
                            hard=hard,
                        ),
                    )
                ],
                metrics={"sequence_length": int(lengths[index])},
            )
            for index in range(len(log_ratios))
        ]
        return cls(results=results, vocab=vocab)


class ParadeStabilityGradientConfig(RelaxationConfig, ParadeCheckpointConfig):
    """Configuration for the PARADE 3' UTR stability gradient tool.

    Attributes:
        checkpoint (str): Inherited; your own ``.ckpt`` path or https link, used instead of the
            pinned stability checkpoint.
        batch_size (int): Inherited; sequences per GPU forward pass.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


def example_input() -> ParadeStabilityGradientInput:
    """Minimal valid input for testing and examples."""
    return ParadeStabilityGradientInput(inputs=[NucleotideSequenceLogits([[0.25] * 4] * 50)])


@tool(
    key="parade-stability-gradient",
    label="PARADE Stability Gradient",
    category="sequence_scoring",
    input_class=ParadeStabilityGradientInput,
    config_class=ParadeStabilityGradientConfig,
    output_class=ParadeStabilityGradientOutput,
    description="Compute PARADE 3' UTR stability gradients for relaxed DNA sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=32,
    metrics_class=ParadeStabilityGradientMetrics,
)
def run_parade_stability_gradient(
    inputs: ParadeStabilityGradientInput,
    config: ParadeStabilityGradientConfig,
    instance: Any = None,
) -> ParadeStabilityGradientOutput:
    """Compute PARADE stability gradients with respect to relaxed 3' UTR logits."""
    logger.debug("Using local worker for PARADE stability gradient (%d candidate(s))", len(inputs.inputs))
    # The objective runs through the same stability checkpoint as parade-stability.
    checkpoint_path, url, md5, filename = resolve_checkpoint_source("stability", config.checkpoint)

    result = ToolInstance.dispatch(
        "parade",
        {
            "operation": "stability_gradient",
            "logits": [item.root for item in inputs.inputs],
            "temperature": inputs.temperature,
            "checkpoint_path": checkpoint_path,
            "checkpoint_url": url,
            "checkpoint_md5": md5,
            "checkpoint_filename": filename,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )
    return ParadeStabilityGradientOutput.from_batch(
        result,
        matrices=[item.root for item in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
