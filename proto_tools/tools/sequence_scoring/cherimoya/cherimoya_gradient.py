"""Differentiable Cherimoya accessibility objectives for relaxed DNA logits."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from proto_tools.tools.sequence_scoring.cherimoya.catalog import (
    CHERIMOYA_CONTEXT,
    CHERIMOYA_OUTPUT,
    CHERIMOYA_OUTPUT_FLANK,
    DEFAULT_CHERIMOYA_MODEL,
    CherimoyaCollectionKey,
    CherimoyaFold,
    is_checkpoint_path,
    validate_model_reference,
)
from proto_tools.tools.sequence_scoring.cherimoya.shared_data_models import (
    MAX_CHERIMOYA_MODELS,
    CherimoyaModelInfo,
    checkpoint_specs,
    model_infos,
    validate_folds,
)
from proto_tools.tools.sequence_scoring.shared_data_models import validate_dna_sequence
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    DNA_NUCLEOTIDES,
    BatchedNucleotideGradientInput,
    ConfigField,
    GradientOutput,
    GradientResult,
    NucleotideSequenceLogits,
    RelaxationConfig,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Metrics, MetricSpec

CherimoyaQuantity = Literal["log_counts", "log_target_counts"]
CherimoyaObjectiveDirection = Literal["max", "min"]


class CherimoyaGradientLossTerm(BaseModel):
    """One differentiable Cherimoya objective term.

    Both quantities are natural-log read counts, which depend on each experiment's
    sequencing depth, so set ``sigmoid_center`` per model and quantity. The default 4.5 is
    the median fold 0 ``log_counts`` of the default model (ENCSR000EOT, K562 DNase-seq) over
    5,000 sampled hg38 SCREEN cCREs; a target span shorter than 1,000 bp holds fewer reads.

    Attributes:
        model (str): ENCODE experiment accession or local ``.torch`` checkpoint path.
        folds (list[CherimoyaFold]): Folds averaged for this term (ignored for a checkpoint path).
        quantity (CherimoyaQuantity): ``"log_counts"``, ln(reads + 1) over the 1,000 bp output
            window, or ``"log_target_counts"``, ln(expected reads + 1) in the config's target span.
        direction (CherimoyaObjectiveDirection): ``"max"`` minimizes
            ``1 - sigmoid((value - center) / scale)``; ``"min"`` minimizes
            ``sigmoid((value - center) / scale)``.
        weight (float): Non-negative scalar applied before terms are summed.
        sigmoid_center (float): Value of ``quantity`` where the sigmoid is 0.5.
        sigmoid_scale (float): Positive scale of the sigmoid in ``quantity`` units.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default=DEFAULT_CHERIMOYA_MODEL, title="Model", description="ENCODE experiment accession or .torch path"
    )
    folds: list[CherimoyaFold] = Field(
        default=[0], min_length=1, title="Folds", description="Folds averaged for this term"
    )
    quantity: CherimoyaQuantity = Field(
        default="log_counts",
        title="Quantity",
        description="ln(reads + 1) over the output window (log_counts) or the target span (log_target_counts)",
    )
    direction: CherimoyaObjectiveDirection = Field(
        default="max", title="Direction", description="Whether to maximize ('max') or minimize ('min') the quantity"
    )
    weight: float = Field(default=1.0, ge=0.0, title="Weight", description="Relative weight of this term (>= 0)")
    sigmoid_center: float = Field(
        default=4.5, title="Sigmoid Center", description="Quantity value at the sigmoid midpoint; set per model"
    )
    sigmoid_scale: float = Field(
        default=1.0, gt=0.0, title="Sigmoid Scale", description="Sigmoid steepness in quantity units"
    )


class CherimoyaGradientInput(BatchedNucleotideGradientInput):
    """Input for differentiable Cherimoya sequence optimization.

    Attributes:
        inputs (list[NucleotideSequenceLogits]): One relaxed ``(L, 4)`` state per candidate in
            ``A,C,G,T`` order, for the designed segment only. Every item must share one length,
            because the backend evaluates them in one vectorized pass.
        temperature (float | None): Softmax temperature shared by every item. The default ``1.0``
            reads each state as unnormalized and applies ``softmax(logits / T)``, returning
            logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """


class CherimoyaGradientConfig(RelaxationConfig):
    """Configuration for differentiable Cherimoya objectives.

    The model input is ``context_left + designed segment + context_right``, which must be
    2,114 bp. Only the designed segment is differentiated.

    Attributes:
        collection (CherimoyaCollectionKey): Collection the accessions in ``loss_terms`` come from.
        loss_terms (list[CherimoyaGradientLossTerm]): Objective terms summed into one loss.
        context_left (str): Fixed DNA placed before the designed segment.
        context_right (str): Fixed DNA placed after the designed segment.
        target_start (int | None): Start of the ``log_target_counts`` span, relative to the
            designed segment; ``None`` means its start.
        target_end (int | None): End of the ``log_target_counts`` span, relative to the
            designed segment; ``None`` means its end.
        reverse_complement_average (bool): Average forward and reverse-complement predictions.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient_models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run the backward pass and return the gradient.
        deterministic (bool): Use plain PyTorch kernels so repeated gradients are bit-identical.
        device (str): Device used for inference and backpropagation.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run Cherimoya on",
        include_in_key=False,
    )
    collection: CherimoyaCollectionKey = ConfigField(
        title="Collection",
        default="catv1",
        description="Model collection the accessions in loss_terms come from.",
    )
    loss_terms: list[CherimoyaGradientLossTerm] = ConfigField(
        default_factory=lambda: [CherimoyaGradientLossTerm()],
        min_length=1,
        max_length=MAX_CHERIMOYA_MODELS,
        title="Loss Terms",
        description="Per-model Cherimoya objectives summed before backpropagation.",
    )
    context_left: str = ConfigField(
        title="Left Context",
        default="",
        description="Fixed DNA before the designed segment; with it, the input must total 2,114 bp.",
    )
    context_right: str = ConfigField(
        title="Right Context",
        default="",
        description="Fixed DNA after the designed segment; with it, the input must total 2,114 bp.",
    )
    target_start: int | None = ConfigField(
        title="Target Start",
        default=None,
        ge=0,
        description="log_target_counts span start, relative to the designed segment; None is its start.",
    )
    target_end: int | None = ConfigField(
        title="Target End",
        default=None,
        ge=1,
        description="log_target_counts span end, relative to the designed segment; None is its end.",
    )
    reverse_complement_average: bool = ConfigField(
        title="Reverse-Complement Average",
        default=False,
        description="Average forward and reverse-complement predictions (twice the compute).",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
    )
    deterministic: bool = ConfigField(
        title="Deterministic",
        default=False,
        description="Plain PyTorch kernels instead of Triton so repeated GPU gradients are bit-identical.",
    )

    @field_validator("context_left", "context_right")
    @classmethod
    def validate_context(cls, value: str) -> str:
        """Validate and uppercase fixed context DNA."""
        return validate_dna_sequence(value) if value else value

    @field_validator("loss_terms")
    @classmethod
    def resolve_term_models(
        cls, terms: list[CherimoyaGradientLossTerm], info: ValidationInfo
    ) -> list[CherimoyaGradientLossTerm]:
        """Resolve each term's model and folds against the collection."""
        collection = info.data.get("collection", "catv1")
        resolved = []
        for term in terms:
            validate_folds(list(term.folds), collection)
            resolved.append(term.model_copy(update={"model": validate_model_reference(term.model, collection)}))
        return resolved

    @model_validator(mode="after")
    def validate_target(self) -> "CherimoyaGradientConfig":
        """Require target_end after target_start, and a term that uses the span when either is set."""
        if self.target_start is not None and self.target_end is not None and self.target_end <= self.target_start:
            raise ValueError("target_end must be greater than target_start")
        targeted = any(term.quantity == "log_target_counts" for term in self.loss_terms)
        if (self.target_start is not None or self.target_end is not None) and not targeted:
            raise ValueError("target_start/target_end are only used by loss terms with quantity='log_target_counts'")
        return self

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Checkpoint paths name local files that a remote worker does not have."""
        paths = sorted({term.model for term in self.loss_terms if is_checkpoint_path(term.model)})
        if paths:
            return f"loss_terms models {paths} are local checkpoint files not available on device='{device}'."
        return None


class CherimoyaGradientSampleMetrics(Metrics):
    """Per-sample differentiable Cherimoya objective metrics.

    Metrics documented in ``metric_spec``:
        loss (float): Weighted scalar objective value for one sample.

    Attributes:
        loss_terms (list[dict[str, Any]]): Per-term model, quantity value, sigmoid
            transform, and weighted score for this sample.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "loss": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "description": "Weighted scalar Cherimoya objective value for one sample.",
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="loss",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )
    loss_terms: list[dict[str, Any]] = Field(
        default_factory=list,
        title="Loss Terms",
        description="Per-objective-term values and transform metadata for this sample.",
    )


class CherimoyaGradientOutput(GradientOutput):
    """Differentiable Cherimoya objective output.

    Attributes:
        results (list[GradientResult]): One per input item, in input order, each holding that
            sample's own gradient, weighted sigmoid objective, and per-term values.
        vocab (list[str]): DNA column ordering for logits and gradient.
        models (list[CherimoyaModelInfo]): Provenance of each model the terms used.
    """

    models: list[CherimoyaModelInfo] = Field(
        default_factory=list, title="Models", description="Provenance of each model the loss terms used"
    )

    @classmethod
    def from_batch(cls, payload: Mapping[str, Any], models: list[CherimoyaModelInfo]) -> "CherimoyaGradientOutput":
        """Unbatch one worker payload into per-sample results.

        The worker evaluates the batch in one vectorized pass and reports a stacked ``(B, L, 4)``
        gradient with per-sample losses and term values, so this splits them back into one result
        per designed segment.

        Args:
            payload (Mapping[str, Any]): Worker payload for the whole batch.
            models (list[CherimoyaModelInfo]): Provenance of each model the loss terms used.

        Returns:
            CherimoyaGradientOutput: One result per sample, in input order.
        """
        metrics = payload["metrics"]
        gradients = payload["gradient"] or [None] * len(metrics["losses"])
        results = []
        for gradient, loss, term_rows in zip(gradients, metrics["losses"], metrics["loss_terms"], strict=True):
            sample = CherimoyaGradientSampleMetrics.model_validate({"loss": float(loss), "loss_terms": term_rows})
            results.append(GradientResult(gradient=gradient, loss=float(loss), metrics=sample.model_dump()))
        return cls(results=results, vocab=payload["vocab"], models=models)

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient bundle, including model provenance, as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        payload = self.model_dump(include={"results", "vocab", "models"})
        json_path.write_text(json.dumps(payload, indent=2))


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CherimoyaGradientInput(inputs=[NucleotideSequenceLogits([[0.0] * len(DNA_NUCLEOTIDES)] * CHERIMOYA_CONTEXT)])


def _target_span(config: CherimoyaGradientConfig, length: int) -> tuple[int, int] | None:
    """Return the target span in output-window coordinates, or None when no term needs it."""
    if all(term.quantity != "log_target_counts" for term in config.loss_terms):
        return None
    start = len(config.context_left) + (config.target_start or 0) - CHERIMOYA_OUTPUT_FLANK
    end = len(config.context_left) + (config.target_end or length) - CHERIMOYA_OUTPUT_FLANK
    if config.target_end is not None and config.target_end > length:
        raise ValueError(f"target_end={config.target_end} is past the designed segment of length {length}")
    if start < 0 or end > CHERIMOYA_OUTPUT or start >= end:
        raise ValueError(
            "the log_target_counts span must lie inside the central 1,000 bp output window "
            f"(window positions {CHERIMOYA_OUTPUT_FLANK}-{CHERIMOYA_OUTPUT_FLANK + CHERIMOYA_OUTPUT}); "
            f"got {start + CHERIMOYA_OUTPUT_FLANK}-{end + CHERIMOYA_OUTPUT_FLANK}"
        )
    return start, end


@tool(
    key="cherimoya-gradient",
    label="Cherimoya Gradient",
    category="sequence_scoring",
    input_class=CherimoyaGradientInput,
    config_class=CherimoyaGradientConfig,
    output_class=CherimoyaGradientOutput,
    metrics_class=CherimoyaGradientSampleMetrics,
    description="Compute differentiable Cherimoya accessibility losses and gradients for relaxed DNA logits",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_cherimoya_gradient(
    inputs: CherimoyaGradientInput,
    config: CherimoyaGradientConfig,
    instance: Any = None,
) -> CherimoyaGradientOutput:
    """Compute Cherimoya objective gradients with respect to relaxed DNA logits.

    Args:
        inputs (CherimoyaGradientInput): Batched relaxed logits of the designed segment.
        config (CherimoyaGradientConfig): Loss terms, fixed context, and relaxation settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CherimoyaGradientOutput: Per-sample loss, metrics, and the gradient with respect to
            each designed segment.
    """
    length = len(inputs.inputs[0])
    total = len(config.context_left) + length + len(config.context_right)
    if total != CHERIMOYA_CONTEXT:
        raise ValueError(
            f"context_left ({len(config.context_left)}) + inputs ({length}) + context_right "
            f"({len(config.context_right)}) is {total} bp; Cherimoya needs {CHERIMOYA_CONTEXT}"
        )
    target_span = _target_span(config, length)

    # Each checkpoint is loaded once even when several terms share a model.
    models = list(dict.fromkeys(term.model for term in config.loss_terms))
    specs: list[dict[str, Any]] = []
    for model in models:
        folds = list(dict.fromkeys(fold for term in config.loss_terms if term.model == model for fold in term.folds))
        specs.extend(checkpoint_specs([model], folds, config.collection))
    index = {(spec["model"], spec["fold"]): i for i, spec in enumerate(specs)}
    terms = [
        {
            **term.model_dump(),
            "folds": [] if is_checkpoint_path(term.model) else list(term.folds),
            "checkpoints": [
                index[(spec["model"], spec["fold"])]
                for spec in checkpoint_specs([term.model], list(term.folds), config.collection)
            ],
        }
        for term in config.loss_terms
    ]

    result = ToolInstance.dispatch(
        "cherimoya",
        {
            "operation": "gradient",
            "logits": [item.root for item in inputs.inputs],
            "temperature": inputs.temperature,
            "checkpoints": specs,
            "loss_terms": terms,
            "context_left": config.context_left,
            "context_right": config.context_right,
            "target_span": target_span,
            "reverse_complement_average": config.reverse_complement_average,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "deterministic": config.deterministic,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    return CherimoyaGradientOutput.from_batch(result, model_infos(models, config.collection, specs, result["applied"]))
