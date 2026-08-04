"""Differentiable PARADE UTR activity objectives for gradient-based design."""

import json
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    _CUSTOM_CHECKPOINT_REJECTED,
    PARADE_CELL_TYPES,
    ParadeCellType,
    ParadeConstructType,
    _SafeCheckpointUrlConfigMixin,
    normalize_checkpoint_override,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    DNA_NUCLEOTIDES,
    BaseConfig,
    BaseToolInput,
    ConfigField,
    GradientOutput,
    InputField,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Metrics, MetricSpec

ParadeObjectiveDirection = Literal["max", "min"]
ParadeLogits = list[list[list[float]]]
ParadeGradientValue = list[list[list[float]]] | None


class ParadeGradientLossTerm(BaseModel):
    """One differentiable PARADE UTR-activity objective term.

    Attributes:
        cell_type (ParadeCellType): PARADE cell code to optimize; must be in the
            configured ``construct_type`` panel.
        direction (ParadeObjectiveDirection): ``"max"`` minimizes
            ``1 - sigmoid((raw - center) / scale)``; ``"min"`` minimizes
            ``sigmoid((raw - center) / scale)``.
        weight (float): Non-negative scalar applied before terms are summed.
        sigmoid_center (float): Raw activity value where the sigmoid is 0.5.
        sigmoid_scale (float): Positive scale for the raw activity transform.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    cell_type: ParadeCellType = Field(default="c2", title="Cell Type", description="PARADE cell code to optimize.")
    direction: ParadeObjectiveDirection = Field(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') activity for this cell code",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Weight",
        description="Relative weight of this term in the summed objective (must be non-negative)",
    )
    sigmoid_center: float = Field(
        default=2.0,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="Raw PARADE activity at the sigmoid midpoint (0.5 output)",
    )
    sigmoid_scale: float = Field(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Sigmoid Scale",
        description="Sigmoid steepness in raw-activity units; larger values produce a wider transition",
    )


class ParadeGradientInput(BaseToolInput):
    """Input for differentiable PARADE UTR optimization.

    Attributes:
        logits (ParadeLogits): Batched relaxed UTR logits with shape ``(B, L, 4)`` in
            ``A,C,G,T`` order. Use ``B=1`` for a single design candidate.
        temperature (float): Softmax temperature used to relax logits.
    """

    logits: ParadeLogits = InputField(
        title="Logits",
        description="Batched relaxed UTR logits with shape (B, L, 4) in A,C,G,T order.",
        examples=[[[[0.0] * 4] * 50]],
    )
    temperature: float = InputField(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Temperature",
        description="Softmax temperature used to convert logits into relaxed nucleotide probabilities.",
    )

    @field_validator("logits")
    @classmethod
    def validate_logits(cls, logits: ParadeLogits) -> ParadeLogits:
        """Ensure logits are a non-empty ``B x L x 4`` tensor."""
        if not logits:
            raise ValueError("logits must contain at least one batch item")
        expected_width = len(DNA_NUCLEOTIDES)
        seq_length: int | None = None
        for batch_idx, matrix in enumerate(logits):
            if not matrix:
                raise ValueError(f"logits batch {batch_idx} must contain at least one position")
            if not isinstance(matrix[0], list):
                raise ValueError("logits must have shape (B, L, 4); use batch size 1 for a single candidate")
            if seq_length is None:
                seq_length = len(matrix)
            elif len(matrix) != seq_length:
                raise ValueError(
                    f"all batched logits must have the same sequence length; "
                    f"batch 0 has {seq_length}, batch {batch_idx} has {len(matrix)}"
                )
            for row_idx, row in enumerate(matrix):
                if len(row) != expected_width:
                    raise ValueError(
                        f"logits batch {batch_idx} row {row_idx} must have {expected_width} columns, got {len(row)}"
                    )
                if not all(math.isfinite(value) for value in row):
                    raise ValueError(f"logits batch {batch_idx} row {row_idx} must contain only finite numbers")
        return logits


class ParadeGradientConfig(_SafeCheckpointUrlConfigMixin, BaseConfig):
    """Configuration for differentiable PARADE UTR-activity objectives.

    Attributes:
        device (str): Device used for inference and backpropagation.
        construct_type (ParadeConstructType): UTR model to use — ``"utr5"`` or ``"utr3"``.
        loss_terms (list[ParadeGradientLossTerm]): Per-cell objective terms summed
            into one scalar loss.
        checkpoint (str): Your own checkpoint to use instead of the pinned one, given as a local
            ``.ckpt`` path or an ``https`` download link (a schemeless ``host.tld/path`` is
            accepted). Works on ``device="cpu"``, ``device="cuda"``, and ``device="modal"``.
            Rejected on ``device="proto"``. Leave empty to use the pinned per-target checkpoint.
        soft (float): Blend hard argmax one-hot (0) to softmax probabilities (1).
        hard (float): Straight-through hard-forward coefficient.
        compute_gradient (bool): Run backward pass and return gradient.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run PARADE inference on.",
        include_in_key=False,
    )
    construct_type: ParadeConstructType = ConfigField(
        title="Construct Type",
        default="utr5",
        description="UTR model to use: 'utr5' (5' UTR) or 'utr3' (3' UTR).",
        reload_on_change=True,
    )
    loss_terms: list[ParadeGradientLossTerm] = ConfigField(
        default_factory=lambda: [ParadeGradientLossTerm()],
        title="Loss Terms",
        description="Per-cell PARADE activity objectives to sum before backpropagation.",
    )
    checkpoint: str = ConfigField(
        title="Checkpoint",
        default="",
        description="Your own .ckpt path or https link, used instead of the pinned checkpoint.",
        reload_on_change=True,
        repr=False,
    )
    soft: float = ConfigField(
        title="Soft Mixing",
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend hard argmax one-hot (0) to softmax probabilities (1).",
    )
    hard: float = ConfigField(
        title="Hard Mixing",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Straight-through hard-forward coefficient.",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
    )

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: str) -> str:
        """Validate a checkpoint override: an https link (schemeless allowed) or a local path."""
        return normalize_checkpoint_override(value)

    @field_validator("construct_type")
    @classmethod
    def validate_construct_type(cls, value: ParadeConstructType, info: ValidationInfo) -> ParadeConstructType:
        """Reject a construct update that would invalidate the current loss terms."""
        loss_terms = info.data.get("loss_terms", [])
        offpanel = sorted({term.cell_type for term in loss_terms} - set(PARADE_CELL_TYPES[value]))
        if offpanel:
            raise ValueError(
                f"loss_terms cell codes {offpanel} are not in the {value} panel {list(PARADE_CELL_TYPES[value])}"
            )
        return value

    @field_validator("loss_terms")
    @classmethod
    def validate_loss_terms(
        cls, value: list[ParadeGradientLossTerm], info: ValidationInfo
    ) -> list[ParadeGradientLossTerm]:
        """Require at least one loss term whose cell code is in the construct panel."""
        if not value:
            raise ValueError("loss_terms cannot be empty")
        construct_type = info.data.get("construct_type", "utr5")
        panel = PARADE_CELL_TYPES[construct_type]
        offpanel = sorted({term.cell_type for term in value} - set(panel))
        if offpanel:
            raise ValueError(f"loss_terms cell codes {offpanel} are not in the {construct_type} panel {list(panel)}")
        return value

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Proto will not load a custom checkpoint; a deployment the caller owns may."""
        if device == "proto" and self.checkpoint:
            return _CUSTOM_CHECKPOINT_REJECTED
        return None


class ParadeGradientSampleMetrics(Metrics):
    """Per-sample differentiable PARADE objective metrics.

    Raw per-cell activity predictions are stored as metrics when the corresponding cell
    code appears in the objective. Per-term transformation metadata is kept in
    ``loss_terms`` so metric values remain scalar and easy to index.

    Metrics documented in ``metric_spec``:
        loss (float): Weighted scalar objective value for one sample.
        c1, c2, c4, c6, c17, c13 (float): Raw PARADE activity for that cell code.

    Attributes:
        loss_terms (list[dict[str, Any]]): Per-objective-term metadata, including
            direction, weight, sigmoid transform, and weighted score.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "loss": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "description": "Weighted scalar PARADE objective value for one sample.",
            "better_values_are": "lower",
        },
        **{
            code: {
                "availability": "when requested",
                "type": "float",
                "min": None,
                "max": None,
                "description": f"Raw PARADE UTR activity for cell code {code}.",
                "better_values_are": "context-dependent",
            }
            for code in ("c1", "c2", "c4", "c6", "c17", "c13")
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
        description="Per-objective-term transform metadata for this sample.",
    )


def _build_gradient_sample_metrics(metrics: dict[str, Any]) -> list[ParadeGradientSampleMetrics]:
    """Build per-sample metric containers from the standalone gradient payload."""
    raw_scores_by_sample = metrics.get("raw_scores")
    losses = metrics.get("losses")
    loss_terms_by_sample = metrics.get("loss_terms")
    if not isinstance(raw_scores_by_sample, list) or not isinstance(losses, list):
        return []

    sample_metrics = []
    for sample_idx, loss in enumerate(losses):
        raw_scores = raw_scores_by_sample[sample_idx] if sample_idx < len(raw_scores_by_sample) else {}
        loss_terms = (
            loss_terms_by_sample[sample_idx]
            if isinstance(loss_terms_by_sample, list) and sample_idx < len(loss_terms_by_sample)
            else []
        )
        metric_values: dict[str, float] = {"loss": float(loss)}
        if isinstance(raw_scores, dict):
            metric_values.update(
                {code: float(raw_scores[code]) for code in ("c1", "c2", "c4", "c6", "c17", "c13") if code in raw_scores}
            )
        sample_metrics.append(ParadeGradientSampleMetrics.model_validate({**metric_values, "loss_terms": loss_terms}))
    return sample_metrics


class ParadeGradientOutput(GradientOutput):
    """Differentiable PARADE UTR-activity output.

    Attributes:
        gradient (ParadeGradientValue): Gradient tensor matching input UTR logits, or
            ``None`` when ``compute_gradient=False``.
        loss (float): Sum of per-sample weighted scalar objective values. Per-sample
            values are available in ``sample_metrics``.
        sample_metrics (list[ParadeGradientSampleMetrics]): Per-sample metric containers
            with scalar loss and raw per-cell activity.
        metrics (dict[str, Any]): Auxiliary metadata bundle from the standalone worker,
            including raw scores, objective-term metadata, and relaxation parameters.
        vocab (list[str]): DNA column ordering for logits and gradient.
    """

    loss: float = Field(
        title="Loss",
        description="Sum of per-sample weighted PARADE sigmoid objective values.",
    )
    gradient: ParadeGradientValue = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )
    sample_metrics: list[ParadeGradientSampleMetrics] = Field(
        default_factory=list,
        title="Sample Metrics",
        description="Per-sample PARADE objective metrics.",
    )

    @model_validator(mode="after")
    def populate_sample_metrics(self) -> "ParadeGradientOutput":
        """Populate metric containers from the standalone metrics payload."""
        if not self.sample_metrics:
            object.__setattr__(self, "sample_metrics", _build_gradient_sample_metrics(self.metrics))
        return self

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient bundle as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        payload = self.model_dump(include={"gradient", "loss", "sample_metrics", "metrics", "vocab"})
        json_path.write_text(json.dumps(payload, indent=2))


def example_gradient_input() -> Any:
    """Minimal valid input for differentiable PARADE examples."""
    return ParadeGradientInput(logits=[[[0.0] * len(DNA_NUCLEOTIDES)] * 50])


@tool(
    key="parade-gradient",
    label="PARADE UTR Activity Gradient",
    category="sequence_scoring",
    input_class=ParadeGradientInput,
    config_class=ParadeGradientConfig,
    output_class=ParadeGradientOutput,
    metrics_class=ParadeGradientSampleMetrics,
    description="Compute differentiable PARADE UTR-activity losses and gradients for relaxed UTR logits",
    uses_gpu=True,
    example_input=example_gradient_input,
    cacheable=False,
)
def run_parade_gradient(
    inputs: ParadeGradientInput,
    config: ParadeGradientConfig,
    instance: Any = None,
) -> ParadeGradientOutput:
    """Compute PARADE UTR-activity gradients with respect to relaxed UTR logits.

    Args:
        inputs (ParadeGradientInput): Batched relaxed UTR logits and temperature.
        config (ParadeGradientConfig): PARADE differentiable objective configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ParadeGradientOutput: Loss, optional gradient, and per-sample activity metrics.
    """
    inputs = ParadeGradientInput.model_validate(inputs.model_dump())
    # Revalidate a dumped copy so nested/container mutation cannot reach the worker
    # without re-running the cross-field and finite-number checks.
    config = ParadeGradientConfig.model_validate(config.model_dump())
    # The objective runs through the same per-construct activity checkpoint as parade-activity.
    checkpoint_path, url, md5, filename = resolve_checkpoint_source(config.construct_type, config.checkpoint)

    result = ToolInstance.dispatch(
        "parade",
        {
            "operation": "gradient",
            "logits": inputs.logits,
            "temperature": inputs.temperature,
            "construct_type": config.construct_type,
            "loss_terms": [term.model_dump() for term in config.loss_terms],
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
    return ParadeGradientOutput(**result)
