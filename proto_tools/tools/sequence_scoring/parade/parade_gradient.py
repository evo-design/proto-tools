"""Differentiable PARADE UTR activity objectives for gradient-based design."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    PARADE_CELL_PANEL,
    PARADE_CELL_TYPES,
    ParadeCellType,
    ParadeConstructType,
    _SafeCheckpointUrlConfigMixin,
    local_checkpoint_on_remote,
    normalize_checkpoint_override,
    resolve_cell_type,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    DNA_NUCLEOTIDES,
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
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Directionality


class ParadeActivityTerm(DifferentiableTerm):
    """One differentiable PARADE UTR-activity quantity: a single cell's predicted activity.

    Attributes:
        id (Literal["activity"]): Fixed identifier for this kind of quantity. Every cell's term
            reports the same one; the ``cell_type`` selector is what tells two of them apart.
        cell_type (ParadeCellType): PARADE cell whose activity is differentiated, given by code
            (``"c2"``) or cell-line name (``"HepG2"``, case-insensitive); resolved to its canonical
            code, which must be in the configured ``construct_type`` panel.
    """

    #: Whether more activity is wanted depends on whether the cell is a target or an off-target.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Predicted PARADE UTR activity for one cell line"

    id: Literal["activity"] = Field(
        default="activity",
        title="Term ID",
        description="Identifies this quantity as a PARADE UTR activity prediction.",
    )
    cell_type: ParadeCellType = Field(
        default="c2",
        title="Cell Type",
        description=f"Cell to differentiate: {PARADE_CELL_PANEL}",
    )

    @field_validator("cell_type", mode="before")
    @classmethod
    def _resolve_cell_type(cls, value: Any) -> Any:
        """Accept a cell code ('c2') or cell-line name ('HepG2') and resolve it to its code."""
        if isinstance(value, str):
            return resolve_cell_type(value)
        return value


class ParadeGradientInput(BatchedNucleotideGradientInput):
    """Input for differentiable PARADE UTR optimization.

    Attributes:
        inputs (list[NucleotideSequenceLogits]): One relaxed ``(L, 4)`` UTR state per
            candidate in ``A,C,G,T`` order. Every item must share one length, because the
            backend evaluates them in one vectorized pass.
        temperature (float | None): Softmax temperature shared by every item. The default ``1.0``
            reads each state as unnormalized and applies ``softmax(logits / T)``, returning
            logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """


class ParadeGradientConfig(_SafeCheckpointUrlConfigMixin, RelaxationConfig):
    """Configuration for differentiable PARADE UTR-activity objectives.

    Attributes:
        device (str): Device used for inference and backpropagation.
        construct_type (ParadeConstructType): UTR model to use — ``"utr5"`` or ``"utr3"``.
        gradient_terms (list[ParadeActivityTerm]): Cells whose activity to differentiate. A bare
            cell code or cell-line name is also accepted and normalized into a term.
        model_checkpoint (str): ``"parade"``, the bundled checkpoints (the worker picks the one
            for ``construct_type``), or your own checkpoint as a local ``.ckpt`` path or an
            ``https`` download link (a schemeless ``host.tld/path`` is accepted). A link also works
            on ``device="modal"``; a local path runs only locally.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
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
    gradient_terms: list[ParadeActivityTerm] = ConfigField(
        default_factory=lambda: [ParadeActivityTerm()],
        min_length=1,
        title="Gradient Terms",
        description="Cells whose predicted activity to differentiate, by code or cell-line name.",
    )
    model_checkpoint: str = ConfigField(
        title="Model / Checkpoint",
        default="parade",
        description="Bundled model name ('parade') or your own .ckpt path or https link.",
        reload_on_change=True,
        repr=False,
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
    )

    @field_validator("model_checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: str) -> str:
        """Validate a bundled name, an https link (schemeless allowed), or a local path."""
        return normalize_checkpoint_override(value)

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A local checkpoint path isn't present on a remote worker."""
        return local_checkpoint_on_remote(self.model_checkpoint, device)

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def _normalize_terms(cls, value: Any) -> Any:
        """Accept bare cell codes or names alongside full term objects."""
        if isinstance(value, list):
            return [
                ParadeActivityTerm(cell_type=cast(ParadeCellType, item)) if isinstance(item, str) else item
                for item in value
            ]
        return value

    @field_validator("construct_type")
    @classmethod
    def validate_construct_type(cls, value: ParadeConstructType, info: ValidationInfo) -> ParadeConstructType:
        """Reject a construct that strands an already-selected cell, before the assignment lands.

        This runs *before* the field is written, so a rejected assignment leaves the config
        unchanged. The model validator below cannot do that: it runs after Pydantic has already
        mutated the field, so raising there would leave an invalid construct in place. At
        construction time ``info.data`` has no ``gradient_terms`` yet, which is why both exist.
        """
        selected = info.data.get("gradient_terms", [])
        offpanel = sorted({term.cell_type for term in selected} - set(PARADE_CELL_TYPES[value]))
        if offpanel:
            raise ValueError(
                f"gradient_terms cell codes {offpanel} are not in the {value} panel {list(PARADE_CELL_TYPES[value])}"
            )
        return value

    @model_validator(mode="after")
    def validate_terms_against_panel(self) -> Self:
        """Require every selected cell to be in the construct's panel, and each to appear once."""
        panel = PARADE_CELL_TYPES[self.construct_type]
        offpanel = sorted({term.cell_type for term in self.gradient_terms} - set(panel))
        if offpanel:
            raise ValueError(
                f"gradient_terms cell codes {offpanel} are not in the {self.construct_type} panel {list(panel)}"
            )

        cells = [term.cell_type for term in self.gradient_terms]
        duplicates = sorted({cell for cell in cells if cells.count(cell) > 1})
        if duplicates:
            raise ValueError(f"gradient_terms names cells {duplicates} more than once; each cell yields one term")
        return self


class ParadeTermGradient(TermGradient):
    """One PARADE cell's differentiated activity.

    Attributes:
        term (ParadeActivityTerm): The resolved cell-selection term.
    """

    term: ParadeActivityTerm = Field(title="Term", description="Resolved cell-selection term.")


class ParadeGradientResult(GradientResult):
    """One evaluated UTR and every cell activity differentiated for it.

    Attributes:
        terms (list[ParadeTermGradient]): One entry per selected cell, in requested order.
    """

    terms: list[ParadeTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Gradient Terms",
        description="One result per selected cell, each with its raw activity and gradient.",
    )


class ParadeGradientOutput(GradientOutput):
    """Differentiable PARADE UTR-activity output.

    Attributes:
        results (list[ParadeGradientResult]): One per input UTR, in input order. Contains one term
            per selected cell, each carrying that cell's raw predicted activity and its own
            gradient (``None`` in forward-only mode).
        vocab (list[str]): DNA column ordering for logits and gradient.
    """

    results: list[ParadeGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input UTR, in input order, each carrying one term per cell.",
    )

    @classmethod
    def from_batch(
        cls,
        payload: Mapping[str, Any],
        *,
        matrices: Sequence[Sequence[Sequence[float]]],
        temperature: float | None,
        soft: float = 1.0,
        hard: float = 0.0,
    ) -> "ParadeGradientOutput":
        """Unbatch one worker payload into per-sample results.

        The worker reports one entry per cell, each holding that cell's value and gradient for
        every sample, so this transposes cells x samples into one result per UTR.

        Args:
            payload (Mapping[str, Any]): Worker payload for the whole batch.
            matrices (Sequence[Sequence[Sequence[float]]]): The relaxed UTR each sample was
                differentiated at, in input order. Every cell's gradient for one sample is a
                derivative with respect to the same state, which is what lets them be combined.
            temperature (float | None): The input's temperature, which fixes the coordinate space.
            soft (float): Relaxation blend in force.
            hard (float): Straight-through coefficient in force.

        Returns:
            ParadeGradientOutput: One result per sample, in input order.
        """
        entries = list(payload["terms"])
        batch_size = int(payload["metrics"]["batch_size"])
        results = [
            ParadeGradientResult(
                terms=[
                    ParadeTermGradient(
                        term=ParadeActivityTerm(cell_type=entry["cell_type"]),
                        term_value=float(entry["values"][index]),
                        gradient=entry["gradients"][index] if entry["gradients"] else None,
                        metadata=GradientMetadata.from_input(
                            matrices[index],
                            vocab=payload["vocab"],
                            temperature=temperature,
                            soft=soft,
                            hard=hard,
                        ),
                    )
                    for entry in entries
                ]
            )
            for index in range(batch_size)
        ]
        return cls(results=results, vocab=payload["vocab"])

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient bundle as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        payload = self.model_dump(include={"results", "vocab"})
        json_path.write_text(json.dumps(payload, indent=2))


def example_gradient_input() -> Any:
    """Minimal valid input for differentiable PARADE examples."""
    return ParadeGradientInput(inputs=[NucleotideSequenceLogits([[0.0] * len(DNA_NUCLEOTIDES)] * 50)])


@tool(
    key="parade-gradient",
    label="PARADE UTR Activity Gradient",
    category="sequence_scoring",
    input_class=ParadeGradientInput,
    config_class=ParadeGradientConfig,
    output_class=ParadeGradientOutput,
    description="Compute per-cell PARADE UTR activities and their gradients for relaxed UTR logits",
    uses_gpu=True,
    example_input=example_gradient_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=32,
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
        ParadeGradientOutput: One result per input UTR, each carrying one term per selected
            cell with that cell's raw activity and its own gradient.
    """
    inputs = ParadeGradientInput.model_validate(inputs.model_dump())
    # Revalidate a dumped copy so nested/container mutation cannot reach the worker
    # without re-running the cross-field and finite-number checks.
    config = ParadeGradientConfig.model_validate(config.model_dump())
    # The objective runs through the same per-construct activity checkpoint as parade-activity.
    result = ToolInstance.dispatch(
        "parade",
        {
            "operation": "gradient",
            "logits": [item.root for item in inputs.inputs],
            "temperature": inputs.temperature,
            "construct_type": config.construct_type,
            "gradient_terms": [term.model_dump() for term in config.gradient_terms],
            "model_checkpoint": config.model_checkpoint,
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
    return ParadeGradientOutput.from_batch(
        result,
        matrices=[item.root for item in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
