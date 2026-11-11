"""Differentiable Cherimoya accessibility objectives for relaxed DNA logits."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator

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
    DifferentiableTerm,
    GradientConfig,
    GradientMetadata,
    GradientOutput,
    GradientResult,
    NucleotideSequenceLogits,
    TermCatalog,
    TermGradient,
    ToolInstance,
    resolve_terms,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Directionality


class CherimoyaCountsTerm(DifferentiableTerm):
    """Shared selectors for a differentiable Cherimoya read-count quantity.

    Both quantities are natural-log read counts, which depend on each experiment's sequencing
    depth. The value reported is that log count itself, in its own units; a bounded objective
    over it is a goal the caller composes, and the centre it should saturate around is
    experiment-specific. For the default model (ENCSR000EOT, K562 DNase-seq) the median fold 0
    ``log_counts`` over 5,000 sampled hg38 SCREEN cCREs is 4.5; a target span shorter than
    1,000 bp holds fewer reads, and a BATv1 ChIP-seq model run with its all-zero control
    has its own scale.

    Attributes:
        model (str): ENCODE experiment accession or local ``.torch`` checkpoint path.
        folds (list[CherimoyaFold]): Folds averaged for this term (ignored for a checkpoint path).
    """

    #: Whether more signal is wanted depends on whether the experiment is a target or an off-target.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    model: str = Field(
        default=DEFAULT_CHERIMOYA_MODEL, title="Model", description="ENCODE experiment accession or .torch path"
    )
    folds: list[CherimoyaFold] = Field(
        default=[0], min_length=1, title="Folds", description="Folds averaged for this term"
    )


class CherimoyaLogCountsTerm(CherimoyaCountsTerm):
    """ln(reads + 1) predicted over Cherimoya's full 1,000 bp output window.

    Attributes:
        id (Literal["log_counts"]): Fixed identifier for this kind of quantity.
    """

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Predicted ln(reads + 1) over Cherimoya's full output window"

    id: Literal["log_counts"] = Field(
        default="log_counts",
        title="Term ID",
        description="Identifies this quantity as ln(reads + 1) over the full output window.",
    )


class CherimoyaLogTargetCountsTerm(CherimoyaCountsTerm):
    """ln(expected reads + 1) predicted inside the config's target span.

    Requires ``target_start``/``target_end`` on the config to name a span; without them the
    span is the whole designed segment.

    Attributes:
        id (Literal["log_target_counts"]): Fixed identifier for this kind of quantity.
    """

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Predicted ln(expected reads + 1) inside the target span"

    id: Literal["log_target_counts"] = Field(
        default="log_target_counts",
        title="Term ID",
        description="Identifies this quantity as ln(expected reads + 1) inside the target span.",
    )


class CherimoyaTermGradient(TermGradient):
    """One Cherimoya read-count quantity and its raw gradient.

    Attributes:
        term (CherimoyaLogCountsTerm | CherimoyaLogTargetCountsTerm): The model-and-quantity
            selector this gradient answers.
    """

    term: CherimoyaLogCountsTerm | CherimoyaLogTargetCountsTerm = Field(
        title="Term", description="Model-and-quantity selector answered."
    )


class CherimoyaGradientResult(GradientResult):
    """One evaluated sample and every Cherimoya quantity differentiated for it.

    Attributes:
        terms (list[CherimoyaTermGradient]): One entry per requested term, in the order requested.
    """

    terms: list[CherimoyaTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Terms",
        description="One gradient per requested model-and-quantity, in the order requested.",
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


class CherimoyaGradientConfig(GradientConfig):
    """Configuration for differentiable Cherimoya objectives.

    The model input is ``context_left + designed segment + context_right``, which must be
    2,114 bp. Only the designed segment is differentiated.

    Attributes:
        collection (CherimoyaCollectionKey): Collection the accessions in ``gradient_terms`` come from.
        gradient_terms (list[CherimoyaLogCountsTerm | CherimoyaLogTargetCountsTerm]): Quantities to
            differentiate, by name or as instances.
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
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
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
    TERMS: ClassVar[TermCatalog] = {
        "log_counts": CherimoyaLogCountsTerm,
        "log_target_counts": CherimoyaLogTargetCountsTerm,
    }

    collection: CherimoyaCollectionKey = ConfigField(
        title="Collection",
        default="catv1",
        description="Collection of the gradient_terms models: catv1 (accessibility) or batv1 (ChIP-seq).",
    )
    gradient_terms: list[CherimoyaLogCountsTerm | CherimoyaLogTargetCountsTerm] = ConfigField(
        default_factory=lambda: [CherimoyaLogCountsTerm()],
        min_length=1,
        max_length=MAX_CHERIMOYA_MODELS,
        title="Gradient Terms",
        description="Cherimoya read-count quantities to differentiate, by name or as instances.",
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

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept a term by name or class, leaving an already-built instance as it is."""
        if not isinstance(value, list):
            return value
        catalog = cls.supported_terms()
        return [
            resolve_terms([entry], catalog)[0]  # type: ignore[arg-type]
            if isinstance(entry, str | type)
            else entry
            for entry in value
        ]

    @field_validator("gradient_terms")
    @classmethod
    def resolve_term_models(cls, terms: list[Any], info: ValidationInfo) -> list[Any]:
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
        targeted = any(term.id == "log_target_counts" for term in self.gradient_terms)
        if (self.target_start is not None or self.target_end is not None) and not targeted:
            raise ValueError("target_start/target_end are only used by gradient terms named 'log_target_counts'")
        asked = [(term.id, term.model, tuple(term.folds)) for term in self.gradient_terms]
        repeated = sorted({item for item in asked if asked.count(item) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same quantity, model and folds more than once: {repeated}")
        return self

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Checkpoint paths name local files that a remote worker does not have."""
        paths = sorted({term.model for term in self.gradient_terms if is_checkpoint_path(term.model)})
        if paths:
            return f"gradient_terms models {paths} are local checkpoint files not available on device='{device}'."
        return None


class CherimoyaGradientOutput(GradientOutput):
    """Differentiable Cherimoya objective output.

    Attributes:
        results (list[CherimoyaGradientResult]): One per input item, in input order, each
            holding that sample's requested read-count quantities and their raw gradients
            (``None`` in forward-only mode).
        vocab (list[str]): DNA column ordering for logits and gradient.
        models (list[CherimoyaModelInfo]): Provenance of each model the terms used.
    """

    results: list[CherimoyaGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input item, in input order, each carrying its read-count quantities.",
    )
    models: list[CherimoyaModelInfo] = Field(
        default_factory=list, title="Models", description="Provenance of each model the gradient terms used"
    )

    @classmethod
    def from_batch(
        cls,
        payload: Mapping[str, Any],
        models: list[CherimoyaModelInfo],
        *,
        matrices: Sequence[Sequence[Sequence[float]]],
        temperature: float | None,
        soft: float = 1.0,
        hard: float = 0.0,
    ) -> "CherimoyaGradientOutput":
        """Unbatch one worker payload into per-sample results.

        The worker reports one entry per requested quantity, each holding that quantity's value
        and gradient for every sample, so this transposes terms x samples into one result per
        designed segment.

        Args:
            payload (Mapping[str, Any]): Worker payload for the whole batch.
            models (list[CherimoyaModelInfo]): Provenance of each model the gradient terms used.
            matrices (Sequence[Sequence[Sequence[float]]]): The relaxed segment each sample was
                differentiated at, in input order. Every term's gradient for one sample is a
                derivative with respect to the same state, which is what lets them be combined.
            temperature (float | None): The input's temperature, which fixes the coordinate space.
            soft (float): Relaxation blend in force.
            hard (float): Straight-through coefficient in force.

        Returns:
            CherimoyaGradientOutput: One result per sample, in input order.
        """
        entries = list(payload["terms"])
        batch_size = int(payload["metrics"]["batch_size"])
        # The config's catalog is the one place the ids are declared; narrow it to the two
        # concrete classes, since the shared base pins no id of its own.
        term_classes = cast(
            dict[str, type[CherimoyaLogCountsTerm | CherimoyaLogTargetCountsTerm]],
            CherimoyaGradientConfig.supported_terms(),
        )
        results = [
            CherimoyaGradientResult(
                terms=[
                    CherimoyaTermGradient(
                        term=term_classes[entry["id"]](model=entry["model"], folds=entry["folds"]),
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
    if all(term.id != "log_target_counts" for term in config.gradient_terms):
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
    description="Compute differentiable Cherimoya accessibility losses and gradients for relaxed DNA logits",
    uses_gpu=True,
    pin_visible_devices=True,
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
    models = list(dict.fromkeys(term.model for term in config.gradient_terms))
    specs: list[dict[str, Any]] = []
    for model in models:
        folds = list(
            dict.fromkeys(fold for term in config.gradient_terms if term.model == model for fold in term.folds)
        )
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
        for term in config.gradient_terms
    ]

    result = ToolInstance.dispatch(
        "cherimoya",
        {
            "operation": "gradient",
            "logits": [item.root for item in inputs.inputs],
            "temperature": inputs.temperature,
            "checkpoints": specs,
            "gradient_terms": terms,
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
    return CherimoyaGradientOutput.from_batch(
        result,
        model_infos(models, config.collection, specs, result["applied"]),
        matrices=[item.root for item in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
