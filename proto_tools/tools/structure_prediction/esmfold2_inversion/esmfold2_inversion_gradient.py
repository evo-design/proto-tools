"""ESMFold2 inversion structure-objective gradient tool."""

import logging
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    ConfigField,
    DifferentiableTerm,
    GradientConfig,
    GradientInput,
    GradientMetadata,
    GradientOutput,
    GradientResult,
    InputField,
    ProteinSequenceLogits,
    TermCatalog,
    TermGradient,
    ToolInstance,
    coerce_to_item_list,
    resolve_terms,
)
from proto_tools.utils.sequence_logits import SequenceLogits
from proto_tools.utils.tool_io import Directionality, Metrics, MetricSpec

logger = logging.getLogger(__name__)

#: Inversion checkpoints this tool differentiates. The maintained ESMFold2 prediction
#: checkpoints are a separate family and are not interchangeable with these.
ESMFold2InversionCheckpoint = Literal["ESMFold2-Experimental-Fast", "ESMFold2-Experimental-Fast-Cutoff2025"]
INVERSION_CHECKPOINTS: tuple[str, ...] = get_args(ESMFold2InversionCheckpoint)


class ESMFold2InversionTerm(DifferentiableTerm):
    """One differentiable ESMFold2 inversion quantity, read off the predicted distogram.

    Attributes:
        id (str): Fixed identifier, pinned by each concrete quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"


class ESMFold2IntraContactTerm(ESMFold2InversionTerm):
    """Binder self-contact: each binder residue's two best contacts within its own chain.

    Attributes:
        id (Literal["intra_contact"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "Binder self-contact entropy over its two best intra-chain contacts"

    id: Literal["intra_contact"] = Field(
        default="intra_contact",
        title="Term ID",
        description="Identifies this quantity as the binder intra-chain contact objective.",
    )


class ESMFold2InterContactTerm(ESMFold2InversionTerm):
    """Interface contact: each target residue's single best contact with the binder.

    Attributes:
        id (Literal["inter_contact"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "Interface contact entropy between the target and the binder"

    id: Literal["inter_contact"] = Field(
        default="inter_contact",
        title="Term ID",
        description="Identifies this quantity as the target-binder interface contact objective.",
    )


class ESMFold2GlobularityTerm(ESMFold2InversionTerm):
    """Excess radius of gyration over a packed globular protein of the same length.

    Attributes:
        id (Literal["globularity"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "Binder radius of gyration in excess of a packed protein"
    UNITS: ClassVar[str | None] = "angstroms"

    id: Literal["globularity"] = Field(
        default="globularity",
        title="Term ID",
        description="Identifies this quantity as the binder globularity objective.",
    )


class ESMFold2EpitopeTerm(ESMFold2InversionTerm):
    """Contact surprisal between the binder and the requested target hotspots.

    Selecting this term requires ``target_hotspot_ids`` on the candidate; there is nothing to
    aim at otherwise.

    Attributes:
        id (Literal["epitope"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "Contact surprisal between the binder and the target hotspots"

    id: Literal["epitope"] = Field(
        default="epitope",
        title="Term ID",
        description="Identifies this quantity as the target-hotspot contact objective.",
    )


class ESMFold2PseudoPerplexityTerm(ESMFold2InversionTerm):
    """ESM C masked pseudo-perplexity of the designed chain, as a sequence prior.

    Selecting this term loads a second model alongside the folding trunk; leave it out and no
    language model is loaded at all.

    Attributes:
        id (Literal["plm_nll"]): Fixed identifier for this kind of quantity.
    """

    DESCRIPTION: ClassVar[str] = "ESM C masked pseudo-perplexity of the designed chain"
    UNITS: ClassVar[str | None] = "nats per masked position"

    id: Literal["plm_nll"] = Field(
        default="plm_nll",
        title="Term ID",
        description="Identifies this quantity as the ESM C pseudo-perplexity sequence prior.",
    )


ESMFold2InversionTermType = (
    ESMFold2IntraContactTerm
    | ESMFold2InterContactTerm
    | ESMFold2GlobularityTerm
    | ESMFold2EpitopeTerm
    | ESMFold2PseudoPerplexityTerm
)


class ESMFold2InversionTermGradient(TermGradient):
    """One ESMFold2 inversion quantity and its gradient.

    Attributes:
        term (ESMFold2InversionTermType): The quantity this gradient answers for.
    """

    term: ESMFold2InversionTermType = Field(title="Term", description="Inversion quantity this gradient answers.")


class ESMFold2InversionGradientResult(GradientResult):
    """One evaluated binder and every inversion quantity differentiated for it.

    Attributes:
        terms (list[ESMFold2InversionTermGradient]): One entry per requested quantity, in the
            order requested.
    """

    terms: list[ESMFold2InversionTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Gradient Terms",
        description="One gradient per requested quantity, in the order requested.",
    )


class ESMFold2InversionCandidate(BaseModel):
    """One relaxed binder designed against one fixed target chain.

    The binder is the only mutable entity: the target is supplied as a discrete sequence and
    never receives a gradient.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed ``(L, 20)`` binder state in ``alphabet`` order.
        target_sequence (str): Fixed target chain, in canonical one-letter amino acids.
        target_hotspot_ids (list[str] | None): Target residues the binder should contact, each a
            one-letter residue followed by its 1-indexed position, such as ``"L15"``.
        alphabet (str): Column order of ``logits``, as a permutation of the 20 canonical amino
            acids. The gradient comes back in this same order.
    """

    model_config = ConfigDict(extra="forbid")

    logits: ProteinSequenceLogits = Field(
        title="Logits",
        description="Relaxed (L, 20) binder state in alphabet column order.",
    )
    target_sequence: str = Field(
        title="Target Sequence",
        description="Fixed target chain in canonical one-letter amino acids.",
    )
    target_hotspot_ids: list[str] | None = Field(
        default=None,
        title="Target Hotspots",
        description="Target residues to contact, residue letter plus 1-indexed position (L15).",
    )
    alphabet: str = Field(
        default=PROTEIN_AMINO_ACIDS,
        title="Alphabet",
        description="Logit column order, a permutation of the 20 canonical amino acids.",
    )

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this candidate's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this candidate's matrix.
        """
        yield "logits", self.logits

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        """Reject non-canonical sequences, a bad alphabet, and hotspots the target contradicts.

        Returns:
            Self: The validated candidate.

        Raises:
            ValueError: If the target is not canonical, the alphabet is not a permutation of the
                canonical amino acids, or a hotspot is out of range or names the wrong residue.
        """
        if not self.target_sequence or set(self.target_sequence) - set(PROTEIN_AMINO_ACIDS):
            raise ValueError("target_sequence must be non-empty and contain only canonical amino acids")
        if sorted(self.alphabet) != sorted(PROTEIN_AMINO_ACIDS):
            raise ValueError("alphabet must be a permutation of the 20 canonical amino acids")
        for hotspot in self.target_hotspot_ids or []:
            # A hotspot names the residue it expects, so a stale selection fails here rather
            # than silently aiming the binder at whatever now sits at that position.
            if len(hotspot) < 2 or not hotspot[0].isalpha() or not hotspot[1:].isdigit():
                raise ValueError(f"hotspot {hotspot!r} must be a residue letter and a 1-indexed position, like L15")
            position = int(hotspot[1:]) - 1
            if not 0 <= position < len(self.target_sequence):
                raise ValueError(f"hotspot {hotspot!r} is outside the target sequence")
            if self.target_sequence[position] != hotspot[0]:
                raise ValueError(
                    f"hotspot {hotspot!r} does not match the target, which has "
                    f"{self.target_sequence[position]!r} at that position"
                )
        return self


class ESMFold2InversionGradientInput(GradientInput):
    """Input for the ESMFold2 inversion gradient tool: one or more binder/target pairs.

    Attributes:
        inputs (list[ESMFold2InversionCandidate]): One relaxed binder and its target per
            candidate. A single candidate may be given directly.
        temperature (float | None): Inherited softmax temperature; ``None`` consumes the logits
            as probabilities.
    """

    inputs: list[ESMFold2InversionCandidate] = InputField(
        title="Inputs",
        description="One relaxed binder with its fixed target per candidate",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Accept one candidate where a list is expected."""
        return coerce_to_item_list(value)

    @classmethod
    def build_candidate(
        cls,
        probabilities: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> ESMFold2InversionCandidate:
        """Build one candidate pairing the converted binder matrix with its fixed target.

        The target belongs to the candidate, so it arrives here either shared across the call or
        named in this candidate's own mapping.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``target_sequence``.

        Returns:
            ESMFold2InversionCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``target_sequence`` is missing.
        """
        if "target_sequence" not in fields:
            raise ValueError("esmfold2-inversion-gradient from_probabilities requires the target sequence")
        matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return ESMFold2InversionCandidate(logits=matrix, **fields)


class ESMFold2InversionGradientMetrics(Metrics):
    """Per-candidate ESMFold2 inversion metrics.

    Metrics documented in ``metric_spec``:
        binder_length (int): Residues in the designed chain.
        target_length (int): Residues in the fixed target chain.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "binder_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Residues in the designed binder chain.",
            "better_values_are": "context-dependent",
        },
        "target_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Residues in the fixed target chain.",
            "better_values_are": "context-dependent",
        },
    }

    primary_metric: str | None = Field(
        default=None,
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class ESMFold2InversionGradientOutput(GradientOutput):
    """ESMFold2 inversion output: one result per candidate, each carrying its requested terms.

    Attributes:
        results (list[ESMFold2InversionGradientResult]): One per input candidate, in input
            order, each holding the requested quantities and their gradients with respect to
            that candidate's relaxed binder state.
        vocab (list[str]): Amino-acid column ordering for the logits and gradients.
    """

    results: list[ESMFold2InversionGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input candidate, in input order, each carrying its terms.",
    )


class ESMFold2InversionGradientConfig(GradientConfig):
    """Configuration for the ESMFold2 inversion gradient tool.

    Attributes:
        gradient_terms (list[ESMFold2InversionTermType]): Quantities to differentiate, given as
            catalog names (``"intra_contact"``) or as instances. Each comes back with its own
            raw value and gradient.
        model_name (ESMFold2InversionCheckpoint): Inversion checkpoint to fold with.
        epitope_contact_distance (float): Distance in angstroms defining a hotspot contact.
        plm_passes (int): Independent mask draws averaged by the pseudo-perplexity term.
        plm_batch_size (int): Masked sequences per language-model forward pass.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run the backward passes; ``False`` returns values only.
        seed (int): Seed for checkpoint loading, dropout and mask draws.
        device (str): Device to run the model on.
    """

    TERMS: ClassVar[TermCatalog] = {
        "intra_contact": ESMFold2IntraContactTerm,
        "inter_contact": ESMFold2InterContactTerm,
        "globularity": ESMFold2GlobularityTerm,
        "epitope": ESMFold2EpitopeTerm,
        "plm_nll": ESMFold2PseudoPerplexityTerm,
    }

    gradient_terms: list[ESMFold2InversionTermType] = ConfigField(
        title="Gradient Terms",
        default_factory=lambda: [
            ESMFold2IntraContactTerm(),
            ESMFold2InterContactTerm(),
            ESMFold2GlobularityTerm(),
        ],
        min_length=1,
        description="Quantities to differentiate, by catalog name or as instances",
    )

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept catalog names and term classes alongside instances."""
        if not isinstance(value, list):
            return value
        catalog = cls.supported_terms()
        return [
            resolve_terms([entry], catalog)[0]  # type: ignore[arg-type]
            if isinstance(entry, str | type)
            else entry
            for entry in value
        ]

    @model_validator(mode="after")
    def reject_repeated_terms(self) -> Self:
        """The same quantity twice would be one question asked twice."""
        asked = [term.id for term in self.gradient_terms]
        repeated = sorted({name for name in asked if asked.count(name) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same quantity more than once: {repeated}")
        return self

    model_name: ESMFold2InversionCheckpoint = ConfigField(
        title="Model Name",
        default=INVERSION_CHECKPOINTS[0],
        description="Inversion checkpoint to fold with",
        reload_on_change=True,
    )
    epitope_contact_distance: float = ConfigField(
        title="Epitope Contact Distance",
        default=12.0,
        gt=0,
        description="Distance in angstroms defining a hotspot-binder contact",
    )
    plm_passes: int = ConfigField(
        title="Pseudo-perplexity Passes",
        default=4,
        ge=1,
        description="Independent mask draws averaged by the pseudo-perplexity term",
    )
    plm_batch_size: int = ConfigField(
        title="Pseudo-perplexity Batch Size",
        default=128,
        ge=1,
        description="Masked sequences per language-model forward pass",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward passes and return gradients; False scores the terms only",
    )
    seed: int = ConfigField(
        title="Seed",
        default=0,
        ge=0,
        description="Seed for checkpoint loading, dropout and mask draws",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


def example_input() -> ESMFold2InversionGradientInput:
    """Build a small binder/target pair for the registry's example surface.

    Returns:
        ESMFold2InversionGradientInput: One uniform 24-residue binder against a short target.
    """
    return ESMFold2InversionGradientInput(
        inputs=[
            ESMFold2InversionCandidate(
                logits=ProteinSequenceLogits([[0.0] * 20] * 24),
                target_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            )
        ]
    )


@tool(
    key="esmfold2-inversion-gradient",
    label="ESMFold2 Inversion Gradient",
    category="structure_prediction",
    input_class=ESMFold2InversionGradientInput,
    config_class=ESMFold2InversionGradientConfig,
    output_class=ESMFold2InversionGradientOutput,
    metrics_class=ESMFold2InversionGradientMetrics,
    description="Differentiate ESMFold2 inversion structure objectives w.r.t. a relaxed binder",
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
    stochastic=True,
)
def run_esmfold2_inversion_gradient(
    inputs: ESMFold2InversionGradientInput,
    config: ESMFold2InversionGradientConfig,
    instance: Any = None,
) -> ESMFold2InversionGradientOutput:
    """Differentiate ESMFold2 inversion structure objectives with respect to a relaxed binder.

    Each requested quantity comes back with its own raw value and its own gradient, taken from
    one folding pass. Combining them into a single objective is the caller's to do, with
    ``GradientResult.compose``.

    Args:
        inputs (ESMFold2InversionGradientInput): Relaxed binders with their fixed targets.
        config (ESMFold2InversionGradientConfig): Which quantities to differentiate, and how.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        ESMFold2InversionGradientOutput: One result per candidate, in input order.

    Raises:
        ValueError: If the epitope quantity is requested for a candidate with no hotspots.
    """
    requested = [term.id for term in config.gradient_terms]
    # Pydantic cannot see across the input and the config, so the pairing is checked here.
    if "epitope" in requested:
        missing = [index for index, item in enumerate(inputs.inputs) if not item.target_hotspot_ids]
        if missing:
            raise ValueError(f"the epitope quantity needs target_hotspot_ids, absent on candidates {missing}")

    results = []
    for candidate in inputs.inputs:
        reply = ToolInstance.dispatch(
            "esmfold2_inversion",
            {
                "operation": "structure_gradient",
                "logits": candidate.logits.root,
                "alphabet": candidate.alphabet,
                "target_sequence": candidate.target_sequence,
                "target_hotspot_ids": candidate.target_hotspot_ids,
                "gradient_terms": requested,
                "temperature": inputs.temperature,
                "soft": config.soft,
                "hard": config.hard,
                "compute_gradient": config.compute_gradient,
                "model_name": config.model_name,
                "epitope_contact_distance": config.epitope_contact_distance,
                "plm_passes": config.plm_passes,
                "plm_batch_size": config.plm_batch_size,
                "seed": config.seed,
                "device": config.device,
                "verbose": config.verbose,
            },
            instance=instance,
            config=config,
        )
        # Every quantity is read off one folding pass, so they share one description of the
        # point they were taken at.
        basis = GradientMetadata.from_input(
            candidate.logits.root,
            vocab=list(candidate.alphabet),
            temperature=inputs.temperature,
            soft=config.soft,
            hard=config.hard,
        )
        # The config's catalog is the one place the ids are declared; narrow it to the concrete
        # classes, since the shared base pins no id of its own.
        catalog = cast(
            dict[str, type[ESMFold2InversionTermType]],
            ESMFold2InversionGradientConfig.supported_terms(),
        )
        results.append(
            ESMFold2InversionGradientResult(
                terms=[
                    ESMFold2InversionTermGradient(
                        term=catalog[entry["id"]](),
                        term_value=float(entry["value"]),
                        gradient=entry["gradient"],
                        metadata=basis,
                    )
                    for entry in reply["terms"]
                ],
                metrics=reply["metrics"],
            )
        )
    return ESMFold2InversionGradientOutput(results=results, vocab=list(inputs.inputs[0].alphabet))
