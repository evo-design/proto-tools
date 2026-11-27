"""Binder-specific ESMFold2 inversion: a designed binder against one fixed target."""

import logging
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.structure_prediction.esmfold2.inversion_terms import (
    ESMFold2EpitopeTerm,
    ESMFold2GlobularityTerm,
    ESMFold2InterContactTerm,
    ESMFold2IntraContactTerm,
    ESMFold2InversionConfigBase,
    ESMFold2PseudoPerplexityTerm,
    basis_metadata,
    resolve_catalog_names,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    ConfigField,
    GradientInput,
    GradientOutput,
    GradientResult,
    InputField,
    ProteinSequenceLogits,
    TermCatalog,
    TermGradient,
    ToolInstance,
    coerce_to_item_list,
)
from proto_tools.utils.base_config import RANDOM_SEED_UPPER_BOUND
from proto_tools.utils.fanout import derive_item_seed
from proto_tools.utils.sequence_logits import SequenceLogits
from proto_tools.utils.tool_io import Metrics, MetricSpec
from proto_tools.utils.weights_layout import toolkit_hf_hub_cache

logger = logging.getLogger(__name__)

#: The binder tool folds exactly two chains, in this order.
TARGET_CHAIN_ID = "target"
BINDER_CHAIN_ID = "binder"

ESMFold2BinderTermType = (
    ESMFold2IntraContactTerm
    | ESMFold2GlobularityTerm
    | ESMFold2PseudoPerplexityTerm
    | ESMFold2InterContactTerm
    | ESMFold2EpitopeTerm
)


class ESMFold2BinderCandidate(BaseModel):
    """One relaxed binder designed against one fixed target chain.

    The binder is the only mutable entity: the target is supplied as a discrete sequence and
    never receives a gradient. The two chain roles are fixed by the model rather than configured,
    which is the point of having a binder-specific tool.

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
        """Expose the binder's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and the binder's matrix.
        """
        yield "logits", self.logits

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        """Reject non-canonical sequences, a bad alphabet, and hotspots the target contradicts.

        Returns:
            Self: The validated candidate.

        Raises:
            ValueError: If the target is not canonical, the alphabet is not a permutation of the
                canonical amino acids, or a hotspot is malformed, out of range, or names a
                residue the target does not have at that position.
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


class ESMFold2BinderGradientInput(GradientInput):
    """Input for the binder inversion tool: one or more binder/target pairs.

    Attributes:
        inputs (list[ESMFold2BinderCandidate]): One relaxed binder and its target per candidate.
        temperature (float | None): Inherited softmax temperature; ``None`` consumes the logits
            as probabilities.
    """

    inputs: list[ESMFold2BinderCandidate] = InputField(
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
    ) -> ESMFold2BinderCandidate:
        """Build one candidate pairing the converted binder matrix with its fixed target.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``target_sequence``.

        Returns:
            ESMFold2BinderCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``target_sequence`` is missing.
        """
        if "target_sequence" not in fields:
            raise ValueError("esmfold2-inversion-binder-gradient from_probabilities requires the target sequence")
        matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return ESMFold2BinderCandidate(logits=matrix, **fields)


class ESMFold2BinderGradientMetrics(Metrics):
    """Per-candidate metrics for the binder inversion tool.

    Metrics documented in ``metric_spec``:
        binder_length (int): Residues in the designed binder chain.
        target_length (int): Residues in the fixed target chain.
        distogram_iptm_proxy (float): Interface confidence read off the distogram, in ``[0, 1]``;
            present whenever the complex was folded.
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
        "distogram_iptm_proxy": {
            "availability": "when a structure term is requested",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "description": "Interface confidence from the binder-target distogram (1 - mean entropy of the sharpest "
            "pairs under 22 A, over its maximum); absent when only plm_nll was requested.",
            "better_values_are": "higher",
        },
    }

    primary_metric: str | None = Field(
        default=None,
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class ESMFold2BinderTermGradient(TermGradient):
    """One binder-design quantity and its gradient.

    Attributes:
        term (ESMFold2BinderTermType): The quantity this gradient answers for.
    """

    term: ESMFold2BinderTermType = Field(title="Term", description="Binder quantity this gradient answers.")


class ESMFold2BinderGradientResult(GradientResult):
    """One evaluated binder and every quantity differentiated for it.

    Attributes:
        terms (list[ESMFold2BinderTermGradient]): One entry per requested quantity, in order.
    """

    terms: list[ESMFold2BinderTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Gradient Terms",
        description="One gradient per requested quantity, in the order requested.",
    )


class ESMFold2BinderGradientOutput(GradientOutput):
    """Binder inversion output: one result per candidate.

    Attributes:
        results (list[ESMFold2BinderGradientResult]): One per input candidate, in input order.
        vocab (list[str]): Amino-acid column ordering for the logits and gradients.
    """

    results: list[ESMFold2BinderGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input candidate, in input order, each carrying its terms.",
    )


class ESMFold2BinderGradientConfig(ESMFold2InversionConfigBase):
    """Configuration for the binder-specific ESMFold2 inversion gradient tool.

    Attributes:
        gradient_terms (list[ESMFold2BinderTermType]): Quantities to differentiate, given as
            catalog names or as instances. The three single-chain quantities apply to the binder.
        epitope_contact_distance (float): Distance in angstroms defining a hotspot contact.
        model_checkpoint (ESMFold2Checkpoint): Inherited ESMFold2 variant.
        lm_dropout (float): Inherited pair-representation dropout; experimental checkpoints only.
        plm_passes (int): Inherited mask draws for the pseudo-perplexity term.
        plm_batch_size (int): Inherited masked sequences per language-model pass.
        compute_gradient (bool): Inherited; ``False`` returns values only.
        soft (float): Inherited relaxation blend.
        hard (float): Inherited straight-through coefficient.
        seed (int): Inherited seed.
        device (str): Inherited device.
    """

    TERMS: ClassVar[TermCatalog] = {
        "intra_contact": ESMFold2IntraContactTerm,
        "globularity": ESMFold2GlobularityTerm,
        "plm_nll": ESMFold2PseudoPerplexityTerm,
        "inter_contact": ESMFold2InterContactTerm,
        "epitope": ESMFold2EpitopeTerm,
    }

    gradient_terms: list[ESMFold2BinderTermType] = ConfigField(
        title="Gradient Terms",
        default_factory=lambda: [
            ESMFold2IntraContactTerm(),
            ESMFold2InterContactTerm(),
            ESMFold2GlobularityTerm(),
        ],
        min_length=1,
        description="Quantities to differentiate, by catalog name or as instances",
    )
    epitope_contact_distance: float = ConfigField(
        title="Epitope Contact Distance",
        default=12.0,
        gt=0,
        description="Distance in angstroms defining a hotspot-binder contact",
    )

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept catalog names and term classes alongside instances."""
        return resolve_catalog_names(value, cls.supported_terms())

    @model_validator(mode="after")
    def validate_gradient_terms(self) -> Self:
        """Reject repeats, and chain selections that do not name the binder.

        Returns:
            Self: The validated config.

        Raises:
            ValueError: If a quantity is asked for twice, or a single-chain quantity names a
                chain other than the binder.
        """
        asked = [term.id for term in self.gradient_terms]
        repeated = sorted({name for name in asked if asked.count(name) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same quantity more than once: {repeated}")
        # This tool folds exactly two chains with fixed roles, so the only selectable chain is
        # the binder; anything else belongs in the general tool.
        misdirected = [
            term.id for term in self.gradient_terms if getattr(term, "chain_id", None) not in (None, BINDER_CHAIN_ID)
        ]
        if misdirected:
            raise ValueError(
                f"the binder tool evaluates single-chain quantities on the binder; "
                f"{misdirected} named another chain. Use esmfold2-inversion-gradient for a general complex."
            )
        return self


def example_input() -> ESMFold2BinderGradientInput:
    """Build a small binder/target pair for the registry's example surface.

    Returns:
        ESMFold2BinderGradientInput: One uniform 24-residue binder against a short target.
    """
    return ESMFold2BinderGradientInput(
        inputs=[
            ESMFold2BinderCandidate(
                logits=ProteinSequenceLogits([[0.0] * 20] * 24),
                target_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            )
        ]
    )


@tool(
    key="esmfold2-inversion-binder-gradient",
    label="ESMFold2 Inversion Binder Gradient",
    category="structure_prediction",
    input_class=ESMFold2BinderGradientInput,
    config_class=ESMFold2BinderGradientConfig,
    output_class=ESMFold2BinderGradientOutput,
    metrics_class=ESMFold2BinderGradientMetrics,
    description="Differentiate ESMFold2 binder-design objectives against a fixed protein target",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
    stochastic=True,
)
def run_esmfold2_inversion_binder_gradient(
    inputs: ESMFold2BinderGradientInput,
    config: ESMFold2BinderGradientConfig,
    instance: Any = None,
) -> ESMFold2BinderGradientOutput:
    """Differentiate ESMFold2 binder-design objectives with respect to a relaxed binder.

    Each requested quantity comes back with its own raw value and its own gradient, taken from
    one folding pass. Combining them into a single objective is the caller's to do, with
    ``GradientResult.compose``.

    Args:
        inputs (ESMFold2BinderGradientInput): Relaxed binders with their fixed targets.
        config (ESMFold2BinderGradientConfig): Which quantities to differentiate, and how.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        ESMFold2BinderGradientOutput: One result per candidate, in input order.

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
    for index, candidate in enumerate(inputs.inputs):
        reply = ToolInstance.dispatch(
            "esmfold2",
            {
                "operation": "complex_gradient",
                # The binder tool is the two-chain case of the general path: a fixed target
                # followed by the designed binder.
                "chains": [
                    {"chain_id": TARGET_CHAIN_ID, "sequence": candidate.target_sequence, "logits": None},
                    {"chain_id": BINDER_CHAIN_ID, "sequence": None, "logits": candidate.logits.root},
                ],
                "alphabet": candidate.alphabet,
                "gradient_terms": [{"id": name, "chain_id": BINDER_CHAIN_ID} for name in requested],
                "target_chain_id": TARGET_CHAIN_ID,
                "target_hotspot_ids": candidate.target_hotspot_ids,
                "epitope_contact_distance": config.epitope_contact_distance,
                "temperature": inputs.temperature,
                "soft": config.soft,
                "hard": config.hard,
                "compute_gradient": config.compute_gradient,
                "model_checkpoint": config.model_checkpoint,
                "lm_dropout": config.lm_dropout,
                # The experimental heads load ESM C 6B; read the esmc toolkit's copy, not a second one.
                "esmc_cache_dir": toolkit_hf_hub_cache("esmc"),
                "plm_passes": config.plm_passes,
                "plm_batch_size": config.plm_batch_size,
                # One seed per candidate, so duplicates in a batch draw differently (stochastic=True).
                "seed": derive_item_seed(config.seed, index, RANDOM_SEED_UPPER_BOUND),
                "device": config.device,
                "verbose": config.verbose,
            },
            instance=instance,
            config=config,
        )
        basis = basis_metadata(
            [candidate.logits.root], candidate.alphabet, inputs.temperature, config.soft, config.hard
        )
        catalog = cast(dict[str, type[ESMFold2BinderTermType]], ESMFold2BinderGradientConfig.supported_terms())
        results.append(
            ESMFold2BinderGradientResult(
                terms=[
                    ESMFold2BinderTermGradient(
                        term=catalog[entry["id"]](),
                        term_value=float(entry["value"]),
                        gradient=entry["gradient"],
                        metadata=basis,
                    )
                    for entry in reply["terms"]
                ],
                metrics={
                    "binder_length": len(candidate.logits.root),
                    "target_length": len(candidate.target_sequence),
                    **(
                        {"distogram_iptm_proxy": reply["metrics"]["distogram_iptm_proxy"]}
                        if "distogram_iptm_proxy" in (reply.get("metrics") or {})
                        else {}
                    ),
                },
            )
        )
    return ESMFold2BinderGradientOutput(results=results, vocab=list(inputs.inputs[0].alphabet))
