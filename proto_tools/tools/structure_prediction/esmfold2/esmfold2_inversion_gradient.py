"""General ESMFold2 inversion: per-chain objectives over a protein complex."""

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.structure_prediction.esmfold2.inversion_terms import (
    ESMFold2GlobularityTerm,
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

ESMFold2ChainTermType = ESMFold2IntraContactTerm | ESMFold2GlobularityTerm | ESMFold2PseudoPerplexityTerm


class ESMFold2Chain(BaseModel):
    """One chain of the complex, either held fixed or designed.

    A chain is fixed when it carries a discrete ``sequence`` and designable when it carries
    relaxed ``logits``; exactly one of the two is given. Fixed chains are folded as context and
    receive no gradient.

    Attributes:
        chain_id (str): Identifier for this chain, unique within the complex, and what a term
            names when it selects a chain.
        sequence (str | None): Discrete sequence, for a fixed chain.
        logits (ProteinSequenceLogits | None): Relaxed ``(L, 20)`` state, for a designable chain.
        alphabet (str): Column order of ``logits``, a permutation of the canonical amino acids.
    """

    model_config = ConfigDict(extra="forbid")

    chain_id: str = Field(title="Chain ID", description="Identifier for this chain, unique in the complex.")
    sequence: str | None = Field(
        default=None,
        title="Sequence",
        description="Discrete sequence for a fixed chain, in canonical one-letter amino acids.",
    )
    logits: ProteinSequenceLogits | None = Field(
        default=None,
        title="Logits",
        description="Relaxed (L, 20) state for a designable chain, in alphabet column order.",
    )
    alphabet: str = Field(
        default=PROTEIN_AMINO_ACIDS,
        title="Alphabet",
        description="Logit column order, a permutation of the 20 canonical amino acids.",
    )

    @property
    def designable(self) -> bool:
        """Whether this chain carries a relaxed state and so receives a gradient."""
        return self.logits is not None

    @model_validator(mode="after")
    def validate_chain(self) -> Self:
        """Require exactly one of ``sequence`` and ``logits``, and a usable alphabet.

        Returns:
            Self: The validated chain.

        Raises:
            ValueError: If both or neither of the two are given, the sequence is not canonical,
                or the alphabet is not a permutation of the canonical amino acids.
        """
        if (self.sequence is None) == (self.logits is None):
            raise ValueError(f"chain {self.chain_id!r} needs exactly one of sequence (fixed) or logits (designable)")
        if self.sequence is not None and (not self.sequence or set(self.sequence) - set(PROTEIN_AMINO_ACIDS)):
            raise ValueError(f"chain {self.chain_id!r} sequence must be non-empty canonical amino acids")
        if sorted(self.alphabet) != sorted(PROTEIN_AMINO_ACIDS):
            raise ValueError(f"chain {self.chain_id!r} alphabet must be a permutation of the canonical amino acids")
        return self


class ESMFold2ComplexCandidate(BaseModel):
    """One protein complex, folded as a unit, with at least one chain being designed.

    Attributes:
        chains (list[ESMFold2Chain]): Chains in the order they are folded. Their order fixes the
            row order of the complex, and of the concatenated gradient.
    """

    model_config = ConfigDict(extra="forbid")

    chains: list[ESMFold2Chain] = Field(
        min_length=1,
        title="Chains",
        description="Chains in fold order; at least one must be designable.",
    )

    @model_validator(mode="after")
    def validate_complex(self) -> Self:
        """Require unique chain ids, one shared alphabet, and something to design.

        Returns:
            Self: The validated complex.

        Raises:
            ValueError: If chain ids repeat, no chain is designable, or designable chains
                disagree about their column order.
        """
        ids = [chain.chain_id for chain in self.chains]
        repeated = sorted({name for name in ids if ids.count(name) > 1})
        if repeated:
            raise ValueError(f"chain ids must be unique within a complex, repeated: {repeated}")
        designable = [chain for chain in self.chains if chain.designable]
        if not designable:
            raise ValueError("at least one chain must be designable, otherwise there is nothing to differentiate")
        # The gradient is one matrix over the concatenated designable rows, so its columns can
        # only mean one thing.
        alphabets = {chain.alphabet for chain in designable}
        if len(alphabets) > 1:
            raise ValueError(f"designable chains must share one alphabet, got {sorted(alphabets)}")
        return self

    @property
    def designable_chains(self) -> list[ESMFold2Chain]:
        """Designable chains, in fold order, which is the order their gradient rows appear in."""
        return [chain for chain in self.chains if chain.designable]

    @property
    def alphabet(self) -> str:
        """The column order the designable chains share."""
        return self.designable_chains[0].alphabet

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose every designable chain's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The chain id and that chain's matrix.
        """
        for index, chain in enumerate(self.chains):
            if chain.logits is not None:
                yield f"chains[{index}].logits", chain.logits


class ESMFold2InversionGradientInput(GradientInput):
    """Input for the general ESMFold2 inversion tool: one or more protein complexes.

    Attributes:
        inputs (list[ESMFold2ComplexCandidate]): One complex per candidate. A single complex may
            be given directly.
        temperature (float | None): Inherited softmax temperature; ``None`` consumes the logits
            as probabilities.
    """

    inputs: list[ESMFold2ComplexCandidate] = InputField(
        title="Inputs",
        description="One protein complex per candidate, each with at least one designable chain",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Accept one complex where a list is expected."""
        return coerce_to_item_list(value)

    @classmethod
    def candidate_specs(cls, probabilities: Any) -> list[dict[str, Any]]:
        """Accept a candidate keyed by the path its matrix lives at, as well as by name.

        A complex keeps its matrices inside ``chains`` rather than in a field of its own, so
        ``sequence_logits()`` reports them by path. A caller round-tripping those paths back in
        is naming the same thing as ``probabilities``, so they are treated alike.

        Args:
            probabilities (Any): One matrix, or a list of matrices or candidate mappings.

        Returns:
            list[dict[str, Any]]: One mapping per candidate, each carrying ``probabilities``.
        """
        if isinstance(probabilities, Sequence) and probabilities and isinstance(probabilities[0], Mapping):
            rekeyed = []
            for spec in probabilities:
                entry = dict(spec)
                if "probabilities" not in entry:
                    # Several designable chains arrive keyed by the paths sequence_logits()
                    # reports, so they are collected into one mapping the factory can place.
                    paths = [key for key in entry if key.endswith(".logits")]
                    if paths:
                        entry["probabilities"] = {path: entry.pop(path) for path in paths}
                rekeyed.append(entry)
            probabilities = rekeyed
        return super().candidate_specs(probabilities)

    @classmethod
    def build_candidate(
        cls,
        probabilities: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> ESMFold2ComplexCandidate:
        """Build a one-designable-chain complex from a distribution plus its context chains.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): The candidate's own context. ``chains`` rebuilds an existing complex
                with this matrix on its first designed chain; otherwise ``chain_id`` names a
                single designed chain.

        Returns:
            ESMFold2ComplexCandidate: Complex whose designed chain reproduces ``probabilities``.
        """
        existing = [ESMFold2Chain.model_validate(entry) for entry in fields.pop("chains", [])]
        if not existing:
            matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
            return ESMFold2ComplexCandidate(
                chains=[ESMFold2Chain(chain_id=fields.pop("chain_id", "A"), logits=matrix, **fields)]
            )

        # Rebuild the complex as given. A mapping places each distribution on the chain whose
        # path it was reported under; a bare matrix goes on the first designed chain.
        converted: dict[int, ProteinSequenceLogits] = {}
        if isinstance(probabilities, Mapping):
            for path, rows in probabilities.items():
                index = int(path.split("[", 1)[1].split("]", 1)[0])
                converted[index] = ProteinSequenceLogits.from_probabilities(rows, temperature=temperature)
        else:
            first = next(index for index, chain in enumerate(existing) if chain.designable)
            converted[first] = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)

        rebuilt = [
            chain.model_copy(update={"logits": converted[index]}) if index in converted else chain
            for index, chain in enumerate(existing)
        ]
        return ESMFold2ComplexCandidate(chains=rebuilt)


class ESMFold2InversionGradientMetrics(Metrics):
    """Per-candidate metrics for the general inversion tool.

    Metrics documented in ``metric_spec``:
        complex_length (int): Residues in the folded complex.
        designable_length (int): Rows the returned gradient covers.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "complex_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Residues in the folded complex, across every chain.",
            "better_values_are": "context-dependent",
        },
        "designable_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Rows the returned gradient covers, summed over designable chains.",
            "better_values_are": "context-dependent",
        },
    }

    primary_metric: str | None = Field(
        default=None,
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class ESMFold2ChainTermGradient(TermGradient):
    """One per-chain quantity and its gradient.

    Attributes:
        term (ESMFold2ChainTermType): The quantity this gradient answers for.
    """

    term: ESMFold2ChainTermType = Field(title="Term", description="Per-chain quantity this gradient answers.")


class ESMFold2InversionGradientResult(GradientResult):
    """One evaluated complex and every per-chain quantity differentiated for it.

    Each term's gradient covers the designable chains' rows concatenated in fold order;
    ``chain_rows`` says where each chain's rows begin and end.

    Attributes:
        terms (list[ESMFold2ChainTermGradient]): One entry per requested quantity, in order.
        chain_rows (dict[str, list[int]]): Chain id to its ``[start, stop)`` rows in the gradient.
    """

    terms: list[ESMFold2ChainTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Gradient Terms",
        description="One gradient per requested quantity, in the order requested.",
    )
    chain_rows: dict[str, list[int]] = Field(
        default_factory=dict,
        title="Chain Rows",
        description="Chain id to the [start, stop) rows it occupies in each gradient.",
    )


class ESMFold2InversionGradientOutput(GradientOutput):
    """General ESMFold2 inversion output: one result per complex.

    Attributes:
        results (list[ESMFold2InversionGradientResult]): One per input complex, in input order.
        vocab (list[str]): Amino-acid column ordering for the logits and gradients.
    """

    results: list[ESMFold2InversionGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input complex, in input order, each carrying its terms.",
    )


class ESMFold2InversionGradientConfig(ESMFold2InversionConfigBase):
    """Configuration for the general ESMFold2 inversion gradient tool.

    Attributes:
        gradient_terms (list[ESMFold2ChainTermType]): Per-chain quantities to differentiate,
            given as catalog names or as instances naming a ``chain_id``.
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
    }

    gradient_terms: list[ESMFold2ChainTermType] = ConfigField(
        title="Gradient Terms",
        default_factory=lambda: [ESMFold2IntraContactTerm(), ESMFold2GlobularityTerm()],
        min_length=1,
        description="Per-chain quantities to differentiate, by catalog name or as instances",
    )

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept catalog names and term classes alongside instances."""
        return resolve_catalog_names(value, cls.supported_terms())

    @model_validator(mode="after")
    def reject_repeated_terms(self) -> Self:
        """One quantity on one chain is one question; asking twice is a mistake."""
        asked = [(term.id, term.chain_id) for term in self.gradient_terms]
        repeated = sorted({item for item in asked if asked.count(item) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same quantity and chain more than once: {repeated}")
        return self


def example_input() -> ESMFold2InversionGradientInput:
    """Build a two-chain complex, one fixed and one designed, for the registry's example surface.

    Returns:
        ESMFold2InversionGradientInput: One small complex with a single designable chain.
    """
    return ESMFold2InversionGradientInput(
        inputs=[
            ESMFold2ComplexCandidate(
                chains=[
                    ESMFold2Chain(chain_id="A", sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"),
                    ESMFold2Chain(chain_id="B", logits=ProteinSequenceLogits([[0.0] * 20] * 24)),
                ]
            )
        ]
    )


def _resolve_chain(term: ESMFold2ChainTermType, candidate: ESMFold2ComplexCandidate) -> str:
    """Name the chain a term evaluates, defaulting to the only designable one.

    Args:
        term (ESMFold2ChainTermType): The quantity being placed.
        candidate (ESMFold2ComplexCandidate): The complex it is evaluated over.

    Returns:
        str: The chain id to evaluate.

    Raises:
        ValueError: If the term names no chain and the choice is ambiguous, or names a chain the
            complex does not have.
    """
    known = {chain.chain_id for chain in candidate.chains}
    if term.chain_id is None:
        designable = candidate.designable_chains
        if len(designable) != 1:
            raise ValueError(
                f"term {term.id!r} must name a chain_id: this complex designs "
                f"{sorted(chain.chain_id for chain in designable)}"
            )
        return designable[0].chain_id
    if term.chain_id not in known:
        raise ValueError(f"term {term.id!r} names chain {term.chain_id!r}, which this complex does not have")
    return term.chain_id


@tool(
    key="esmfold2-inversion-gradient",
    label="ESMFold2 Inversion Gradient",
    category="structure_prediction",
    input_class=ESMFold2InversionGradientInput,
    config_class=ESMFold2InversionGradientConfig,
    output_class=ESMFold2InversionGradientOutput,
    metrics_class=ESMFold2InversionGradientMetrics,
    description="Differentiate per-chain ESMFold2 inversion objectives over a protein complex",
    uses_gpu=True,
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
    """Differentiate per-chain ESMFold2 objectives over a complex, with respect to its design.

    The complex is folded once per candidate. Each requested quantity is evaluated over the
    chain it names and differentiated separately, and every gradient covers the designable
    chains' rows concatenated in fold order.

    Args:
        inputs (ESMFold2InversionGradientInput): Complexes to fold and differentiate.
        config (ESMFold2InversionGradientConfig): Which quantities to evaluate, and how.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        ESMFold2InversionGradientOutput: One result per complex, in input order.
    """
    results = []
    for index, candidate in enumerate(inputs.inputs):
        chains = [
            {
                "chain_id": chain.chain_id,
                "sequence": chain.sequence,
                "logits": None if chain.logits is None else chain.logits.root,
            }
            for chain in candidate.chains
        ]
        requested = [{"id": term.id, "chain_id": _resolve_chain(term, candidate)} for term in config.gradient_terms]
        reply = ToolInstance.dispatch(
            "esmfold2",
            {
                "operation": "complex_gradient",
                "chains": chains,
                "alphabet": candidate.alphabet,
                "gradient_terms": requested,
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
                # One seed per complex, so duplicates in a batch draw differently (stochastic=True).
                "seed": derive_item_seed(config.seed, index, RANDOM_SEED_UPPER_BOUND),
                "device": config.device,
                "verbose": config.verbose,
            },
            instance=instance,
            config=config,
        )
        designable = candidate.designable_chains
        basis = basis_metadata(
            [chain.logits.root for chain in designable if chain.logits is not None],
            candidate.alphabet,
            inputs.temperature,
            config.soft,
            config.hard,
        )
        catalog = cast(dict[str, type[ESMFold2ChainTermType]], ESMFold2InversionGradientConfig.supported_terms())
        results.append(
            ESMFold2InversionGradientResult(
                terms=[
                    ESMFold2ChainTermGradient(
                        term=catalog[entry["id"]](chain_id=entry["chain_id"]),
                        term_value=float(entry["value"]),
                        gradient=entry["gradient"],
                        metadata=basis,
                    )
                    for entry in reply["terms"]
                ],
                chain_rows=reply["chain_rows"],
                metrics=reply["metrics"],
            )
        )
    return ESMFold2InversionGradientOutput(results=results, vocab=list(inputs.inputs[0].alphabet))
