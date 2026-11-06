"""proto_tools/tools/structure_prediction/alphafold2/alphafold2_fixbb_gradient.py.

Differentiable AlphaFold2 fixed-backbone design: agreement between a sequence and a reference fold.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import AlphaFold2Metrics
from proto_tools.tools.structure_prediction.alphafold2.gradient_helpers import (
    AlphaFold2ContactTerm,
    AlphaFold2DesignGradientConfig,
    AlphaFold2DgramCCETerm,
    AlphaFold2ExpResTerm,
    AlphaFold2FapeTerm,
    AlphaFold2HelixTerm,
    AlphaFold2PaeTerm,
    AlphaFold2PlddtTerm,
    AlphaFold2RmsdTerm,
    AlphaFold2StructureGradientOutput,
    build_structure,
    build_terms,
    dispatch_design_gradient,
)
from proto_tools.tools.structure_prediction.shared_data_models import StructureGradientResult
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    ConfigField,
    GradientInput,
    InputField,
    ProteinSequenceLogits,
    SequenceLogits,
    TermCatalog,
    TermGradient,
    coerce_to_item_list,
    one_hot_protein_logits,
)

logger = logging.getLogger(__name__)

_AF2_FIXTURE_PDB = Path(__file__).resolve().parent / "example_input_fixture.pdb"


#: The quantities ColabDesign's fixbb protocol computes against a reference backbone.
AlphaFold2FixBBTerm = Annotated[
    AlphaFold2PlddtTerm
    | AlphaFold2PaeTerm
    | AlphaFold2ContactTerm
    | AlphaFold2ExpResTerm
    | AlphaFold2HelixTerm
    | AlphaFold2DgramCCETerm
    | AlphaFold2FapeTerm
    | AlphaFold2RmsdTerm,
    Field(discriminator="id"),
]

_CATALOG: TermCatalog = {
    "plddt": AlphaFold2PlddtTerm,
    "pae": AlphaFold2PaeTerm,
    "con": AlphaFold2ContactTerm,
    "exp_res": AlphaFold2ExpResTerm,
    "helix": AlphaFold2HelixTerm,
    "dgram_cce": AlphaFold2DgramCCETerm,
    "fape": AlphaFold2FapeTerm,
    "rmsd": AlphaFold2RmsdTerm,
}


class AlphaFold2FixBBGradientCandidate(BaseModel):
    """One sequence scored against the backbone it is meant to reproduce.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed ``(L, 20)`` state for the chain being redesigned,
            in canonical amino-acid order.
        reference_pdb (Structure): Structure holding the backbone to reproduce.
        reference_chain (str): Which chain of ``reference_pdb`` defines that backbone.
    """

    model_config = ConfigDict(extra="forbid")

    logits: ProteinSequenceLogits = Field(title="Logits", description="Relaxed (L, 20) state for the redesigned chain.")
    reference_pdb: Structure = Field(
        title="Reference Structure", description="Structure whose backbone the sequence should reproduce."
    )
    reference_chain: str = Field(
        default="A", title="Reference Chain", description="Chain of the reference defining the backbone."
    )

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this candidate's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this candidate's matrix.
        """
        yield "logits", self.logits

    @model_validator(mode="after")
    def validate_chain(self) -> Self:
        """Require the named chain to exist in the reference structure."""
        present = set(self.reference_pdb.get_chain_ids())
        if self.reference_chain not in present:
            raise ValueError(
                f"reference_chain={self.reference_chain!r} is not a chain in reference_pdb "
                f"(available: {sorted(present)})."
            )
        return self


class AlphaFold2FixBBGradientInput(GradientInput):
    """Sequences and the reference backbones they are scored against.

    Attributes:
        inputs (list[AlphaFold2FixBBGradientCandidate]): One sequence-and-reference pair per
            candidate.
        temperature (float | None): Inherited softmax temperature; see :class:`GradientInput`.
    """

    inputs: list[AlphaFold2FixBBGradientCandidate] = InputField(
        title="Candidates",
        description="One relaxed sequence with its reference backbone per candidate",
        min_length=1,
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one candidate as well as a list of them."""
        return coerce_to_item_list(value)

    @classmethod
    def build_candidate(
        cls,
        probabilities: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> AlphaFold2FixBBGradientCandidate:
        """Build one candidate pairing the converted matrix with the backbone it must reproduce.

        The reference belongs to the candidate, so it and its chain selection arrive here either
        shared across the call or named in this candidate's own mapping.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``reference_pdb``.

        Returns:
            AlphaFold2FixBBGradientCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``reference_pdb`` is missing.
        """
        if "reference_pdb" not in fields:
            raise ValueError("alphafold2-fixbb-gradient from_probabilities requires the reference structure")
        matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return AlphaFold2FixBBGradientCandidate(logits=matrix, **fields)


class AlphaFold2FixBBTermGradient(TermGradient):
    """One fixbb quantity and its raw gradient.

    Attributes:
        term (AlphaFold2FixBBTerm): The quantity this gradient answers.
    """

    term: AlphaFold2FixBBTerm = Field(title="Term", description="Quantity this gradient answers.")


class AlphaFold2FixBBGradientResult(StructureGradientResult):
    """One redesigned sequence: its quantities, metrics and predicted structure.

    Attributes:
        terms (list[AlphaFold2FixBBTermGradient]): One per requested quantity, in order.
    """

    terms: list[AlphaFold2FixBBTermGradient] = Field(  # type: ignore[assignment]
        default_factory=list,
        title="Terms",
        description="One gradient per requested quantity, in the order requested.",
    )


class AlphaFold2FixBBGradientOutput(AlphaFold2StructureGradientOutput):
    """Differentiable AlphaFold2 fixbb output.

    Attributes:
        results (list[AlphaFold2FixBBGradientResult]): One per candidate, in input order.
        vocab (list[str]): Amino-acid column ordering for logits and gradients.
    """

    results: list[AlphaFold2FixBBGradientResult] = Field(  # type: ignore[assignment]
        title="Results",
        description="One result per input candidate, in input order.",
    )


class AlphaFold2FixBBGradientConfig(AlphaFold2DesignGradientConfig):
    """Configuration for differentiable AlphaFold2 fixbb.

    Attributes:
        gradient_terms (list[AlphaFold2FixBBTerm]): Quantities to differentiate, by name,
            class or instance.
        compute_gradient (bool): Inherited backward-pass switch; see
            :class:`AlphaFold2DesignGradientConfig`.
        num_recycles (int): Inherited recycling iterations; see :class:`AlphaFold2DesignGradientConfig`.
        recycle_mode (Literal["last", "sample", "average", "first"]): Inherited recycle selection;
            see :class:`AlphaFold2DesignGradientConfig`.
        model_num (int): Inherited parameter set; see :class:`AlphaFold2DesignGradientConfig`.
        sample_models (bool): Inherited random parameter choice; see
            :class:`AlphaFold2DesignGradientConfig`.
        include_pae_matrix (bool): Inherited PAE matrix switch; see
            :class:`AlphaFold2DesignGradientConfig`.
        excluded_amino_acids (list[AminoAcid] | None): Inherited softmax-branch exclusions; see
            :class:`AlphaFold2DesignGradientConfig`.
        soft (float): Inherited relaxation blend; see :class:`RelaxationConfig`.
        hard (float): Inherited straight-through coefficient.
        device (str): Inherited device; see :class:`AlphaFold2DesignGradientConfig`.
    """

    TERMS: ClassVar[TermCatalog] = _CATALOG

    gradient_terms: list[AlphaFold2FixBBTerm] = ConfigField(
        default_factory=lambda: [AlphaFold2PlddtTerm()],
        min_length=1,
        title="Gradient Terms",
        description="Quantities to differentiate, by name, class or instance",
    )


def example_input() -> AlphaFold2FixBBGradientInput:
    """Minimal valid input for testing and examples."""
    reference = Structure.from_file(_AF2_FIXTURE_PDB)
    length = len(reference.get_chain_sequence("B"))
    return AlphaFold2FixBBGradientInput(
        inputs=[
            AlphaFold2FixBBGradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits("A" * length, sharpness=2.0)),
                reference_pdb=reference,
                reference_chain="B",
            )
        ]
    )


@tool(
    key="alphafold2-fixbb-gradient",
    label="AlphaFold2 FixBB Gradient",
    category="structure_prediction",
    input_class=AlphaFold2FixBBGradientInput,
    config_class=AlphaFold2FixBBGradientConfig,
    output_class=AlphaFold2FixBBGradientOutput,
    metrics_class=AlphaFold2Metrics,
    description="Differentiate AlphaFold2 structure quantities for a sequence against a reference backbone",
    uses_gpu=True,
    stochastic=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
)
def run_alphafold2_fixbb_gradient(
    inputs: AlphaFold2FixBBGradientInput,
    config: AlphaFold2FixBBGradientConfig,
    instance: Any = None,
) -> AlphaFold2FixBBGradientOutput:
    """Differentiate AlphaFold2 quantities for sequences scored against a reference backbone."""
    # ColabDesign reads structures from disk, so each temp file must outlive the dispatch call.
    with ExitStack() as stack:
        items: list[dict[str, Any]] = [
            {
                "logits": item.logits.root,
                "prep_kwargs": {
                    "pdb_filename": str(stack.enter_context(item.reference_pdb.temp_file())),
                    "chain": item.reference_chain,
                },
            }
            for item in inputs.inputs
        ]
        worker_results = dispatch_design_gradient(
            protocol="fixbb",
            items=items,
            config=config,
            temperature=inputs.temperature,
            tool_key="alphafold2-fixbb-gradient",
            instance=instance,
        )

    results = []
    for candidate, reply in zip(inputs.inputs, worker_results, strict=True):
        built = build_terms(
            reply["terms"],
            config.gradient_terms,
            candidate.logits.root,
            temperature=inputs.temperature,
            soft=config.soft,
            hard=config.hard,
        )
        results.append(
            AlphaFold2FixBBGradientResult(
                terms=[AlphaFold2FixBBTermGradient(**kwargs) for kwargs in built],
                metrics=reply["metrics"],
                structure=build_structure(reply, source="alphafold2-fixbb-gradient"),
            )
        )
    return AlphaFold2FixBBGradientOutput(results=results, vocab=list(PROTEIN_AMINO_ACIDS))
