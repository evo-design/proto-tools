"""proto_tools/tools/structure_prediction/alphafold2/alphafold2_binder_gradient.py.

Differentiable AlphaFold2 de novo binder design against a fixed target structure.
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
    AlphaFold2BetaStrandTerm,
    AlphaFold2ContactTerm,
    AlphaFold2DesignGradientConfig,
    AlphaFold2ExpResTerm,
    AlphaFold2FrameworkContactTerm,
    AlphaFold2HelixTerm,
    AlphaFold2InterfaceContactTerm,
    AlphaFold2InterfacePaeTerm,
    AlphaFold2InterfacePtmTerm,
    AlphaFold2PaeTerm,
    AlphaFold2PlddtTerm,
    AlphaFold2RgTerm,
    AlphaFold2StructureGradientOutput,
    AlphaFold2TerminiDistanceTerm,
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


#: The quantities ColabDesign's binder protocol computes against a fixed target.
AlphaFold2BinderTerm = Annotated[
    AlphaFold2PlddtTerm
    | AlphaFold2PaeTerm
    | AlphaFold2ContactTerm
    | AlphaFold2ExpResTerm
    | AlphaFold2HelixTerm
    | AlphaFold2InterfacePaeTerm
    | AlphaFold2InterfaceContactTerm
    | AlphaFold2RgTerm
    | AlphaFold2TerminiDistanceTerm
    | AlphaFold2BetaStrandTerm
    | AlphaFold2InterfacePtmTerm
    | AlphaFold2FrameworkContactTerm,
    Field(discriminator="id"),
]

_CATALOG: TermCatalog = {
    "plddt": AlphaFold2PlddtTerm,
    "pae": AlphaFold2PaeTerm,
    "con": AlphaFold2ContactTerm,
    "exp_res": AlphaFold2ExpResTerm,
    "helix": AlphaFold2HelixTerm,
    "i_pae": AlphaFold2InterfacePaeTerm,
    "i_con": AlphaFold2InterfaceContactTerm,
    "rg": AlphaFold2RgTerm,
    "NC": AlphaFold2TerminiDistanceTerm,
    "beta_strand": AlphaFold2BetaStrandTerm,
    "i_ptm": AlphaFold2InterfacePtmTerm,
    "fw_i_con": AlphaFold2FrameworkContactTerm,
}


class AlphaFold2BinderGradientCandidate(BaseModel):
    """One de novo binder designed against a fixed target structure.

    The binder's length is the number of rows in ``logits``; it has no template and no reference
    coordinates, which is what keeps the confidence quantities live.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed ``(L_binder, 20)`` state for the binder, in
            canonical amino-acid order.
        target_pdb (Structure): Fixed target structure.
        target_chain (str): Chain or comma-separated chains defining the target.
        target_hotspot (str | None): Optional target residue selection, in ColabDesign's
            convention, biasing the interface toward those residues.
    """

    model_config = ConfigDict(extra="forbid")

    logits: ProteinSequenceLogits = Field(title="Logits", description="Relaxed (L, 20) state for the de novo binder.")
    target_pdb: Structure = Field(title="Target Structure", description="Fixed target the binder is designed against.")
    target_chain: str = Field(
        default="A", title="Target Chain", description="Chain or chains defining the target in target_pdb."
    )
    target_hotspot: str | None = Field(
        default=None, title="Target Hotspot", description="Optional target residues to bias the interface toward."
    )

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this candidate's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this candidate's matrix.
        """
        yield "logits", self.logits

    @model_validator(mode="after")
    def validate_chains(self) -> Self:
        """Require every named target chain to exist, and store the form ColabDesign parses."""
        present = set(self.target_pdb.get_chain_ids())
        named = [part.strip() for part in self.target_chain.split(",") if part.strip()]
        missing = [part for part in named if part not in present]
        if missing:
            raise ValueError(f"target_chain names {missing} absent from target_pdb (available: {sorted(present)}).")
        # ColabDesign splits on "," without stripping, so only the normalized form is safe to send.
        self.target_chain = ",".join(named)
        return self


class AlphaFold2BinderGradientInput(GradientInput):
    """Binder sequences and the fixed targets they are designed against.

    Attributes:
        inputs (list[AlphaFold2BinderGradientCandidate]): One binder-and-target pair per
            candidate.
        temperature (float | None): Inherited softmax temperature; see :class:`GradientInput`.
    """

    inputs: list[AlphaFold2BinderGradientCandidate] = InputField(
        title="Candidates",
        description="One relaxed binder sequence with its fixed target per candidate",
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
    ) -> "AlphaFold2BinderGradientCandidate":
        """Build one candidate pairing the converted binder matrix with its fixed target.

        The target belongs to the candidate, so it and its selections arrive here either shared
        across the call or named in this candidate's own mapping.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``target_pdb``.

        Returns:
            AlphaFold2BinderGradientCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``target_pdb`` is missing.
        """
        if "target_pdb" not in fields:
            raise ValueError("alphafold2-binder-gradient from_probabilities requires the target structure")
        matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return AlphaFold2BinderGradientCandidate(logits=matrix, **fields)


class AlphaFold2BinderTermGradient(TermGradient):
    """One binder quantity and its raw gradient.

    Attributes:
        term (AlphaFold2BinderTerm): The quantity this gradient answers.
    """

    term: AlphaFold2BinderTerm = Field(title="Term", description="Quantity this gradient answers.")


class AlphaFold2BinderGradientResult(StructureGradientResult):
    """One de novo binder: its quantities, metrics and predicted complex.

    Attributes:
        terms (list[AlphaFold2BinderTermGradient]): One per requested quantity, in order.
    """

    terms: list[AlphaFold2BinderTermGradient] = Field(  # type: ignore[assignment]
        default_factory=list,
        title="Terms",
        description="One gradient per requested quantity, in the order requested.",
    )


class AlphaFold2BinderGradientOutput(AlphaFold2StructureGradientOutput):
    """Differentiable AlphaFold2 binder output.

    Attributes:
        results (list[AlphaFold2BinderGradientResult]): One per candidate, in input order.
        vocab (list[str]): Amino-acid column ordering for logits and gradients.
    """

    results: list[AlphaFold2BinderGradientResult] = Field(  # type: ignore[assignment]
        title="Results",
        description="One result per input candidate, in input order.",
    )


class AlphaFold2BinderGradientConfig(AlphaFold2DesignGradientConfig):
    """Configuration for differentiable AlphaFold2 binder design.

    Attributes:
        gradient_terms (list[AlphaFold2BinderTerm]): Quantities to differentiate, by name,
            class or instance.
        use_multimer (bool): Run the binder protocol on AlphaFold-Multimer parameters.
        rm_target_seq (bool): Mask the target's template sequence during prep.
        rm_target_sc (bool): Mask the target's template side chains during prep.
        rm_template_ic (bool): Mask inter-chain template contacts during prep.
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
        soft (float): Inherited relaxation blend; see :class:`RelaxationConfig`.
        hard (float): Inherited straight-through coefficient.
        device (str): Inherited device; see :class:`AlphaFold2DesignGradientConfig`.
    """

    TERMS: ClassVar[TermCatalog] = _CATALOG

    gradient_terms: list[AlphaFold2BinderTerm] = ConfigField(
        default_factory=lambda: [AlphaFold2PlddtTerm()],
        min_length=1,
        title="Gradient Terms",
        description="Quantities to differentiate, by name, class or instance",
    )
    use_multimer: bool = ConfigField(
        title="Use Multimer",
        default=True,
        description="Use AlphaFold multimer parameters for the binder protocol",
        reload_on_change=True,
    )
    rm_target_seq: bool = ConfigField(
        title="Mask Target Sequence",
        default=True,
        description="Mask the target's template sequence during prep",
    )
    rm_target_sc: bool = ConfigField(
        title="Mask Target Side Chains",
        default=False,
        description="Mask the target's template side chains during prep",
    )
    rm_template_ic: bool = ConfigField(
        title="Mask Inter-chain Contacts",
        default=True,
        description="Mask inter-chain template contacts during prep",
    )


def example_input() -> AlphaFold2BinderGradientInput:
    """Minimal valid input for testing and examples."""
    return AlphaFold2BinderGradientInput(
        inputs=[
            AlphaFold2BinderGradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits("A" * 12, sharpness=2.0)),
                target_pdb=Structure.from_file(_AF2_FIXTURE_PDB),
                target_chain="A",
            )
        ]
    )


@tool(
    key="alphafold2-binder-gradient",
    label="AlphaFold2 Binder Gradient",
    category="structure_prediction",
    input_class=AlphaFold2BinderGradientInput,
    config_class=AlphaFold2BinderGradientConfig,
    output_class=AlphaFold2BinderGradientOutput,
    metrics_class=AlphaFold2Metrics,
    description="Differentiate AlphaFold2 structure quantities for a de novo binder against a target",
    uses_gpu=True,
    stochastic=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
)
def run_alphafold2_binder_gradient(
    inputs: AlphaFold2BinderGradientInput,
    config: AlphaFold2BinderGradientConfig,
    instance: Any = None,
) -> AlphaFold2BinderGradientOutput:
    """Differentiate AlphaFold2 quantities for de novo binders against a fixed target."""
    # ColabDesign reads structures from disk, so each temp file must outlive the dispatch call.
    with ExitStack() as stack:
        items: list[dict[str, Any]] = []
        for item in inputs.inputs:
            prep_kwargs: dict[str, Any] = {
                "pdb_filename": str(stack.enter_context(item.target_pdb.temp_file())),
                "target_chain": item.target_chain,
                "binder_len": len(item.logits),
                "rm_target_seq": config.rm_target_seq,
                "rm_target_sc": config.rm_target_sc,
                "rm_template_ic": config.rm_template_ic,
            }
            if item.target_hotspot:
                prep_kwargs["hotspot"] = item.target_hotspot
            items.append({"logits": item.logits.root, "prep_kwargs": prep_kwargs})

        worker_results = dispatch_design_gradient(
            protocol="binder",
            items=items,
            config=config,
            temperature=inputs.temperature,
            tool_key="alphafold2-binder-gradient",
            extra_payload={"use_multimer": config.use_multimer},
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
            AlphaFold2BinderGradientResult(
                terms=[AlphaFold2BinderTermGradient(**kwargs) for kwargs in built],
                metrics=reply["metrics"],
                structure=build_structure(reply, source="alphafold2-binder-gradient"),
            )
        )
    return AlphaFold2BinderGradientOutput(results=results, vocab=list(PROTEIN_AMINO_ACIDS))
