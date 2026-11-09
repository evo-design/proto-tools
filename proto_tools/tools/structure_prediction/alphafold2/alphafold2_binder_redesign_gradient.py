"""proto_tools/tools/structure_prediction/alphafold2/alphafold2_binder_redesign_gradient.py.

Differentiable AlphaFold2 template-based binder redesign.

The binder is an existing chain of a template complex (for example a nanobody docked on its
target). ColabDesign's binder protocol is prepared with that chain as ``binder_chain``, so its
backbone is offered as a template while its sequence and side chains are masked, and the designed
segment is the whole chain. Which residues may actually change is the caller's decision.

Under redesign ColabDesign parses the binder from the template, so its unsupervised masks are
empty and its own ``plddt``, ``pae`` and ``con`` would read zero; the worker computes those three
over the binder instead. ``exp_res``, ``helix`` and ``rmsd`` stay masked to nothing here, so this
tool does not offer them. ColabDesign's redesign defaults (``dgram_cce=1``, the rest 0) weight
only its fused loss, which this tool never reports: each term's value and gradient is taken from
the unweighted loss dictionary.
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
    AlphaFold2DgramCCETerm,
    AlphaFold2FapeTerm,
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


#: The quantities ColabDesign's binder protocol computes when redesigning a template chain.
AlphaFold2BinderRedesignTerm = Annotated[
    AlphaFold2PlddtTerm
    | AlphaFold2PaeTerm
    | AlphaFold2ContactTerm
    | AlphaFold2InterfacePaeTerm
    | AlphaFold2InterfaceContactTerm
    | AlphaFold2RgTerm
    | AlphaFold2TerminiDistanceTerm
    | AlphaFold2HelixTerm
    | AlphaFold2BetaStrandTerm
    | AlphaFold2InterfacePtmTerm
    | AlphaFold2FrameworkContactTerm
    | AlphaFold2DgramCCETerm
    | AlphaFold2FapeTerm,
    Field(discriminator="id"),
]

_CATALOG: TermCatalog = {
    "plddt": AlphaFold2PlddtTerm,
    "pae": AlphaFold2PaeTerm,
    "con": AlphaFold2ContactTerm,
    "i_pae": AlphaFold2InterfacePaeTerm,
    "i_con": AlphaFold2InterfaceContactTerm,
    "rg": AlphaFold2RgTerm,
    "NC": AlphaFold2TerminiDistanceTerm,
    "helix": AlphaFold2HelixTerm,
    "beta_strand": AlphaFold2BetaStrandTerm,
    "i_ptm": AlphaFold2InterfacePtmTerm,
    "fw_i_con": AlphaFold2FrameworkContactTerm,
    "dgram_cce": AlphaFold2DgramCCETerm,
    "fape": AlphaFold2FapeTerm,
}


class AlphaFold2BinderRedesignGradientCandidate(BaseModel):
    """One template binder chain redesigned against its target.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed ``(L_binder, 20)`` state covering every residue of
            ``binder_chain``, in canonical amino-acid order.
        template_pdb (Structure): Complex holding the target and the binder template.
        target_chain (str): Chain or comma-separated chains defining the target.
        binder_chain (str): Chain of ``template_pdb`` being redesigned.
        target_hotspot (str | None): Optional target residue selection, in ColabDesign's
            convention, biasing the interface toward those residues.
    """

    model_config = ConfigDict(extra="forbid")

    logits: ProteinSequenceLogits = Field(title="Logits", description="Relaxed (L, 20) state for the binder chain.")
    template_pdb: Structure = Field(title="Template Structure", description="Complex of the target and binder.")
    target_chain: str = Field(default="A", title="Target Chain", description="Chain or chains defining the target.")
    binder_chain: str = Field(default="B", title="Binder Chain", description="Template chain being redesigned.")
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
        """Require the named chains to exist and differ, and store the form ColabDesign parses."""
        present = set(self.template_pdb.get_chain_ids())
        named = [part.strip() for part in self.target_chain.split(",") if part.strip()]
        missing = [part for part in [*named, self.binder_chain] if part not in present]
        if missing:
            raise ValueError(f"chains {missing} are absent from template_pdb (available: {sorted(present)}).")
        if self.binder_chain in named:
            raise ValueError(f"binder_chain {self.binder_chain!r} is also named as a target chain.")
        # ColabDesign splits on "," without stripping, so only the normalized form is safe to send.
        self.target_chain = ",".join(named)
        return self


class AlphaFold2BinderRedesignGradientInput(GradientInput):
    """Template binder chains and the targets they are redesigned against.

    Attributes:
        inputs (list[AlphaFold2BinderRedesignGradientCandidate]): One template complex per
            candidate.
        temperature (float | None): Inherited softmax temperature; see :class:`GradientInput`.
    """

    inputs: list[AlphaFold2BinderRedesignGradientCandidate] = InputField(
        title="Candidates",
        description="One relaxed binder sequence with its template complex per candidate",
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
    ) -> "AlphaFold2BinderRedesignGradientCandidate":
        """Build one candidate pairing the converted binder matrix with its template complex.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``template_pdb``.

        Returns:
            AlphaFold2BinderRedesignGradientCandidate: Candidate whose logits reproduce
                ``probabilities``.

        Raises:
            ValueError: If ``template_pdb`` is missing.
        """
        if "template_pdb" not in fields:
            raise ValueError("alphafold2-binder-redesign-gradient from_probabilities requires the template structure")
        matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return AlphaFold2BinderRedesignGradientCandidate(logits=matrix, **fields)


class AlphaFold2BinderRedesignTermGradient(TermGradient):
    """One redesign quantity and its raw gradient.

    Attributes:
        term (AlphaFold2BinderRedesignTerm): The quantity this gradient answers.
    """

    term: AlphaFold2BinderRedesignTerm = Field(title="Term", description="Quantity this gradient answers.")


class AlphaFold2BinderRedesignGradientResult(StructureGradientResult):
    """One redesigned binder: its quantities, metrics and predicted complex.

    Attributes:
        terms (list[AlphaFold2BinderRedesignTermGradient]): One per requested quantity, in order.
    """

    terms: list[AlphaFold2BinderRedesignTermGradient] = Field(  # type: ignore[assignment]
        default_factory=list,
        title="Terms",
        description="One gradient per requested quantity, in the order requested.",
    )


class AlphaFold2BinderRedesignGradientOutput(AlphaFold2StructureGradientOutput):
    """Differentiable AlphaFold2 template-redesign output.

    Attributes:
        results (list[AlphaFold2BinderRedesignGradientResult]): One per candidate, in input order.
        vocab (list[str]): Amino-acid column ordering for logits and gradients.
    """

    results: list[AlphaFold2BinderRedesignGradientResult] = Field(  # type: ignore[assignment]
        title="Results",
        description="One result per input candidate, in input order.",
    )


class AlphaFold2BinderRedesignGradientConfig(AlphaFold2DesignGradientConfig):
    """Configuration for differentiable AlphaFold2 template-based binder redesign.

    Inter-chain template contacts are always masked: the interface terms are only measured over
    pairs the template does not supply, so revealing them would leave ``i_pae`` and ``i_con``
    with nothing to measure.

    Attributes:
        gradient_terms (list[AlphaFold2BinderRedesignTerm]): Quantities to differentiate, by name,
            class or instance.
        use_multimer (bool): Run the binder protocol on AlphaFold-Multimer parameters.
        rm_target_seq (bool): Mask the target's template sequence during prep.
        rm_target_sc (bool): Mask the target's template side chains during prep.
        rm_binder (bool): Mask the binder's template backbone during prep.
        rm_binder_seq (bool): Mask the binder's template sequence during prep.
        rm_binder_sc (bool): Mask the binder's template side chains during prep.
        compute_gradient (bool): Inherited backward-pass switch; see
            :class:`AlphaFold2DesignGradientConfig`.
        num_recycles (int): Inherited recycling iterations; see :class:`AlphaFold2DesignGradientConfig`.
        recycle_mode (Literal["last", "sample", "average", "first"]): Inherited recycle selection;
            see :class:`AlphaFold2DesignGradientConfig`.
        model_num (int): Inherited parameter set; see :class:`AlphaFold2DesignGradientConfig`.
        sample_models (bool): Inherited random parameter choice; see
            :class:`AlphaFold2DesignGradientConfig`.
        dropout (bool): Inherited gradient-pass dropout switch; see
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

    gradient_terms: list[AlphaFold2BinderRedesignTerm] = ConfigField(
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
        title="Mask Target Sequence", default=True, description="Mask the target's template sequence during prep"
    )
    rm_target_sc: bool = ConfigField(
        title="Mask Target Side Chains", default=False, description="Mask the target's template side chains during prep"
    )
    rm_binder: bool = ConfigField(
        title="Mask Binder Backbone", default=False, description="Mask the binder's template backbone during prep"
    )
    rm_binder_seq: bool = ConfigField(
        title="Mask Binder Sequence", default=True, description="Mask the binder's template sequence during prep"
    )
    rm_binder_sc: bool = ConfigField(
        title="Mask Binder Side Chains", default=True, description="Mask the binder's template side chains during prep"
    )


def example_input() -> AlphaFold2BinderRedesignGradientInput:
    """Minimal valid input for testing and examples."""
    template = Structure.from_file(_AF2_FIXTURE_PDB)
    binder = template.get_chain_sequence("B")
    return AlphaFold2BinderRedesignGradientInput(
        inputs=[
            AlphaFold2BinderRedesignGradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits(binder, sharpness=2.0)),
                template_pdb=template,
                target_chain="A",
                binder_chain="B",
            )
        ]
    )


@tool(
    key="alphafold2-binder-redesign-gradient",
    label="AlphaFold2 Binder Redesign Gradient",
    category="structure_prediction",
    input_class=AlphaFold2BinderRedesignGradientInput,
    config_class=AlphaFold2BinderRedesignGradientConfig,
    output_class=AlphaFold2BinderRedesignGradientOutput,
    metrics_class=AlphaFold2Metrics,
    description="Differentiate AlphaFold2 structure quantities for a template binder chain against its target",
    uses_gpu=True,
    stochastic=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
)
def run_alphafold2_binder_redesign_gradient(
    inputs: AlphaFold2BinderRedesignGradientInput,
    config: AlphaFold2BinderRedesignGradientConfig,
    instance: Any = None,
) -> AlphaFold2BinderRedesignGradientOutput:
    """Differentiate AlphaFold2 quantities for template binder chains against their targets."""
    # ColabDesign reads structures from disk, so each temp file must outlive the dispatch call.
    with ExitStack() as stack:
        items: list[dict[str, Any]] = []
        for item in inputs.inputs:
            prep_kwargs: dict[str, Any] = {
                "pdb_filename": str(stack.enter_context(item.template_pdb.temp_file())),
                "target_chain": item.target_chain,
                "binder_chain": item.binder_chain,
                "rm_target_seq": config.rm_target_seq,
                "rm_target_sc": config.rm_target_sc,
                "rm_binder": config.rm_binder,
                "rm_binder_seq": config.rm_binder_seq,
                "rm_binder_sc": config.rm_binder_sc,
                "rm_template_ic": True,
            }
            if item.target_hotspot:
                prep_kwargs["hotspot"] = item.target_hotspot
            items.append({"logits": item.logits.root, "prep_kwargs": prep_kwargs})

        worker_results = dispatch_design_gradient(
            protocol="binder",
            items=items,
            config=config,
            temperature=inputs.temperature,
            tool_key="alphafold2-binder-redesign-gradient",
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
            AlphaFold2BinderRedesignGradientResult(
                terms=[AlphaFold2BinderRedesignTermGradient(**kwargs) for kwargs in built],
                metrics=reply["metrics"],
                structure=build_structure(reply, source="alphafold2-binder-redesign-gradient"),
            )
        )
    return AlphaFold2BinderRedesignGradientOutput(results=results, vocab=list(PROTEIN_AMINO_ACIDS))
