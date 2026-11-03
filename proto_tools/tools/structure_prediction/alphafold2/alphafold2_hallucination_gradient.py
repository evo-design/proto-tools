"""proto_tools/tools/structure_prediction/alphafold2/alphafold2_hallucination_gradient.py.

Differentiable AlphaFold2 hallucination: structure quantities for a sequence with no reference.
"""

import logging
from typing import Annotated, Any, ClassVar

from pydantic import Field, field_validator

from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import AlphaFold2Metrics
from proto_tools.tools.structure_prediction.alphafold2.gradient_helpers import (
    AlphaFold2ContactTerm,
    AlphaFold2DesignGradientConfig,
    AlphaFold2ExpResTerm,
    AlphaFold2HelixTerm,
    AlphaFold2PaeTerm,
    AlphaFold2PlddtTerm,
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
    TermCatalog,
    TermGradient,
    coerce_to_item_list,
    one_hot_protein_logits,
)

logger = logging.getLogger(__name__)

#: The quantities ColabDesign's hallucination protocol computes over a free sequence.
AlphaFold2HallucinationTerm = Annotated[
    AlphaFold2PlddtTerm | AlphaFold2PaeTerm | AlphaFold2ContactTerm | AlphaFold2ExpResTerm | AlphaFold2HelixTerm,
    Field(discriminator="id"),
]

_CATALOG: TermCatalog = {
    "plddt": AlphaFold2PlddtTerm,
    "pae": AlphaFold2PaeTerm,
    "con": AlphaFold2ContactTerm,
    "exp_res": AlphaFold2ExpResTerm,
    "helix": AlphaFold2HelixTerm,
}


class AlphaFold2HallucinationGradientInput(GradientInput):
    """Relaxed protein sequences to hallucinate structures for.

    Attributes:
        inputs (list[ProteinSequenceLogits]): One relaxed ``(L, 20)`` state per candidate in
            canonical amino-acid order. Candidates are independent and may differ in length.
        temperature (float | None): Inherited softmax temperature; see :class:`GradientInput`.
    """

    inputs: list[ProteinSequenceLogits] = InputField(
        title="Relaxed Sequences",
        description="One relaxed (L, 20) protein state per candidate",
        min_length=1,
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one relaxed state as well as a list of them."""
        return coerce_to_item_list(value)


class AlphaFold2HallucinationTermGradient(TermGradient):
    """One hallucination quantity and its raw gradient.

    Attributes:
        term (AlphaFold2HallucinationTerm): The quantity this gradient answers.
    """

    term: AlphaFold2HallucinationTerm = Field(title="Term", description="Quantity this gradient answers.")


class AlphaFold2HallucinationGradientResult(StructureGradientResult):
    """One hallucinated candidate: its quantities, metrics and predicted structure.

    Attributes:
        terms (list[AlphaFold2HallucinationTermGradient]): One per requested quantity, in order.
    """

    terms: list[AlphaFold2HallucinationTermGradient] = Field(  # type: ignore[assignment]
        default_factory=list,
        title="Terms",
        description="One gradient per requested quantity, in the order requested.",
    )


class AlphaFold2HallucinationGradientOutput(AlphaFold2StructureGradientOutput):
    """Differentiable AlphaFold2 hallucination output.

    Attributes:
        results (list[AlphaFold2HallucinationGradientResult]): One per candidate, in input order.
        vocab (list[str]): Amino-acid column ordering for logits and gradients.
    """

    results: list[AlphaFold2HallucinationGradientResult] = Field(  # type: ignore[assignment]
        title="Results",
        description="One result per input candidate, in input order.",
    )


class AlphaFold2HallucinationGradientConfig(AlphaFold2DesignGradientConfig):
    """Configuration for differentiable AlphaFold2 hallucination.

    Attributes:
        gradient_terms (list[AlphaFold2HallucinationTerm]): Quantities to differentiate, by name,
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
        soft (float): Inherited relaxation blend; see :class:`RelaxationConfig`.
        hard (float): Inherited straight-through coefficient.
        device (str): Inherited device; see :class:`AlphaFold2DesignGradientConfig`.
    """

    TERMS: ClassVar[TermCatalog] = _CATALOG

    gradient_terms: list[AlphaFold2HallucinationTerm] = ConfigField(
        default_factory=lambda: [AlphaFold2PlddtTerm()],
        min_length=1,
        title="Gradient Terms",
        description="Quantities to differentiate, by name, class or instance",
    )


def example_input() -> AlphaFold2HallucinationGradientInput:
    """Minimal valid input for testing and examples."""
    return AlphaFold2HallucinationGradientInput(
        inputs=[ProteinSequenceLogits(one_hot_protein_logits("MKTLLILAVVAAALA", sharpness=2.0))]
    )


@tool(
    key="alphafold2-hallucination-gradient",
    label="AlphaFold2 Hallucination Gradient",
    category="structure_prediction",
    input_class=AlphaFold2HallucinationGradientInput,
    config_class=AlphaFold2HallucinationGradientConfig,
    output_class=AlphaFold2HallucinationGradientOutput,
    metrics_class=AlphaFold2Metrics,
    description="Differentiate AlphaFold2 structure quantities for a free protein sequence, with no reference",
    uses_gpu=True,
    stochastic=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
)
def run_alphafold2_hallucination_gradient(
    inputs: AlphaFold2HallucinationGradientInput,
    config: AlphaFold2HallucinationGradientConfig,
    instance: Any = None,
) -> AlphaFold2HallucinationGradientOutput:
    """Differentiate AlphaFold2 confidence and geometry quantities for sequences with no reference."""
    worker_results = dispatch_design_gradient(
        protocol="hallucination",
        items=[{"logits": matrix.root, "prep_kwargs": {"length": len(matrix)}} for matrix in inputs.inputs],
        config=config,
        temperature=inputs.temperature,
        tool_key="alphafold2-hallucination-gradient",
        instance=instance,
    )

    results = []
    for matrix, item in zip(inputs.inputs, worker_results, strict=True):
        built = build_terms(
            item["terms"],
            _CATALOG,  # type: ignore[arg-type]
            matrix.root,
            temperature=inputs.temperature,
            soft=config.soft,
            hard=config.hard,
        )
        results.append(
            AlphaFold2HallucinationGradientResult(
                terms=[AlphaFold2HallucinationTermGradient(**kwargs) for kwargs in built],
                metrics=item["metrics"],
                structure=build_structure(item, source="alphafold2-hallucination-gradient"),
            )
        )
    return AlphaFold2HallucinationGradientOutput(results=results, vocab=list(PROTEIN_AMINO_ACIDS))
