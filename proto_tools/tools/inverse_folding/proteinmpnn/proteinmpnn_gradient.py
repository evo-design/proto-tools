"""ProteinMPNN structure-conditioned perplexity gradient tool."""

import logging
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from proto_tools.entities.structures import ChainSelection, ResidueSelection, Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    BaseConfig,
    BaseToolInput,
    ConfigField,
    GradientOutput,
    InputField,
    ToolInstance,
    one_hot_protein_logits,
)

logger = logging.getLogger(__name__)

ProteinMPNNModelChoice = Literal["proteinmpnn", "v_48_002", "v_48_010", "v_48_030", "abmpnn", "soluble"]


class ProteinMPNNGradientInput(BaseToolInput):
    """Input for relaxed ProteinMPNN perplexity gradients.

    Attributes:
        logits (list[list[float]]): Relaxed sequence state, shape ``L x 20`` in
            canonical amino-acid order ``ACDEFGHIKLMNPQRSTVWY``.
        structure (Structure): Backbone structure to condition ProteinMPNN on.
        chains_to_redesign (ChainSelection | None): Chains to score/design. If
            ``None``, all chains in ``structure`` are used.
        fixed_positions (ResidueSelection | None): Per-chain positions excluded
            from the perplexity objective.
        temperature (float | None): Optional softmax temperature. When set,
            applies ``softmax(input / temperature)`` before evaluating the
            relaxed sequence. When ``None``, the input is used as-is.
    """

    logits: list[list[float]] = InputField(
        title="Logits",
        description="Relaxed sequence logits with shape (L, 20) in canonical amino-acid order.",
        examples=[[[0.0] * 20, [0.0] * 20]],
    )
    structure: Structure = InputField(
        title="Input Structure",
        description="Backbone structure for ProteinMPNN conditioning.",
    )
    chains_to_redesign: ChainSelection | None = InputField(
        default=None,
        title="Chains to Redesign",
        description="Structure chains to score. Defaults to all chains in the structure.",
    )
    fixed_positions: ResidueSelection | None = InputField(
        default=None,
        title="Fixed Positions",
        description="Per-chain 1-indexed residue positions excluded from the ProteinMPNN objective.",
    )
    temperature: float | None = InputField(
        default=None,
        title="Temperature",
        description="Softmax temperature. Applies softmax(input / T) when set.",
        gt=0.0,
    )

    @field_validator("logits")
    @classmethod
    def validate_logits(cls, logits: list[list[float]]) -> list[list[float]]:
        """Ensure logits are a non-empty rectangular ``L x 20`` matrix."""
        if not logits:
            raise ValueError("logits must contain at least one position")

        expected_width = len(PROTEIN_AMINO_ACIDS)
        for idx, row in enumerate(logits):
            if len(row) != expected_width:
                raise ValueError(f"logits row {idx} must have {expected_width} columns, got {len(row)}")
        return logits

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_chain_ids(cls, data: Any) -> Any:
        """Accept legacy ``chain_ids`` as an alias for ``chains_to_redesign``."""
        if not isinstance(data, dict) or "chain_ids" not in data:
            return data
        if "chains_to_redesign" in data:
            raise ValueError("Specify only one of chain_ids or chains_to_redesign")
        data = dict(data)
        data["chains_to_redesign"] = data.pop("chain_ids")
        return data

    @model_validator(mode="after")
    def validate_selections(self) -> "ProteinMPNNGradientInput":
        """Validate chain and residue selections against the backbone."""
        if self.chains_to_redesign is not None:
            self.chains_to_redesign.validate_against(self.structure, label="chains_to_redesign")
        if self.fixed_positions is not None:
            self.fixed_positions.validate_against(self.structure, label="fixed_positions")
        return self

    @property
    def chain_ids_to_redesign(self) -> list[str]:
        """Resolved chain IDs for the standalone ProteinMPNN worker."""
        if self.chains_to_redesign is not None:
            return list(self.chains_to_redesign.chains)
        return self.structure.get_chain_ids()


class ProteinMPNNGradientOutput(GradientOutput):
    """ProteinMPNN mean-NLL output; gradient is optional in forward-only mode.

    Attributes:
        gradient (list[list[float]] | None): Gradient w.r.t. input logits. None
            when ``compute_gradient=False``.
        loss (float): Mean negative log-likelihood over ProteinMPNN-scored positions.
        metrics (dict[str, Any]): Log-likelihood, perplexity, sequence length, and objective details.
        vocab (list[str]): Canonical amino-acid column ordering for logits and gradients.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )


class ProteinMPNNGradientConfig(BaseConfig):
    """Configuration for relaxed ProteinMPNN perplexity gradients.

    Attributes:
        model_choice (ProteinMPNNModelChoice): ProteinMPNN weight variant.
        use_ste (bool): Use hard one-hot forward pass with soft-probability gradients.
        compute_gradient (bool): Return gradients when true; run forward scoring only when false.
        device (str): Device for ProteinMPNN execution.
    """

    model_choice: ProteinMPNNModelChoice = ConfigField(
        title="Model Choice",
        default="proteinmpnn",
        description="Weights: proteinmpnn (=v_48_020), v_48_{002,010,030} noise variants, abmpnn, soluble",
        reload_on_change=True,
        examples=["proteinmpnn", "v_48_010", "abmpnn", "soluble"],
    )
    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=True,
        description="Hard one-hot forward pass with soft-probability gradients.",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only log-likelihood.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


def example_input() -> ProteinMPNNGradientInput:
    """Minimal valid input for testing and examples."""
    from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_score import example_input as scoring_example_input

    pair = scoring_example_input().sequence_structure_pairs[0]
    return ProteinMPNNGradientInput(
        logits=one_hot_protein_logits(pair.sequence, sharpness=2.0),
        structure=pair.structure,
    )


@tool(
    key="proteinmpnn-gradient",
    label="ProteinMPNN Gradient",
    category="inverse_folding",
    input_class=ProteinMPNNGradientInput,
    config_class=ProteinMPNNGradientConfig,
    output_class=ProteinMPNNGradientOutput,
    description="Compute ProteinMPNN structure-conditioned perplexity gradient for relaxed protein sequences",
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    stochastic=True,
)
def run_proteinmpnn_gradient(
    inputs: ProteinMPNNGradientInput,
    config: ProteinMPNNGradientConfig,
    instance: Any = None,
) -> ProteinMPNNGradientOutput:
    """Compute ProteinMPNN mean-NLL gradient with respect to relaxed sequence logits."""
    chain_ids = inputs.chain_ids_to_redesign
    seed = config.seed if config.seed is not None else config.get_random_int()
    logger.debug("Using local worker for ProteinMPNN gradient: %s", config.model_choice)
    with inputs.structure.temp_file() as pdb_path:
        result = ToolInstance.dispatch(
            "proteinmpnn",
            {
                "operation": "compute_gradient",
                "pdb_path": str(pdb_path),
                "chain_ids": chain_ids,
                "logits": inputs.logits,
                "temperature": inputs.temperature,
                "use_ste": config.use_ste,
                "compute_gradient": config.compute_gradient,
                "fixed_positions": inputs.fixed_positions.chains if inputs.fixed_positions is not None else None,
                "model_choice": config.model_choice,
                "seed": seed,
                "device": config.device,
                "verbose": config.verbose,
            },
            instance=instance,
            config=config,
        )
    return ProteinMPNNGradientOutput(**result)
