"""ProteinMPNN structure-conditioned perplexity gradient tool."""

import logging
from typing import Any, Literal

from pydantic import model_validator

from proto_tools.entities.structures import ChainSelection, ResidueSelection, Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    GradientOutput,
    InputField,
    ProteinGradientInput,
    RelaxationConfig,
    ToolInstance,
    one_hot_protein_logits,
)

logger = logging.getLogger(__name__)

ProteinMPNNModelChoice = Literal["proteinmpnn", "v_48_002", "v_48_010", "v_48_030", "abmpnn", "soluble"]


class ProteinMPNNGradientInput(ProteinGradientInput):
    """Input for relaxed ProteinMPNN perplexity gradients.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed sequence state, shape ``L x 20`` in
            canonical amino-acid order ``ACDEFGHIKLMNPQRSTVWY``.
        temperature (float | None): Softmax temperature. The default ``1.0`` reads ``logits``
            as unnormalized and applies ``softmax(logits / T)``, returning a logit-space
            gradient; ``None`` reads them as probabilities and returns a probability-space
            gradient.
        structure (Structure): Backbone structure to condition ProteinMPNN on.
        chains_to_redesign (ChainSelection | None): Chains to score/design. If
            ``None``, all chains in ``structure`` are used.
        fixed_positions (ResidueSelection | None): Per-chain positions excluded
            from the perplexity objective.
    """

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
        results (list[GradientResult]): One per input item, each holding its gradient w.r.t.
            the supplied logits (``None`` in forward-only mode), its mean negative log-likelihood
            over ProteinMPNN-scored positions, and its log-likelihood, perplexity, sequence
            length and objective details.
        vocab (list[str]): Canonical amino-acid column ordering for logits and gradients.
    """


class ProteinMPNNGradientConfig(RelaxationConfig):
    """Configuration for relaxed ProteinMPNN perplexity gradients.

    Attributes:
        model_choice (ProteinMPNNModelChoice): ProteinMPNN weight variant.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
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
                "logits": inputs.logits.root,
                "temperature": inputs.temperature,
                "soft": config.soft,
                "hard": config.hard,
                "compute_gradient": config.compute_gradient,
                "fixed_positions": (
                    inputs.fixed_positions.to_residue_numbers(inputs.structure)
                    if inputs.fixed_positions is not None
                    else None
                ),
                "model_choice": config.model_choice,
                "seed": seed,
                "device": config.device,
                "verbose": config.verbose,
            },
            instance=instance,
            config=config,
        )
    return ProteinMPNNGradientOutput.from_single(result)
