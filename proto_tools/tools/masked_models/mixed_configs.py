"""Configuration and preprocessing for mixed protein/DNA masked models."""

import logging
import math
from typing import Any, ClassVar, Literal, cast

from pydantic import model_validator
from standalone_helpers.mixed_sequence import (
    DNA_TOKENS,
    MIXED_VOCAB,
    PROTEIN_TOKENS,
    STRAND_TOKENS,
    tokenize_mixed_sequence,
)

from proto_tools.transforms.masking import MaskingInput, MaskingStrategy
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.base_config import _marked_preprocessed

logger = logging.getLogger(__name__)


class MixedModelConfig(BaseConfig):
    """Shared execution configuration; registered tools specialize checkpoint choices.

    Attributes:
        model_checkpoint (str): Public Hugging Face checkpoint.
        batch_size (int): Sequences or masked PLL variants per forward pass.
        device (str): Device used for model inference.
    """

    context_limits: ClassVar[dict[str, int]] = {}
    layer_counts: ClassVar[dict[str, int]] = {}

    model_checkpoint: str = ConfigField(
        title="Model Checkpoint", description="Public mixed-modality model checkpoint", reload_on_change=True
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Sequences or masked PLL variants per forward pass; reduce if memory is limited",
    )
    device: str = ConfigField(
        default="cuda", title="Device", description="Device used for model inference", include_in_key=False
    )

    def preprocess(self, inputs: Any) -> Any:
        """Check context limits before loading weights and warn on a missing leading strand marker."""
        sequences = inputs.sequences if hasattr(inputs, "sequences") else [inputs.sequence]
        limit = self.context_limits[self.model_checkpoint]
        for index, sequence in enumerate(sequences):
            tokens = tokenize_mixed_sequence(sequence, allow_masks=True)
            if len(tokens) > limit:
                raise ValueError(
                    f"{self.model_checkpoint}: sequence {index} has {len(tokens)} tokens; limit is {limit}"
                )
            if tokens[0] not in STRAND_TOKENS:
                logger.warning(
                    "Sequence %d does not start with a strand marker (+ or -). "
                    "Generally this means the input was malformed.",
                    index,
                )
        return inputs


class MixedEmbeddingsConfig(MixedModelConfig):
    """Embedding extraction options.

    Attributes:
        return_logits (bool): Include token-aligned biological logits.
        repr_layer (int): Representation layer: 0 is the embedding table, 1..N are
            transformer outputs, and -1 selects the last transformer output.
    """

    return_logits: bool = ConfigField(
        default=False,
        title="Return Logits",
        description="Include token-aligned logits over the 24 canonical biological tokens",
    )
    repr_layer: int = ConfigField(
        default=-1,
        ge=-1,
        title="Representation Layer",
        description="0=embedding table, 1..N=transformer outputs, -1=last transformer output",
    )

    @model_validator(mode="after")
    def validate_layer(self) -> "MixedEmbeddingsConfig":
        """Reject layers absent from the selected checkpoint."""
        if self.repr_layer > self.layer_counts[self.model_checkpoint]:
            raise ValueError("repr_layer exceeds the selected model's transformer depth")
        return self


class MixedScoringConfig(MixedModelConfig):
    """Masked pseudo-log-likelihood scoring options.

    Attributes:
        return_logits (bool): Return masked logits; unscored context rows contain zeros.
    """

    return_logits: bool = ConfigField(
        default=False,
        title="Return Logits",
        description="Include masked biological logits; unscored context rows contain zeros",
    )


class MixedSampleConfig(MixedModelConfig):
    """Modality-preserving sampling with the existing masking and refinement strategies.

    Attributes:
        masking_strategy (MaskingStrategy): Select editable token positions when no masks are supplied.
        sampling_method (Literal['single_pass', 'iterative_refinement']): Sampling algorithm.
        temperature (float): Temperature for sampling alternatives within each position's modality.
        top_p (float): Nucleus threshold for iterative refinement.
        num_steps (int): Number of iterative refinement rounds.
        schedule (Literal['cosine', 'linear']): Iterative unmask schedule.
        strategy (Literal['random', 'entropy']): Select commitments randomly or by confidence.
        temperature_annealing (bool): Cool temperature during iterative refinement.
        return_logits (bool): Return biological logits, evaluated on the completed sequence.
    """

    masking_inputs: ClassVar[frozenset[MaskingInput]] = frozenset({MaskingInput.LOGITS})
    masking_strategy: MaskingStrategy = ConfigField(
        default_factory=MaskingStrategy,
        title="Masking Strategy",
        description="Select editable model-token positions; explicit '_' masks bypass selection",
    )
    sampling_method: Literal["single_pass", "iterative_refinement"] = ConfigField(
        default="single_pass",
        title="Sampling Method",
        description="Fill all masks once, or progressively commit predictions across refinement rounds",
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Temperature",
        description="Sampling temperature within each position's protein or DNA alphabet",
    )
    top_p: float = ConfigField(
        default=1.0,
        gt=0.0,
        le=1.0,
        title="Top P",
        description="Iterative nucleus threshold; 1.0 disables nucleus filtering",
    )
    num_steps: int = ConfigField(
        default=20, ge=1, title="Num Steps", description="Number of iterative refinement rounds"
    )
    schedule: Literal["cosine", "linear"] = ConfigField(
        default="cosine", title="Schedule", description="Iterative unmask schedule"
    )
    strategy: Literal["random", "entropy"] = ConfigField(
        default="random", title="Strategy", description="Commit masked sites randomly or by lowest predictive entropy"
    )
    temperature_annealing: bool = ConfigField(
        default=True,
        title="Temperature Annealing",
        description="Cool sampling temperature over iterative refinement rounds",
    )
    return_logits: bool = ConfigField(
        default=False,
        title="Return Logits",
        description="Include biological logits from a final forward pass over each completed sequence",
    )

    def preprocess(self, inputs: Any) -> Any:
        """Select tokens once, preserving original modalities in the normalized input."""
        inputs = super().preprocess(inputs)
        if any("_" in sequence for sequence in inputs.sequences):
            if self.masking_strategy != MaskingStrategy():
                logger.warning(
                    "Sequences already contain mask tokens ('_'); ignoring custom masking_strategy. "
                    "Remove '_' tokens to use the strategy, or omit masking_strategy to silence this warning."
                )
            return inputs
        # One-character tokens make string indices token positions; strand markers stay fixed.
        sequences = ["".join(tokenize_mixed_sequence(sequence)) for sequence in inputs.sequences]
        eligibility = [[token in MIXED_VOCAB for token in sequence] for sequence in sequences]
        position_score_fn = None
        if self.masking_strategy.method != "random":
            # Resolve the matching operation through the registry, using this tool's toolkit.
            from proto_tools.tools.tool_registry import ToolRegistry

            if self.tool_key is None:
                raise ValueError("Automatic masking requires a registered sampling config")
            embedding_key = self.tool_key.removesuffix("-sample") + "-embedding"
            spec = ToolRegistry.get(embedding_key)
            # This preprocess already checked these sequences, so the scoring call skips its own.
            result = spec.function(
                spec.input_model.model_validate({"sequences": sequences}),
                _marked_preprocessed(
                    cast(
                        BaseConfig,
                        spec.config_model(
                            model_checkpoint=self.model_checkpoint,
                            batch_size=self.batch_size,
                            device=self.device,
                            return_logits=True,
                        ),
                    )
                ),
            )
            restricted_logits = []
            for row_tokens, item in zip(sequences, result.results, strict=True):
                rows = []
                for token, logits in zip(row_tokens, item.logits, strict=True):
                    permitted = range(20, 24) if token in DNA_TOKENS else range(20)
                    rows.append([value if j in permitted else -math.inf for j, value in enumerate(logits)])
                restricted_logits.append(rows)

            def position_score_fn(_sequences: list[str]) -> list[list[list[float]]]:
                return restricted_logits

        masked = self.masking_strategy.mask(
            sequences,
            position_score_fn=position_score_fn,
            seed=self.seed,
            eligibility=eligibility,
        )
        modalities = [
            {i + 1: ("protein" if before[i] in PROTEIN_TOKENS else "dna") for i, t in enumerate(after) if t == "_"}
            for before, after in zip(sequences, masked, strict=True)
        ]
        return inputs.model_copy(update={"sequences": masked, "mask_modalities": modalities})


class MixedGradientConfig(MixedModelConfig):
    """Differentiable masked pseudo-log-likelihood options.

    Attributes:
        use_ste (bool): Use hard forward tokens with a soft backward derivative.
        compute_gradient (bool): Compute a backward pass, or return only the objective.
    """

    use_ste: bool = ConfigField(
        default=False,
        title="Straight-Through Estimator",
        description="Use hard one-hot forward tokens with soft-probability gradients",
    )
    compute_gradient: bool = ConfigField(
        default=True,
        title="Compute Gradient",
        description="Run backward and return the gradient; False evaluates only the masked PLL objective",
    )
