"""AbLang embeddings tool."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelOutput,
    SequenceEmbedding,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)

ABLANG_MODEL_CHOICES = Literal[
    "auto",
    "ablang1-heavy",
    "ablang1-light",
    "ablang2-paired",
]


def _resolve_model_choice(model_choice: str, sequences: list[str]) -> str:
    """Resolve 'auto' model choice based on sequence format.

    Paired sequences (containing '|') route to ablang2-paired.
    Single-chain sequences route to ablang1-heavy (the more common use case).
    """
    if model_choice != "auto":
        return model_choice
    if any("|" in seq for seq in sequences):
        return "ablang2-paired"
    return "ablang1-heavy"


# ============================================================================
# Data Models
# ============================================================================
class AbLangEmbeddingsInput(BaseToolInput):
    """Input for AbLang antibody embedding extraction.

    Attributes:
        sequences (list[str]): Antibody sequence(s) to process. Format depends
            on the model choice:

            - ``ablang1-heavy``: Heavy chain sequences (e.g., ``"EVQLVESGGGLVQPGG..."``)
            - ``ablang1-light``: Light chain sequences (e.g., ``"DIQMTQSPSSLSAS..."``)
            - ``ablang2-paired``: Paired heavy|light chains separated by pipe
              (e.g., ``"EVQLVES...|DIQMTQS..."``)

            Can be provided as a single string or a list of strings.
    """

    sequences: list[str] = InputField(
        description="Antibody sequence(s) to embed. For paired models, use 'heavy|light' format.",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v: Any) -> Any:
        """Convert single string to list."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v


class AbLangEmbeddingsConfig(MaskedModelConfig):
    """Configuration for AbLang antibody embedding extraction.

    AbLang provides antibody-specific language model embeddings trained on
    paired antibody sequence data from the Observed Antibody Space (OAS).

    Attributes:
        model_choice (ABLANG_MODEL_CHOICES): AbLang model variant to use:

            - ``"auto"``: Automatically select based on input format (default).
              Paired sequences (``"heavy|light"``) use ``ablang2-paired``,
              single-chain sequences use ``ablang1-heavy``.
            - ``"ablang1-heavy"``: Heavy chain only model
            - ``"ablang1-light"``: Light chain only model
            - ``"ablang2-paired"``: Paired heavy+light chain model

        batch_size (int): Number of sequences to process per forward pass.
            Default: ``1``.

        device (str): Device to run on. Default: ``"cuda"``.
    """

    model_choice: ABLANG_MODEL_CHOICES = ConfigField(
        title="Model Choice",
        default="auto",
        description="Model variant: 'auto', 'ablang1-heavy', 'ablang1-light', or 'ablang2-paired'",
        reload_on_change=True,
    )


class AbLangEmbeddingsOutput(MaskedModelOutput):
    """Output from AbLang antibody embedding extraction.

    Inherits from ``MaskedModelOutput``.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results. Each
            ``SequenceEmbedding`` contains:

            - ``mean_embedding``: Mean-pooled embedding vector (768-dim for ablang1, 480-dim for ablang2-paired)
            - ``attention_mask``: Binary mask (1 = valid, 0 = padding)
            - ``logits``: Optional per-position amino acid logits
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> AbLangEmbeddingsInput:
    """Minimal valid input for testing and examples."""
    return AbLangEmbeddingsInput(sequences=["EVQLVESGGGLVQPGG"])


@tool(
    key="ablang-embedding",
    label="AbLang Embeddings",
    category="masked_models",
    input_class=AbLangEmbeddingsInput,
    config_class=AbLangEmbeddingsConfig,
    output_class=AbLangEmbeddingsOutput,
    description="Extract antibody sequence embeddings using AbLang",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_ablang_embeddings(
    inputs: AbLangEmbeddingsInput,
    config: AbLangEmbeddingsConfig,
    instance: Any = None,
) -> AbLangEmbeddingsOutput:
    """Extract antibody sequence embeddings using AbLang.

    Uses AbLang, an antibody-specific language model trained on paired
    antibody sequences from OAS, to extract contextualized embeddings.
    Supports heavy-only, light-only, and paired heavy+light chain models.

    Args:
        inputs (AbLangEmbeddingsInput): Validated input containing antibody
            sequences to embed.
        config (AbLangEmbeddingsConfig): Configuration specifying model variant,
            batch size, and device.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AbLangEmbeddingsOutput: Structured output containing per-sequence
            embeddings (768-dim for ablang1, 480-dim for ablang2-paired).

    Examples:
        >>> inputs = AbLangEmbeddingsInput(sequences=["EVQLVESGGGLVQPGG"])
        >>> config = AbLangEmbeddingsConfig(model_choice="ablang1-heavy")
        >>> result = run_ablang_embeddings(inputs, config)
        >>> print(f"Embedding dim: {len(result.results[0].mean_embedding)}")

    See Also:
        - AbLang2 GitHub: https://github.com/oxpig/AbLang2
    """
    resolved_model = _resolve_model_choice(config.model_choice, inputs.sequences)
    logger.debug(f"Using local venv for AbLang embeddings: {resolved_model}")
    outputs = ToolInstance.dispatch(
        "ablang",
        {
            "operation": "embeddings",
            "sequences": inputs.sequences,
            "batch_size": config.batch_size,
            "model_choice": resolved_model,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    results = [
        SequenceEmbedding(
            mean_embedding=outputs["mean_embeddings"][i],
            attention_mask=outputs["attention_masks"][i],
            logits=None,
        )
        for i in range(len(inputs.sequences))
    ]

    return AbLangEmbeddingsOutput(
        metadata={
            "model_choice": resolved_model,
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,
            "device": config.device,
        },
        results=results,
    )
