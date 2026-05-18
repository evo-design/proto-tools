"""proto_tools/tools/rna_splicing/splice_transformer/splice_transformer.py.

Tissue-specific splice site prediction using SpliceTransformer.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Enums
# ============================================================================
CONTEXT_LENGTH = 4000
TARGET_LENGTH = 1000


class SpliceTransformerType(Enum):
    """Splice site classification type (neither, acceptor, or donor)."""

    NEITHER = 0
    ACCEPTOR = 1
    DONOR = 2


SpliceTransformerTissue = Literal[
    "AVERAGE",
    "ADIPOSE_TISSUE",
    "BLOOD",
    "BLOOD_VESSEL",
    "BRAIN",
    "COLON",
    "HEART",
    "KIDNEY",
    "LIVER",
    "LUNG",
    "MUSCLE",
    "NERVE",
    "SMALL_INTESTINE",
    "SKIN",
    "SPLEEN",
    "STOMACH",
]

# Prediction channels:
# [0: neither, 1: acceptor, 2: donor, 3+: tissue-specific channels].
SPLICE_TISSUE_CHANNEL_INDEX: dict[SpliceTransformerTissue, int | None] = {
    "AVERAGE": None,
    "ADIPOSE_TISSUE": 3,
    "BLOOD": 4,
    "BLOOD_VESSEL": 5,
    "BRAIN": 6,
    "COLON": 7,
    "HEART": 8,
    "KIDNEY": 9,
    "LIVER": 10,
    "LUNG": 11,
    "MUSCLE": 12,
    "NERVE": 13,
    "SMALL_INTESTINE": 14,
    "SKIN": 15,
    "SPLEEN": 16,
    "STOMACH": 17,
}


# ============================================================================
# Data Models
# ============================================================================
class SpliceTransformerInput(BaseToolInput):
    """Input object for SpliceTransformer splice site prediction.

    This class defines the input parameters for predicting splice sites in RNA
    sequences using SpliceTransformer, a deep learning model for tissue-specific
    splicing prediction.

    Attributes:
        target_seqs (list[str]): RNA or DNA sequence(s) on which to make splicing
            predictions. These are the central sequences where splice sites will
            be predicted at single-nucleotide resolution. All sequences in the
            batch should have the same length (typically 1000bp).

        left_contexts (list[str]): Sequence(s) providing left (5') context for
            each target sequence. Must have the same number of sequences as
            ``target_seqs``. All left context sequences must have the same length
            (typically 4000bp) to provide sufficient context for accurate prediction.

        right_contexts (list[str]): Sequence(s) providing right (3') context for
            each target sequence. Must have the same number of sequences as
            ``target_seqs``. All right context sequences must have the same length
            (typically 4000bp) matching the left context.

    Note:
        The total sequence length provided to SpliceTransformer is
        ``context_length + target_length + context_length``. The model makes
        predictions only over the ``target_length`` region (the target sequence),
        but uses the full context for accurate splice site identification.
    """

    target_seqs: list[str] = InputField(
        description="Sequence(s) on which to make splicing predictions",
    )
    left_contexts: list[str] = InputField(
        description="Sequence(s) of the left context. Must be the same length as target_seqs",
    )
    right_contexts: list[str] = InputField(
        description="Sequence(s) of the right context. Must be the same length as target_seqs",
    )

    @model_validator(mode="after")
    def validate_input_format(self) -> Any:
        """Validate sequence-list parity and per-sequence length against the published checkpoint."""
        if len(self.target_seqs) != len(self.left_contexts):
            raise ValueError("Number of target sequences must be the same as the number of left context sequences")
        if len(self.target_seqs) != len(self.right_contexts):
            raise ValueError("Number of target sequences must be the same as the number of right context sequences")
        for i, (t, lf, rt) in enumerate(zip(self.target_seqs, self.left_contexts, self.right_contexts, strict=True)):
            if len(t) != TARGET_LENGTH:
                raise ValueError(f"target_seqs[{i}] length {len(t)} != expected {TARGET_LENGTH}")
            if len(lf) != CONTEXT_LENGTH:
                raise ValueError(f"left_contexts[{i}] length {len(lf)} != expected {CONTEXT_LENGTH}")
            if len(rt) != CONTEXT_LENGTH:
                raise ValueError(f"right_contexts[{i}] length {len(rt)} != expected {CONTEXT_LENGTH}")
        return self


class SpliceTransformerConfig(BaseConfig):
    """Configuration for SpliceTransformer.

    Attributes:
        device (str): Device to run the model on. Override of ``BaseConfig.device``
            because SpliceTransformer is a GPU tool (default ``cuda``).
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


class SpliceTransformerOutput(BaseToolOutput):
    """Output from SpliceTransformer splice site prediction.

    This class encapsulates the results of SpliceTransformer prediction, providing
    per-position probabilities for splice site types and tissue-specific splicing.

    Attributes:
        prediction (list[list[list[float]]]): Prediction tensor of shape ``[batch, target_length, 18]``
            where:

            - ``batch``: Number of input sequences
            - ``target_length``: Length of each target sequence (e.g., 1000)
            - ``18``: Number of prediction channels (3 splice types + 15 tissues)

            The 18 channels are organized as follows:

            **Splice type predictions (channels 0-2):**

            - Channel 0: Probability of neither acceptor nor donor
            - Channel 1: Probability of acceptor splice site
            - Channel 2: Probability of donor splice site

            **Tissue-specific predictions (channels 3-17):**

            - Channel 3: Adipose tissue
            - Channel 4: Blood
            - Channel 5: Blood vessel
            - Channel 6: Brain
            - Channel 7: Colon
            - Channel 8: Heart
            - Channel 9: Kidney
            - Channel 10: Liver
            - Channel 11: Lung
            - Channel 12: Muscle
            - Channel 13: Nerve
            - Channel 14: Small intestine
            - Channel 15: Skin
            - Channel 16: Spleen
            - Channel 17: Stomach

            All probabilities are in the range [0, 1].
    """

    prediction: list[list[list[float]]] = Field(
        description="Matrix of [batch, target_length, 18]",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["npy", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "npy"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "npy":
            import numpy as np

            np.save(path, np.array(self.prediction))

        elif file_format == "json":
            with open(path, "w") as f:
                json.dump(self.prediction, f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SpliceTransformerInput(
        target_seqs=["ATCG" * 250],
        left_contexts=["CGTA" * 1000],
        right_contexts=["TACG" * 1000],
    )


@tool(
    key="splice-transformer-prediction",
    label="SpliceTransformer Splicing Prediction",
    category="rna_splicing",
    input_class=SpliceTransformerInput,
    config_class=SpliceTransformerConfig,
    output_class=SpliceTransformerOutput,
    description="Tissue-specific splicing prediction using SpliceTransformer",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
)
def run_splice_transformer(
    inputs: SpliceTransformerInput,
    config: SpliceTransformerConfig,
    instance: Any = None,
) -> SpliceTransformerOutput:
    """Predict splice sites in RNA/DNA sequences using SpliceTransformer.

    Uses SpliceTransformer, a transformer-based deep learning model, to predict
    splice acceptor and donor sites with tissue-specific probabilities. The model
    analyzes sequences with flanking context to identify canonical and alternative
    splicing patterns across 15 different human tissues.

    Args:
        inputs (SpliceTransformerInput): Validated input containing target sequences
            and their left/right context sequences.
        config (SpliceTransformerConfig): Validated SpliceTransformer configuration
            (inherits device / verbose / timeout / seed from BaseConfig).

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SpliceTransformerOutput: Structured output containing:
            - ``prediction``: Nested list of shape ``[batch, target_length, 18]`` with
            per-position probabilities for splice types and tissue-specific splicing

    See Also:
        - SpliceTransformer paper: https://www.nature.com/articles/s41467-024-53088-6
        - Model repository: https://github.com/ShenLab-Genomics/SpliceTransformer

    Example:
        >>> inputs = SpliceTransformerInput(
        ...     target_seqs=["ATGC" * 250],  # 1000bp
        ...     left_contexts=["CGTA" * 1000],  # 4000bp
        ...     right_contexts=["TACG" * 1000],  # 4000bp
        ... )
        >>> config = SpliceTransformerConfig()
        >>> result = run_splice_transformer(inputs, config)
        >>> # Extract donor site probabilities (channel 2) for first sequence
        >>> import numpy as np
        >>> pred = np.array(result.prediction)
        >>> donor_probs = pred[0, :, 2]
        >>> donor_sites = np.where(donor_probs > 0.5)[0]

    Note:
        - GPU is strongly recommended (CPU inference is very slow)
        - Context length of 4000bp is recommended for best accuracy
        - Target length is typically 1000bp but can vary
        - Each subprocess is fresh (no in-process caching)
    """
    logger.debug("Using local device for SpliceTransformer inference (context_length=%d)", CONTEXT_LENGTH)

    input_data = {
        "operation": "predict",
        "target_seqs": inputs.target_seqs,
        "left_contexts": inputs.left_contexts,
        "right_contexts": inputs.right_contexts,
        "context_length": CONTEXT_LENGTH,
        "device": config.device,
        "verbose": config.verbose,
    }

    output_data = ToolInstance.dispatch(
        "splice_transformer",
        input_data,
        instance=instance,
        config=config,
    )

    return SpliceTransformerOutput(prediction=output_data["prediction"])
