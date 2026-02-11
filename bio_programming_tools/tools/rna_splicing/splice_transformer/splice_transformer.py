"""
splice_transformer.py

Tissue-specific splice site prediction using SpliceTransformer.
"""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import List, Literal

import numpy as np
from pydantic import Field, field_serializer, model_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField, use_cloud_gpu
from bio_programming_tools.utils.tool_cache import tool_cache
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Enums
# ============================================================================
CONTEXT_LENGTH = 4000
TARGET_LENGTH = 1000
TISSUE_INDEX_OFFSET = 3  # Offset of tissue predictions in SpliceTransformer output.


class SpliceTransformerType(Enum):
    NEITHER = 0
    ACCEPTOR = 1
    DONOR = 2


class SpliceTransformerTissue(Enum):
    AVERAGE = -1
    ADIPOSE_TISSUE = 0
    BLOOD = 1
    BLOOD_VESSEL = 2
    BRAIN = 3
    COLON = 4
    HEART = 5
    KIDNEY = 6
    LIVER = 7
    LUNG = 8
    MUSCLE = 9
    NERVE = 10
    SMALL_INTESTINE = 11
    SKIN = 12
    SPLEEN = 13
    STOMACH = 14

    @classmethod
    def as_literal(cls) -> Literal[str]:
        """Generate Literal type from enum member names."""
        names = tuple(member.name for member in cls)
        return Literal[names]


# ============================================================================
# Data Models
# ============================================================================
class SpliceTransformerInput(BaseToolInput):
    """Input object for SpliceTransformer splice site prediction.

    This class defines the input parameters for predicting splice sites in RNA
    sequences using SpliceTransformer, a deep learning model for tissue-specific
    splicing prediction.

    Attributes:
        target_seqs (List[str]): RNA or DNA sequence(s) on which to make splicing
            predictions. These are the central sequences where splice sites will
            be predicted at single-nucleotide resolution. All sequences in the
            batch should have the same length (typically 1000bp).

        left_contexts (List[str]): Sequence(s) providing left (5') context for
            each target sequence. Must have the same number of sequences as
            ``target_seqs``. All left context sequences must have the same length
            (typically 4000bp) to provide sufficient context for accurate prediction.

        right_contexts (List[str]): Sequence(s) providing right (3') context for
            each target sequence. Must have the same number of sequences as
            ``target_seqs``. All right context sequences must have the same length
            (typically 4000bp) matching the left context.

    Note:
        The total sequence length provided to SpliceTransformer is
        ``context_length + target_length + context_length``. The model makes
        predictions only over the ``target_length`` region (the target sequence),
        but uses the full context for accurate splice site identification.
    """

    target_seqs: List[str] = Field(
        description="Sequence(s) on which to make splicing predictions",
    )
    left_contexts: List[str] = Field(
        description="Sequence(s) of the left context. Must be the same length as target_seqs",
    )
    right_contexts: List[str] = Field(
        description="Sequence(s) of the right context. Must be the same length as target_seqs",
    )

    @model_validator(mode="after")
    def validate_input_format(self):
        if len(self.target_seqs) != len(self.left_contexts):
            raise ValueError(
                "Number of target sequences must be the same as the number of "
                "left context sequences"
            )
        if len(self.target_seqs) != len(self.right_contexts):
            raise ValueError(
                "Number of target sequences must be the same as the number of "
                "right context sequences"
            )
        return self


class SpliceTransformerConfig(BaseConfig):
    """Configuration object for SpliceTransformer model.

    This class defines configuration parameters for running SpliceTransformer,
    a transformer-based model for predicting tissue-specific splice sites in
    RNA sequences.

    Attributes:
        context_length (int): Length of context sequence on both left (5') and
            right (3') sides of the target sequence. All sequences in ``left_contexts``
            and ``right_contexts`` must match this length. Longer contexts generally
            improve prediction accuracy. Default: 4000.

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution), or specific GPU devices like ``"cuda:0"``.
            GPU is strongly recommended for acceptable inference speed. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during model execution,
            including loading progress and timing information. Default: ``False``.

    Note:
        The context length determines the receptive field for splice site prediction.
        Standard value is 4000bp, which balances accuracy and computational cost.
    """

    context_length: int = ConfigField(
        title="Context Length",
        default=CONTEXT_LENGTH,
        description="Context length on both left and right of target sequence.",  # All sequences in left_contexts and right_contexts must be this length
    )
    device: str = (
        ConfigField(  # TODO: Device management should be managed elsewhere eventually
            default="cuda",
            title="Device",
            description="Device to run the model on (e.g., 'cuda', 'cpu')",
            hidden=True,
        )
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages during execution",
        hidden=True,
    )


class SpliceTransformerOutput(BaseToolOutput):
    """Output from SpliceTransformer splice site prediction.

    This class encapsulates the results of SpliceTransformer prediction, providing
    per-position probabilities for splice site types and tissue-specific splicing.

    Attributes:
        prediction (np.ndarray): Prediction tensor of shape ``(batch, target_length, 18)``
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

    prediction: np.ndarray = Field(
        description="Matrix of (batch, target_length, 18)",
    )

    @field_serializer('prediction')
    def serialize_prediction(self, value: np.ndarray) -> list:
        return value.tolist()

    @property
    def output_format_options(self) -> List[str]:
        return ["npy", "json"]

    @property
    def output_format_default(self) -> str:
        return "npy"

    def _export_output(self, export_path: str | Path, file_format: str):
        from pathlib import Path

        import numpy as np

        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "npy":
            np.save(path, self.prediction)

        elif file_format == "json":
            import json

            # Convert to list for JSON
            json_data = self.prediction.tolist()
            with open(path, "w") as f:
                json.dump(json_data, f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="splice-transformer-prediction",
    label="SpliceTransformer Splicing Prediction",
    input=SpliceTransformerInput,
    config=SpliceTransformerConfig,
    output=SpliceTransformerOutput,
    description="Tissue-specific splicing prediction using SpliceTransformer",
)
@tool_cache("splice-transformer-prediction")
def run_splice_transformer(
    inputs: SpliceTransformerInput,
    config: SpliceTransformerConfig,
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
            specifying context length and device settings.

    Returns:
        SpliceTransformerOutput: Structured output containing:
            - ``prediction``: Tensor of shape ``(batch, target_length, 18)`` with
            per-position probabilities for splice types and tissue-specific splicing

    See Also:
        - SpliceTransformer paper: https://www.nature.com/articles/s41467-024-53088-6
        - Model repository: https://github.com/ShenLab-Genomics/SpliceTransformer

    Example:
        >>> inputs = SpliceTransformerInput(
        ...     target_seqs=["ATGC" * 250],  # 1000bp
        ...     left_contexts=["CGTA" * 1000],  # 4000bp
        ...     right_contexts=["TACG" * 1000]  # 4000bp
        ... )
        >>> config = SpliceTransformerConfig(
        ...     context_length=4000,
        ...     verbose=True
        ... )
        >>> result = run_splice_transformer(inputs, config)
        >>> # Extract donor sites (channel 2)
        >>> donor_probs = result.prediction[0, :, 2]
        >>> # Find high-confidence donor sites
        >>> import numpy as np
        >>> donor_sites = np.where(donor_probs > 0.5)[0]

    Note:
        - GPU is strongly recommended (CPU inference is very slow)
        - Context length of 4000bp is recommended for best accuracy
        - Target length is typically 1000bp but can vary
        - the cloud runtime GPU execution is automatically used when configured via environment
        - For local execution, each subprocess is fresh (no in-process caching)
    """
    if config.device == "cuda" and use_cloud_gpu():
        # the cloud runtime
        logger.debug(
            f"Using the cloud runtime for SpliceTransformer inference (context_length={config.context_length})"
        )
        import _gpu_runtime

        SpliceTransformerService = _gpu_runtime.Cls.from_name(
            "bio-programming", "SpliceTransformerService"
        )
        prediction = SpliceTransformerService().run.remote(
            target_seqs=inputs.target_seqs,
            left_contexts=inputs.left_contexts,
            right_contexts=inputs.right_contexts,
            verbose=config.verbose,
        )
    else:
        # Local GPU/CPU via standalone venv
        from bio_programming_tools.utils.env_manager import EnvManager

        logger.debug(
            f"Using local device for SpliceTransformer inference (context_length={config.context_length})"
        )

        venv_manager = EnvManager(model_name="splice_transformer")

        input_data = {
            "target_seqs": inputs.target_seqs,
            "left_contexts": inputs.left_contexts,
            "right_contexts": inputs.right_contexts,
            "context_length": config.context_length,
            "device": config.device,
            "verbose": config.verbose,
        }

        output_data = venv_manager.call_standalone_script_in_venv(
            script_path=Path(__file__).parent / "standalone" / "run.py",
            input_dict=input_data,
            device=config.device,
        )

        prediction = np.array(output_data["prediction"])

    return SpliceTransformerOutput(
        metadata={"context_length": config.context_length},
        prediction=prediction,
    )
