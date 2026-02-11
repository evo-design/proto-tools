"""ESM3 structure prediction tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import Field

from bio_programming_tools.tools.infra.env_manager import EnvManager
from bio_programming_tools.tools.infra.tool_io import BaseToolOutput
from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.tools.utils import ConfigField, use_cloud_gpu

from .standalone.inference import ESM3_MODEL_CHECKPOINTS

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM3StructurePredictionInput = MaskedModelInput

# Output:
class ESM3StructurePredictionOutput(BaseToolOutput):
    """Output from ESM3 structure prediction.

    TODO: Remove or move to structure prediction module

    This class encapsulates the results of ESM3 structure prediction, providing
    predicted 3D structures with confidence metrics for each input sequence.

    Attributes:
        structures (List[Dict[str, Any]]): Predicted structures for each input
            sequence. Each structure is a dictionary containing:

            - ``"sequence"``: The input protein sequence
            - ``"pdb_string"``: PDB format string of the predicted structure
            - ``"avg_plddt"``: Average predicted Local Distance Difference Test
            (pLDDT) score (0-100). Higher values indicate higher confidence:

            - ``> 90``: Very high confidence
            - ``70-90``: High confidence
            - ``50-70``: Low confidence
            - ``< 50``: Very low confidence

            - ``"ptm"``: Predicted TM-score (0-1). Measures overall fold quality:

            - ``> 0.8``: High confidence in overall fold
            - ``0.5-0.8``: Medium confidence
            - ``< 0.5``: Low confidence in overall fold

        num_sequences (int): Total number of sequences that were processed and
            for which structures were predicted.

    Note:
        PDB strings can be saved directly to ``.pdb`` files for visualization
        in tools like PyMOL, Chimera, or mol* viewer.
    """
    structures: List[Dict[str, Any]] = Field(
        description="Predicted structures for each sequence with confidence metrics",
    )
    num_sequences: int = Field(
        description="Number of sequences processed",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path)

        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for i, struct in enumerate(self.structures):
                out_file = path / f"structure_{i}.pdb"
                with open(out_file, "w") as f:
                    f.write(struct["pdb_string"])

        elif file_format == "json":
            path = path.with_suffix(".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(path, "w") as f:
                json.dump(self.structures, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {file_format}")

# Config:
class ESM3StructurePredictionConfig(MaskedModelConfig):
    """Configuration object for ESM3 structure prediction.

    This class defines configuration parameters for running ESM3's generative
    structure prediction. Uses ESM3's ``GenerationConfig(track="structure")`` to
    predict 3D protein structures. This is a separate forward pass from embeddings
    extraction.

    Inherits from ``MaskedModelConfig``.

    Attributes:
        batch_size (int): Number of sequences to process in parallel. Structure
            prediction is memory-intensive, so smaller batch sizes (1-4) are
            typically recommended. Default: 128 (but lower values recommended).

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution), ``"mps"`` (Apple Metal), or specific GPU devices
            like ``"cuda:0"``. Structure prediction requires significant compute,
            so GPU is strongly recommended. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during structure prediction,
            including progress updates and timing information. Default: ``False``.

        model_checkpoint (str): ESM3 model checkpoint to use. Currently available:

            - ``"esm3_sm_open_v1"``: Small open-source ESM3 model (default)

            Default: ``"esm3_sm_open_v1"``.

    Note:
        Structure prediction is computationally expensive and may take several
        minutes per sequence depending on length and hardware.
    """
    model_checkpoint: Literal[ESM3_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM3 Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 model checkpoint to use",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="esm3-structure-prediction",
    label="ESM3 Structure Prediction",
    input=ESM3StructurePredictionInput,
    config=ESM3StructurePredictionConfig,
    output=ESM3StructurePredictionOutput,
    description="Predict protein 3D structures using ESM3 generative model",
)
def run_esm3_structure_prediction(
    inputs: ESM3StructurePredictionInput, config: ESM3StructurePredictionConfig
) -> ESM3StructurePredictionOutput:
    """Predict protein 3D structures using ESM3 generative model.

    Uses ESM3's generative capabilities to predict 3D structures from amino acid
    sequences. This is a separate operation from embeddings extraction and uses
    ``batch_generate`` with ``GenerationConfig(track="structure")`` to produce
    structure predictions with confidence metrics.

    Args:
        inputs (MaskedModelInput): Validated input containing one or more protein
            sequences (amino acid sequences) for structure prediction.
        config (ESM3StructurePredictionConfig): Validated ESM3 structure prediction
            configuration specifying model variant, batch size, and device settings.

    Returns:
        ESM3StructurePredictionOutput: Structured output containing:
            - ``structures``: List of predicted structures with PDB strings and confidence scores
            - ``num_sequences``: Number of sequences processed

    Examples:
        >>> # Predict structure for single sequence
        >>> inputs = MaskedModelInput(
        ...     sequences=["MVLSPADKTNVKAAW"]
        ... )
        >>> config = ESM3StructurePredictionConfig(verbose=True)
        >>> result = run_esm3_structure_prediction(inputs, config)
        >>> print(f"Average pLDDT: {result.structures[0]['avg_plddt']:.2f}")
        >>> print(f"PTM score: {result.structures[0]['ptm']:.2f}")
        >>>
        >>> # Save structure to PDB file
        >>> with open("predicted_structure.pdb", "w") as f:
        ...     f.write(result.structures[0]['pdb_string'])
        >>>
        >>> # Batch structure prediction
        >>> inputs = MaskedModelInput(
        ...     sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"]
        ... )
        >>> config = ESM3StructurePredictionConfig(batch_size=2)
        >>> result = run_esm3_structure_prediction(inputs, config)
        >>>
        >>> # Filter high-confidence predictions
        >>> high_conf = [
        ...     s for s in result.structures
        ...     if s['avg_plddt'] > 70 and s['ptm'] > 0.5
        ... ]

    Note:
        - Structure prediction is computationally expensive (minutes per sequence)
        - Use smaller batch sizes (1-4) for structure prediction vs embeddings
        - the cloud runtime GPU execution is automatically used when configured via environment
        - Very long sequences (>1000 residues) may fail or produce low-quality structures
    """

    # Choose execution mode
    if use_cloud_gpu():
        # the cloud runtime
        logger.debug(f"Using the cloud runtime for ESM3 structure prediction: {config.model_checkpoint}")
        import _gpu_runtime

        ESM3Service = _gpu_runtime.Cls.from_name("bio-programming", "ESM3Service")
        structures = ESM3Service().predict_structure.remote(
            sequences=inputs.sequences,
            batch_size=config.batch_size,
            verbose=config.verbose,
        )
    else:
        # Local venv execution
        logger.debug(f"Using local venv for ESM3 structure prediction: {config.model_checkpoint}")
        venv_manager = EnvManager("esm3")
        script_path = Path(__file__).parent / "standalone" / "inference.py"
        structures = venv_manager.call_standalone_script_in_venv(
            script_path=script_path,
            input_dict={
                "operation": "predict_structure",
                "sequences": inputs.sequences,
                "batch_size": config.batch_size,
                "model_checkpoint": config.model_checkpoint,
                "device": config.device,
                "verbose": config.verbose,
            },
            device=config.device,
            verbose=config.verbose,
        )

    return ESM3StructurePredictionOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,
            "device": config.device,
            "used_cloud": use_cloud_gpu(),
        },
        structures=structures,
        num_sequences=len(inputs.sequences),
    )
