"""DeepPBS DNA specificity prediction tool."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

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


class DeepPBSSpecificityInput(BaseToolInput):
    """Input for DeepPBS specificity prediction.

    Attributes:
        pdb_paths (list[str]): PDB paths for protein-DNA structures to score. A
            single path string is normalized to a one-element list. At least one
            path is required.
    """

    pdb_paths: list[str] = InputField(
        title="PDB Paths",
        description="PDB paths for protein-DNA structures to score",
    )

    @field_validator("pdb_paths", mode="before")
    @classmethod
    def normalize_paths(cls, value: Any) -> Any:
        """Normalize single path input into list form."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("pdb_paths")
    @classmethod
    def validate_paths(cls, value: list[str]) -> list[str]:
        """Require at least one input PDB path."""
        if not value:
            raise ValueError("pdb_paths must contain at least one path")
        return value


class DeepPBSSpecificityConfig(BaseConfig):
    """Configuration for DeepPBS specificity prediction.

    The DeepPBS checkout (with its bundled X3DNA/DSSR binaries, process/predict
    configs, and inference weights) is provisioned automatically by the standalone
    setup into the managed weights cache, so no repository, config, or X3DNA path is
    configured here.

    Attributes:
        output_directory (str | None): Optional directory for canonical NPZ
            artifacts. A temporary directory is used when unset.
        keep_intermediate (bool): Keep intermediate process and predict files.
        no_clean_protein (bool): Pass --no_cleanp to DeepPBS preprocessing to skip
            pdb2pqr-dependent protein cleaning.
        allow_fallback (bool): On missing DeepPBS deps/outputs, return a uniform
            fallback PPM instead of raising.
        device (str): Device to run DeepPBS inference on (inherited).
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run DeepPBS inference on",
        include_in_key=False,
    )
    output_directory: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional output directory for canonical NPZ artifacts",
        include_in_key=False,
    )
    keep_intermediate: bool = ConfigField(
        title="Keep Intermediate",
        default=False,
        description="Keep intermediate process and predict files",
        include_in_key=False,
    )
    no_clean_protein: bool = ConfigField(
        title="Skip Protein Cleaning",
        default=False,
        description="Pass --no_cleanp to skip pdb2pqr-dependent protein cleaning",
    )
    allow_fallback: bool = ConfigField(
        title="Allow Fallback",
        default=False,
        description="On missing DeepPBS deps/outputs, return a uniform fallback PPM instead of raising",
        include_in_key=False,
    )

    def cloud_unsupported_reason(self) -> str | None:
        """DeepPBS needs a local repository and X3DNA install not staged to cloud."""
        return (
            "DeepPBS requires a local DeepPBS checkout and its bundled X3DNA install, "
            "which are not available on device='cloud'. Run locally with device='cpu'."
        )


class DeepPBSSpecificityResult(BaseModel):
    """Canonicalized DeepPBS specificity result for one structure.

    Attributes:
        input_name (str): Basename of the scored input structure.
        source_method (str): Source method tag, always "deeppbs".
        output_npz_path (str): Path to canonical NPZ output.
        predicted_ppm (list[list[float]]): Canonical DNA PPM in A,C,G,T order.
        true_sequence (list[int]): Canonical DNA truth indices in 0..3.
        mask (list[int]): Valid residue mask for canonical rows.
        dna_mask (list[int]): DNA residue mask for canonical rows.
        chain_labels (list[int]): Canonical chain IDs for canonical rows.
        used_fallback (bool): Whether canonical output came from fallback logic.
        fallback_reason (str | None): Reason for fallback when used_fallback is true.
    """

    input_name: str = Field(
        title="Input Name",
        description="Basename of the scored input structure",
    )
    source_method: str = Field(
        title="Source Method",
        description="Source method tag for this prediction",
    )
    output_npz_path: str = Field(
        title="Output NPZ Path",
        description="Path to canonical NPZ output",
    )
    predicted_ppm: list[list[float]] = Field(
        title="Predicted PPM",
        description="Canonical DNA PPM in A,C,G,T order",
    )
    true_sequence: list[int] = Field(
        title="True Sequence",
        description="Canonical DNA truth indices in 0..3",
    )
    mask: list[int] = Field(
        title="Mask",
        description="Valid residue mask for canonical rows",
    )
    dna_mask: list[int] = Field(
        title="DNA Mask",
        description="DNA residue mask for canonical rows",
    )
    chain_labels: list[int] = Field(
        title="Chain Labels",
        description="Canonical chain IDs for canonical rows",
    )
    used_fallback: bool = Field(
        default=False,
        title="Used Fallback",
        description="Whether canonical output was generated from fallback logic",
    )
    fallback_reason: str | None = Field(
        default=None,
        title="Fallback Reason",
        description="Reason for fallback generation when used_fallback is true",
    )


class DeepPBSSpecificityOutput(BaseToolOutput):
    """Output from DeepPBS specificity prediction.

    Attributes:
        results (list[DeepPBSSpecificityResult]): Canonicalized results per input,
            index-aligned with ``inputs.pdb_paths``.
    """

    results: list[DeepPBSSpecificityResult] = Field(
        default_factory=list,
        title="Results",
        description="Canonicalized results per input structure",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(
        self,
        export_path: str | Path,
        file_format: str,
    ) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        rows = [result.model_dump() for result in self.results]
        if file_format == "json":
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(rows, handle, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "input_name",
                        "source_method",
                        "output_npz_path",
                        "predicted_ppm",
                        "true_sequence",
                        "mask",
                        "dna_mask",
                        "chain_labels",
                        "used_fallback",
                        "fallback_reason",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> DeepPBSSpecificityInput:
    """Minimal valid input for testing and examples."""
    return DeepPBSSpecificityInput(pdb_paths=["complex.pdb"])


@tool(
    key="deeppbs-specificity",
    label="DeepPBS Specificity",
    category="sequence_scoring",
    input_class=DeepPBSSpecificityInput,
    config_class=DeepPBSSpecificityConfig,
    output_class=DeepPBSSpecificityOutput,
    description="Predict DNA specificity (PPM) from protein-DNA structures using DeepPBS",
    example_input=example_input,
    uses_gpu=True,
)
def run_deeppbs_specificity(
    inputs: DeepPBSSpecificityInput,
    config: DeepPBSSpecificityConfig,
    instance: Any = None,
) -> DeepPBSSpecificityOutput:
    """Run DeepPBS specificity prediction and return canonicalized outputs.

    Runs the local DeepPBS preprocessing and prediction scripts on each input
    protein-DNA structure and returns a canonical DNA-only position probability
    matrix (PPM) in A,C,G,T order. When a required DeepPBS dependency (X3DNA
    binaries) is missing, or preprocessing/prediction fails to produce output,
    the runner falls back to a conservative uniform PPM derived from the DNA
    residues in the PDB and flags the result via ``used_fallback`` /
    ``fallback_reason``.

    Args:
        inputs (DeepPBSSpecificityInput): Input structure paths for prediction.
        config (DeepPBSSpecificityConfig): Runtime and path configuration values.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        DeepPBSSpecificityOutput: Canonicalized specificity outputs per structure,
            index-aligned with ``inputs.pdb_paths``. Each result exposes
            ``predicted_ppm`` (L x 4), ``output_npz_path``, and the
            ``used_fallback`` / ``fallback_reason`` fallback markers.

    Example:
        >>> inputs = DeepPBSSpecificityInput(pdb_paths=["/path/to/complex.pdb"])
        >>> config = DeepPBSSpecificityConfig()
        >>> out = run_deeppbs_specificity(inputs, config)
        >>> ppm = out.results[0].predicted_ppm  # L x 4, ACGT order
    """
    logger.debug("Using local venv for deeppbs_specificity prediction")
    input_data = {
        "pdb_paths": inputs.pdb_paths,
        "output_directory": config.output_directory,
        "keep_intermediate": config.keep_intermediate,
        "no_clean_protein": config.no_clean_protein,
        "allow_fallback": config.allow_fallback,
        "device": config.device,
        "verbose": config.verbose,
    }

    output_data = ToolInstance.dispatch(
        "deeppbs_specificity",
        input_data,
        instance=instance,
        config=config,
    )

    results = [DeepPBSSpecificityResult(**item) for item in output_data["results"]]
    return DeepPBSSpecificityOutput(
        results=results,
        metadata={"num_inputs": len(results)},
    )
