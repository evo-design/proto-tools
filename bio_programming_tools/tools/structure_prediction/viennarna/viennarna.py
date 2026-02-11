"""
viennarna.py

RNA secondary structure prediction using ViennaRNA.

This module provides standardized interfaces for RNA secondary structure
prediction using ViennaRNA's minimum free energy (MFE) folding algorithm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ViennaRNAInput(BaseToolInput):
    """Input object for ViennaRNA secondary structure prediction.

    This class defines the input parameters for predicting RNA secondary
    structures using ViennaRNA's MFE folding algorithm.

    Attributes:
        sequences (List[str]): List of RNA sequences to fold. Each sequence
            should contain only valid RNA nucleotides (A, U, G, C) or DNA
            nucleotides (A, T, G, C) which will be automatically converted
            to RNA (T -> U). Lowercase letters are also accepted.

    Note:
        ViennaRNA can handle both RNA and DNA sequences. DNA sequences
        (containing T) will be converted to RNA (T -> U) before folding
        unless DNA parameters are explicitly loaded.
    """
    sequences: List[str] = Field(description="List of input RNA sequences")

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: List[str]) -> List[str]:
        """
        Validates that sequences contain only valid nucleotides.

        Checks:
        - Non-empty sequences
        - Valid nucleotide characters (A, U, G, C, T, N)
        """
        if not sequences:
            raise ValueError("At least one sequence is required")

        valid_chars = set("AUGCTNaugctn")

        for seq_idx, seq in enumerate(sequences):
            invalid_chars = set(seq) - valid_chars
            if invalid_chars:
                raise ValueError(
                    f"Invalid nucleotide characters in sequence {seq_idx}: "
                    f"{', '.join(sorted(invalid_chars))}"
                )

        return sequences

# Output:
class ViennaRNAResult(BaseModel):
    """Result for a single RNA fold prediction.

    TODO: Consider abstracting into a base class if other RNA secondary structure
    predictors are implemented

    Attributes:
        sequence: The input RNA sequence.
        structure: Predicted secondary structure in dot-bracket notation.
        mfe: Minimum free energy in kcal/mol.
    """
    sequence: str
    structure: Optional[str] = None
    mfe: Optional[float] = None

# Output:
class ViennaRNAOutput(BaseToolOutput):
    """Output object for ViennaRNA secondary structure prediction.

    Attributes:
        results (List[ViennaRNAResult]): List of fold results, one per input
            sequence. Each result contains the sequence, predicted structure
            in dot-bracket notation, and the minimum free energy.
        metadata (dict): Additional information about the prediction run.
    """
    results: List[ViennaRNAResult] = Field(description="List of ViennaRNA results")

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json", "fasta"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence", "structure", "mfe"])
                for r in self.results:
                    writer.writerow([r.sequence, r.structure, r.mfe])

        elif file_format == "json":
             import json
             data = [r.model_dump() for r in self.results]
             with open(path, "w") as f:
                 json.dump(data, f, indent=2)

        elif file_format == "fasta":
             with open(path, "w") as f:
                 for i, r in enumerate(self.results):
                     f.write(f">seq_{i} mfe={r.mfe}\n")
                     f.write(f"{r.sequence}\n")
                     if r.structure:
                        f.write(f"{r.structure}\n")
        else:
             raise ValueError(f"Unsupported format: {file_format}")

# Config:
class ViennaRNAConfig(BaseConfig):
    """Configuration object for ViennaRNA secondary structure prediction.

    This class defines configuration parameters for running ViennaRNA's
    minimum free energy (MFE) folding algorithm.

    Attributes:
        temperature (float): Temperature in Celsius for energy calculations.
            Affects the thermodynamic parameters used in folding. Default: 37.0
            (physiological temperature).

        use_dna_params (bool): Whether to use DNA energy parameters instead of
            RNA parameters. When True, loads DNA_Mathews2004 parameters.
            Default: False (use RNA_Turner2004 parameters).

        no_lonely_pairs (bool): Disallow lonely base pairs (helices of length 1).
            This can reduce artifacts in structure prediction. Default: False.

        verbose (bool): Whether to print status messages during execution.
            Default: False.
    """
    temperature: float = ConfigField(
        title="Temperature",
        default=37.0,
        ge=-273.15,
        description="Temperature in Celsius for energy calculations",
    )
    use_dna_params: bool = ConfigField(
        title="Use DNA Parameters",
        default=False,
        description="Use DNA energy parameters instead of RNA parameters",
    )
    no_lonely_pairs: bool = ConfigField(
        title="No Lonely Pairs",
        default=False,
        description="Disallow lonely base pairs (helices of length 1)",
        advanced=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Print status messages",
    )

# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="viennarna",
    label="ViennaRNA Secondary Structure Prediction",
    input=ViennaRNAInput,
    config=ViennaRNAConfig,
    output=ViennaRNAOutput,
    description="RNA secondary structure prediction using ViennaRNA MFE folding",
)
def run_viennarna(
    inputs: ViennaRNAInput,
    config: ViennaRNAConfig,
) -> ViennaRNAOutput:
    """
    Predict RNA secondary structures using ViennaRNA's MFE algorithm.

    This function uses ViennaRNA's minimum free energy (MFE) algorithm to
    predict the most thermodynamically stable secondary structure for each
    input RNA sequence.

    Args:
        inputs: Input containing RNA sequences to fold.
        config: Configuration parameters for ViennaRNA.

    Returns:
        ViennaRNAOutput: Contains:
            results (List[ViennaRNAResult]): One result per input sequence with:
                sequence (str): The input sequence (converted to RNA if needed).
                structure (str): Predicted structure in dot-bracket notation.
                    '.' = unpaired, '(' and ')' = base pair.
                mfe (float): Minimum free energy in kcal/mol. More negative
                    values indicate more stable structures.

            metadata (dict): Run information including number of sequences
                processed and parameters used.

    Raises:
        ValueError: If sequences contain invalid characters.
        ImportError: If ViennaRNA (RNA module) is not installed.

    Example:
        >>> inputs = ViennaRNAInput(sequences=["GCGCUUUUGCGC"])
        >>> config = ViennaRNAConfig(temperature=37.0)
        >>> result = run_viennarna(inputs, config)
        >>> print(f"Structure: {result.results[0].structure}")
        >>> print(f"MFE: {result.results[0].mfe:.2f} kcal/mol")

    Note:
        - Structures are in dot-bracket notation where '.' means unpaired,
          '(' is the 5' partner, and ')' is the 3' partner of a base pair.
        - MFE values are in kcal/mol; more negative = more stable.
        - DNA sequences (containing T) are converted to RNA (U) unless
          use_dna_params is True.
    """
    logger.debug("Using standalone venv for ViennaRNA structure prediction...")

    from bio_programming_tools.utils.env_manager import EnvManager

    venv_manager = EnvManager(model_name="viennarna")

    # Prepare input data for inference script
    input_data = {
        "sequences": inputs.sequences,
        "temperature": config.temperature,
        "use_dna_params": config.use_dna_params,
        "no_lonely_pairs": config.no_lonely_pairs,
        "verbose": config.verbose,
    }

    # Call the inference script in the isolated venv
    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "inference.py",
        input_dict=input_data,
        device="cpu",  # ViennaRNA doesn't use GPU
        verbose=config.verbose,
    )

    # Convert results back to ViennaRNAResult objects
    results = [
        ViennaRNAResult(
            sequence=r["sequence"],
            structure=r["structure"],
            mfe=r["mfe"],
        )
        for r in output_data["results"]
    ]

    return ViennaRNAOutput(
        results=results,
        metadata=output_data["metadata"],
    )
