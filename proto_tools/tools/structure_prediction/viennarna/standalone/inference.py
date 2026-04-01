"""Standalone inference script for ViennaRNA secondary structure prediction.

This script provides a standalone interface for ViennaRNA that can be executed
in an isolated virtual environment with JSON-based I/O.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ViennaRNAModel:
    """ViennaRNA secondary structure prediction model.

    This class provides a lightweight wrapper around ViennaRNA's MFE folding
    algorithm with lazy loading and JSON-compatible I/O.
    """

    def __init__(self) -> None:
        """Initialize ViennaRNA model (lazy loading)."""
        self._RNA = None
        logger.debug("ViennaRNAModel initialized (lazy loading)")

    def load(self) -> None:
        """Load ViennaRNA package if not already loaded."""
        if self._RNA is None:
            import RNA

            self._RNA = RNA
            logger.debug("ViennaRNA package loaded")

    def __call__(
        self,
        sequences: list[str],
        temperature: float = 37.0,
        use_dna_params: bool = False,
        no_lonely_pairs: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Predict RNA secondary structures using ViennaRNA's MFE algorithm.

        Args:
            sequences: List of RNA sequences to fold. Each sequence should
                contain only valid RNA nucleotides (A, U, G, C) or DNA
                nucleotides (A, T, G, C). Lowercase letters are also accepted.
            temperature: Temperature in Celsius for energy calculations.
                Default: 37.0 (physiological temperature).
            use_dna_params: Whether to use DNA energy parameters instead of
                RNA parameters. Default: False (use RNA_Turner2004 parameters).
            no_lonely_pairs: Disallow lonely base pairs (helices of length 1).
                Default: False.
            verbose: Whether to print status messages during execution.
                Default: False.

        Returns:
            Dictionary containing:
                - results: List of dicts, each with:
                    - sequence: The input sequence (normalized)
                    - structure: Predicted structure in dot-bracket notation
                    - mfe: Minimum free energy in kcal/mol
                - metadata: Run information (num_sequences, parameters used)

        Raises:
            ValueError: If sequences contain invalid characters or list is empty.
        """
        # Ensure ViennaRNA is loaded
        self.load()
        RNA = self._RNA

        # Set logging level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)

        # Validate input sequences
        if not sequences:
            raise ValueError("At least one sequence is required")

        valid_chars = set("AUGCTNaugctn")
        for seq_idx, seq in enumerate(sequences):
            invalid_chars = set(seq) - valid_chars
            if invalid_chars:
                raise ValueError(
                    f"Invalid nucleotide characters in sequence {seq_idx}: {', '.join(sorted(invalid_chars))}"
                )

        # Configure model details
        md = RNA.md()  # type: ignore[attr-defined]
        md.temperature = temperature
        md.noLP = 1 if no_lonely_pairs else 0

        # Load appropriate energy parameters
        if use_dna_params:
            RNA.params_load_DNA_Mathews2004()  # type: ignore[attr-defined]
            logger.debug("Using DNA energy parameters (Mathews 2004)")
        else:
            RNA.params_load_RNA_Turner2004()  # type: ignore[attr-defined]
            logger.debug("Using RNA energy parameters (Turner 2004)")

        # Fold each sequence
        results = []
        for seq_idx, sequence in enumerate(sequences):
            if not sequence:
                logger.debug(f"Warning: Found empty sequence (index {seq_idx})")
                results.append(
                    {
                        "sequence": sequence,
                        "structure": None,
                        "mfe": None,
                    }
                )
                continue

            # Normalize sequence: uppercase and convert T to U for RNA
            normalized_seq = sequence.upper()
            if not use_dna_params:
                normalized_seq = normalized_seq.replace("T", "U")

            logger.debug(f"Folding sequence {seq_idx + 1}/{len(sequences)} ({len(normalized_seq)} nt)")

            # Create fold compound and compute MFE structure
            fc = RNA.fold_compound(normalized_seq, md)  # type: ignore[attr-defined]
            structure, mfe = fc.mfe()

            results.append(
                {
                    "sequence": normalized_seq,  # type: ignore[dict-item]
                    "structure": structure,
                    "mfe": float(mfe),  # type: ignore[dict-item]
                }
            )

            if verbose:
                logger.info(f"Sequence {seq_idx + 1}: {normalized_seq[:20]}... MFE={mfe:.2f} kcal/mol")

        return {
            "results": results,
            "metadata": {
                "num_sequences": len(results),
                "temperature": temperature,
                "use_dna_params": use_dna_params,
                "no_lonely_pairs": no_lonely_pairs,
            },
        }


# ============================================================================
# Dispatch
# ============================================================================
_model: ViennaRNAModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ViennaRNAModel()

    kwargs = dict(input_dict)
    operation = kwargs.pop("operation", "predict")
    kwargs.pop("device", None)  # ViennaRNA is CPU-only, doesn't use device
    if operation == "predict":
        return _model(**kwargs)
    raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Standalone Entry Point
# ============================================================================


def to_device(device: str) -> dict[str, Any]:
    """Passthrough - tool does not maintain persistent state."""
    return {"success": True, "device": device}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
