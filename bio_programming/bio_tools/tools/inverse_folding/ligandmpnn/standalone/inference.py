"""
LigandMPNN inference implementation using Foundry.
"""
from __future__ import annotations

import json
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional
import gc
import torch

logger = getLogger(__name__)

DEFAULT_TEMPERATURE = 0.1
# TODO: Standardize where this seed is set
DEFAULT_SEED = 42

# Alphabet ordering for logits interpretation (standard MPNN)
MPNN_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"

class LigandMPNNModel:
    """LigandMPNN model for ligand-aware protein sequence design using Foundry."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
    ):
        self._loaded = False
        self._engine = None
        self.device = None
        self.checkpoint_path = checkpoint_path

    def sample(
        self,
        pdb_structure: str,
        chain_ids: List[str],
        batch_size: int,
        temperature: float = DEFAULT_TEMPERATURE,
        fixed_positions: Optional[Dict[str, List[int]]] = None,
        excluded_amino_acids: Optional[List[str]] = None,
        seed: int = DEFAULT_SEED,
        device: str = "cuda",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample protein sequences using LigandMPNN.

        Args:
            pdb_structure: Path to PDB file containing the structure.
            chain_ids: List of chain IDs to design.
            batch_size: Number of sequences to generate.
            temperature: Sampling temperature (default: 0.1).
            fixed_positions: Dict mapping chain IDs to fixed residue positions.
            excluded_amino_acids: List of amino acids to exclude.
            seed: Random seed for reproducibility.
            device: Device to run on ('cuda' or 'cpu').
            verbose: Whether to print status messages.

        Returns:
            Dictionary with keys: sequences, metrics
        """
        # Lazy load the model
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.load(device, verbose)

        # Build fixed_residues list from fixed_positions dict
        fixed_residues = None
        if fixed_positions:
            fixed_residues = [
                f"{chain}{pos}"
                for chain, positions in fixed_positions.items()
                for pos in positions
            ]

        # Build input dict for Foundry engine
        # NOTE: Cannot mix residue-based (fixed_residues) and chain-based (designed_chains)
        # design constraints in the same input. If fixed_residues is provided, we omit
        # designed_chains - the designable scope is implicitly defined by unfixed residues.
        input_dict = {
            "structure_path": pdb_structure,
            "name": "design",
            "seed": seed,
            "batch_size": batch_size,
            "number_of_batches": 1,
            "temperature": temperature,
            "omit_aa": excluded_amino_acids,
        }

        if fixed_residues:
            # Use residue-based constraints only
            input_dict["fixed_residues"] = fixed_residues
            if chain_ids:
                logger.warning(
                    "Both fixed_positions and chain_ids were provided. LigandMPNN does not support mixing residue-based and chain-based "
                    "design constraints. The chain_ids parameter will be ignored; designable scope is determined by residues NOT in fixed_positions."
                )
        else:
            # Use chain-based constraints only
            input_dict["designed_chains"] = chain_ids

        # Run inference
        results = self._engine.run(input_dicts=[input_dict])

        # Extract sequences and metrics
        sequences: List[str] = []
        metrics: List[Dict[str, Any]] = []
        for output in results:
            sequences.append(output.output_dict["designed_sequence"])
            metrics.append(output.output_dict)

        self.unload()
        return {"sequences": sequences, "metrics": metrics}

    def score(
        self,
        pdb_structure: str,
        chain_ids: List[str],
        sequence: str,
        fixed_positions: Optional[Dict[str, List[int]]] = None,
        seed: int = DEFAULT_SEED,
        device: str = "cuda",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Score a protein sequence against a structure via forward pass.

        Computes logits for the sequence given the structure, then calculates
        scoring metrics (log_likelihood, avg_log_likelihood, perplexity) from
        the logits.

        Args:
            pdb_structure: Path to PDB file containing the structure.
            chain_ids: List of chain IDs to score.
            sequence: Protein sequence to score.
            fixed_positions: Dict mapping chain IDs to fixed residue positions.
            seed: Random seed for reproducibility.
            device: Device to run on ('cuda' or 'cpu').
            verbose: Whether to print status messages.

        Returns:
            Dictionary with keys:
                - logits: Per-position logits array (seq_len, vocab_size)
                - metrics: Dict with log_likelihood, avg_log_likelihood, perplexity
        """
        # TODO: Implement LigandMPNN scoring
        pass

    def load(self, device: str = "cuda", verbose: bool = False):
        """Load the LigandMPNN model via Foundry."""
        if verbose:
            logger.info(f"Loading LigandMPNN model on {device}")

        from mpnn.inference_engines.mpnn import MPNNInferenceEngine

        self._engine = MPNNInferenceEngine(
            model_type="ligand_mpnn",
            checkpoint_path=self.checkpoint_path,
            is_legacy_weights=True,
            device=device,
            write_fasta=False,
            write_structures=False,
        )
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("LigandMPNN model loaded successfully")

    def unload(self):
        """Unload the model to free GPU memory."""
        self._engine = None
        self._loaded = False
        self.device = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Get operation type
    operation = input_data.get("operation", "sample")

    # Handle pdb_contents -> temp file for standalone execution
    pdb_contents = input_data.get("pdb_contents")
    pdb_structure = input_data.get("pdb_structure")

    with tempfile.TemporaryDirectory() as temp_dir:
        if pdb_contents and not pdb_structure:
            # Write contents to temp file
            pdb_path = Path(temp_dir) / "input.pdb"
            pdb_path.write_text(pdb_contents)
            pdb_structure = str(pdb_path)

        # Create model
        model = LigandMPNNModel()

        if operation == "sample":
            result = model.sample(
                pdb_structure=pdb_structure,
                chain_ids=input_data.get("chain_ids", []),
                batch_size=input_data.get("batch_size", 1),
                temperature=input_data.get("temperature", DEFAULT_TEMPERATURE),
                fixed_positions=input_data.get("fixed_positions"),
                excluded_amino_acids=input_data.get("excluded_amino_acids"),
                seed=input_data.get("seed", DEFAULT_SEED),
                device=input_data.get("device", "cuda"),
                verbose=True,
            )
        elif operation == "score":
            result = model.score(
                pdb_structure=pdb_structure,
                chain_ids=input_data.get("chain_ids", []),
                sequence=input_data.get("sequence"),
                fixed_positions=input_data.get("fixed_positions"),
                seed=input_data.get("seed", DEFAULT_SEED),
                device=input_data.get("device", "cuda"),
                verbose=True,
            )

            # Convert logits to list for JSON serialization
            logits = result["logits"]
            if hasattr(logits, "tolist"):
                logits = logits.tolist()
            result = {
                "logits": logits,
                "metrics": result["metrics"],
            }
        else:
            raise ValueError(f"Unknown operation: {operation}. Must be 'sample' or 'score'.")

    with open(output_json_path, "w") as f:
        json.dump(result, f)
