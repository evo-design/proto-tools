"""LigandMPNN inference implementation using Foundry."""

import gc
import json
import os
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from standalone_helpers import move_model_to_device, serialize_output

logger = getLogger(__name__)

DEFAULT_TEMPERATURE = 0.1

# Alphabet ordering for logits interpretation (standard MPNN)
MPNN_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"


class LigandMPNNModel:
    """LigandMPNN model for ligand-aware protein sequence design using Foundry."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
    ):
        """Initialize LigandMPNNModel."""
        self._loaded = False
        self._engine = None
        self.device: str | None = None
        self.checkpoint_path = checkpoint_path

    def sample(
        self,
        pdb_structure: str,
        chain_ids: list[str],
        batch_size: int,
        temperature: float = DEFAULT_TEMPERATURE,
        fixed_positions: dict[str, list[int]] | None = None,
        excluded_amino_acids: list[str] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Sample protein sequences using LigandMPNN.

        Args:
            pdb_structure: Path to PDB file containing the structure.
            chain_ids: List of chain IDs to design.
            batch_size: Number of sequences to generate.
            temperature: Sampling temperature (default: 0.1).
            fixed_positions: Dict mapping chain IDs to fixed residue positions.
            excluded_amino_acids: List of amino acids to exclude.
            seed: Random seed for reproducibility (required — Foundry engine
                expects an int).
            device: Device to run on ('cuda' or 'cpu').
            verbose: Whether to print status messages.

        Returns:
            Dictionary with keys: sequences, metrics
        """
        if seed is None:
            raise ValueError("LigandMPNNModel.sample requires an explicit int seed.")

        # Lazy load the model
        if not self._loaded or self.device != device:
            self.load(device, verbose)

        # Build fixed_residues list from fixed_positions dict
        fixed_residues = None
        if fixed_positions:
            fixed_residues = [f"{chain}{pos}" for chain, positions in fixed_positions.items() for pos in positions]

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
        results = self._engine.run(input_dicts=[input_dict])  # type: ignore[attr-defined]

        # Extract sequences and metrics
        sequences: list[str] = []
        metrics: list[dict[str, Any]] = []
        for output in results:
            sequences.append(output.output_dict["designed_sequence"])
            metrics.append(output.output_dict)

        self.unload()
        return {"sequences": sequences, "metrics": metrics}

    def score(
        self,
        pdb_structure: str,
        chain_ids: list[str],
        sequence: str,
        fixed_positions: dict[str, list[int]] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Score a protein sequence against a structure via forward pass.

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
        raise NotImplementedError(
            "LigandMPNN scoring is not yet implemented. Use proteinmpnn-score for protein-only contexts."
        )

    def load(self, device: str = "cuda", verbose: bool = False) -> None:
        """Load the LigandMPNN model via Foundry."""
        if verbose:
            logger.info(f"Loading LigandMPNN model on {device}")

        # Set FOUNDRY_CHECKPOINT_DIRS so Foundry finds BPT-managed weights
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("ligandmpnn")
        if weights_dir:
            os.environ["FOUNDRY_CHECKPOINT_DIRS"] = weights_dir

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

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        For LigandMPNN, this requires reloading the Foundry engine with the new device.
        """
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            # LigandMPNN uses Foundry engine which doesn't support standard .to() movement
            # Use helper for consistency (it will handle gracefully), then reload engine
            self._engine = move_model_to_device(self._engine, self.device, device)
            # Foundry engine requires full reload for device change
            self.load(device, verbose=False)

    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        self._engine = None
        self._loaded = False
        self.device = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: LigandMPNNModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = LigandMPNNModel(
            checkpoint_path=input_dict.get("checkpoint_path"),
        )

    # Handle pdb_contents -> temp file
    pdb_contents = input_dict.get("pdb_contents")
    pdb_structure = input_dict.get("pdb_structure")

    with tempfile.TemporaryDirectory() as temp_dir:
        if pdb_contents and not pdb_structure:
            pdb_path = Path(temp_dir) / "input.pdb"
            pdb_path.write_text(pdb_contents)
            pdb_structure = str(pdb_path)

        operation = input_dict["operation"]
        if operation == "sample":
            return _model.sample(
                pdb_structure=pdb_structure,  # type: ignore[arg-type]
                chain_ids=input_dict["chain_ids"],
                batch_size=input_dict["batch_size"],
                temperature=input_dict["temperature"],
                fixed_positions=input_dict.get("fixed_positions"),
                excluded_amino_acids=input_dict.get("excluded_amino_acids"),
                seed=input_dict["seed"],
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
        if operation == "score":
            return _model.score(
                pdb_structure=pdb_structure,  # type: ignore[arg-type]
                chain_ids=input_dict["chain_ids"],
                sequence=input_dict["sequence"],
                fixed_positions=input_dict.get("fixed_positions"),
                seed=input_dict["seed"],
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
        raise ValueError(f"Unknown operation: {operation}")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    # Model not loaded yet - will use device on next call
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
