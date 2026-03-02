"""
ProteinMPNN standalone inference implementation for venv execution.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

# JAX memory settings - prevent preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

logger = getLogger(__name__)

DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 42

# Alphabet ordering for logits interpretation
ALPHAFOLD_VOCAB: List[str] = list("ARNDCQEGHILKMFPSTWYVX")  # ColabDesign autoconverts to Alphafold alphabet for ProteinMPNN scoring

class ProteinMPNNModel:
    """ProteinMPNN model for structure-conditioned protein sequence design."""

    def __init__(self):
        self._loaded = False
        self.device = None
        self.params = None
        self.model = None
        self.jax = None

    def sample(
        self,
        pdb_structure: str,
        chain_ids: List[str],
        batch_size: int,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        fixed_positions: Optional[Dict[str, List[int]]] = None,
        excluded_amino_acids: Optional[List[str]] = None,
        seed: Optional[int] = DEFAULT_SEED,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample protein sequences from the ProteinMPNN model.

        Args:
            pdb_structure: Path to PDB file or PDB content string
            chain_ids: List of chain IDs to design
            batch_size: Number of sequences to generate
            temperature: Sampling temperature
            fixed_positions: Dict mapping chain IDs to fixed residue positions
            excluded_amino_acids: List of amino acids to exclude
            seed: Random seed
            device: Device to run on ('cuda' or 'cpu')
            verbose: Whether to print status messages
            return_logits: Whether to include logits in the output

        Returns:
            Dictionary with keys: seq, score, seqid, and optionally logits
        """
        # Lazy load the model
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        fix_pos = (
            ",".join(
                f"{chain}{idx}"
                for chain, positions in fixed_positions.items()
                for idx in positions
            )
            if fixed_positions is not None
            else None
        )

        # Load the PDB file
        self.model.prep_inputs(
            pdb_structure,
            fix_pos=fix_pos,
            chain=",".join(chain_ids),
            rm_aa=",".join(excluded_amino_acids) if excluded_amino_acids else None,
        )

        # Sample sequences
        sequences = self.model.sample_parallel(
            batch=batch_size,
            temperature=temperature,
            key=self.jax.random.PRNGKey(seed),
        )

        self.unload()
        return {
            "seq": sequences["seq"],
            "score": sequences["score"],
            "seqid": sequences["seqid"],
            "logits": sequences["logits"] if return_logits else None,
        }

    def score(
        self,
        pdb_structure: str,
        chain_ids: List[str],
        sequence: str,
        fixed_positions: Optional[Dict[str, List[int]]] = None,
        seed: int = DEFAULT_SEED,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Score a protein sequence against a structure.

        Args:
            pdb_structure: Path to PDB file or PDB content string
            chain_ids: List of chain IDs
            sequence: Sequence to score
            fixed_positions: Dict mapping chain IDs to fixed positions
            seed: Random seed
            device: Device to run on
            verbose: Whether to print status messages
            return_logits: Whether to include logits in the output

        Returns:
            Dictionary with keys: metrics (log_likelihood, avg_log_likelihood, perplexity),
            and optionally logits.
        """
        # Lazy load the model
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        fix_pos = (
            ",".join(
                f"{chain}{idx}"
                for chain, positions in fixed_positions.items()
                for idx in positions
            )
            if fixed_positions is not None
            else None
        )

        # Prepare input
        self.model.prep_inputs(
            pdb_structure,
            fix_pos=fix_pos,
            chain=",".join(chain_ids),
        )

        # Score the sequence (model returns "score" (negative avg log likelihood), "logits")
        output = self.model.score(
            seq=sequence,
            key=self.jax.random.PRNGKey(seed),
        )

        neg_avg_ll = float(output["score"])
        metrics = {
            "log_likelihood": -neg_avg_ll * len(sequence),
            "avg_log_likelihood": -neg_avg_ll,
            "perplexity": float(math.exp(neg_avg_ll)),
        }

        self.unload()
        return {
            "logits": output["logits"] if return_logits else None,
            "metrics": metrics,
            "vocab": ALPHAFOLD_VOCAB,
        }

    def load(self, device: str, verbose: bool = False):
        """Load ProteinMPNN model to device."""
        self.verbose = verbose

        if self.verbose:
            logger.info(f"Loading ProteinMPNN model on {device}")

        import jax
        self.jax = jax

        # Lazy import ProteinMPNN from ColabDesign
        from colabdesign.mpnn import mk_mpnn_model

        # Load the Flax module (params land on CPU by default)
        self.model = mk_mpnn_model()
        self.params = self.model._model.params
        self.device = "cpu"

        # Move the model parameters to the selected device
        self.to_device(device)
        self._loaded = True

        if self.verbose:
            logger.info("ProteinMPNN model loaded successfully")

    def to_device(self, device: str):
        """Move the model params to the selected device via move_model_to_device."""
        from standalone_helpers import move_model_to_device

        if self.model is None:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")
        if self.device == device:
            return

        if self.verbose:
            logger.info(f"Moving ProteinMPNN to {device}")

        # params is a dict pytree — move_model_to_device handles via device_put
        self.params = move_model_to_device(self.params, self.device, device)
        self.device = device

    def unload(self):
        """Move ProteinMPNN params back to CPU and free GPU HBM."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            return

        if self.verbose:
            logger.info("Unloading ProteinMPNN to CPU")

        self.params = move_model_to_device(self.params, self.device, "cpu")
        self.device = "cpu"


def _serialize_output(value: Any) -> Any:
    """Recursively serialize tensors and arrays to JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _serialize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_output(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


# ============================================================================
# Dispatch
# ============================================================================
_model: ProteinMPNNModel | None = None


def dispatch(input_dict: dict) -> dict:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ProteinMPNNModel()

    # Handle pdb_contents -> temp file
    pdb_contents = input_dict.get("pdb_contents")
    pdb_structure = input_dict.get("pdb_structure")

    with tempfile.TemporaryDirectory() as temp_dir:
        if pdb_contents and not pdb_structure:
            pdb_path = Path(temp_dir) / "input.pdb"
            pdb_path.write_text(pdb_contents)
            pdb_structure = str(pdb_path)

        operation = input_dict.get("operation", "sample")
        if operation == "sample":
            return _model.sample(
                pdb_structure=pdb_structure,
                chain_ids=input_dict.get("chain_ids", []),
                batch_size=input_dict.get("batch_size", 1),
                temperature=input_dict.get("temperature", DEFAULT_TEMPERATURE),
                fixed_positions=input_dict.get("fixed_positions"),
                excluded_amino_acids=input_dict.get("excluded_amino_acids"),
                seed=input_dict.get("seed", DEFAULT_SEED),
                device=input_dict.get("device", "cuda"),
                verbose=input_dict.get("verbose", False),
                return_logits=input_dict.get("return_logits", False),
            )
        elif operation == "score":
            return _model.score(
                pdb_structure=pdb_structure,
                chain_ids=input_dict.get("chain_ids", []),
                sequence=input_dict.get("sequence"),
                fixed_positions=input_dict.get("fixed_positions"),
                seed=input_dict.get("seed", DEFAULT_SEED),
                device=input_dict.get("device", "cuda"),
                verbose=input_dict.get("verbose", False),
                return_logits=input_dict.get("return_logits", False),
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")



def to_device(device: str) -> dict:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and hasattr(_model, "to_device"):
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_jax_memory_stats

    return get_jax_memory_stats(device_index=0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(_serialize_output(result), f)
