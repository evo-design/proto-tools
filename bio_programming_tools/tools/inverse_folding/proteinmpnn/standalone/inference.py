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

        # Load the Flax module
        self.model = mk_mpnn_model()
        self.params = self.model._model.params

        # Move the model parameters to the selected device
        self.to_device(device)
        self._loaded = True

        if self.verbose:
            logger.info("ProteinMPNN model loaded successfully")

    def to_device(self, device: str):
        """Move the model to the selected device."""
        if self.model is None:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")
        if self.device == device:
            return

        if self.verbose:
            logger.info(f"Moving ProteinMPNN to {device}")

        device_obj = self.jax.devices(device)[0]
        self.params = self.jax.device_put(self.params, device_obj)
        self.device = device

    def unload(self):
        """Move ProteinMPNN params back to CPU and free GPU HBM."""
        if not self._loaded:
            return

        if self.verbose:
            logger.info("Unloading ProteinMPNN to CPU")

        cpu = self.jax.devices("cpu")[0]
        self.params = self.jax.device_put(self.params, cpu)
        self.device = "cpu"


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

        model = ProteinMPNNModel()

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
                verbose=input_data.get("verbose", False),
                return_logits=input_data.get("return_logits", False),
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
                return_logits=input_data.get("return_logits", False),
            )

        else:
            raise ValueError(f"Unknown operation: {operation}. Must be 'sample' or 'score'.")

    # Serialize for JSON output (convert numpy/JAX arrays to lists)
    clean_result = {}
    for k, v in result.items():
        if v is not None and hasattr(v, "tolist"):
            clean_result[k] = v.tolist()
        else:
            clean_result[k] = v

    with open(output_json_path, "w") as f:
        json.dump(clean_result, f)
