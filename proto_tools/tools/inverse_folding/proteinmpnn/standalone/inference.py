"""ProteinMPNN standalone inference implementation for venv execution."""

import json
import math
import os
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any

# JAX memory settings - prevent preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

logger = getLogger(__name__)

DEFAULT_TEMPERATURE = 1.0

# Alphabet ordering for logits interpretation
ALPHAFOLD_VOCAB: list[str] = list(
    "ARNDCQEGHILKMFPSTWYVX"
)  # ColabDesign autoconverts to Alphafold alphabet for ProteinMPNN scoring

# Maps model_choice to ColabDesign's (model_name, weights) parameters
_MODEL_CONFIG: dict[str, tuple[str, str]] = {
    "proteinmpnn": ("v_48_020", "original"),
    "abmpnn": ("abmpnn", "original"),
    "soluble": ("v_48_020", "soluble"),
}


class ProteinMPNNModel:
    """ProteinMPNN model for structure-conditioned protein sequence design."""

    def __init__(self) -> None:
        """Initialize ProteinMPNNModel."""
        self._loaded = False
        self._model_choice: str | None = None
        self.device: str | None = None
        self.params: Any = None
        self.model: Any = None

    def sample(
        self,
        pdb_structure: str,
        chain_ids: list[str],
        batch_size: int,
        temperature: float | None = DEFAULT_TEMPERATURE,
        fixed_positions: dict[str, list[int]] | None = None,
        excluded_amino_acids: list[str] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        model_choice: str = "proteinmpnn",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Sample protein sequences from the ProteinMPNN model.

        Args:
            pdb_structure: Path to PDB file or PDB content string
            chain_ids: List of chain IDs to design
            batch_size: Number of sequences to generate
            temperature: Sampling temperature
            fixed_positions: Dict mapping chain IDs to fixed residue positions
            excluded_amino_acids: List of amino acids to exclude
            seed: Random seed
            device: Device to run on ('cuda' or 'cpu')
            model_choice: Model weights ('proteinmpnn', 'abmpnn', or 'soluble')
            verbose: Whether to print status messages
            return_logits: Whether to include logits in the output

        Returns:
            Dictionary with keys: seq, score, seqid, and optionally logits
        """
        from standalone_helpers import set_jax_seed

        key = set_jax_seed(seed)
        if key is None:
            raise ValueError("proteinmpnn: sample requires an explicit int seed (jax.random.PRNGKey rejects None)")

        # Lazy load the model (reload if model_choice changed)
        if not self._loaded or self._model_choice != model_choice:
            self.load(device, model_choice, verbose)
        elif self.device != device:
            self.to_device(device)

        fix_pos = (
            ",".join(f"{chain}{idx}" for chain, positions in fixed_positions.items() for idx in positions)
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
            key=key,
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
        chain_ids: list[str],
        sequence: str,
        fixed_positions: dict[str, list[int]] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        model_choice: str = "proteinmpnn",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Score a protein sequence against a structure.

        Args:
            pdb_structure: Path to PDB file or PDB content string
            chain_ids: List of chain IDs
            sequence: Sequence to score
            fixed_positions: Dict mapping chain IDs to fixed positions
            seed: Random seed (required — jax.random.PRNGKey rejects None)
            device: Device to run on
            model_choice: Model weights ('proteinmpnn', 'abmpnn', or 'soluble')
            verbose: Whether to print status messages
            return_logits: Whether to include logits in the output

        Returns:
            Dictionary with keys: metrics (log_likelihood, avg_log_likelihood, perplexity),
            and optionally logits.
        """
        from standalone_helpers import set_jax_seed

        key = set_jax_seed(seed)
        if key is None:
            raise ValueError("proteinmpnn: score requires an explicit int seed (jax.random.PRNGKey rejects None)")

        # Lazy load the model (reload if model_choice changed)
        if not self._loaded or self._model_choice != model_choice:
            self.load(device, model_choice, verbose)
        elif self.device != device:
            self.to_device(device)

        fix_pos = (
            ",".join(f"{chain}{idx}" for chain, positions in fixed_positions.items() for idx in positions)
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
            key=key,
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

    def load(self, device: str, model_choice: str = "proteinmpnn", verbose: bool = False) -> None:
        """Load ProteinMPNN model to device.

        Args:
            device: Device to load the model on.
            model_choice: Model weights ('proteinmpnn', 'abmpnn', or 'soluble').
            verbose: Whether to print status messages.
        """
        self.verbose = verbose
        model_name, weights = _MODEL_CONFIG.get(model_choice, ("v_48_020", "original"))

        if self.verbose:
            logger.info(f"Loading {model_choice} (model_name={model_name}, weights={weights}) on {device}")

        # Lazy import ProteinMPNN from ColabDesign
        from colabdesign.mpnn import mk_mpnn_model

        # Load the Flax module (params land on CPU by default)
        self.model = mk_mpnn_model(model_name=model_name, weights=weights)
        self.params = self.model._model.params
        self.device = "cpu"
        self._model_choice = model_choice

        # Move the model parameters to the selected device
        self.to_device(device)
        self._loaded = True

        if self.verbose:
            logger.info(f"{model_choice} model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move the model params to the selected device via move_model_to_device."""
        from standalone_helpers import move_model_to_device

        if self.model is None:
            raise ValueError("proteinmpnn: cannot move unloaded model to device — call load() first")
        if self.device == device:
            return

        if self.verbose:
            logger.info(f"Moving ProteinMPNN to {device}")

        # params is a dict pytree; move_model_to_device handles via device_put
        self.params = move_model_to_device(self.params, self.device, device)
        self.device = device

    def unload(self) -> None:
        """Move ProteinMPNN params back to CPU and free GPU HBM."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            return

        if self.verbose:
            logger.info("Unloading ProteinMPNN to CPU")

        self.params = move_model_to_device(self.params, self.device, "cpu")
        self.device = "cpu"


# ============================================================================
# Dispatch
# ============================================================================
_model: ProteinMPNNModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
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

        operation = input_dict["operation"]
        model_choice = input_dict["model_choice"]
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
                model_choice=model_choice,
                verbose=input_dict["verbose"],
                return_logits=input_dict["return_logits"],
            )
        if operation == "score":
            return _model.score(
                pdb_structure=pdb_structure,  # type: ignore[arg-type]
                chain_ids=input_dict["chain_ids"],
                sequence=input_dict["sequence"],
                fixed_positions=input_dict.get("fixed_positions"),
                seed=input_dict["seed"],
                device=input_dict["device"],
                model_choice=model_choice,
                verbose=input_dict["verbose"],
                return_logits=input_dict["return_logits"],
            )
        raise ValueError(f"proteinmpnn: unknown operation {operation!r}; valid: ['sample', 'score']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and hasattr(_model, "to_device"):
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_jax_memory_stats

    return get_jax_memory_stats(device_index=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("proteinmpnn: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    from standalone_helpers import serialize_output

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
