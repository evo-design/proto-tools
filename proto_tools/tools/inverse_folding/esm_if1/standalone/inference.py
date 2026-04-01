"""ESM-IF/ProteinDPO inference implementation for standalone venv execution."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from standalone_helpers import move_model_to_device

logger = getLogger(__name__)

DEFAULT_TEMPERATURE = 0.1
DEFAULT_SEED = 42


class ESMIF1Model:
    """ESM-IF1/ProteinDPO model for structure-conditioned inverse folding."""

    def __init__(self) -> None:
        """Initialize ESMIF1Model."""
        self._loaded = False
        self.device = None
        self.model = None
        self.alphabet = None
        self._weights_variant = None

    def _load_structure(
        self,
        pdb_structure: str,
        chain_ids: list[str],
    ) -> Any:
        """Load structure and extract coords for the complex.

        Returns:
            Tuple of (all_coords, all_native_seqs, target_chain).
        """
        import biotite.structure
        import esm.inverse_folding.multichain_util
        import esm.inverse_folding.util

        structure = esm.inverse_folding.util.load_structure(pdb_structure)
        structure = biotite.structure.array([atom for atom in structure if not atom.hetero])
        all_coords, all_native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        target_chain = chain_ids[0] if chain_ids else next(iter(all_coords.keys()))
        return all_coords, all_native_seqs, target_chain

    def _sample_with_fixed_positions(
        self,
        all_coords: dict[str, Any],
        all_native_seqs: dict[str, Any],
        target_chain: str,
        fixed_pos: list[int],
        temperature: float,
    ) -> str:
        """Sample a sequence with certain positions fixed to native residues.

        Uses the model's partial_seq support: positions pre-filled with amino
        acid tokens are kept fixed during autoregressive decoding.

        Args:
            all_coords: Coords dict from extract_coords_from_complex.
            all_native_seqs: Native sequences dict.
            target_chain: Chain to design.
            fixed_pos: 1-indexed positions to keep fixed.
            temperature: Sampling temperature.

        Returns:
            Sampled sequence string.
        """
        import esm.inverse_folding.multichain_util

        native_seq = all_native_seqs[target_chain]
        # Convert 1-indexed to 0-indexed
        fixed_indices = {p - 1 for p in fixed_pos}

        # Build partial_seq following multichain_util convention:
        # target chain positions get '<mask>' (sampled) or native residue (fixed),
        # other chain positions get '<pad>' (context only)
        all_chain_ids = list(all_coords.keys())
        partial_seq = []
        for chain_id in all_chain_ids:
            chain_seq = all_native_seqs[chain_id]
            if chain_id == target_chain:
                for i, res in enumerate(chain_seq):
                    if i in fixed_indices:
                        partial_seq.append(res)
                    else:
                        partial_seq.append("<mask>")
            else:
                partial_seq.extend(["<pad>"] * len(chain_seq))

        # Build coords tensor matching multichain_util._concatenate_coords
        all_coords_concat = esm.inverse_folding.multichain_util._concatenate_coords(all_coords, target_chain)

        sampled_seq = self.model.sample(  # type: ignore[attr-defined]
            all_coords_concat,
            partial_seq=partial_seq,
            temperature=temperature,
        )
        # Extract only target chain portion
        target_len = len(native_seq)
        return sampled_seq[:target_len]  # type: ignore[no-any-return]

    def sample(
        self,
        pdb_structure: str,
        chain_ids: list[str],
        batch_size: int,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        device: str = "cuda",
        weights_variant: str = "protein_dpo",
        verbose: bool = False,
        fixed_positions: dict[str, list[int]] | None = None,
    ) -> dict[str, Any]:
        """Sample sequences using ESM-IF autoregressive decoder.

        Args:
            pdb_structure: Path to PDB file containing the structure.
            chain_ids: List of chain IDs. First chain is the target for design.
            batch_size: Number of sequences to generate.
            temperature: Sampling temperature (default: 0.1).
            seed: Random seed for reproducibility.
            device: Device to run on ('cuda' or 'cpu').
            weights_variant: 'esmif' for vanilla or 'protein_dpo' for DPO weights.
            verbose: Whether to print status messages.
            fixed_positions: Optional dict mapping chain IDs to lists of
                1-indexed positions to keep fixed at native residue identity.

        Returns:
            Dictionary with keys: sequences, log_likelihoods
        """
        if not self._loaded or self._weights_variant != weights_variant:
            self.load(device, weights_variant, verbose)
        elif self.device != device:  # type: ignore[unreachable]
            self.to_device(device)

        import esm.inverse_folding.multichain_util

        all_coords, all_native_seqs, target_chain = self._load_structure(pdb_structure, chain_ids)

        sequences = []
        log_likelihoods = []

        torch.manual_seed(seed)
        for _ in range(batch_size):
            if fixed_positions and target_chain in fixed_positions:
                sampled_seq = self._sample_with_fixed_positions(
                    all_coords,
                    all_native_seqs,
                    target_chain,
                    fixed_positions[target_chain],
                    temperature,
                )
            else:
                sampled_seq = esm.inverse_folding.multichain_util.sample_sequence_in_complex(
                    self.model,
                    all_coords,
                    target_chain,
                    temperature=temperature,
                )
            sequences.append(sampled_seq)

            # Score sampled sequence to get log-likelihood
            avg_ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                self.model,
                self.alphabet,
                all_coords,
                target_chain,
                sampled_seq,
            )
            log_likelihoods.append(float(avg_ll))

        return {
            "sequences": sequences,
            "log_likelihoods": log_likelihoods,
        }

    def score(
        self,
        pdb_structure: str,
        chain_ids: list[str],
        sequence: str,
        device: str = "cuda",
        weights_variant: str = "protein_dpo",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Score a sequence against a structure using the score_complex approach.

        Uses score_sequence_in_complex to score the full sequence with
        multi-chain structural context (matching ProteinDPO score_complex.py
        --no_mutations path, equivalent to LigandMPNN default scoring).

        Args:
            pdb_structure: Path to PDB file containing the structure.
            chain_ids: List of chain IDs. First chain is the target for scoring.
            sequence: Protein sequence to score.
            device: Device to run on.
            weights_variant: 'esmif' for vanilla or 'protein_dpo' for DPO weights.
            verbose: Whether to print status messages.

        Returns:
            Dictionary with keys: metrics (dict of scalar metrics)
        """
        if not self._loaded or self._weights_variant != weights_variant:
            self.load(device, weights_variant, verbose)
        elif self.device != device:  # type: ignore[unreachable]
            self.to_device(device)

        import esm.inverse_folding.multichain_util

        all_coords, _all_native_seqs, target_chain = self._load_structure(pdb_structure, chain_ids)

        # Score the sequence in the complex context
        avg_ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
            self.model,
            self.alphabet,
            all_coords,
            target_chain,
            sequence,
        )

        metrics = {
            "avg_log_likelihood": float(avg_ll),
            "perplexity": float(math.exp(-avg_ll)),
        }

        return {"metrics": metrics}

    def load(
        self,
        device: str,
        weights_variant: str = "protein_dpo",
        verbose: bool = False,
    ) -> None:
        """Load ESM-IF1 model, optionally with ProteinDPO weights."""
        if verbose:
            logger.info(f"Loading ESM-IF ({weights_variant}) on {device}")

        import esm
        import esm.pretrained

        # Load base ESM-IF1 model from pre-downloaded weights
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("esm_if1")
        if not weights_dir:
            venv = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH", ".")
            weights_dir = os.path.join(venv, "weights")
        base_weights_path = os.path.join(weights_dir, "esm_if1_gvp4_t16_142M_UR50.pt")

        # Load with weights_only=False for PyTorch 2.6+ compatibility
        # (ESM-IF checkpoint contains argparse.Namespace objects)
        model_data = torch.load(base_weights_path, map_location="cpu", weights_only=False)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_core(
            "esm_if1_gvp4_t16_142M_UR50", model_data
        )

        # Apply ProteinDPO weights if requested
        if weights_variant == "protein_dpo":
            dpo_weights_path = os.path.join(weights_dir, "paired_weights.pt")
            if os.path.exists(dpo_weights_path):
                state_dict = torch.load(dpo_weights_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state_dict, strict=True)  # type: ignore[attr-defined]
                if verbose:
                    logger.info(f"Loaded ProteinDPO weights from {dpo_weights_path}")
            else:
                raise FileNotFoundError(
                    f"ProteinDPO weights not found at {dpo_weights_path}. "
                    f"Run setup.sh or set PROTO_ESM_IF1_WEIGHTS_DIR."
                )

        self.model = self.model.to(device)  # type: ignore[attr-defined]
        self.model.eval()  # type: ignore[attr-defined]
        self.device = device  # type: ignore[assignment]
        self._weights_variant = weights_variant  # type: ignore[assignment]
        self._loaded = True

        if verbose:
            logger.info("ESM-IF model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device  # type: ignore[assignment]

    def unload(self) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            self.model = self.model.to("cpu")  # type: ignore[attr-defined]
            self.device = "cpu"  # type: ignore[assignment]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
_model: ESMIF1Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ESMIF1Model()

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
                pdb_structure=pdb_structure,  # type: ignore[arg-type]
                chain_ids=input_dict.get("chain_ids", []),
                batch_size=input_dict.get("batch_size", 1),
                temperature=input_dict.get("temperature", DEFAULT_TEMPERATURE),
                seed=input_dict.get("seed", DEFAULT_SEED),
                device=input_dict.get("device", "cuda"),
                weights_variant=input_dict.get("weights_variant", "protein_dpo"),
                verbose=input_dict.get("verbose", False),
                fixed_positions=input_dict.get("fixed_positions"),
            )
        if operation == "score":
            return _model.score(
                pdb_structure=pdb_structure,  # type: ignore[arg-type]
                chain_ids=input_dict.get("chain_ids", []),
                sequence=input_dict.get("sequence"),  # type: ignore[arg-type]
                device=input_dict.get("device", "cuda"),
                weights_variant=input_dict.get("weights_variant", "protein_dpo"),
                verbose=input_dict.get("verbose", False),
            )
        raise ValueError(f"Unknown operation: {operation}")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
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
        json.dump(_serialize_output(result), f)
