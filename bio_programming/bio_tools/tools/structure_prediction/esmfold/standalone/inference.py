"""
ESMFold standalone inference implementation.

This script can be run independently in an isolated venv with only ESMFold
dependencies installed. It communicates via JSON files for input/output.

Usage:
    python inference.py <input_json_path> <output_json_path>
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)
# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class ESMFoldModel:
    """
    ESMFold model for protein structure prediction.
    """

    def __init__(self):
        """Initialize ESMFold model wrapper."""
        self._loaded = False
        self.tokenizer = None
        self.device = None
        self.model = None

    def __call__(
        self,
        batch_data: List[Dict[str, Any]],
        residue_idx_offset: int,
        chain_linker: str,
        device: str = "cuda",
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run ESMFold structure prediction on protein sequences.

        Args:
            batch_data: List of dicts with keys: linked_seq, chains, seq_lengths
            residue_idx_offset: Offset between chains in residue numbering
            chain_linker: Sequence used to link chains
            device: Device to run on
            verbose: Whether to print status messages

        Returns:
            List of dicts with keys: pdb, avg_plddt, ptm
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Extract sequences
        linked_sequences = [item["linked_seq"] for item in batch_data]

        if verbose:
            logger.info(f"Starting ESMFold inference on {len(batch_data)} structure(s)...")

        # Enable trunk chunking for long sequences
        max_seq_len = max(len(seq) for seq in linked_sequences)
        if max_seq_len > 1200:
            if verbose:
                logger.info(f"Long sequence detected ({max_seq_len} residues), enabling trunk chunking (chunk_size=64)")
            self.model.trunk.set_chunk_size(64)

        # Use progress bar for batch processing
        with torch.inference_mode(), _allow_tf32():
            # Tokenize all sequences
            tokenized_inputs = self.tokenizer(
                linked_sequences,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            )
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

            # Build position_ids and linker_masks
            position_ids, linker_masks = self._build_batch_tensors(
                batch_data,
                residue_idx_offset,
                chain_linker
            )
            tokenized_inputs["position_ids"] = position_ids

            # Forward pass
            outputs = self.model(**tokenized_inputs)

            # Apply linker masking
            outputs["atom37_atom_exists"] = outputs["atom37_atom_exists"] * linker_masks[:, :, None]

        # Extract per-complex results
        results = []
        for idx in range(len(batch_data)):
            results.append(self._extract_result(outputs, idx))

        return results

    def _build_batch_tensors(
        self,
        batch_data: List[Dict[str, Any]],
        residue_idx_offset: int,
        chain_linker: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build position_ids and linker_masks for entire batch.

        Returns:
            position_ids: (batch_size, max_length) - residue numbering with offsets
            linker_masks: (batch_size, max_length) - 0 for linkers, 1 for residues
        """
        batch_size = len(batch_data)
        max_length = max(len(item["linked_seq"]) for item in batch_data)

        position_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        linker_masks = torch.zeros(batch_size, max_length, dtype=torch.float32, device=self.device)

        for batch_idx, item in enumerate(batch_data):
            seq_len = len(item["linked_seq"])

            # Build tensors for this complex
            pos_ids, mask = self._build_single_tensors(
                item["chains"],
                residue_idx_offset,
                chain_linker
            )

            # Fill batch tensors (rest stays 0/1 for padding)
            position_ids[batch_idx, :seq_len] = pos_ids
            linker_masks[batch_idx, :seq_len] = mask

        return position_ids, linker_masks

    def _build_single_tensors(
        self,
        chains: List[str],
        residue_idx_offset: int,
        chain_linker: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build position_ids and linker_mask for a single complex.

        Args:
            chains: List of chain sequences
            residue_idx_offset: Offset between chains in residue numbering
            chain_linker: Sequence used to link chains

        Returns:
            Tuple of (position_ids, linker_mask) tensors
        """
        # Calculate total length
        seq_length = sum(len(chain) for chain in chains)
        if len(chains) > 1:
            seq_length += len(chain_linker) * (len(chains) - 1)

        # Initialize
        position_ids = torch.arange(seq_length, device=self.device)
        linker_mask = torch.ones(seq_length, dtype=torch.float32, device=self.device)

        # Apply residue offsets and mask linkers
        pos = 0
        for chain_idx, chain in enumerate(chains):
            chain_len = len(chain)

            # Apply offset to this chain (only if offset > 0)
            if residue_idx_offset > 0:
                position_ids[pos : pos + chain_len] += chain_idx * residue_idx_offset
            pos += chain_len

            # Mask linker (if not last chain) - ALWAYS do this regardless of offset
            if chain_idx < len(chains) - 1:
                linker_len = len(chain_linker)
                linker_mask[pos : pos + linker_len] = 0
                pos += linker_len

        return position_ids, linker_mask

    def _extract_result(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        Extract results for a single complex from batched outputs.

        Returns:
            Dict with keys: pdb, avg_plddt, ptm, avg_pae
        """
        # Structure module tensors have shape (num_blocks, batch, ...) - batch is dim 1.
        # All other tensors have shape (batch, ...) - batch is dim 0.
        structure_module_tensors = {
            "positions", "frames", "sidechain_frames",
            "unnormalized_angles", "angles", "states", "lddt_head",
        }
        complex_output = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:  # Scalar
                    complex_output[key] = value
                elif key in structure_module_tensors:
                    complex_output[key] = value[:, batch_idx:batch_idx + 1, ...] # structure module tensors have batch as dim 1
                else:
                    complex_output[key] = value[batch_idx:batch_idx + 1] # all other tensors have batch as dim 0
            else:
                complex_output[key] = value

        # Convert to PDB
        pdb_output = self.model.output_to_pdb(complex_output)[0]

        # Calculate average pLDDT
        atom_exists = complex_output["atom37_atom_exists"]
        plddt = complex_output["plddt"]
        avg_plddt = ((plddt * atom_exists).sum() / atom_exists.sum().clamp(min=1)).item()

        # Extract PTM score
        ptm_tensor = complex_output.get("ptm")
        ptm = ptm_tensor.item() if ptm_tensor is not None else None

        # Calculate average pAE (masked to exclude padding)
        pae = complex_output.get("predicted_aligned_error")
        if pae is not None:
            # Create 1D mask from atom_exists (any atom existing means residue is valid)
            residue_mask = atom_exists.any(dim=-1)  # (1, seq_len)
            # Create 2D mask for valid (i, j) residue pairs
            pae_mask = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)  # (1, seq_len, seq_len)
            avg_pae = float((pae * pae_mask).sum() / pae_mask.sum().clamp(min=1))
        else:
            avg_pae = None

        return {
            "pdb": pdb_output,
            "avg_plddt": float(avg_plddt),
            "ptm": float(ptm) if ptm is not None else None,
            "avg_pae": avg_pae,
        }

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False):
        """Load ESMFold model and tokenizer to device."""
        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError:
            raise ImportError(
                "Could not import transformers. Make sure ESMFold dependencies "
                "are installed in the current environment."
            )

        if verbose:
            logger.info(f"Loading ESMFold model: facebook/esmfold_v1 on {device}")

        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", trust_remote_code=True
        )

        self.model = self.model.to(device)
        self.model.esm = self.model.esm.half()  # Convert to half precision for faster inference
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ESMFold model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.model = self.model.to(device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@contextmanager
def _allow_tf32():
    """Temporarily enable TF32 for matmul operations."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


# ============================================================================
# Standalone Script Entry Point
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python inference.py <input_json_path> <output_json_path>"
        )

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Create model and run inference
    model = ESMFoldModel()

    # Build kwargs for model call
    model_kwargs = {
        "batch_data": input_data["batch_data"],
        "residue_idx_offset": input_data["residue_idx_offset"],
        "chain_linker": input_data["chain_linker"],
        "device": input_data.get("device", "cuda"),
        "verbose": True,
    }

    results = model(**model_kwargs)

    # Write output to json file
    with open(output_json_path, "w") as f:
        json.dump({"results": results}, f)
