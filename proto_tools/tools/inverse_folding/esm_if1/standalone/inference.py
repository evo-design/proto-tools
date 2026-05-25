"""ESM-IF/ProteinDPO inference implementation for standalone venv execution."""

import gc
import json
import math
import os
import sys
from typing import Any

import torch
from standalone_helpers import (
    get_logger,
    log_likelihood_metrics,
    move_model_to_device,
    serialize_output,
    set_torch_seed,
)

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 0.1


class ESMIF1Model:
    """ESM-IF1/ProteinDPO model for structure-conditioned inverse folding."""

    def __init__(self) -> None:
        """Initialize ESMIF1Model."""
        self._loaded = False
        self.device: str | None = None
        self.model: Any = None
        self.alphabet: Any = None
        self._weights_variant: str | None = None

    def _load_structure(
        self,
        pdb_path: str,
        chain_ids: list[str],
        target_chain: str | None = None,
    ) -> Any:
        """Load structure and extract coords for the complex.

        Args:
            pdb_path: Path to PDB file containing the structure.
            chain_ids: Chain IDs present in the structure.
            target_chain: Chain ID to designate as the target. When ``None``,
                falls back to the first entry of ``chain_ids`` (used by the
                sample path, which treats the first chain as the design target).

        Returns:
            Tuple of (all_coords, all_native_seqs, target_chain).
        """
        import biotite.structure
        import esm.inverse_folding.multichain_util
        import esm.inverse_folding.util
        from biotite.sequence import ProteinSequence

        def _is_known_aa(res_name: str) -> bool:
            try:
                ProteinSequence.convert_letter_3to1(res_name)
                return True
            except KeyError:
                return False

        structure = esm.inverse_folding.util.load_structure(pdb_path)
        # Drop residues biotite cannot convert to a 1-letter code.
        structure = biotite.structure.array([atom for atom in structure if _is_known_aa(atom.res_name)])
        all_coords, all_native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        if target_chain is None:
            target_chain = chain_ids[0] if chain_ids else next(iter(all_coords.keys()))
        if target_chain not in all_coords:
            raise ValueError(
                f"esm-if1: target_chain {target_chain!r} not found in structure coords "
                f"(available chains: {list(all_coords.keys())})"
            )
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

        sampled_seq = self.model.sample(
            all_coords_concat,
            partial_seq=partial_seq,
            temperature=temperature,
        )
        # Extract only target chain portion
        target_len = len(native_seq)
        return sampled_seq[:target_len]  # type: ignore[no-any-return]

    def sample(
        self,
        pdb_path: str,
        chain_ids: list[str],
        batch_size: int,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
        device: str = "cuda",
        weights_variant: str = "protein_dpo",
        verbose: bool = False,
        fixed_positions: dict[str, list[int]] | None = None,
    ) -> dict[str, Any]:
        """Sample sequences using ESM-IF autoregressive decoder.

        Args:
            pdb_path: Path to PDB file containing the structure.
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
        elif self.device != device:
            self.to_device(device)

        import esm.inverse_folding.multichain_util

        all_coords, all_native_seqs, target_chain = self._load_structure(pdb_path, chain_ids)

        sequences = []
        metrics = []

        set_torch_seed(seed)
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

            # Score sampled sequence; avg_ll is the mean per-position log-likelihood.
            avg_ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                self.model,
                self.alphabet,
                all_coords,
                target_chain,
                sampled_seq,
            )
            avg_ll = float(avg_ll)
            metrics.append(
                {
                    "log_likelihood": avg_ll * len(sampled_seq),
                    "avg_log_likelihood": avg_ll,
                    "perplexity": math.exp(-avg_ll),
                }
            )

        return {
            "sequences": sequences,
            "metrics": metrics,
        }

    def score(
        self,
        pdb_path: str,
        chain_ids: list[str],
        target_chain: str,
        sequence: str,
        seed: int | None = None,
        device: str = "cuda",
        weights_variant: str = "protein_dpo",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Score a sequence against a structure using the score_complex approach.

        Uses score_sequence_in_complex to score one chain's sequence within the
        full multi-chain structural context (matching ProteinDPO score_complex.py
        --no_mutations path).

        Args:
            pdb_path: Path to PDB file containing the structure.
            chain_ids: Chain IDs present in the structure.
            target_chain: Chain ID whose sequence is being scored.
            sequence: Target-chain-only sequence to score. Must have length equal
                to the number of residues in ``target_chain``.
            seed: Random seed.
            device: Device to run on.
            weights_variant: 'esmif' for vanilla or 'protein_dpo' for DPO weights.
            verbose: Whether to print status messages.

        Returns:
            Dictionary with keys: metrics (dict of scalar metrics)
        """
        if not self._loaded or self._weights_variant != weights_variant:
            self.load(device, weights_variant, verbose)
        elif self.device != device:
            self.to_device(device)

        import esm.inverse_folding.multichain_util

        all_coords, _all_native_seqs, target_chain = self._load_structure(pdb_path, chain_ids, target_chain)

        set_torch_seed(seed)
        # Score the sequence in the complex context
        avg_ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
            self.model,
            self.alphabet,
            all_coords,
            target_chain,
            sequence,
        )

        return {"metrics": log_likelihood_metrics(float(avg_ll), len(sequence))}

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
                self.model.load_state_dict(state_dict, strict=True)
                if verbose:
                    logger.info(f"Loaded ProteinDPO weights from {dpo_weights_path}")
            else:
                raise FileNotFoundError(
                    f"esm-if1: ProteinDPO weights not found at {dpo_weights_path}; "
                    f"run setup.sh or set PROTO_ESM_IF1_WEIGHTS_DIR"
                )

        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self._weights_variant = weights_variant
        self._loaded = True

        if verbose:
            logger.info("ESM-IF model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise ValueError("esm-if1: cannot move unloaded model to device — call load() first")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            self.model = self.model.to("cpu")
            self.device = "cpu"
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: ESMIF1Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ESMIF1Model()

    pdb_path = input_dict["pdb_path"]
    operation = input_dict["operation"]
    if operation == "sample":
        return _model.sample(
            pdb_path=pdb_path,
            chain_ids=input_dict["chain_ids"],
            batch_size=input_dict["batch_size"],
            temperature=input_dict["temperature"],
            seed=input_dict["seed"],
            device=input_dict["device"],
            weights_variant=input_dict["weights_variant"],
            verbose=input_dict["verbose"],
            fixed_positions=input_dict.get("fixed_positions"),
        )
    if operation == "score":
        return _model.score(
            pdb_path=pdb_path,
            chain_ids=input_dict["chain_ids"],
            target_chain=input_dict["target_chain"],
            sequence=input_dict["sequence"],
            seed=input_dict["seed"],
            device=input_dict["device"],
            weights_variant=input_dict["weights_variant"],
            verbose=input_dict["verbose"],
        )
    raise ValueError(f"esm-if1: unknown operation {operation!r}; valid: ['sample', 'score']")


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
        raise ValueError("esm_if1: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
