"""LigandMPNN inference implementation using Foundry."""

import gc
import json
import os
import sys
from typing import Any

import numpy as np
import torch
from standalone_helpers import get_logger, log_likelihood_metrics, move_model_to_device, serialize_output

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 0.1

SCORING_CAUSALITY = {
    "single_aa": "conditional_minus_self",
    "autoregressive": "auto_regressive",
}


def _fixed_residues(fixed_positions: dict[str, list[int]] | None) -> list[str] | None:
    if not fixed_positions:
        return None
    return [f"{chain}{pos}" for chain, positions in fixed_positions.items() for pos in positions]


def _sequence_tokens(sequence: str) -> tuple[list[str], list[int]]:
    from atomworks.constants import DICT_THREE_TO_ONE  # type: ignore[import-not-found]
    from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING

    vocab = [DICT_THREE_TO_ONE[MPNN_TOKEN_ENCODING.idx_to_token[idx]] for idx in range(MPNN_TOKEN_ENCODING.n_tokens)]
    token_by_letter = {letter: idx for idx, letter in enumerate(vocab)}
    try:
        return vocab, [token_by_letter[aa] for aa in sequence]
    except KeyError as exc:
        raise ValueError(f"ligandmpnn: unsupported residue {exc.args[0]!r}") from exc


class LigandMPNNModel:
    """LigandMPNN model for ligand-aware protein sequence design using Foundry."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
    ):
        """Initialize LigandMPNNModel."""
        self._loaded = False
        self._engine: Any = None
        self.device: str | None = None
        self.checkpoint_path = checkpoint_path
        self._model_type: str | None = None

    def sample(
        self,
        pdb_path: str,
        chain_ids: list[str],
        batch_size: int,
        temperature: float = DEFAULT_TEMPERATURE,
        fixed_positions: dict[str, list[int]] | None = None,
        excluded_amino_acids: list[str] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
        model_type: str = "ligand_mpnn",
        ligand_mpnn_use_atom_context: bool = True,
        ligand_mpnn_use_side_chain_context: bool = False,
        ligand_mpnn_cutoff_for_score: float = 8.0,
    ) -> dict[str, Any]:
        """Sample protein sequences using LigandMPNN.

        Args:
            pdb_path: Path to PDB file containing the structure.
            chain_ids: List of chain IDs to design.
            batch_size: Number of sequences to generate.
            temperature: Sampling temperature (default: 0.1).
            fixed_positions: Dict mapping chain IDs to fixed residue positions.
            excluded_amino_acids: List of amino acids to exclude.
            seed: Random seed for reproducibility (required — Foundry engine
                expects an int).
            device: Device to run on ('cuda' or 'cpu').
            verbose: Whether to print status messages.
            model_type: LigandMPNN variant to load ('ligand_mpnn',
                'per_residue_label_membrane_mpnn', 'global_label_membrane_mpnn').
            ligand_mpnn_use_atom_context: Encode ligand atom context.
            ligand_mpnn_use_side_chain_context: Condition on sidechain atoms of
                fixed residues.
            ligand_mpnn_cutoff_for_score: Ligand-residue distance cutoff (A) for
                ligand-interface recovery scoring.

        Returns:
            dict[str, Any]: Dictionary with keys ``chain_sequences`` (per
                design, an ordered list of ``{"id": str, "sequence": str}``
                dicts) and ``metrics`` (per design, the raw output_dict).
        """
        if seed is None:
            raise ValueError("ligandmpnn: sample requires an explicit int seed")

        # Lazy load the model (reload if model_type changed)
        if not self._loaded or self.device != device or self._model_type != model_type:
            self.load(device, verbose, model_type=model_type)

        fixed_residues = _fixed_residues(fixed_positions)

        # Foundry can't mix residue-based (fixed_residues) and chain-based (designed_chains) constraints;
        # when fixed_residues is set, designed_chains is implicit (the unfixed residues).
        input_dict = {
            "structure_path": pdb_path,
            "name": "design",
            "seed": seed,
            "batch_size": batch_size,
            "number_of_batches": 1,
            "temperature": temperature,
            "omit_aa": excluded_amino_acids,
            "ligand_mpnn_use_atom_context": int(ligand_mpnn_use_atom_context),
            "ligand_mpnn_use_side_chain_context": int(ligand_mpnn_use_side_chain_context),
            "ligand_mpnn_cutoff_for_score": ligand_mpnn_cutoff_for_score,
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

        from atomworks.ml.utils.token import get_token_starts  # type: ignore[import-not-found]

        # Split designed_sequence by the non-atomized token level (1:1 with it).
        chain_sequences: list[list[dict[str, str]]] = []
        metrics: list[dict[str, Any]] = []
        for output in results:
            arr = output.atom_array
            # Drop atomized residues; retained backbone atoms cover every non-atomized residue.
            non_atom = arr[~arr.atomize]
            starts = get_token_starts(non_atom)
            # One row per non-atomized residue, 1:1 with designed_sequence.
            tok = non_atom[starts]
            per_residue_chain = [str(cid) for cid in tok.chain_id]
            designed_sequence = output.output_dict["designed_sequence"]
            if len(per_residue_chain) != len(designed_sequence):
                raise ValueError(
                    f"ligandmpnn: non-atomized token count {len(per_residue_chain)} does not match "
                    f"designed_sequence length {len(designed_sequence)}; upstream contract changed."
                )
            # Group designed_sequence into contiguous per-chain runs, preserving order.
            groups: list[dict[str, str]] = []
            current_chain: str | None = None
            buffer: list[str] = []
            for cid, aa in zip(per_residue_chain, designed_sequence, strict=True):
                if cid != current_chain:
                    if current_chain is not None:
                        groups.append({"id": current_chain, "sequence": "".join(buffer)})
                    current_chain = cid
                    buffer = []
                buffer.append(aa)
            if current_chain is not None:
                groups.append({"id": current_chain, "sequence": "".join(buffer)})
            chain_sequences.append(groups)
            metrics.append(output.output_dict)

        self.unload()
        return {"chain_sequences": chain_sequences, "metrics": metrics}

    def score(
        self,
        pdb_path: str,
        chain_ids: list[str],
        sequence: str,
        fixed_positions: dict[str, list[int]] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
        model_type: str = "ligand_mpnn",
        return_logits: bool = False,
        scoring_mode: str = "single_aa",
    ) -> dict[str, Any]:
        """Score a sequence against a structure."""
        from mpnn.collate.feature_collator import FeatureCollator
        from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
        from mpnn.utils.inference import MPNNInferenceInput

        if seed is None:
            raise ValueError("ligandmpnn: score requires an explicit int seed")
        if scoring_mode not in SCORING_CAUSALITY:
            raise ValueError(f"ligandmpnn: scoring_mode must be one of {sorted(SCORING_CAUSALITY)}")

        if not self._loaded or self.device != device or self._model_type != model_type:
            self.load(device, verbose, model_type=model_type)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        input_dict = {
            "structure_path": pdb_path,
            "name": "score",
            "seed": seed,
            "batch_size": 1,
            "number_of_batches": 1,
            "temperature": 1.0,
            "decode_type": "teacher_forcing",
            "causality_pattern": SCORING_CAUSALITY[scoring_mode],
            "initialize_sequence_embedding_with_ground_truth": True,
            "features_to_return": {
                "input_features": ["S", "mask_for_loss"],
                "decoder_features": ["logits", "log_probs"],
            },
        }
        fixed_residues = _fixed_residues(fixed_positions)
        if fixed_residues:
            input_dict["fixed_residues"] = fixed_residues
        else:
            input_dict["designed_chains"] = chain_ids

        inference_input = MPNNInferenceInput.from_atom_array_and_dict(input_dict=input_dict)
        input_dict = inference_input.input_dict
        pipeline = build_mpnn_transform_pipeline(
            model_type=self._engine.model_type,
            is_inference=True,
            minimal_return=True,
            device=self._engine.device,
        )
        network_input = FeatureCollator()(
            [
                pipeline(
                    {
                        "atom_array": inference_input.atom_array.copy(),
                        "structure_noise": input_dict["structure_noise"],
                        "decode_type": input_dict["decode_type"],
                        "causality_pattern": input_dict["causality_pattern"],
                        "initialize_sequence_embedding_with_ground_truth": input_dict[
                            "initialize_sequence_embedding_with_ground_truth"
                        ],
                        "atomize_side_chains": input_dict["atomize_side_chains"],
                        "repeat_sample_num": input_dict["repeat_sample_num"],
                        "features_to_return": input_dict["features_to_return"],
                    }
                )
            ]
        )

        vocab, token_ids = _sequence_tokens(sequence)
        parsed_len = int(network_input["input_features"]["S"].shape[1])
        if len(token_ids) != parsed_len:
            raise ValueError(f"Sequence length {len(sequence)} does not match structure ({parsed_len} residues).")

        target = torch.tensor([token_ids], dtype=network_input["input_features"]["S"].dtype, device=self._engine.device)
        network_input["input_features"]["S"] = target

        with torch.no_grad():
            output = self._engine.model(network_input)

        mask = output["input_features"]["mask_for_loss"][0].bool()
        log_probs = output["decoder_features"]["log_probs"][0]
        selected = log_probs[torch.arange(parsed_len, device=log_probs.device), target[0]][mask]
        effective_length = int(mask.sum().item())
        if effective_length == 0:
            raise ValueError("ligandmpnn: no residues available to score")

        avg_log_likelihood = float(selected.sum().item()) / effective_length

        self.unload()
        return {
            "logits": output["decoder_features"]["logits"][0] if return_logits else None,
            "metrics": log_likelihood_metrics(avg_log_likelihood, effective_length),
            "vocab": vocab,
        }

    def load(self, device: str = "cuda", verbose: bool = False, model_type: str = "ligand_mpnn") -> None:
        """Load the LigandMPNN model via Foundry.

        Args:
            device: Device to load the model on.
            verbose: Whether to print status messages.
            model_type: LigandMPNN variant ('ligand_mpnn',
                'per_residue_label_membrane_mpnn', 'global_label_membrane_mpnn').
        """
        if verbose:
            logger.info(f"Loading LigandMPNN model_type={model_type} on {device}")

        # Set FOUNDRY_CHECKPOINT_DIRS so Foundry finds BPT-managed weights
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("ligandmpnn")
        if weights_dir:
            os.environ["FOUNDRY_CHECKPOINT_DIRS"] = weights_dir

        from mpnn.inference_engines.mpnn import MPNNInferenceEngine

        self._engine = MPNNInferenceEngine(
            model_type=model_type,
            checkpoint_path=self.checkpoint_path,
            is_legacy_weights=True,
            device=device,
            write_fasta=False,
            write_structures=False,
        )
        self.device = device
        self._model_type = model_type
        self._loaded = True

        if verbose:
            logger.info("LigandMPNN model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device.

        For LigandMPNN, this requires reloading the Foundry engine with the new device.
        """
        if not self._loaded:
            raise ValueError("ligandmpnn: cannot move unloaded model to device — call load() first")

        if self.device != device:
            # LigandMPNN uses Foundry engine which doesn't support standard .to() movement
            # Use helper for consistency (it will handle gracefully), then reload engine
            self._engine = move_model_to_device(self._engine, self.device, device)
            # Foundry engine requires full reload for device change; preserve model_type
            self.load(device, verbose=False, model_type=self._model_type or "ligand_mpnn")

    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        self._engine = None
        self._loaded = False
        self.device = None
        self._model_type = None

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

    pdb_path = input_dict["pdb_path"]
    operation = input_dict["operation"]
    if operation == "sample":
        return _model.sample(
            pdb_path=pdb_path,
            chain_ids=input_dict["chain_ids"],
            batch_size=input_dict["batch_size"],
            temperature=input_dict["temperature"],
            fixed_positions=input_dict.get("fixed_positions"),
            excluded_amino_acids=input_dict.get("excluded_amino_acids"),
            seed=input_dict["seed"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            model_type=input_dict["model_type"],
            ligand_mpnn_use_atom_context=input_dict["ligand_mpnn_use_atom_context"],
            ligand_mpnn_use_side_chain_context=input_dict["ligand_mpnn_use_side_chain_context"],
            ligand_mpnn_cutoff_for_score=input_dict["ligand_mpnn_cutoff_for_score"],
        )
    if operation == "score":
        return _model.score(
            pdb_path=pdb_path,
            chain_ids=input_dict["chain_ids"],
            sequence=input_dict["sequence"],
            fixed_positions=input_dict.get("fixed_positions"),
            seed=input_dict["seed"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            model_type=input_dict["model_type"],
            return_logits=input_dict["return_logits"],
            scoring_mode=input_dict["scoring_mode"],
        )
    raise ValueError(f"ligandmpnn: unknown operation {operation!r}; valid: ['sample', 'score']")


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
        raise ValueError("ligandmpnn: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
