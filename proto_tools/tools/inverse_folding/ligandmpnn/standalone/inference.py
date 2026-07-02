"""LigandMPNN inference implementation using Foundry."""

import gc
import importlib
import json
import os
import random
import sys
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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


def _atom_array_to_pdb(atom_array: Any) -> str:
    """Serialize a Foundry/Biotite atom array to a PDB string."""
    from biotite.structure.io.pdb import PDBFile

    finite_atoms = np.isfinite(np.asarray(atom_array.coord)).all(axis=-1)
    atom_array = atom_array[finite_atoms]
    if len(atom_array) == 0:
        raise ValueError("ligandmpnn: sampled structure contains no finite-coordinate atoms.")
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    buffer = StringIO()
    pdb_file.write(buffer)
    return buffer.getvalue()


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
        chains_explicitly_set: bool = False,
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
            model_type: LigandMPNN variant to load (currently only
                'ligand_mpnn' is wired through the tool layer).
            ligand_mpnn_use_atom_context: Encode ligand atom context.
            ligand_mpnn_use_side_chain_context: Condition on sidechain atoms of
                fixed residues.
            ligand_mpnn_cutoff_for_score: Ligand-residue distance cutoff (A) for
                ligand-interface recovery scoring.
            chains_explicitly_set: Whether the caller explicitly chose chains to
                redesign (vs the all-chains default); gates the residue/chain
                mixing warning so it only fires on a real conflict.

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
            if chain_ids and chains_explicitly_set:
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
        pdb_strings: list[str] = []
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
            pdb_strings.append(_atom_array_to_pdb(arr))

        self.unload()
        return {"chain_sequences": chain_sequences, "metrics": metrics, "pdb_strings": pdb_strings}

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
        ligand_mpnn_use_atom_context: bool = True,
        ligand_mpnn_use_side_chain_context: bool = False,
        ligand_mpnn_cutoff_for_score: float = 8.0,
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
            "ligand_mpnn_use_atom_context": int(ligand_mpnn_use_atom_context),
            "ligand_mpnn_use_side_chain_context": int(ligand_mpnn_use_side_chain_context),
            "ligand_mpnn_cutoff_for_score": ligand_mpnn_cutoff_for_score,
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
        pipeline_args = {}
        if input_dict["occupancy_threshold_sidechain"] is not None:
            pipeline_args["occupancy_threshold_sidechain"] = input_dict["occupancy_threshold_sidechain"]
        if input_dict["occupancy_threshold_backbone"] is not None:
            pipeline_args["occupancy_threshold_backbone"] = input_dict["occupancy_threshold_backbone"]
        if input_dict["undesired_res_names"] is not None:
            pipeline_args["undesired_res_names"] = input_dict["undesired_res_names"]
        pipeline = build_mpnn_transform_pipeline(
            model_type=self._engine.model_type,
            is_inference=True,
            minimal_return=True,
            device=self._engine.device,
            **pipeline_args,
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
            model_type: LigandMPNN variant (currently only 'ligand_mpnn'
                is wired through the tool layer).
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


def _set_reference_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _reference_package_path(reference_backend_path: str | None) -> Path:
    if reference_backend_path is None:
        raise ValueError("ligandmpnn reference backend requires reference_backend_path")
    root = Path(reference_backend_path).expanduser().resolve()
    candidates = [
        root / "models" / "ligandmpnn",
        root / "ligandmpnn",
        root,
    ]
    for candidate in candidates:
        if (candidate / "ligandmpnn.py").is_file() and (candidate / "model_utils.py").is_file():
            return candidate
    raise ValueError(
        "ligandmpnn reference backend path must contain ligandmpnn.py and model_utils.py "
        f"(checked {', '.join(str(path) for path in candidates)})"
    )


def _load_reference_modules(reference_backend_path: str | None) -> dict[str, Any]:
    package_path = _reference_package_path(reference_backend_path)
    parent = str(package_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    package = package_path.name
    for module_name, module in list(sys.modules.items()):
        if module_name != package and not module_name.startswith(f"{package}."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not Path(module_file).resolve().is_relative_to(package_path):
            del sys.modules[module_name]
    return {
        "ligandmpnn": importlib.import_module(f"{package}.ligandmpnn"),
        "pdb_utils": importlib.import_module(f"{package}.pdb_utils"),
        "sc_utils": importlib.import_module(f"{package}.sc_utils"),
        "model_utils": importlib.import_module(f"{package}.model_utils"),
        "data_utils": importlib.import_module(f"{package}.data_utils"),
        "package_path": package_path,
    }


def _reference_checkpoint_path(
    reference_backend_path: str | None,
    checkpoint_path: str | None,
    filename: str,
) -> str:
    if checkpoint_path is not None:
        return checkpoint_path
    package_path = _reference_package_path(reference_backend_path)
    root_candidates = [package_path, package_path.parent, package_path.parent.parent]
    for root in root_candidates:
        candidate = root / "model_params" / filename
        if candidate.is_file():
            return str(candidate)
        candidate = root / "models" / "ligandmpnn" / "model_params" / filename
        if candidate.is_file():
            return str(candidate)
    raise ValueError(f"ligandmpnn reference backend could not locate {filename}; pass an explicit checkpoint path")


def _fixed_residue_strings(fixed_positions: dict[str, list[int]] | None) -> list[str]:
    if not fixed_positions:
        return []
    return [f"{chain}{position}" for chain, positions in fixed_positions.items() for position in positions]


def _chain_sequences_from_reference(protein_dict: dict[str, Any], sequence: str) -> list[dict[str, str]]:
    groups: list[dict[str, str]] = []
    current_chain: str | None = None
    buffer: list[str] = []
    for chain_id, aa in zip([str(chain) for chain in protein_dict["chain_letters"]], sequence, strict=True):
        if chain_id != current_chain:
            if current_chain is not None:
                groups.append({"id": current_chain, "sequence": "".join(buffer)})
            current_chain = chain_id
            buffer = []
        buffer.append(aa)
    if current_chain is not None:
        groups.append({"id": current_chain, "sequence": "".join(buffer)})
    return groups


def _reference_sequence_recovery(output_dict: dict[str, Any]) -> float:
    native = output_dict["native_sequence"].detach().cpu().numpy()
    generated = torch.transpose(output_dict["generated_sequence"], 1, 2)[0].squeeze().detach().cpu().numpy()
    mask = (output_dict["mask"] * output_dict["chain_mask"]).detach().cpu().numpy().astype(bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(native[mask] == generated[mask]))


def _run_reference_ligandmpnn(
    ligandmpnn_module: Any,
    args: Any,
    protein_dict: dict[str, Any],
    feature_dict: dict[str, Any],
    model: Any,
    *,
    device: str,
) -> dict[str, Any]:
    original_format_float_positional = ligandmpnn_module.np.format_float_positional

    def format_float_positional_compat(x: Any, *args_: Any, **kwargs: Any) -> str:
        array = np.asarray(x)
        if array.size == 1:
            x = array.reshape(()).item()
        return str(original_format_float_positional(x, *args_, **kwargs))

    ligandmpnn_module.np.format_float_positional = format_float_positional_compat
    try:
        return cast(
            "dict[str, Any]",
            ligandmpnn_module.run_ligandmpnn(args, protein_dict, feature_dict, model, device=device),
        )
    finally:
        ligandmpnn_module.np.format_float_positional = original_format_float_positional


class ReferenceLigandMPNNModel:
    """Reference LigandMPNN backend used for compatibility experiments."""

    def __init__(
        self,
        reference_backend_path: str | None,
        checkpoint_path: str | None = None,
        packer_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize reference backend paths and lazy model state."""
        self.reference_backend_path = reference_backend_path
        self.checkpoint_path = _reference_checkpoint_path(
            reference_backend_path,
            checkpoint_path,
            "ligandmpnn_v_32_020_25.pt",
        )
        self.packer_checkpoint_path = _reference_checkpoint_path(
            reference_backend_path,
            packer_checkpoint_path,
            "ligandmpnn_sc_v_32_002_16.pt",
        )
        self._modules = _load_reference_modules(reference_backend_path)
        self._loaded = False
        self.device: str | None = None
        self.model: Any = None
        self.packer: Any = None
        self.atom_context_num: int = 16

    def load(self, device: str) -> None:
        """Load the reference LigandMPNN sequence model and side-chain packer."""
        if self._loaded and self.device == device:
            return
        if self._loaded:
            self.unload()

        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.atom_context_num = int(checkpoint["atom_context_num"])
        k_neighbors = checkpoint["num_edges"]
        protein_mpnn = self._modules["model_utils"].ProteinMPNN
        self.model = protein_mpnn(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=device,
            atom_context_num=self.atom_context_num,
            model_type="ligand_mpnn",
            ligand_mpnn_use_side_chain_context=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(device)

        checkpoint_packer = torch.load(self.packer_checkpoint_path, map_location=device)
        packer_cls = self._modules["sc_utils"].Packer
        self.packer = packer_cls(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=device,
            num_mix=3,
        )
        self.packer.load_state_dict(checkpoint_packer["model_state_dict"])
        self.packer.eval()
        self.packer.to(device)

        self.device = device
        self._loaded = True

    def to_device(self, device: str) -> None:
        """Reload the reference models on a different device."""
        if not self._loaded:
            raise ValueError("ligandmpnn reference backend cannot move an unloaded model; call load() first")
        if self.device != device:
            self.unload()
            self.load(device)

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
        ligand_mpnn_cutoff_for_score: float = 20.0,
        sc_num_denoising_steps: int = 8,
        sc_num_samples: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        """Sample fixed-backbone sequences and packed structures with the reference backend."""
        self.load(device)
        _set_reference_seed(seed)
        args = self._args(
            chain_ids=chain_ids,
            fixed_positions=fixed_positions,
            excluded_amino_acids=excluded_amino_acids,
            temperature=temperature,
            ligand_mpnn_cutoff_for_score=ligand_mpnn_cutoff_for_score,
            sc_num_denoising_steps=sc_num_denoising_steps,
            sc_num_samples=sc_num_samples,
            verbose=verbose,
            seed=seed,
        )
        protein_dict, other_atoms, icodes, feature_dict = self._prepare(pdb_path, args, device)

        chain_sequences: list[list[dict[str, str]]] = []
        metrics: list[dict[str, Any]] = []
        pdb_strings: list[str] = []
        with tempfile.TemporaryDirectory(prefix="ligandmpnn_reference_") as tmpdir:
            for index in range(batch_size):
                feature_dict["randn"] = torch.randn([1, feature_dict["mask"].shape[1]], device=device)
                output_dict = _run_reference_ligandmpnn(
                    self._modules["ligandmpnn"],
                    args,
                    protein_dict,
                    feature_dict,
                    self.model,
                    device=device,
                )
                sequence = output_dict["generated_sequence_str"]
                outfile = os.path.join(tmpdir, f"reference_{index}.pdb")
                self._modules["pdb_utils"].pack_sc(
                    args,
                    self.packer,
                    protein_dict,
                    output_dict["generated_sequence"],
                    other_atoms,
                    icodes,
                    outfile=outfile,
                    device=device,
                )
                chain_sequences.append(_chain_sequences_from_reference(protein_dict, sequence))
                metrics.append(
                    {
                        "sequence_recovery": _reference_sequence_recovery(output_dict),
                        "ligand_interface_sequence_recovery": float("nan"),
                        "pmpnn": float(output_dict["loss_np"]),
                    }
                )
                pdb_strings.append(Path(outfile).read_text())
        return {"chain_sequences": chain_sequences, "metrics": metrics, "pdb_strings": pdb_strings}

    def score(
        self,
        pdb_path: str,
        chain_ids: list[str],
        sequence: str,
        fixed_positions: dict[str, list[int]] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
        scoring_mode: str = "single_aa",
        ligand_mpnn_cutoff_for_score: float = 20.0,
        **_: Any,
    ) -> dict[str, Any]:
        """Score a sequence against a structure with the reference backend."""
        self.load(device)
        _set_reference_seed(seed)
        args = self._args(
            chain_ids=chain_ids,
            fixed_positions=fixed_positions,
            excluded_amino_acids=None,
            temperature=1.0,
            ligand_mpnn_cutoff_for_score=ligand_mpnn_cutoff_for_score,
            sc_num_denoising_steps=8,
            sc_num_samples=1,
            verbose=verbose,
            seed=seed,
        )
        protein_dict, _other_atoms, _icodes, feature_dict = self._prepare(pdb_path, args, device)
        data_utils = self._modules["data_utils"]
        token_ids = [data_utils.restype_str_to_int[aa] for aa in sequence]
        if len(token_ids) != int(feature_dict["mask"].shape[1]):
            raise ValueError(
                f"Sequence length {len(token_ids)} does not match structure ({int(feature_dict['mask'].shape[1])} residues)."
            )
        feature_dict["S"] = torch.tensor([token_ids], device=device, dtype=torch.int32)
        with torch.inference_mode():
            if scoring_mode == "single_aa":
                score_dict = self.model.single_aa_score(feature_dict, use_sequence=1)
            else:
                score_dict = self.model.score(feature_dict, use_sequence=1)

        probs_2d = torch.mean(torch.exp(score_dict["log_probs"]), 0)
        seq_tensor = feature_dict["S"].squeeze()
        selected = probs_2d[torch.arange(len(token_ids), device=device), seq_tensor].detach().cpu().numpy()
        chain_mask = protein_dict["chain_mask"].detach().cpu().numpy().astype(bool)
        selected_masked = selected[chain_mask]
        if selected_masked.size == 0:
            raise ValueError("ligandmpnn reference score found no residues available to score")
        avg_log_likelihood = float(np.mean(np.log(np.maximum(selected_masked, 1e-12))))
        logits = None
        if return_logits:
            logits = np.log(np.maximum(probs_2d.detach().cpu().numpy(), 1e-12)).tolist()
        return {
            "logits": logits,
            "metrics": {
                **log_likelihood_metrics(avg_log_likelihood, int(selected_masked.size)),
                "mean_current_probability": float(np.mean(selected_masked)),
            },
            "vocab": [data_utils.restype_int_to_str[idx] for idx in range(len(data_utils.restype_int_to_str))],
        }

    def _args(
        self,
        *,
        chain_ids: list[str],
        fixed_positions: dict[str, list[int]] | None,
        excluded_amino_acids: list[str] | None,
        temperature: float,
        ligand_mpnn_cutoff_for_score: float,
        sc_num_denoising_steps: int,
        sc_num_samples: int,
        verbose: bool,
        seed: int | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            seq_model="ligandmpnn",
            seed=seed,
            model_path=self.checkpoint_path,
            packer_path=self.packer_checkpoint_path,
            var_residues="",
            fixed_residues=" ".join(_fixed_residue_strings(fixed_positions)),
            bias_AA="",
            bias_AA_per_residue="",
            omit_AA="".join(excluded_amino_acids or []),
            omit_AA_per_residue="",
            symmetry_residues="",
            symmetry_weights="",
            homo_oligomer=0,
            sc_num_denoising_steps=sc_num_denoising_steps,
            sc_num_samples=sc_num_samples,
            ligand_mpnn_cutoff_for_score=ligand_mpnn_cutoff_for_score,
            temperature=temperature,
            zero_indexed=0,
            autoregressive_score=0,
            single_aa_score=1,
            chains_to_design=",".join(chain_ids),
            parse_these_chains_only="",
            parse_atoms_with_zero_occupancy=0,
            verbose=verbose,
        )

    def _prepare(
        self,
        pdb_path: str,
        args: SimpleNamespace,
        device: str,
    ) -> tuple[dict[str, Any], Any, list[str], dict[str, Any]]:
        ligandmpnn = self._modules["ligandmpnn"]
        protein_dict, _, other_atoms, icodes, _ = self._modules["data_utils"].parse_PDB(
            pdb_path,
            device=device,
            chains=[],
            parse_all_atoms=True,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )
        encoded_residues, encoded_residue_dict, _ = ligandmpnn.get_encoded_residues(protein_dict, icodes)
        design_params = {
            "var_residues": ligandmpnn.get_var_residues(args, pdb_path),
            "fixed_residues": ligandmpnn.get_fixed_residues(args, pdb_path),
            "bias_AA": ligandmpnn.get_bias_aa(args, pdb_path, device)[0],
            "bias_AA_per_residue": {},
            "omit_AA": ligandmpnn.get_omit_aa(args, pdb_path, device)[0],
            "omit_AA_per_residue": {},
            "parse_these_chains_only_list": [],
        }
        protein_dict, feature_dict = ligandmpnn.prepare_ligandmpnn(
            args,
            protein_dict,
            icodes,
            design_params,
            self.atom_context_num,
            device=device,
        )
        omit_per_residue = ligandmpnn.omit_aa(args, encoded_residues, encoded_residue_dict, {}, device)
        feature_dict["bias"] = feature_dict["bias"] - 1e8 * omit_per_residue[None]
        return protein_dict, other_atoms, icodes, feature_dict

    def unload(self) -> None:
        """Unload reference models and clear cached GPU memory."""
        self.model = None
        self.packer = None
        self._loaded = False
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: Any | None = None
_model_key: tuple[Any, ...] | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model, _model_key
    backend = input_dict.get("backend", "foundry")
    checkpoint_path = input_dict.get("checkpoint_path")
    reference_backend_path = input_dict.get("reference_backend_path")
    packer_checkpoint_path = input_dict.get("packer_checkpoint_path")
    model_key = (backend, checkpoint_path, reference_backend_path, packer_checkpoint_path)
    if _model is not None and _model_key != model_key:
        _model.unload()
        _model = None
        _model_key = None
    if _model is None:
        if backend == "reference":
            _model = ReferenceLigandMPNNModel(
                reference_backend_path=reference_backend_path,
                checkpoint_path=checkpoint_path,
                packer_checkpoint_path=packer_checkpoint_path,
            )
        elif backend == "foundry":
            _model = LigandMPNNModel(
                checkpoint_path=checkpoint_path,
            )
        else:
            raise ValueError("ligandmpnn backend must be 'foundry' or 'reference'")
        _model_key = model_key

    pdb_path = input_dict["pdb_path"]
    operation = input_dict["operation"]
    if operation == "sample":
        sample_kwargs = {
            "pdb_path": pdb_path,
            "chain_ids": input_dict["chain_ids"],
            "chains_explicitly_set": input_dict.get("chains_explicitly_set", False),
            "batch_size": input_dict["batch_size"],
            "temperature": input_dict["temperature"],
            "fixed_positions": input_dict.get("fixed_positions"),
            "excluded_amino_acids": input_dict.get("excluded_amino_acids"),
            "seed": input_dict["seed"],
            "device": input_dict["device"],
            "verbose": input_dict["verbose"],
            "model_type": input_dict["model_type"],
            "ligand_mpnn_use_atom_context": input_dict["ligand_mpnn_use_atom_context"],
            "ligand_mpnn_use_side_chain_context": input_dict["ligand_mpnn_use_side_chain_context"],
            "ligand_mpnn_cutoff_for_score": input_dict["ligand_mpnn_cutoff_for_score"],
        }
        if backend == "reference":
            sample_kwargs["sc_num_denoising_steps"] = input_dict.get("sc_num_denoising_steps", 8)
            sample_kwargs["sc_num_samples"] = input_dict.get("sc_num_samples", 1)
        return _model.sample(**sample_kwargs)
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
            ligand_mpnn_use_atom_context=input_dict.get("ligand_mpnn_use_atom_context", True),
            ligand_mpnn_use_side_chain_context=input_dict.get("ligand_mpnn_use_side_chain_context", False),
            ligand_mpnn_cutoff_for_score=input_dict.get("ligand_mpnn_cutoff_for_score", 8.0),
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
