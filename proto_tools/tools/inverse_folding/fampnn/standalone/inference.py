"""FAMPNN standalone inference implementation for venv execution.

Handles all four FAMPNN operations:
  - sample: Iterative sequence design with sidechain co-generation
  - pack: Sidechain packing given backbone + sequence
  - score_all_mutations: Score every possible single mutation
  - score_mutations: Score specific mutations from a list
"""

import json
import os
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any

from standalone_helpers import get_random_int, serialize_output, set_torch_seed

logger = getLogger(__name__)


class FAMPNNModel:
    """FAMPNN model for full-atom protein sequence design and sidechain packing."""

    def __init__(self) -> None:
        """Initialize FAMPNNModel."""
        self._loaded = False
        self._model_variant: str | None = None
        self.device: str | None = None
        self.model: Any = None

    def sample(
        self,
        pdb_path: str,
        chain_ids: list[str] | None = None,  # noqa: ARG002 — reserved for future chain filtering
        fixed_positions: dict[str, list[int]] | None = None,
        fixed_sidechain_positions: dict[str, list[int]] | None = None,
        num_sequences: int = 1,
        temperature: float = 0.1,
        num_steps: int = 100,
        seq_only: bool = False,
        repack_last: bool = True,
        psce_threshold: float = 0.3,
        scn_diffusion_steps: int = 50,
        scn_step_scale: float = 1.5,
        seed: int | None = None,
        model_variant: str = "0.3",
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Sample sequences with optional sidechain co-generation."""
        if not self._loaded or self._model_variant != model_variant:
            self.load(model_variant, device, verbose)
        elif self.device != device:
            self.to_device(device)

        import torch
        from fampnn import sampling_utils
        from fampnn.data import residue_constants as rc
        from fampnn.data.data import (
            load_feats_from_pdb,
            pad_to_max_len,
            process_single_pdb,
        )
        from fampnn.model.sd_model import SeqDenoiser

        set_torch_seed(seed)
        sampling_seed = seed if seed is not None else get_random_int()
        sampling_utils.seed_everything(sampling_seed)

        # Build timestep schedules
        t_seq = sampling_utils.get_timesteps_from_schedule(
            num_steps=num_steps,
            mode="linear",
            t_start=0.0,
            t_end=1.0,
        )
        t_scd = sampling_utils.get_timesteps_from_schedule(
            num_steps=scn_diffusion_steps,
            mode="linear",
            t_start=0.0,
            t_end=1.0,
        )
        scd_inputs_template = {
            "num_steps": scn_diffusion_steps,
            "timesteps": None,
            "step_scale": scn_step_scale,
            "churn_cfg": {
                "s_churn": 0,
                "s_noise": 1.0,
                "s_t_min": 0.01,
                "s_t_max": 50.0,
                "num_steps": scn_diffusion_steps,
            },
        }

        # Build fixed position CSV dataframe if needed
        import pandas as pd

        fixed_pos_df = pd.DataFrame(columns=["fixed_pos_seq", "fixed_pos_scn"])

        pdb_name = Path(pdb_path).stem
        if fixed_positions or fixed_sidechain_positions:
            seq_str = ""
            scn_str = ""
            if fixed_positions:
                # Convert {chain: [1-indexed positions]} to "A1-5,A10-20" format
                parts = [f"{chain_id}{pos}" for chain_id, positions in fixed_positions.items() for pos in positions]
                seq_str = ",".join(parts)
            if fixed_sidechain_positions:
                parts = [
                    f"{chain_id}{pos}" for chain_id, positions in fixed_sidechain_positions.items() for pos in positions
                ]
                scn_str = ",".join(parts)
            fixed_pos_df = pd.DataFrame(
                {"fixed_pos_seq": [seq_str], "fixed_pos_scn": [scn_str]},
                index=[pdb_name],
            )

        # Load PDB
        data = load_feats_from_pdb(pdb_path)
        single = process_single_pdb(data)
        chain_id_mapping = data["chain_id_mapping"]

        model_input_keys = [
            "x",
            "aatype",
            "seq_mask",
            "missing_atom_mask",
            "residue_index",
            "chain_index",
            "interface_residue_mask",
        ]

        all_sequences = []
        all_pdb_strings: list[str] = []
        all_psce = []

        B = num_sequences

        # Replicate for batch
        batch_list = [single] * B

        max_len = max(b["x"].shape[0] for b in batch_list)
        batch_list_padded = [
            pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len) for b in batch_list
        ]
        batch = {k: torch.cat([b[k] for b in batch_list_padded], dim=0) for k in model_input_keys}
        batch = {k: batch[k].to(self.device) for k in model_input_keys}

        scd_inputs = dict(scd_inputs_template)
        scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(self.device)

        timesteps = t_seq[None].expand(B, -1).to(self.device)

        pdb_names = [pdb_name] * B
        batch_chain_id_mapping = [chain_id_mapping] * B

        aatype_override_mask, scn_override_mask = sampling_utils.get_override_masks(
            batch,
            pdb_names,
            batch_chain_id_mapping,
            fixed_pos_df,
            verbose=False,
            mode="seq_design",
        )

        with torch.no_grad():
            x_denoised, aatype_denoised, aux = self.model.sample(
                batch["x"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                missing_atom_mask=batch["missing_atom_mask"],
                residue_index=batch["residue_index"],
                chain_index=batch["chain_index"],
                timesteps=timesteps,
                seq_only=seq_only,
                temperature=temperature,
                repack_last=repack_last,
                psce_threshold=psce_threshold,
                aatype_override_mask=aatype_override_mask,
                scn_override_mask=scn_override_mask,
                scd_inputs=scd_inputs,
            )

        samples = {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
        }

        # Extract sequences and PDB strings
        for j in range(B):
            seq_mask_j = samples["seq_mask"][j].cpu()
            pred_aatype_j = samples["pred_aatype"][j].cpu()
            pred_aatype_j = pred_aatype_j[seq_mask_j.bool()]
            pred_seq = "".join(rc.restypes_with_x[a] for a in pred_aatype_j)
            all_sequences.append(pred_seq)

            # Extract per-residue pSCE (mean over atoms)
            psce_j = aux["psce"][j].cpu()
            psce_j = psce_j[seq_mask_j.bool()]
            all_psce.append(psce_j.mean(dim=-1).tolist())

        # Save PDB strings via temp files
        # Detach tensors to avoid "Can't call numpy() on Tensor that
        # requires grad" in upstream pdb_utils (PyTorch >=2.x compat)
        detached_samples = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_paths_out = [f"{tmpdir}/sample_{j}.pdb" for j in range(B)]
            SeqDenoiser.save_samples_to_pdb(detached_samples, pdb_paths_out)
            all_pdb_strings.extend(Path(pdb_out).read_text() for pdb_out in pdb_paths_out)

        return {
            "sequences": all_sequences,
            "pdb_strings": all_pdb_strings,
            "psce": all_psce,
        }

    def pack(
        self,
        pdb_path: str,
        fixed_positions: dict[str, list[int]] | None = None,
        fixed_sidechain_positions: dict[str, list[int]] | None = None,
        num_samples: int = 1,
        scn_diffusion_steps: int = 50,
        scn_step_scale: float = 1.5,
        seed: int | None = None,
        model_variant: str = "0.0",
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Pack sidechains onto a backbone given known sequence."""
        if not self._loaded or self._model_variant != model_variant:
            self.load(model_variant, device, verbose)
        elif self.device != device:
            self.to_device(device)

        import pandas as pd
        import torch
        from fampnn import sampling_utils
        from fampnn.data.data import (
            load_feats_from_pdb,
            pad_to_max_len,
            process_single_pdb,
        )
        from fampnn.model.sd_model import SeqDenoiser

        set_torch_seed(seed)
        sampling_seed = seed if seed is not None else get_random_int()
        sampling_utils.seed_everything(sampling_seed)

        t_scd = sampling_utils.get_timesteps_from_schedule(
            num_steps=scn_diffusion_steps,
            mode="linear",
            t_start=0.0,
            t_end=1.0,
        )
        scd_inputs_template = {
            "num_steps": scn_diffusion_steps,
            "timesteps": None,
            "step_scale": scn_step_scale,
            "churn_cfg": {
                "s_churn": 0,
                "s_noise": 1.0,
                "s_t_min": 0.01,
                "s_t_max": 50.0,
                "num_steps": scn_diffusion_steps,
            },
        }

        # Build fixed positions
        pdb_name = Path(pdb_path).stem
        fixed_pos_df = pd.DataFrame(columns=["fixed_pos_seq", "fixed_pos_scn"])
        if fixed_positions or fixed_sidechain_positions:
            seq_str = ""
            scn_str = ""
            if fixed_positions:
                parts = [f"{chain_id}{pos}" for chain_id, positions in fixed_positions.items() for pos in positions]
                seq_str = ",".join(parts)
            if fixed_sidechain_positions:
                parts = [
                    f"{chain_id}{pos}" for chain_id, positions in fixed_sidechain_positions.items() for pos in positions
                ]
                scn_str = ",".join(parts)
            fixed_pos_df = pd.DataFrame(
                {"fixed_pos_seq": [seq_str], "fixed_pos_scn": [scn_str]},
                index=[pdb_name],
            )

        data = load_feats_from_pdb(pdb_path)
        single = process_single_pdb(data)
        chain_id_mapping = data["chain_id_mapping"]

        model_input_keys = [
            "x",
            "aatype",
            "seq_mask",
            "missing_atom_mask",
            "residue_index",
            "chain_index",
            "interface_residue_mask",
        ]

        all_pdb_strings: list[str] = []
        all_psce = []

        B = num_samples

        batch_list = [single] * B
        max_len = max(b["x"].shape[0] for b in batch_list)
        batch_list_padded = [
            pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len) for b in batch_list
        ]
        batch = {k: torch.cat([b[k] for b in batch_list_padded], dim=0) for k in model_input_keys}
        batch = {k: batch[k].to(self.device) for k in model_input_keys}

        scd_inputs = dict(scd_inputs_template)
        scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(self.device)

        pdb_names = [pdb_name] * B
        batch_chain_id_mapping = [chain_id_mapping] * B

        aatype_override_mask, scn_override_mask = sampling_utils.get_override_masks(
            batch,
            pdb_names,
            batch_chain_id_mapping,
            fixed_pos_df,
            verbose=False,
            mode="packing",
        )

        with torch.no_grad():
            x_denoised, aatype_denoised, aux = self.model.sidechain_pack(
                batch["x"],
                batch["aatype"],
                seq_mask=batch["seq_mask"],
                missing_atom_mask=batch["missing_atom_mask"],
                residue_index=batch["residue_index"],
                chain_index=batch["chain_index"],
                aatype_override_mask=aatype_override_mask,
                scn_override_mask=scn_override_mask,
                scd_inputs=scd_inputs,
            )

        samples = {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": torch.zeros_like(batch["missing_atom_mask"]),
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
        }

        for j in range(B):
            seq_mask_j = samples["seq_mask"][j].cpu()
            psce_j = aux["psce"][j].cpu()
            psce_j = psce_j[seq_mask_j.bool()]
            all_psce.append(psce_j.mean(dim=-1).tolist())

        detached_samples = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_paths_out = [f"{tmpdir}/packed_{j}.pdb" for j in range(B)]
            SeqDenoiser.save_samples_to_pdb(detached_samples, pdb_paths_out)
            all_pdb_strings.extend(Path(pdb_out).read_text() for pdb_out in pdb_paths_out)

        return {
            "pdb_strings": all_pdb_strings,
            "psce": all_psce,
        }

    def score_all_mutations(
        self,
        pdb_path: str,
        batch_size: int = 16,
        seed: int | None = None,
        model_variant: str = "0.3_cath",
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Score every possible single mutation at every position."""
        if not self._loaded or self._model_variant != model_variant:
            self.load(model_variant, device, verbose)
        elif self.device != device:
            self.to_device(device)

        import torch
        from fampnn import scoring_utils
        from fampnn.data import residue_constants as rc
        from fampnn.data.data import load_feats_from_pdb, process_single_pdb
        from fampnn.sampling_utils import seed_everything

        set_torch_seed(seed)
        sampling_seed = seed if seed is not None else get_random_int()
        seed_everything(sampling_seed)

        data = load_feats_from_pdb(pdb_path)
        batch = process_single_pdb(data)

        model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index"]
        model_inputs = {k: batch[k].to(self.device) for k in model_input_keys}

        B = batch_size

        def repeat_fn(x: Any) -> Any:
            return x[None, ...].repeat(B, *([1] * len(x.shape)))

        model_inputs = {k: repeat_fn(v) for k, v in model_inputs.items()}

        num_positions = int(torch.sum(model_inputs["seq_mask"][0]).item())
        total_positions = torch.arange(num_positions)
        batched_positions = total_positions.split(B)

        scores_dict = {}
        for positions in batched_positions:
            x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = [
                model_inputs[k][: len(positions), ...].clone() for k in model_input_keys
            ]

            x_masked, aatype_masked, missing_atom_mask_masked, scn_mlm_mask = scoring_utils.mask_positions(
                x,
                aatype,
                seq_mask,
                missing_atom_mask,
                positions,
            )

            with torch.no_grad():
                logprobs = self.model.score(
                    x=x_masked,
                    aatype=aatype_masked,
                    missing_atom_mask=missing_atom_mask_masked,
                    seq_mask=seq_mask,
                    scn_mlm_mask=scn_mlm_mask,
                    residue_index=residue_index,
                    chain_index=chain_index,
                )

            wt_aatype = aatype[torch.arange(len(positions)), positions]
            wt_logprobs = logprobs[torch.arange(len(positions)), positions, wt_aatype][:, None]
            mut_logprobs = logprobs[torch.arange(len(positions)), positions, :]
            scores = mut_logprobs - wt_logprobs

            for score, position, wt_res in zip(scores, positions, wt_aatype, strict=False):
                # Key format: "1A" = 1-indexed position + wild-type residue
                key = f"{position.item() + 1}{rc.idx_to_restype_with_x[wt_res.item()]}"
                scores_dict[key] = {
                    rc.restypes_with_x[res_num]: score[res_num].item() for res_num in range(rc.restype_num)
                }

        return {"scores": scores_dict}

    @staticmethod
    def _convert_mutation_1indexed_to_0indexed(mutation_str: str) -> str:
        """Convert 1-indexed mutation string to 0-indexed for FAMPNN internals.

        Input format:  'A1V' or 'A1V:G5L' (1-indexed, user-facing)
        Output format: 'A0V' or 'A0V:G4L' (0-indexed, FAMPNN internal)

        Special case: 'wt' is passed through unchanged.
        """
        if mutation_str.lower() == "wt":
            return mutation_str

        import re

        parts = mutation_str.split(":")
        converted = []
        for part in parts:
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", part)
            if not match:
                raise ValueError(
                    f"fampnn: invalid mutation format {part!r}; expected '<WT><position><MUT>' (e.g., 'A1V')"
                )
            wt, pos, mut = match.groups()
            converted.append(f"{wt}{int(pos) - 1}{mut}")
        return ":".join(converted)

    def score_mutations(
        self,
        pdb_path: str,
        mutations: list[str],
        batch_size: int = 16,
        seq_only: bool = False,
        scn_diffusion_steps: int = 50,
        scn_step_scale: float = 1.5,
        seed: int | None = None,
        model_variant: str = "0.3_cath",
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Score specific mutations (format: 'A1V' or 'A1V:G5L' for multi-site, 1-indexed)."""
        if not self._loaded or self._model_variant != model_variant:
            self.load(model_variant, device, verbose)
        elif self.device != device:
            self.to_device(device)

        import numpy as np
        from fampnn import sampling_utils, scoring_utils
        from fampnn.data.data import load_feats_from_pdb, process_single_pdb

        set_torch_seed(seed)
        sampling_seed = seed if seed is not None else get_random_int()
        sampling_utils.seed_everything(sampling_seed)

        # Convert 1-indexed mutations to 0-indexed for FAMPNN internals
        internal_mutations = [self._convert_mutation_1indexed_to_0indexed(m) for m in mutations]

        data = load_feats_from_pdb(pdb_path)
        batch = process_single_pdb(data)

        model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index"]
        model_inputs = {k: batch[k].to(self.device) for k in model_input_keys}

        B = batch_size

        def repeat_fn(x: Any) -> Any:
            return x[None, ...].repeat(B, *([1] * len(x.shape)))

        model_inputs = {k: repeat_fn(v) for k, v in model_inputs.items()}

        t_scd = sampling_utils.get_timesteps_from_schedule(
            num_steps=scn_diffusion_steps,
            mode="linear",
            t_start=0.0,
            t_end=1.0,
        )
        scd_inputs = {
            "num_steps": scn_diffusion_steps,
            "timesteps": None,
            "step_scale": scn_step_scale,
            "churn_cfg": {
                "s_churn": 0,
                "s_noise": 1.0,
                "s_t_min": 0.01,
                "s_t_max": 50.0,
                "num_steps": scn_diffusion_steps,
            },
        }

        num_mutations = len(internal_mutations)
        num_batches = num_mutations // B + (num_mutations % B > 0)
        batched_mutations = np.array_split(internal_mutations, num_batches)

        scores_all = []
        for mutation_batch in batched_mutations:
            x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = [
                model_inputs[k][: len(mutation_batch), ...].clone() for k in model_input_keys
            ]

            scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(self.device)
            scores = (
                scoring_utils.score_seq(
                    model=self.model,
                    x=x,
                    aatype=aatype,
                    seq_mask=seq_mask,
                    residue_index=residue_index,
                    missing_atom_mask=missing_atom_mask,
                    chain_index=chain_index,
                    mutations=mutation_batch.tolist(),
                    scd_inputs=scd_inputs,
                    method="multiple",
                    seq_only=seq_only,
                )
                .cpu()
                .tolist()
            )

            scores_all += scores

        # Return original 1-indexed mutation strings in the output
        return {
            "mutations": mutations,
            "scores": scores_all,
        }

    def load(self, model_variant: str, device: str, verbose: bool = False) -> None:
        """Load FAMPNN model."""
        import torch
        from fampnn.model.sd_model import SeqDenoiser

        self.verbose = verbose

        # Resolve weights path
        variant_map = {
            "0.0": "fampnn_0_0.pt",
            "0.3": "fampnn_0_3.pt",
            "0.3_cath": "fampnn_0_3_cath.pt",
        }
        if model_variant not in variant_map:
            raise ValueError(f"fampnn: unknown model_variant {model_variant!r}; valid: {list(variant_map.keys())}")

        weights_filename = variant_map[model_variant]

        # Search for weights in multiple locations
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("fampnn")
        search_paths = []
        if weights_dir:
            search_paths.append(Path(weights_dir))
        # Fallback locations for NONE mode or missing venv
        venv = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv:
            search_paths.append(Path(venv) / "weights")
        try:
            import fampnn as fampnn_pkg

            pkg_dir = Path(fampnn_pkg.__file__).parent.parent
            search_paths.append(pkg_dir / "weights")
        except ImportError as e:
            logger.debug("fampnn: package not importable, skipping its weights/ as a fallback path (%s)", e)

        checkpoint_path = None
        for search_dir in search_paths:
            candidate = search_dir / weights_filename
            if candidate.exists():
                checkpoint_path = str(candidate)
                break

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"fampnn: weights {weights_filename} not found in {search_paths}; "
                f"set PROTO_FAMPNN_WEIGHTS_DIR or run setup.sh"
            )

        if verbose:
            logger.info(f"Loading FAMPNN model ({model_variant}) from {checkpoint_path}")

        torch_device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        self.model = SeqDenoiser(ckpt["model_cfg"]).to(torch_device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device
        self._loaded = True
        self._model_variant = model_variant

        if verbose:
            logger.info("FAMPNN model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        from standalone_helpers import move_model_to_device

        if self.model is None:
            raise ValueError("fampnn: cannot move unloaded model to device — call load() first")
        if self.device == device:
            return

        self.model = move_model_to_device(self.model, self.device, device)
        self.device = device

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        import gc

        if not self._loaded:
            return

        self.model = None
        self._loaded = False
        self._model_variant = None
        self.device = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ============================================================================
# Dispatch
# ============================================================================
_model: FAMPNNModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = FAMPNNModel()

    pdb_contents = input_dict.get("pdb_contents")

    with tempfile.TemporaryDirectory() as temp_dir:
        pdb_path = input_dict.get("pdb_path")
        if pdb_contents and not pdb_path:
            pdb_path = str(Path(temp_dir) / "input.pdb")
            Path(pdb_path).write_text(pdb_contents)

        operation = input_dict["operation"]

        if operation == "sample":
            return _model.sample(
                pdb_path=pdb_path,  # type: ignore[arg-type]
                chain_ids=input_dict.get("chain_ids"),
                fixed_positions=input_dict.get("fixed_positions"),
                fixed_sidechain_positions=input_dict.get("fixed_sidechain_positions"),
                num_sequences=input_dict["num_sequences"],
                temperature=input_dict["temperature"],
                num_steps=input_dict["num_steps"],
                seq_only=input_dict["seq_only"],
                repack_last=input_dict["repack_last"],
                psce_threshold=input_dict["psce_threshold"],
                scn_diffusion_steps=input_dict["scn_diffusion_steps"],
                scn_step_scale=input_dict["scn_step_scale"],
                seed=input_dict["seed"],
                model_variant=input_dict["model_variant"],
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
        if operation == "pack":
            return _model.pack(
                pdb_path=pdb_path,  # type: ignore[arg-type]
                fixed_positions=input_dict.get("fixed_positions"),
                fixed_sidechain_positions=input_dict.get("fixed_sidechain_positions"),
                num_samples=input_dict["num_samples"],
                scn_diffusion_steps=input_dict["scn_diffusion_steps"],
                scn_step_scale=input_dict["scn_step_scale"],
                seed=input_dict["seed"],
                model_variant=input_dict["model_variant"],
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
        if operation == "score_all_mutations":
            return _model.score_all_mutations(
                pdb_path=pdb_path,  # type: ignore[arg-type]
                batch_size=input_dict["batch_size"],
                seed=input_dict["seed"],
                model_variant=input_dict["model_variant"],
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
        if operation == "score_mutations":
            return _model.score_mutations(
                pdb_path=pdb_path,  # type: ignore[arg-type]
                mutations=input_dict["mutations"],
                batch_size=input_dict["batch_size"],
                seq_only=input_dict["seq_only"],
                scn_diffusion_steps=input_dict["scn_diffusion_steps"],
                scn_step_scale=input_dict["scn_step_scale"],
                seed=input_dict["seed"],
                model_variant=input_dict["model_variant"],
                device=input_dict["device"],
                verbose=input_dict["verbose"],
            )
        raise ValueError(
            f"fampnn: unknown operation {operation!r}; valid: ['sample', 'pack', 'score_all_mutations', 'score_mutations']"
        )


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
        raise ValueError("fampnn: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
