"""Standalone BioEmu inference implementation."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from standalone_helpers import get_logger, set_torch_seed

logger = get_logger(__name__)


class BioEmuModel:
    """BioEmu standalone model wrapper.

    Holds the score model + SDEs across calls so the worker only loads weights
    once. ``__call__`` moves the model to the requested device, runs sampling,
    then moves it back to CPU (matches the ``ProteinMPNNModel`` pattern).
    """

    def __init__(self) -> None:
        """Initialize BioEmuModel."""
        self._loaded = False
        self._model_name: str | None = None
        self.device: str | None = None
        self.score_model: Any = None
        self.sdes: Any = None
        self.verbose: bool = False

    def __call__(
        self,
        sequence: str,
        num_samples: int = 500,
        model_name: str = "bioemu-v1.1",
        filter_samples: bool = True,
        batch_size: int = 10,
        denoiser_type: str = "dpm",
        denoiser_config: str | None = None,
        msa_a3m: str | None = None,
        msa_host_url: str | None = None,
        cache_embeds_dir: str | None = None,
        cache_so3_dir: str | None = None,
        device: str = "cuda",
        output_dir: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Sample a conformational ensemble with BioEmu."""
        if not self._loaded or self._model_name != model_name:
            self.load(
                model_name=model_name,
                device=device,
                cache_so3_dir=cache_so3_dir,
                verbose=verbose,
            )
        elif self.device != device:
            self.to_device(device)

        self.verbose = verbose
        set_torch_seed(seed)

        use_temp_dir = output_dir is None
        if use_temp_dir:
            tmp_obj = tempfile.TemporaryDirectory()
            working_dir = tmp_obj.name
        else:
            working_dir = str(output_dir)
            os.makedirs(working_dir, exist_ok=True)

        try:
            import hydra.utils
            import numpy as np
            import yaml
            from bioemu.convert_chemgraph import save_pdb_and_xtc
            from bioemu.sample import DEFAULT_DENOISER_CONFIG_DIR, generate_batch
            from bioemu.seq_io import check_protein_valid, write_fasta
            from bioemu.steering import log_physicality
            from bioemu.utils import count_samples_in_output_dir, format_npz_samples_filename
            from tqdm import tqdm

            if verbose:
                logger.info(f"Sampling {num_samples} conformations for sequence of length {len(sequence)}")
                logger.info(f"Using model: {self._model_name}, device: {self.device}")

            # Wrapper-supplied MSA → file path lets bioemu skip its internal ColabFold.
            msa_file: str | None = None
            if msa_a3m:
                msa_file = os.path.join(working_dir, "query.a3m")
                with open(msa_file, "w") as handle:
                    handle.write(msa_a3m)

            check_protein_valid(sequence)

            output_path = Path(working_dir).expanduser().resolve()
            output_path.mkdir(parents=True, exist_ok=True)

            fasta_path = output_path / "sequence.fasta"
            if not fasta_path.is_file():
                write_fasta([sequence], fasta_path)

            # Denoiser is config-dependent so it stays per-call (cheap).
            if denoiser_config is None:
                denoiser_config_path = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"
            else:
                denoiser_config_path = Path(denoiser_config).expanduser().resolve()
            with open(denoiser_config_path) as f:
                denoiser_config_obj = yaml.safe_load(f)
            denoiser = hydra.utils.instantiate(denoiser_config_obj)

            # Sequence length scales memory quadratically. max(1, ...) guards
            # against the int truncation falling to 0 for sequences ≳316aa.
            adjusted_batch_size = max(1, min(int(batch_size * (100 / len(sequence)) ** 2), num_samples))

            base_seed = seed if seed is not None else time.time_ns()

            existing_num_samples = count_samples_in_output_dir(output_path)
            for start_idx in tqdm(
                range(existing_num_samples, num_samples, adjusted_batch_size),
                desc="Sampling batches...",
                disable=not verbose,
            ):
                n = min(adjusted_batch_size, num_samples - start_idx)
                npz_path = output_path / format_npz_samples_filename(start_idx, n)
                if npz_path.exists():
                    raise ValueError(
                        f"Not sure why {npz_path} already exists when so far only "
                        f"{existing_num_samples} samples have been generated."
                    )
                batch_seed = base_seed + start_idx
                batch = generate_batch(
                    score_model=self.score_model,
                    sequence=sequence,
                    sdes=self.sdes,
                    batch_size=min(adjusted_batch_size, n),
                    seed=batch_seed,
                    denoiser=denoiser,
                    cache_embeds_dir=cache_embeds_dir,
                    msa_file=msa_file,
                    msa_host_url=msa_host_url,
                    fk_potentials=None,
                    steering_config=None,
                )
                batch_np = {k: v.cpu().numpy() for k, v in batch.items()}
                np.savez(npz_path, **batch_np, sequence=sequence)

            samples_files = sorted(output_path.glob("batch_*.npz"))
            sequences_in_files = [np.load(f)["sequence"].item() for f in samples_files]
            if set(sequences_in_files) != {sequence}:
                raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences_in_files)}")
            positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
            node_orientations = torch.tensor(np.concatenate([np.load(f)["node_orientations"] for f in samples_files]))
            log_physicality(positions, node_orientations, sequence)
            save_pdb_and_xtc(
                pos_nm=positions,
                node_orientations=node_orientations,
                topology_path=output_path / "topology.pdb",
                xtc_path=output_path / "samples.xtc",
                sequence=sequence,
                filter_samples=filter_samples,
            )

            pdb_frames, num_frames, num_residues = self.extract_pdb_frames(working_dir, verbose)
            return {
                "pdb_frames": pdb_frames,
                "num_frames": num_frames,
                "num_residues": num_residues,
            }
        finally:
            # Park weights on CPU between calls (proteinmpnn pattern); next
            # call's to_device() brings them back without reloading from disk.
            self.unload()
            if use_temp_dir:
                tmp_obj.cleanup()

    def extract_pdb_frames(
        self,
        output_dir: str,
        verbose: bool,
    ) -> tuple[list[str], int, int]:
        """Extract PDB frame strings from BioEmu trajectory files."""
        import mdtraj as md

        output_path = Path(output_dir)
        top_path = output_path / "topology.pdb"
        xtc_path = output_path / "samples.xtc"

        if not top_path.exists():
            raise FileNotFoundError(f"bioemu: topology file not found: {top_path}")

        traj = md.load(str(xtc_path), top=str(top_path)) if xtc_path.exists() else md.load(str(top_path))

        if verbose:
            logger.info(f"Loaded ensemble: {traj.n_frames} frames, {traj.n_residues} residues")

        pdb_frames: list[str] = []
        for frame_idx in range(traj.n_frames):
            frame = traj.slice(frame_idx)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as handle:
                tmp_path = handle.name
            try:
                frame.save_pdb(tmp_path)
                with open(tmp_path) as handle:
                    pdb_frames.append(handle.read())
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError as exc:
                    logger.warning(f"Failed to clean up temporary file {tmp_path}: {exc}")

        return pdb_frames, traj.n_frames, traj.n_residues

    def load(
        self,
        model_name: str = "bioemu-v1.1",
        device: str = "cuda",
        cache_so3_dir: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Download weights, build the score model + SDEs, move to ``device``."""
        from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint

        self.verbose = verbose
        if verbose:
            logger.info(f"Loading BioEmu model: {model_name} on {device}")

        ckpt_path, model_config_path = maybe_download_checkpoint(
            model_name=model_name, ckpt_path=None, model_config_path=None
        )
        # load_model uses map_location="cpu" upstream — track that here.
        self.score_model = load_model(ckpt_path, model_config_path)
        self.sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)
        self.device = "cpu"
        self._model_name = model_name
        self.to_device(device)
        self._loaded = True

        if verbose:
            logger.info("BioEmu model initialized successfully")

    def to_device(self, device: str) -> None:
        """Move the held score model + node_orientations SDE to ``device``."""
        from standalone_helpers import move_model_to_device

        if self.score_model is None:
            raise ValueError("bioemu: cannot move unloaded model to device — call load() first")
        if self.device == device:
            return

        if self.verbose:
            logger.info(f"Moving BioEmu to {device}")
        self.score_model = move_model_to_device(self.score_model, self.device, device)
        # sdes["node_orientations"] (SO3SDE) is a torch.nn.Module with precomputed IGSO(3)
        # lookup tables; sdes["pos"] (CosineVPSDE) has no device state.
        if self.sdes is not None and "node_orientations" in self.sdes:
            self.sdes["node_orientations"] = move_model_to_device(self.sdes["node_orientations"], self.device, device)
        self.device = device

    def unload(self) -> None:
        """Park the held score model + SO3 SDE on CPU. Does not reset ``_loaded``."""
        from standalone_helpers import move_model_to_device

        if not self._loaded or self.device == "cpu":
            return

        if self.verbose:
            logger.info("Unloading BioEmu to CPU")
        self.score_model = move_model_to_device(self.score_model, self.device, "cpu")
        if self.sdes is not None and "node_orientations" in self.sdes:
            self.sdes["node_orientations"] = move_model_to_device(self.sdes["node_orientations"], self.device, "cpu")
        self.device = "cpu"


def run_bioemu_batch(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run BioEmu sampling for one or more sequences."""
    assert _model is not None, "run_bioemu_batch() must be reached via dispatch()"
    sequences = input_data["sequences"]
    msa_a3m_contents = input_data["msa_a3m_contents"]
    output_dir = input_data["output_dir"]
    seed = input_data["seed"]
    verbose = input_data["verbose"]

    results: list[dict[str, Any]] = []
    for seq_idx, sequence in enumerate(sequences):
        per_sequence_output_dir = None
        if output_dir:
            per_sequence_output_dir = (
                output_dir if len(sequences) == 1 else str(Path(output_dir) / f"complex_{seq_idx}")
            )

        # Derive a distinct but reproducible seed for each sequence
        per_seq_seed = seed + seq_idx if seed is not None else None

        try:
            result = _model(
                sequence=sequence,
                num_samples=input_data["num_samples"],
                model_name=input_data["model_name"],
                filter_samples=input_data["filter_samples"],
                batch_size=input_data["batch_size"],
                denoiser_type=input_data.get("denoiser_type", "dpm"),
                denoiser_config=input_data.get("denoiser_config"),
                msa_a3m=msa_a3m_contents.get(sequence),
                msa_host_url=input_data.get("msa_host_url"),
                cache_embeds_dir=input_data.get("cache_embeds_dir"),
                cache_so3_dir=input_data.get("cache_so3_dir"),
                device=input_data["device"],
                output_dir=per_sequence_output_dir,
                seed=per_seq_seed,
                verbose=verbose,
            )
        except Exception as e:
            raise RuntimeError(f"bioemu: sequence {seq_idx + 1}/{len(sequences)} failed: {e}") from e
        results.append(result)

    return {"results": results}


# ============================================================================
# Dispatch
# ============================================================================
_model: BioEmuModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = BioEmuModel()

    operation = input_dict["operation"]
    if operation == "sample":
        return run_bioemu_batch(input_dict)
    if operation == "introspect_loaded":
        return {
            "loaded": _model._loaded,
            "model_name": _model._model_name,
            "device": _model.device,
            "model_id": id(_model.score_model) if _model.score_model is not None else None,
        }
    raise ValueError(f"bioemu: unknown operation {operation!r}; valid: ['sample', 'introspect_loaded']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    if _model is not None and _model._loaded:
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("bioemu: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as handle:
        input_payload = json.load(handle)

    output_payload = dispatch(input_payload)

    with open(sys.argv[2], "w") as handle:
        json.dump(output_payload, handle)
