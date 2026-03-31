"""Standalone BioEmu inference implementation."""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from standalone_helpers import move_model_to_device

logger = logging.getLogger(__name__)


class BioEmuModel:
    """BioEmu model wrapper used by the standalone execution path."""

    def __init__(self):
        self._loaded = False
        self._model_name: Optional[str] = None
        self.device: Optional[str] = None

    def __call__(
        self,
        sequence: str,
        num_samples: int = 500,
        model_name: str = "bioemu-v1.1",
        filter_samples: bool = True,
        batch_size: int = 10,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Sample a conformational ensemble with BioEmu."""
        if not self._loaded or self._model_name != model_name:
            self.load(model_name=model_name, device=device, verbose=verbose)
        elif self.device != device:
            self.to_device(device)

        use_temp_dir = output_dir is None
        if use_temp_dir:
            tmp_obj = tempfile.TemporaryDirectory()
            working_dir = tmp_obj.name
        else:
            working_dir = str(output_dir)
            os.makedirs(working_dir, exist_ok=True)

        try:
            from bioemu.sample import main as bioemu_sample

            if verbose:
                logger.info(
                    f"Sampling {num_samples} conformations for sequence of length {len(sequence)}"
                )
                logger.info(f"Using model: {self._model_name}, device: {self.device}")

            bioemu_sample(
                sequence=sequence,
                num_samples=num_samples,
                model_name=self._model_name,
                output_dir=working_dir,
                batch_size_100=batch_size,
                filter_samples=filter_samples,
            )

            pdb_frames, num_frames, num_residues = self.extract_pdb_frames(
                working_dir, verbose
            )
            return {
                "pdb_frames": pdb_frames,
                "num_frames": num_frames,
                "num_residues": num_residues,
            }
        finally:
            if use_temp_dir:
                tmp_obj.cleanup()

    def extract_pdb_frames(
        self,
        output_dir: str,
        verbose: bool,
    ) -> Tuple[List[str], int, int]:
        """Extract PDB frame strings from BioEmu trajectory files."""
        import mdtraj as md

        output_path = Path(output_dir)
        top_path = output_path / "topology.pdb"
        xtc_path = output_path / "samples.xtc"

        if not top_path.exists():
            raise FileNotFoundError(f"BioEmu topology file not found: {top_path}")

        traj = md.load(str(xtc_path), top=str(top_path)) if xtc_path.exists() else md.load(str(top_path))

        if verbose:
            logger.info(
                f"Loaded ensemble: {traj.n_frames} frames, {traj.n_residues} residues"
            )

        pdb_frames: List[str] = []
        for frame_idx in range(traj.n_frames):
            frame = traj.slice(frame_idx)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdb", delete=False
            ) as handle:
                tmp_path = handle.name
            try:
                frame.save_pdb(tmp_path)
                with open(tmp_path, "r") as handle:
                    pdb_frames.append(handle.read())
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError as exc:
                    logger.warning(
                        f"Failed to clean up temporary file {tmp_path}: {exc}"
                    )

        return pdb_frames, traj.n_frames, traj.n_residues

    def load(
        self,
        model_name: str = "bioemu-v1.1",
        device: str = "cuda",
        verbose: bool = False,
    ) -> None:
        """Initialize BioEmu model metadata for runtime."""
        if verbose:
            logger.info(f"Loading BioEmu model: {model_name} on {device}")
        self._model_name = model_name
        self.device = device
        self._loaded = True
        if verbose:
            logger.info("BioEmu model initialized successfully")

    def to_device(self, device: str) -> None:
        """Move model to another device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model. Call load() first.")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Unload model and clear CUDA cache."""
        if self._loaded:
            if verbose:
                logger.info("Unloading BioEmu model")
            self._model_name = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_bioemu_batch(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run BioEmu sampling for one or more sequences."""
    sequences = input_data["sequences"]
    output_dir = input_data.get("output_dir")

    model = BioEmuModel()
    results: List[Dict[str, Any]] = []
    for seq_idx, sequence in enumerate(sequences):
        per_sequence_output_dir = None
        if output_dir:
            per_sequence_output_dir = (
                output_dir
                if len(sequences) == 1
                else str(Path(output_dir) / f"complex_{seq_idx}")
            )

        result = model(
            sequence=sequence,
            num_samples=input_data.get("num_samples", 500),
            model_name=input_data.get("model_name", "bioemu-v1.1"),
            filter_samples=input_data.get("filter_samples", True),
            batch_size=input_data.get("batch_size", 10),
            device=input_data.get("device", "cuda"),
            output_dir=per_sequence_output_dir,
            verbose=input_data.get("verbose", False),
        )
        results.append(result)

    model.unload(verbose=input_data.get("verbose", False))
    return {"results": results}


# ============================================================================
# Dispatch
# ============================================================================


def dispatch(input_dict: dict) -> dict:
    """Entry point for both persistent-worker and one-shot execution."""
    operation = input_dict.get("operation", "sample")
    if operation == "sample":
        return run_bioemu_batch(input_dict)
    else:
        raise ValueError(f"Unknown operation: {operation}")



def to_device(device: str) -> dict:
    """Passthrough for non-persistent tool - loads on each call."""
    # BioEmu creates a new model instance on each call, so there's no
    # persistent model to move. The device is passed in the input_dict.
    return {"success": True, "device": device, "note": "non-persistent tool, loads on each call"}


def get_memory_stats() -> dict:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    # BioEmu doesn't persist models, so just report overall GPU memory state
    return get_pytorch_memory_stats(device=0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1], "r") as handle:
        input_payload = json.load(handle)

    output_payload = dispatch(input_payload)

    with open(sys.argv[2], "w") as handle:
        json.dump(output_payload, handle)
