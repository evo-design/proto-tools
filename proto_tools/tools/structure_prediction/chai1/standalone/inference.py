"""Chai1 inference implementation."""

import logging
import sys
from pathlib import Path
from typing import Any

from standalone_helpers import get_random_int, set_torch_seed

logger = logging.getLogger(__name__)


class Chai1Model:
    """Chai1 model for multi-modal structure prediction."""

    def __init__(self) -> None:
        """Initialize Chai1 model wrapper."""
        self._loaded = False
        self._chai1_run_inference = None
        self.device: str | None = None

    def __call__(
        self,
        fasta_file: Path,
        output_dir: Path,
        use_esm_embeddings: bool = True,
        msa_directory: Path | None = None,
        num_trunk_recycles: int = 3,
        num_diffn_timesteps: int = 200,
        num_diffn_samples: int = 1,
        num_trunk_samples: int = 1,
        seed: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run Chai1 structure prediction.

        Args:
            fasta_file: Path to input FASTA file
            output_dir: Directory to write output to
            use_esm_embeddings: Whether to use ESM embeddings
            msa_directory: Path to directory containing pre-computed MSA files (.aligned.pqt)
            num_trunk_recycles: Number of trunk recycles
            num_diffn_timesteps: Number of diffusion timesteps
            num_diffn_samples: Number of diffusion samples
            num_trunk_samples: Number of trunk samples
            seed: Random seed for reproducibility
            device: Device to run on ('cuda' or 'cpu')
            verbose: Whether to print status messages

        Returns:
            Dictionary containing cif_output and metrics
        """
        # Lazy load on first call
        if not self._loaded:
            self.load(device, verbose)

        logger.debug("\n=== Chai1 Prediction ===")
        logger.debug(f"Input FASTA: {fasta_file}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"MSA directory: {msa_directory}")
        logger.debug("Reading FASTA content...")
        with open(fasta_file) as f:
            fasta_content = f.read()
        logger.debug(f"\n--- Input FASTA ---\n{fasta_content}\n------------------\n")
        sys.stdout.flush()

        # Run the model
        candidates = self._chai1_run_inference(  # type: ignore[misc]
            fasta_file=Path(fasta_file),
            output_dir=Path(output_dir),
            use_esm_embeddings=use_esm_embeddings,
            use_msa_server=False,
            msa_directory=msa_directory,
            constraint_path=None,
            use_templates_server=False,
            template_hits_path=None,
            recycle_msa_subsample=0,
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=num_diffn_samples,
            num_trunk_samples=num_trunk_samples,
            device=self.device,
            seed=seed,
            low_memory=True,
        )

        # Get the best model by score
        candidates = candidates.sorted()
        best_cif_path = candidates.cif_paths[0]
        best_score = candidates.ranking_data[0].aggregate_score.item()
        # Note: These pLDDTs are 0-1, but the B factor outputs are 0-100:
        best_plddt = candidates.plddt[0].mean().item()
        best_ptm = candidates.ranking_data[0].ptm_scores.complex_ptm.item()
        best_iptm = candidates.ranking_data[0].ptm_scores.interface_ptm.item()
        best_pae = candidates.pae[0].mean().item()

        logger.debug("\nChai1 prediction completed successfully")
        logger.debug(f"Best aggregate score: {best_score:.4f}")
        logger.debug(f"pLDDT: {best_plddt:.4f}, pTM: {best_ptm:.4f}, iPTM: {best_iptm:.4f}")
        sys.stdout.flush()

        with open(best_cif_path) as f:
            cif_output = f.read()

        return {
            "cif_output": cif_output,
            "metrics": {
                "avg_plddt": best_plddt,
                "ptm": best_ptm,
                "iptm": best_iptm,
                "avg_pae": best_pae,
                "confidence_score": best_score,
            },
        }

    def load(self, device: str = "cuda", verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load Chai1 model components.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            verbose: Whether to print status messages
        """
        logger.debug("Initializing Chai1")

        try:
            from chai_lab.chai1 import run_inference  # type: ignore[import-not-found]

            self._chai1_run_inference = run_inference
        except ImportError:
            raise ImportError(
                "Could not import chai_lab. Make sure Chai1 is installed in the current environment."
            ) from None

        self.device = device
        self._loaded = True

        logger.debug("Chai1 initialized successfully")


# ============================================================================
# Dispatch
# ============================================================================
_model: Chai1Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = Chai1Model()

    seed = input_dict["seed"]
    set_torch_seed(seed)

    if seed is not None:
        # Chai-lab's diffusion uses scatter/index_add ops that require full
        # deterministic mode beyond cudnn flags alone. Note: these settings
        # are process-global and sticky — chai1 is excluded from the persistent-worker
        # seed reproducibility test (_SEED_PERSISTENT_EXCLUDED_KEYS) precisely
        # because they leak across dispatches within one worker.
        import os

        import torch

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    sampling_seed = seed if seed is not None else get_random_int()

    operation = input_dict["operation"]
    if operation == "predict":
        return _model(
            fasta_file=Path(input_dict["fasta_file"]),
            output_dir=Path(input_dict["output_dir"]),
            use_esm_embeddings=input_dict["use_esm_embeddings"],
            msa_directory=(Path(input_dict["msa_directory"]) if input_dict.get("msa_directory") else None),
            num_trunk_recycles=input_dict["num_trunk_recycles"],
            num_diffn_timesteps=input_dict["num_diffn_timesteps"],
            num_diffn_samples=input_dict["num_diffn_samples"],
            num_trunk_samples=input_dict["num_trunk_samples"],
            seed=sampling_seed,
            device=input_dict["device"],
            verbose=input_dict["verbose"],
        )
    raise ValueError(f"Unknown operation: {operation}")


def to_device(device: str) -> dict[str, Any]:
    """Passthrough - tool does not maintain persistent state."""
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    import json

    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
