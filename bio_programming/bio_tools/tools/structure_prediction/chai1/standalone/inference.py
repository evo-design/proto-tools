"""
Chai1 inference implementation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Chai1Model:
    """Chai1 model for multi-modal structure prediction."""

    def __init__(self):
        """Initialize Chai1 model wrapper."""
        self._loaded = False
        self._chai1_run_inference = None
        self.device = None

    def __call__(
        self,
        fasta_file: Path,
        output_dir: Path,
        use_esm_embeddings: bool = True,
        msa_directory: Optional[Path] = None,
        num_trunk_recycles: int = 3,
        num_diffn_timesteps: int = 200,
        num_diffn_samples: int = 1,
        num_trunk_samples: int = 1,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Chai1 structure prediction.

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

        logger.debug(f"\n=== Chai1 Prediction ===")
        logger.debug(f"Input FASTA: {fasta_file}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"MSA directory: {msa_directory}")
        logger.debug(f"Reading FASTA content...")
        with open(fasta_file, "r") as f:
            fasta_content = f.read()
        logger.debug(f"\n--- Input FASTA ---\n{fasta_content}\n------------------\n")
        sys.stdout.flush()

        # Run the model
        candidates = self._chai1_run_inference(
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

        logger.debug(f"\nChai1 prediction completed successfully")
        logger.debug(f"Best aggregate score: {best_score:.4f}")
        logger.debug(f"pLDDT: {best_plddt:.4f}, pTM: {best_ptm:.4f}, iPTM: {best_iptm:.4f}")
        sys.stdout.flush()

        with open(best_cif_path, "r") as f:
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

    def load(self, device: str = "cuda", verbose: bool = False):
        """
        Load Chai1 model components.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            verbose: Whether to print status messages
        """
        logger.debug("Initializing Chai1")

        try:
            from chai_lab.chai1 import run_inference  # type: ignore

            self._chai1_run_inference = run_inference
        except ImportError:
            raise ImportError(
                "Could not import chai_lab. Make sure Chai1 is installed in the current environment."
            )

        self.device = device
        self._loaded = True

        logger.debug("Chai1 initialized successfully")


# Standalone script entry point for venv execution
if __name__ == "__main__":
    import json

    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python inference.py <input_json_path> <output_json_path>"
        )

    # Get the input and output json paths
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Create model and run inference
    model = Chai1Model()

    # Build kwargs for model call
    model_kwargs = {
        "fasta_file": Path(input_data["fasta_file"]),
        "output_dir": Path(input_data["output_dir"]),
        "use_esm_embeddings": input_data["use_esm_embeddings"],
        "msa_directory": (
            Path(input_data["msa_directory"])
            if input_data.get("msa_directory")
            else None
        ),
        "num_trunk_recycles": input_data["num_trunk_recycles"],
        "num_diffn_timesteps": input_data["num_diffn_timesteps"],
        "num_diffn_samples": input_data["num_diffn_samples"],
        "num_trunk_samples": input_data["num_trunk_samples"],
        "seed": input_data["seed"],
        "device": input_data.get("device", "cuda"),
        "verbose": True,
    }

    result = model(**model_kwargs)

    # Write the output to a json file
    with open(output_json_path, "w") as f:
        json.dump(result, f)
