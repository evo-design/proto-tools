"""
AlphaFold2 standalone inference implementation via ColabDesign.

This script runs in an isolated venv with JAX and ColabDesign installed.
It communicates via JSON files for input/output.

Usage:
    python inference.py <input_json_path> <output_json_path>
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Configure stderr handler so debug output is captured by persistent worker
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

# ColabDesign creates a fresh model per prediction. No persistent state to cache.
# The params directory is resolved once and reused.
_params_dir: str | None = None


def _resolve_params_dir() -> str:
    """Locate the AlphaFold2 parameters directory.

    Checks PROTO_MODEL_CACHE first, then falls back to the venv-local
    location at ``$TOOL_VENV_PATH/data/params/``.
    """
    global _params_dir
    if _params_dir is not None:
        return _params_dir

    from standalone_helpers import resolve_weights_dir

    weights_dir = resolve_weights_dir("alphafold2")
    if weights_dir:
        params_dir = Path(weights_dir) / "params"
    else:
        # NONE mode or no venv: use venv-local default
        venv_path = os.environ.get("TOOL_VENV_PATH")
        if not venv_path:
            raise RuntimeError(
                "TOOL_VENV_PATH not set. AlphaFold2 must be run via ToolInstance."
            )
        params_dir = Path(venv_path) / "data" / "params"

    if not params_dir.exists() or not any(params_dir.glob("*.npz")):
        raise RuntimeError(
            f"AlphaFold2 parameters not found at {params_dir}. "
            "Run the standalone setup.sh script to download parameters."
        )

    _params_dir = str(params_dir)
    return _params_dir


def _configure_device(device: str):
    """Set JAX device via environment variables. Must be called before importing JAX."""
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device.startswith("cuda"):
        if ":" in device:
            gpu_idx = device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx


def _predict_structure(
    complex_data: Dict[str, Any],
    num_recycles: int = 3,
    model_num: int = 1,
    num_ensemble_models: int = 1,
    seed: Optional[int] = None,
    msa_a3m_content: Optional[str] = None,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run AlphaFold2 prediction on a single complex.

    Args:
        complex_data: Dict with keys: chains, seq_lengths, num_chains, total_residues
        num_recycles: Number of recycling iterations
        model_num: Which AF2 model parameter set to use (1-5)
        num_ensemble_models: Number of models to run and average
        seed: Random seed for reproducibility (None for non-deterministic)
        msa_a3m_content: A3M string content for MSA (single-chain only)
        device: Device to run on
        verbose: Whether to print status messages

    Returns:
        Dict with keys: pdb, avg_plddt, ptm, iptm, avg_pae
    """
    import numpy as np
    from colabdesign import mk_afdesign_model as mk_af_model

    params_dir = _resolve_params_dir()

    chains = complex_data["chains"]
    seq_lengths = complex_data["seq_lengths"]
    num_chains = complex_data["num_chains"]
    total_residues = complex_data["total_residues"]
    use_multimer = num_chains > 1
    use_msa = msa_a3m_content is not None

    logger.debug(
        f"Predicting structure: {num_chains} chain(s), "
        f"{total_residues} residues, multimer={use_multimer}, device={device}"
    )

    # Build model kwargs
    model_kwargs = {
        "protocol": "hallucination",
        "use_multimer": use_multimer,
        "data_dir": params_dir,
    }

    if use_msa:
        # Prediction mode: sequence comes from MSA, not optimization
        model_kwargs["optimize_seq"] = False
        model_kwargs["num_msa"] = 512
        model_kwargs["num_extra_msa"] = 1024

    # Create AF2 model (fresh instance per prediction)
    af_model = mk_af_model(**model_kwargs)

    # Prepare inputs
    is_homooligomer = use_multimer and len(set(chains)) == 1
    if is_homooligomer:
        # Homo-oligomer: single length + copies
        af_model.prep_inputs(length=seq_lengths[0], copies=num_chains)
    elif use_multimer:
        # Heteromer: list of unique lengths
        af_model.prep_inputs(length=seq_lengths)
    else:
        af_model.prep_inputs(length=seq_lengths[0])

    # Set sequence or MSA
    if use_msa:
        from colabdesign.shared.parsers import parse_a3m
        msa, deletion_matrix = parse_a3m(a3m_string=msa_a3m_content)
        af_model.set_msa(msa, deletion_matrix)
        # ColabDesign stores raw MSA in _inputs["msa"]
        msa_input = af_model._inputs.get("msa")
        if msa_input is not None:
            logger.debug(
                f"MSA injected: {msa_input.shape[0]} sequences, "
                f"length={msa_input.shape[1]}"
            )
        else:
            logger.warning("MSA not found in model inputs after set_msa")
    elif is_homooligomer:
        logger.debug("Running single-sequence mode (homo-oligomer, no MSA)")
        af_model.set_seq(seq=chains[0])
    else:
        logger.debug("Running single-sequence mode (no MSA)")
        af_model.set_seq(seq="".join(chains))

    # Run prediction
    # When ensembling (num_ensemble_models > 1), don't restrict the model pool;
    # ColabDesign's num_models picks the first N from the available set.
    # When not ensembling, models= selects the specific parameter set (0-indexed).
    predict_kwargs = {
        "num_recycles": num_recycles,
        "seed": seed,
        "verbose": verbose,
    }
    if num_ensemble_models > 1:
        predict_kwargs["num_models"] = num_ensemble_models
    else:
        predict_kwargs["models"] = [model_num - 1]

    af_model.predict(**predict_kwargs)

    # Extract results from aux
    aux = af_model.aux

    # Get PDB output (returns string when filename=None)
    pdb_output = af_model.save_pdb()

    # Extract pLDDT scores (ColabDesign normalizes to 0-1)
    plddt = aux.get("plddt")
    avg_plddt = float(np.mean(plddt)) if plddt is not None else 0.0

    # Extract PTM and iPTM directly from aux
    ptm = float(aux.get("ptm", 0.0))
    iptm = float(aux.get("i_ptm", 0.0)) if use_multimer else None

    # Extract PAE
    pae = aux.get("pae")
    avg_pae = float(np.mean(pae)) if pae is not None else None

    logger.debug(
        f"Prediction complete: avg_plddt={avg_plddt:.4f}, ptm={ptm:.4f}"
    )

    return {
        "pdb": pdb_output,
        "avg_plddt": avg_plddt,
        "ptm": ptm,
        "iptm": iptm,
        "avg_pae": avg_pae,
    }


# ============================================================================
# Dispatch
# ============================================================================
def dispatch(input_dict: dict) -> dict:
    """Entry point for ToolInstance one-shot execution."""
    # Configure device before any JAX imports
    _configure_device(input_dict.get("device", "cuda"))

    return _predict_structure(
        complex_data=input_dict["complex_data"],
        num_recycles=input_dict.get("num_recycles", 3),
        model_num=input_dict.get("model_num", 1),
        num_ensemble_models=input_dict.get("num_ensemble_models", 1),
        seed=input_dict.get("seed"),
        msa_a3m_content=input_dict.get("msa_a3m_content"),
        device=input_dict.get("device", "cuda"),
        verbose=input_dict.get("verbose", False),
    )


def to_device(device: str) -> dict:
    """Passthrough - tool does not maintain persistent state."""
    return {"success": True, "device": device}


def get_memory_stats() -> dict:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_jax_memory_stats

    return get_jax_memory_stats(device_index=0)


# ============================================================================
# Standalone Script Entry Point
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python inference.py <input_json_path> <output_json_path>"
        )

    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
