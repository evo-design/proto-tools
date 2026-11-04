"""Persistent Mol-1 inference in an isolated tool environment."""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import (
    get_logger,
    get_random_int,
    move_model_to_device,
    resolve_weights_dir,
    serialize_output,
    set_torch_seed,
)

logger = get_logger(__name__)


def _extract_structure_and_scores(output_dir: str, include_pae_matrix: bool = False) -> dict[str, Any]:
    """Read the best sample's CIF and confidence values without changing their scales."""
    summaries = sorted(Path(output_dir).rglob("*_summary_confidence_sample_*.json"))
    if not summaries:
        raise FileNotFoundError(f"mol1: prediction produced no confidence files under {output_dir}")
    candidates = [(path, json.loads(path.read_text())) for path in summaries]
    if any(not math.isfinite(float(summary["ranking_score"])) for _, summary in candidates):
        raise ValueError("mol1: prediction produced a non-finite ranking score")
    summary_path, summary = max(candidates, key=lambda candidate: float(candidate[1]["ranking_score"]))
    cif_path = summary_path.with_name(summary_path.name.replace("_summary_confidence", "")).with_suffix(".cif")
    metrics = {
        "ranking_score": float(summary["ranking_score"]),
        "avg_plddt": float(summary["plddt"]),
        "ptm": float(summary["ptm"]),
        "iptm": float(summary["iptm"]),
        "gpde": float(summary["gpde"]),
        "has_clash": bool(summary["has_clash"]),
        "chain_plddt": summary["chain_plddt"],
        "chain_pair_iptm": summary["chain_pair_iptm"],
    }
    if include_pae_matrix:
        full_path = summary_path.with_name(summary_path.name.replace("_summary_confidence", "_full_data"))
        pae = json.loads(full_path.read_text()).get("token_pair_pae")
        if (
            not isinstance(pae, list)
            or not pae
            or any(not isinstance(row, list) or len(row) != len(pae) for row in pae)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for row in pae for value in row)
        ):
            raise ValueError("mol1: token_pair_pae must be a non-empty square matrix of finite numbers")
        metrics["pae"] = pae
        metrics["avg_pae"] = sum(sum(row) for row in pae) / sum(len(row) for row in pae)
    return {"structure_cif_output": cif_path.read_text(), "metrics": metrics}


class Mol1Model:
    """Keep the upstream runner loaded while updating request-specific settings."""

    def __init__(self) -> None:
        """Initialize an unloaded runner."""
        self._loaded = False
        self.runner: Any = None
        self.device: str | None = None
        self.model_checkpoint: str | None = None

    def load(self, input_dict: dict[str, Any], output_dir: str) -> None:
        """Resolve managed assets and load one checkpoint on the assigned device."""
        root = resolve_weights_dir("mol1")
        if root is None:
            raise FileNotFoundError("mol1: unable to resolve the managed weights directory")
        os.environ["MOL_ROOT_DIR"] = root
        os.environ["LAYERNORM_TYPE"] = "torch"

        from mol.config.inference import inference_configs
        from mol.runner.batch_inference import get_default_runner

        checkpoint = input_dict["model_checkpoint"]
        checkpoint_path = (
            Path(root) / "checkpoint" / "mol-1-base.pt" if checkpoint == "mol-1-base" else Path(checkpoint)
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"mol1: checkpoint missing at {checkpoint_path}; re-run setup or supply an existing file"
            )
        inference_configs["dump_dir"] = output_dir
        logger.update_status("Loading Mol-1 checkpoint")
        self.runner = get_default_runner(
            n_cycle=input_dict["num_cycles"],
            n_step=input_dict["num_steps"],
            n_sample=input_dict["num_samples"],
            dtype=input_dict["dtype"],
            device=input_dict["device"],
            load_checkpoint_path=str(checkpoint_path),
            use_msa=input_dict["use_msa"],
            trimul_kernel="torch",
            triatt_kernel="torch",
            need_atom_confidence=input_dict["include_pae_matrix"],
            memory_settings={
                "infer_setting.chunk_size": input_dict["chunk_size"],
                "infer_setting.sample_diffusion_chunk_size": input_dict["sample_diffusion_chunk_size"],
            },
        )
        self.device = str(self.runner.device)
        self.model_checkpoint = checkpoint
        self._loaded = True

    def predict(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict a native job without triggering additional MSA or template searches."""
        with tempfile.TemporaryDirectory(prefix="mol1_") as output_dir:
            if not self._loaded:
                self.load(input_dict, output_dir)
            elif input_dict["model_checkpoint"] != self.model_checkpoint:
                raise ValueError("mol1: changing the checkpoint requires a new worker")
            elif input_dict["device"] != self.device:
                self.to_device(input_dict["device"])

            import torch
            from mol.runner.inference import infer_predict, update_gpu_compatible_configs

            configs = self.runner.configs
            configs.input_json_path = str(Path(output_dir) / "input.json")
            Path(configs.input_json_path).write_text(json.dumps([input_dict["job"]]))
            configs.dump_dir = output_dir
            configs.seeds = [input_dict["sampling_seed"]]
            configs.use_seeds_in_json = False
            configs.deterministic = input_dict.get("seed") is not None
            torch.use_deterministic_algorithms(configs.deterministic)
            configs.model.N_cycle = input_dict["num_cycles"]
            self.runner.model.N_cycle = input_dict["num_cycles"]
            configs.sample_diffusion.N_sample = input_dict["num_samples"]
            configs.sample_diffusion.N_step = input_dict["num_steps"]
            configs.dtype = input_dict["dtype"]
            configs.use_msa = input_dict["use_msa"]
            configs.use_template = False
            configs.use_rna_msa = False
            configs.need_atom_confidence = input_dict["include_pae_matrix"]
            configs.infer_setting.chunk_size = input_dict["chunk_size"]
            configs.infer_setting.sample_diffusion_chunk_size = input_dict["sample_diffusion_chunk_size"]
            update_gpu_compatible_configs(configs)
            self.runner.dump_dir = output_dir
            self.runner.error_dir = str(Path(output_dir) / "ERR")
            self.runner.init_dumper(need_atom_confidence=configs.need_atom_confidence, sorted_by_ranking_score=True)
            self.runner.update_model_configs(configs)
            logger.update_status("Predicting with Mol-1")
            if input_dict["verbose"]:
                logger.info("Mol-1: generating %s samples on %s", input_dict["num_samples"], self.device)
            infer_predict(self.runner, configs)
            return _extract_structure_and_scores(output_dir, include_pae_matrix=input_dict["include_pae_matrix"])

    def to_device(self, device: str) -> None:
        """Move the persistent model and update the upstream runner's device state."""
        from mol.runner.inference import select_inference_device

        target = select_inference_device(device)
        self.runner.model = move_model_to_device(self.runner.model, self.device or "cpu", str(target))
        self.runner.device = target
        self.runner.use_cuda = target.type == "cuda"
        self.runner.configs.device = str(target)
        if target.type == "cuda":
            import torch

            torch.cuda.set_device(target)
        self.device = str(target)


_model: Mol1Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the prediction operation with a reproducible, per-item sampling seed."""
    if input_dict["operation"] != "predict":
        raise ValueError(f"mol1: unknown operation {input_dict['operation']!r}")
    seed = input_dict.get("seed")
    if seed is not None:
        seed = (seed + input_dict.get("item_index", 0)) % (2**32)
    set_torch_seed(seed)
    sampling_seed = seed if seed is not None else get_random_int()
    global _model
    if _model is None:
        _model = Mol1Model()
    result = serialize_output(_model.predict({**input_dict, "sampling_seed": sampling_seed}))
    assert isinstance(result, dict)
    return result


def to_device(device: str) -> dict[str, Any]:
    """Move the loaded model for DeviceManager CPU offload or GPU reassignment."""
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU allocation through the shared PyTorch memory helper."""
    from standalone_helpers import get_pytorch_memory_stats

    stats: dict[str, Any] = get_pytorch_memory_stats(
        _model.device if _model is not None and _model.device is not None else 0
    )
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("mol1: usage: python inference.py <input.json> <output.json>")
    result = dispatch(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(result))
