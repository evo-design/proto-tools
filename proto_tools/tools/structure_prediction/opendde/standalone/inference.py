"""OpenDDE inference implementation."""

import glob
import json
import os
import re
import sys
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

# Bundled model name -> checkpoint file under OPENDDE_ROOT_DIR (auto-downloaded by setup.sh); None = default weights.
_BUNDLED_CHECKPOINTS: dict[str, str | None] = {
    "opendde_v1": None,
    "opendde_abag": "checkpoint/opendde_abag.pt",
}


class OpenDDEExecutionError(Exception):
    """Raised when an OpenDDE prediction fails for a non-OOM reason."""


def _extract_structure_and_scores(output_dir: str, include_pae_matrix: bool = False) -> dict[str, Any]:
    """Select the best-ranked sample and extract its structure + confidence metrics.

    OpenDDE writes, per diffusion sample under
    ``<output_dir>/<job_name>/seed_<seed>/predictions/``, a
    ``<job_name>_sample_<rank>.cif`` and a matching
    ``<job_name>_summary_confidence_sample_<rank>.json``. The summary already
    carries scalar confidence values (``plddt`` is the mean pLDDT on a 0-100
    scale, plus ``ptm``/``iptm``/``gpde``/``ranking_score``); The best
    sample is the one with the highest ``ranking_score``.

    When ``include_pae_matrix`` is set, the matching ``*_full_data_sample_<rank>.json``
    (emitted by ``--need_atom_confidence``) is read for the winning sample and its
    token-pair PAE matrix is returned as ``pae`` (Å) with a derived scalar ``avg_pae``.

    Args:
        output_dir (str): Directory OpenDDE wrote predictions into.
        include_pae_matrix (bool): Also read the per-token PAE matrix + avg_pae.

    Returns:
        dict[str, Any]: ``{"structure_cif_output": <cif str>, "metrics": {...}}``.

    Raises:
        FileNotFoundError: If no summary/CIF outputs are found (or the full-data
            file is missing when ``include_pae_matrix`` is set).
    """
    summary_paths = glob.glob(os.path.join(output_dir, "**", "*_summary_confidence_sample_*.json"), recursive=True)
    if not summary_paths:
        raise FileNotFoundError(
            f"opendde: no '*_summary_confidence_sample_*.json' outputs found under {output_dir}; "
            "the prediction produced no samples"
        )

    # Pick the best sample by ranking score across all seeds
    best_path: str | None = None
    best_summary: dict[str, Any] = {}
    best_score = float("-inf")
    for path in summary_paths:
        with open(path) as f:
            summary = json.load(f)
        score = summary.get("ranking_score")
        if score is None:
            logger.warning("opendde: summary %s has no 'ranking_score'; skipping for selection", path)
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_path = path
            best_summary = summary

    if best_path is None:
        raise FileNotFoundError(
            f"opendde: found {len(summary_paths)} summary file(s) under {output_dir} but none had a 'ranking_score'"
        )

    # Derive the sibling CIF path from the winning summary filename:
    #   <job_name>_summary_confidence_sample_<rank>.json  ->  <job_name>_sample_<rank>.cif
    pred_dir = os.path.dirname(best_path)
    basename = os.path.basename(best_path)
    match = re.search(r"^(?P<prefix>.+)_summary_confidence_sample_(?P<rank>.+)\.json$", basename)
    if match is None:
        raise FileNotFoundError(f"opendde: could not parse sample rank from summary filename {basename!r}")
    cif_path = os.path.join(pred_dir, f"{match.group('prefix')}_sample_{match.group('rank')}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"opendde: structure CIF not found for best sample: {cif_path}")

    metrics: dict[str, Any] = {
        "avg_plddt": float(best_summary["plddt"]),
        "ptm": float(best_summary["ptm"]),
        "iptm": float(best_summary["iptm"]),
        "gpde": float(best_summary["gpde"]),
        "ranking_score": float(best_summary["ranking_score"]),
        "has_clash": bool(best_summary["has_clash"]),
    }

    if include_pae_matrix:
        # Sibling full-data file for the winning sample (written by --need_atom_confidence).
        full_data_path = os.path.join(pred_dir, f"{match.group('prefix')}_full_data_sample_{match.group('rank')}.json")
        if not os.path.exists(full_data_path):
            raise FileNotFoundError(
                f"opendde: PAE requested but full-data file not found: {full_data_path}; "
                "the run must pass --need_atom_confidence"
            )
        with open(full_data_path) as f:
            full_data = json.load(f)
        pae = full_data.get("token_pair_pae")
        if pae is None:
            raise KeyError(f"opendde: 'token_pair_pae' missing from {full_data_path}")
        import numpy as np

        metrics["pae"] = pae
        metrics["avg_pae"] = float(np.asarray(pae, dtype=float).mean())

    with open(cif_path) as f:
        cif_output = f.read()

    return {"structure_cif_output": cif_output, "metrics": metrics}


# ============================================================================
# Worker Protocol - OpenDDE Model Class
# ============================================================================


def _warn_on_chains_without_msa(input_json_path: str) -> None:
    """Warn when a protein chain lacks an MSA path: OpenDDE would then search its own server.

    That search covers every protein chain and overwrites the paths proto-tools wrote.
    """
    with open(input_json_path) as f:
        jobs = json.load(f)
    for job in jobs if isinstance(jobs, list) else [jobs]:
        missing = [
            str(position)
            for position, entry in enumerate(job["sequences"])
            if "proteinChain" in entry
            and not (entry["proteinChain"].get("unpairedMsaPath") or entry["proteinChain"].get("pairedMsaPath"))
        ]
        if missing:
            logger.warning(
                "opendde: protein chain(s) at position %s have no MSA path; OpenDDE will search its own server",
                ", ".join(missing),
            )


class OpenDDEModel:
    """OpenDDE co-folding model held in-process across calls.

    Builds one upstream ``InferenceRunner`` (model + checkpoint) and reuses it; it is rebuilt
    only when a weight- or sampling-affecting field changes (checkpoint, cycles, steps,
    samples, MSA/template flags, PAE output). Per-call inputs (input JSON, output dir, seed)
    are set on the held runner, as the ``opendde pred`` CLI does for each input file.
    """

    def __init__(self) -> None:
        """Initialize with an unresolved OpenDDE root directory and no runner."""
        self._loaded = False
        self.root_dir: str | None = None
        self.runner: Any = None
        self.device: str | None = None
        self._cache_key: tuple[Any, ...] | None = None

    def load(self) -> None:
        """Resolve ``OPENDDE_ROOT_DIR`` (checkpoints + common assets) and verify it holds weights.

        An explicit ``OPENDDE_ROOT_DIR`` env var wins, otherwise the managed weights
        cache is resolved via ``resolve_weights_dir`` (PROTO_OPENDDE_WEIGHTS_DIR /
        PROTO_MODEL_CACHE / PROTO_HOME).

        Raises:
            FileNotFoundError: If no root can be resolved or no checkpoint is present.
        """
        from standalone_helpers import resolve_weights_dir

        root = os.environ.get("OPENDDE_ROOT_DIR")
        if not root:
            root = resolve_weights_dir("opendde")
        if not root:
            raise FileNotFoundError(
                "opendde: unable to resolve OPENDDE_ROOT_DIR; set OPENDDE_ROOT_DIR or "
                "PROTO_OPENDDE_WEIGHTS_DIR, or configure PROTO_MODEL_CACHE"
            )

        ckpt = os.path.join(root, "checkpoint", "opendde.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"opendde: checkpoint not found at {ckpt}; re-run setup.sh to download weights, "
                "or point OPENDDE_ROOT_DIR / PROTO_OPENDDE_WEIGHTS_DIR at a directory containing checkpoint/opendde.pt"
            )

        self.root_dir = root
        # OpenDDE reads checkpoints and common assets from OPENDDE_ROOT_DIR in-process.
        os.environ["OPENDDE_ROOT_DIR"] = root
        self._loaded = True

    def _resolve_checkpoint(self, model_checkpoint: str) -> str:
        """Bundled name -> file under root (``""`` = OpenDDE's default weights); anything else is an explicit path."""
        assert self.root_dir is not None  # set by load()
        if model_checkpoint in _BUNDLED_CHECKPOINTS:
            rel = _BUNDLED_CHECKPOINTS[model_checkpoint]
            resolved = os.path.join(self.root_dir, rel) if rel else ""
        else:
            resolved = model_checkpoint
        if resolved and not os.path.exists(resolved):
            known = ", ".join(sorted(_BUNDLED_CHECKPOINTS))
            raise FileNotFoundError(
                f"opendde: checkpoint not found at {resolved!r} (model_checkpoint={model_checkpoint!r}); "
                f"use a bundled model name ({known}) and re-run setup.sh to fetch weights, "
                "or pass a path to an existing .pt checkpoint"
            )
        return resolved

    def _runner_kwargs(
        self,
        model_checkpoint: str,
        num_samples: int,
        num_steps: int,
        num_cycles: int,
        use_msa: bool,
        use_template: bool,
        use_rna_msa: bool,
        device: str,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """``get_default_runner`` arguments matching the ``opendde pred`` CLI defaults."""
        return {
            "model_name": "opendde_v1",
            "load_checkpoint_path": self._resolve_checkpoint(model_checkpoint),
            "n_sample": num_samples,
            "n_step": num_steps,
            "n_cycle": num_cycles,
            "use_msa": use_msa,
            "use_template": use_template,
            "use_rna_msa": use_rna_msa,
            # Emit the full-data PAE file only when requested (it is O(n_token^2) to write).
            "need_atom_confidence": include_pae_matrix,
            "device": "cpu" if device == "cpu" else "cuda",
        }

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        job_name: str,
        model_checkpoint: str,
        num_samples: int,
        num_steps: int,
        num_cycles: int,
        use_msa: bool,
        use_template: bool,
        use_rna_msa: bool,
        seed: int,
        device: str,
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run OpenDDE prediction and return the best sample's structure + metrics.

        Args:
            input_json_path (str): Path to the OpenDDE input JSON (MSAs already populated).
            output_dir (str): Directory for OpenDDE output files.
            job_name (str): Folding job name (drives OpenDDE's output subdirectory).
            model_checkpoint (str): Bundled model name (``opendde_v1``/``opendde_abag``) or
                a path to a custom ``.pt`` checkpoint.
            num_samples (int): Independent diffusion samples.
            num_steps (int): Diffusion denoising steps.
            num_cycles (int): Recycling iterations.
            use_msa (bool): Featurize the supplied MSA files; every chain has one.
            use_template (bool): Run OpenDDE's template pipeline.
            use_rna_msa (bool): Run OpenDDE's RNA MSA pipeline.
            seed (int): Random seed.
            device (str): Device string (e.g. ``"cuda"``, ``"cpu"``).
            verbose (bool): Log the prediction parameters at INFO.
            include_pae_matrix (bool): Write the full-data file and return the per-token
                PAE matrix (``pae``) plus a derived ``avg_pae`` scalar.

        Returns:
            dict[str, Any]: ``{"structure_cif_output": <cif str>, "metrics": {...}}``.

        Raises:
            OpenDDEExecutionError: If OpenDDE reports a prediction error.
            GpuOutOfMemoryError: If prediction runs out of GPU memory.
        """
        if not self._loaded:
            self.load()
        # After load(): OpenDDE reads OPENDDE_ROOT_DIR into module constants (checkpoint dir, data
        # root) when first imported, so importing it earlier would bind its own fallback root and
        # download the weights there instead of using the managed cache.
        from runner.batch_inference import get_default_runner, preprocess_input  # type: ignore[import-not-found]
        from runner.inference import infer_predict  # type: ignore[import-not-found]
        from standalone_helpers import oom_guard

        kwargs = self._runner_kwargs(
            model_checkpoint=model_checkpoint,
            num_samples=num_samples,
            num_steps=num_steps,
            num_cycles=num_cycles,
            use_msa=use_msa,
            use_template=use_template,
            use_rna_msa=use_rna_msa,
            device=device,
            include_pae_matrix=include_pae_matrix,
        )
        key = tuple(sorted(kwargs.items()))
        os.makedirs(output_dir, exist_ok=True)
        (logger.info if verbose else logger.debug)(
            f"OpenDDE prediction: {kwargs}, seed={seed}, input={input_json_path}"
        )
        if self.runner is None or self._cache_key != key:
            self.close()
            # InferenceRunner.__init__ loads the model and checkpoint once; later calls reuse it.
            with oom_guard("opendde", hint="Reduce num_samples/num_steps/num_cycles or use a larger GPU."):
                self.runner = get_default_runner(seeds=[seed], dump_dir=output_dir, **kwargs)
            self._cache_key = key
            self.device = str(self.runner.device)

        # Re-point the held runner at this call's output dir; the dumper's base_dir is fixed at construction.
        self.runner.dump_dir = output_dir
        self.runner.error_dir = os.path.join(output_dir, "ERR")
        self.runner.dumper.base_dir = output_dir

        if use_msa:
            _warn_on_chains_without_msa(input_json_path)
        configs = self.runner.configs
        configs["input_json_path"] = preprocess_input(
            input_json_path, out_dir=output_dir, use_msa=use_msa, use_template=use_template, use_rna_msa=use_rna_msa
        )
        configs["seeds"] = [seed]
        configs["dump_dir"] = output_dir
        with oom_guard(
            "opendde", hint="Reduce num_samples/num_steps/num_cycles, shorten the complex, or use a larger GPU."
        ):
            infer_predict(self.runner, configs)
        if os.path.isdir(os.path.join(output_dir, job_name, "msa")):
            logger.warning("opendde: OpenDDE ran its own MSA server search, replacing the MSAs proto-tools supplied")

        err_dir = os.path.join(output_dir, "ERR")
        if os.path.isdir(err_dir) and os.listdir(err_dir):
            messages = []
            for name in sorted(os.listdir(err_dir)):
                with open(os.path.join(err_dir, name)) as f:
                    messages.append(f"{name}: {f.read()}")
            raise OpenDDEExecutionError("opendde: prediction failed: " + " | ".join(messages))
        return _extract_structure_and_scores(output_dir, include_pae_matrix=include_pae_matrix)

    def to_device(self, device: str) -> None:
        """Move the held runner's model between devices."""
        import torch
        from standalone_helpers import move_model_to_device

        if self.runner is None or self.device == device:
            return
        self.runner.model = move_model_to_device(self.runner.model, self.device, device)
        self.runner.device = torch.device(device)
        self.device = device

    def close(self) -> None:
        """Release the held runner (its model and GPU memory)."""
        if self.runner is not None:
            self.runner.close()
        self.runner, self._cache_key, self.device = None, None, None


# ============================================================================
# Worker Protocol Entry Points
# ============================================================================

_model: OpenDDEModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution.

    Args:
        input_dict (dict[str, Any]): Input parameters from ``ToolInstance.dispatch()``.

    Returns:
        dict[str, Any]: Dict with ``structure_cif_output`` and ``metrics``.

    Raises:
        ValueError: If the requested operation is unknown.
    """
    global _model
    if _model is None:
        _model = OpenDDEModel()

    operation = input_dict["operation"]
    if operation == "predict":
        return _model(
            input_json_path=input_dict["input_json_path"],
            output_dir=input_dict["output_dir"],
            job_name=input_dict["job_name"],
            model_checkpoint=input_dict["model_checkpoint"],
            num_samples=input_dict["num_samples"],
            num_steps=input_dict["num_steps"],
            num_cycles=input_dict["num_cycles"],
            use_msa=input_dict["use_msa"],
            use_template=input_dict["use_template"],
            use_rna_msa=input_dict["use_rna_msa"],
            seed=input_dict["seed"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict.get("include_pae_matrix", False),
        )
    raise ValueError(f"opendde: unknown operation {operation!r}; valid: ['predict']")


def to_device(device: str) -> dict[str, Any]:
    """DeviceManager callback for moving the held model to a different device.

    Args:
        device (str): Target device (e.g., "cpu", "cuda:0").

    Returns:
        dict[str, Any]: Success status dict.
    """
    if _model is not None:
        _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory held by the persistent model."""
    import torch

    if _model is None or _model.runner is None or not torch.cuda.is_available():
        return {"available": False, "framework": "torch", "reason": "no model loaded on a GPU"}
    return {
        "available": True,
        "framework": "torch",
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


# Worker protocol entry point
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("opendde: usage: python inference.py <input_json_path> <output_json_path>")
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    result = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
