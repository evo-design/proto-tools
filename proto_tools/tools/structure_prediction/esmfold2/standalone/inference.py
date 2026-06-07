"""Local ESMFold2 (Biohub) all-atom complex structure prediction model wrapper."""

import json
import logging
import sys
from typing import Any

import torch
from standalone_helpers import (
    get_logger,
    get_random_int,
    is_cuda_oom,
    release_cuda_memory,
    serialize_output,
    set_torch_seed,
)

logger = get_logger(__name__)

# Suppress noisy library logs
logging.getLogger("esm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# HF repos for the two exposed checkpoints
_CHECKPOINT_REPOS = {
    "esmfold2": "biohub/ESMFold2",
    "esmfold2-fast": "biohub/ESMFold2-Fast",
}


def _build_upstream_input(
    chains: list[dict[str, Any]],
    msas: dict[str, list[str]] | None,
    msas_paired: bool,
    unpaired_msas: dict[str, list[str]] | None = None,
) -> Any:
    """Convert the JSON chains payload into an upstream ``StructurePredictionInput``."""
    from esm.models.esmfold2 import (
        MSA,
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    sequences: list[Any] = []
    for idx, chain in enumerate(chains):
        chain_id = str(chain["id"])
        entity_type = chain["entity_type"]

        if entity_type == "ligand":
            # Prefer CCD code when supplied; fall back to SMILES.
            ccd_code = chain.get("ccd_code")
            smiles = chain.get("smiles")
            if ccd_code:
                sequences.append(LigandInput(id=chain_id, ccd=[ccd_code]))
            elif smiles:
                sequences.append(LigandInput(id=chain_id, smiles=smiles))
            else:
                raise ValueError(f"esmfold2: ligand chain at index {idx} has neither ccd_code nor smiles")
            continue

        sequence = chain["sequence"]
        # proto Modification positions are 1-indexed; upstream expects 0-indexed.
        raw_mods = chain.get("modifications") or []
        mods: list[Any] | None = (
            [Modification(position=m["position"] - 1, ccd=m["ccd_code"]) for m in raw_mods] if raw_mods else None
        )

        if entity_type == "protein":
            msa_obj = None
            chain_msa_rows = msas.get(str(idx)) if msas else None
            if chain_msa_rows:
                from esm.utils.parsing import FastaEntry

                # `key=<row_idx>` headers engage upstream paired_msa.py cross-chain pairing.
                header_fmt = "key={row}" if msas_paired else "seq_{row}"
                entries = [
                    FastaEntry(header=header_fmt.format(row=row_idx), sequence=seq)
                    for row_idx, seq in enumerate(chain_msa_rows)
                ]
                # Append the deep unpaired rows with non-`key=` headers: upstream reads
                # taxonomy -1 for these and places them block-diagonally as per-chain
                # depth after the paired rows (skip the query and any paired duplicate).
                unpaired_rows = unpaired_msas.get(str(idx)) if unpaired_msas else None
                if unpaired_rows:
                    seen = set(chain_msa_rows)
                    for u_seq in unpaired_rows[1:]:
                        if u_seq in seen:
                            continue
                        seen.add(u_seq)
                        entries.append(FastaEntry(header=f"unpaired_{len(entries)}", sequence=u_seq))
                msa_obj = MSA(entries=entries)
            sequences.append(ProteinInput(id=chain_id, sequence=sequence, modifications=mods, msa=msa_obj))
        elif entity_type == "dna":
            sequences.append(DNAInput(id=chain_id, sequence=sequence, modifications=mods))
        elif entity_type == "rna":
            sequences.append(RNAInput(id=chain_id, sequence=sequence, modifications=mods))
        else:
            raise ValueError(f"esmfold2: unsupported entity_type {entity_type!r} at index {idx}")

    return StructurePredictionInput(sequences=sequences)


class ESMFold2Model:
    """ESMFold2 all-atom structure prediction model wrapper."""

    def __init__(self, model_checkpoint: str) -> None:
        """Initialize the model wrapper; defers weight load until first call."""
        if model_checkpoint not in _CHECKPOINT_REPOS:
            raise ValueError(
                f"esmfold2: unknown model_checkpoint {model_checkpoint!r}; valid: {sorted(_CHECKPOINT_REPOS)}"
            )
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.device: str | None = None
        self.model: Any = None
        self.builder: Any = None

    def load(self, device: str, verbose: bool = False) -> None:
        """Load the ESMFold2 model and input builder onto ``device``."""
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from transformers.models.esmfold2.modeling_esmfold2 import (
            ESMFold2Model as _UpstreamESMFold2Model,
        )

        repo = _CHECKPOINT_REPOS[self.model_checkpoint]
        if verbose:
            logger.info(f"Loading ESMFold2 model: {repo} on {device}")

        try:
            self.model = _UpstreamESMFold2Model.from_pretrained(repo).to(device).eval()
        except OSError as e:
            raise RuntimeError(f"esmfold2: HF weight load from {repo!r} failed: {e}") from e

        self.builder = ESMFold2InputBuilder()
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ESMFold2 model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move the loaded model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("esmfold2: cannot move unloaded model to device; call load() first")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def __call__(
        self,
        chains: list[dict[str, Any]],
        msas: dict[str, list[str]] | None,
        msas_paired: bool,
        unpaired_msas: dict[str, list[str]] | None,
        num_loops: int,
        num_sampling_steps: int,
        diffusion_samples: int,
        step_scale: float | None,
        noise_scale: float | None,
        max_inference_sigma: float | None,
        early_exit: bool,
        include_pae_matrix: bool,
        device: str,
        verbose: bool,
        seed: int | None,
    ) -> dict[str, Any]:
        """Fold one complex and return CIF + metrics."""
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Deterministic seed forwarded to the upstream sampler.
        effective_seed = seed if seed is not None else get_random_int()
        set_torch_seed(effective_seed)

        upstream_input = _build_upstream_input(chains, msas, msas_paired, unpaired_msas)

        # Optional sampler overrides: upstream consumes them only when non-None.
        sampler_kwargs: dict[str, Any] = {}
        if step_scale is not None:
            sampler_kwargs["step_scale"] = step_scale
        if noise_scale is not None:
            sampler_kwargs["noise_scale"] = noise_scale
        if max_inference_sigma is not None:
            sampler_kwargs["max_inference_sigma"] = max_inference_sigma

        # Auto-halve num_sampling_steps on CUDA OOM (floor = 1).
        steps = num_sampling_steps
        try:
            while True:
                try:
                    raw = self.builder.fold(
                        self.model,
                        upstream_input,
                        num_loops=num_loops,
                        num_sampling_steps=steps,
                        num_diffusion_samples=diffusion_samples,
                        seed=effective_seed,
                        early_exit=early_exit,
                        **sampler_kwargs,
                    )
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if not is_cuda_oom(exc) or steps <= 1:
                        raise
                    release_cuda_memory()
                    new_steps = max(1, steps // 2)
                    logger.warning(f"esmfold2: CUDA OOM at num_sampling_steps={steps}; retrying with {new_steps}")
                    steps = new_steps
        finally:
            release_cuda_memory()

        # fold() returns a single result for diffusion_samples=1, else a list.
        if isinstance(raw, list):
            # Pick the highest mean-pLDDT sample to return.
            result = max(raw, key=lambda r: float(r.plddt.mean().item()) if r.plddt is not None else 0.0)
        else:
            result = raw

        # iptm is meaningless for single-chain complexes; gate it.
        is_multi_chain = len(chains) > 1

        cif_str = result.complex.to_mmcif()
        plddt_mean = float(result.plddt.mean().item()) if result.plddt is not None else 0.0
        ptm = float(result.ptm) if result.ptm is not None else 0.0
        iptm = float(result.iptm) if (result.iptm is not None and is_multi_chain) else None

        pae_tensor = result.pae
        if pae_tensor is not None:
            avg_pae = float(pae_tensor.mean().item())
            pae_matrix = pae_tensor.tolist() if include_pae_matrix else None
        else:
            avg_pae = 0.0
            pae_matrix = None

        return {
            "structure_cif_output": cif_str,
            "metrics": {
                "plddt": plddt_mean,
                "ptm": ptm,
                "iptm": iptm,
                "avg_pae": avg_pae,
                "pae": pae_matrix,
            },
        }


# ============================================================================
# Dispatch
# ============================================================================
_model: ESMFold2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    requested_checkpoint = input_dict["model_checkpoint"]
    # reload_on_change=True restarts the worker on checkpoint change, but reinit defensively.
    if _model is None or _model.model_checkpoint != requested_checkpoint:
        _model = ESMFold2Model(model_checkpoint=requested_checkpoint)

    operation = input_dict["operation"]
    if operation == "predict":
        return _model(
            chains=input_dict["chains"],
            msas=input_dict.get("msas"),
            msas_paired=input_dict.get("msas_paired", False),
            unpaired_msas=input_dict.get("unpaired_msas"),
            num_loops=input_dict["num_loops"],
            num_sampling_steps=input_dict["num_sampling_steps"],
            diffusion_samples=input_dict.get("diffusion_samples", 1),
            step_scale=input_dict.get("step_scale"),
            noise_scale=input_dict.get("noise_scale"),
            max_inference_sigma=input_dict.get("max_inference_sigma"),
            early_exit=input_dict.get("early_exit", False),
            include_pae_matrix=input_dict["include_pae_matrix"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            seed=input_dict.get("seed"),
        )
    raise ValueError(f"esmfold2: unknown operation {operation!r}; valid: ['predict']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Return PyTorch memory stats for the model's device (used by ToolPool)."""
    from standalone_helpers import get_pytorch_memory_stats

    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esmfold2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
