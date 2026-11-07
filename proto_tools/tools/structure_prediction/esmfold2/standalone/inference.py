"""ESMFold2 worker: folds all-atom complexes and differentiates structure objectives."""

import json
import logging
import sys
from typing import Any

import objective  # type: ignore[import-not-found]  # sibling module, resolved inside the tool env
import torch
from standalone_helpers import (
    get_logger,
    get_random_int,
    is_cuda_oom,
    release_cuda_memory,
    report_msa_rows_loaded,
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
            unpaired_rows = unpaired_msas.get(str(idx)) if unpaired_msas else None
            # A chain with only unpaired rows is sent them as its base alignment, which never pairs.
            pairs = msas_paired and bool(chain_msa_rows)
            if not chain_msa_rows:
                chain_msa_rows, unpaired_rows = unpaired_rows, None
            if chain_msa_rows:
                from esm.utils.parsing import FastaEntry

                # `key=<row_idx>` headers engage upstream paired_msa.py cross-chain pairing.
                header_fmt = "key={row}" if pairs else "seq_{row}"
                entries = [
                    FastaEntry(header=header_fmt.format(row=row_idx), sequence=seq)
                    for row_idx, seq in enumerate(chain_msa_rows)
                ]
                # Append the deep unpaired rows with non-`key=` headers: upstream reads
                # taxonomy -1 for these and stacks them as unpaired depth after the
                # paired rows, never key-paired (skip the query and any paired duplicate).
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
        from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

        repo = _CHECKPOINT_REPOS[self.model_checkpoint]
        if verbose:
            logger.info(f"Loading ESMFold2 model: {repo} on {device}")

        try:
            self.model = EsmFold2Model.from_pretrained(repo, device=device).eval()
        except OSError as e:
            raise RuntimeError(f"esmfold2: HF weight load from {repo!r} failed: {e}") from e

        self.builder = ESMFold2InputBuilder()
        self._log_loaded_msa_rows(self.builder)
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ESMFold2 model loaded successfully")

    def _log_loaded_msa_rows(self, builder: Any) -> None:
        """Wrap ``builder.prepare_input`` to log each protein chain's featurized MSA rows."""
        from esm.models.esmfold2.constants import MSA_GAP_TOKEN_ID

        prepare_input = builder.prepare_input

        def prepare_input_logging_msa_rows(*args: Any, **kwargs: Any) -> Any:
            features, chain_infos = prepare_input(*args, **kwargs)
            sent = getattr(self, "_msa_rows_sent", {})
            if any(rows for rows in sent.values()):
                msa, asym = features["msa"][0], features["asym_id"][0]
                loaded = {
                    str(info.chain_id): int((msa[:, asym == info.asym_id] != MSA_GAP_TOKEN_ID).any(dim=1).sum())
                    for info in chain_infos
                    if str(info.chain_id) in sent
                }
                report_msa_rows_loaded("esmfold2", loaded, sent)
            return features, chain_infos

        builder.prepare_input = prepare_input_logging_msa_rows

    def ensure_loaded(self, device: str, verbose: bool = False, seed: int | None = None) -> Any:
        """Load the checkpoint if needed, move it if the device changed, and return the model.

        A given seed scopes the load, which can initialize runtime-only modules the checkpoint
        does not carry; leaving that unseeded makes a run irreproducible for reasons that never
        show up in the forward pass. Both the folding and the gradient path come through here,
        so one set of weights serves both.

        Args:
            device (str): Device the model must end up on.
            verbose (bool): Log the load.
            seed (int | None): Seed scoping the load; ``None`` loads unseeded.

        Returns:
            Any: The loaded ``EsmFold2Model``.
        """
        if not self._loaded:
            if seed is None:
                self.load(device, verbose)
            else:
                from esm.models.esmfold2.layers import _seed_context as seed_context

                with seed_context(seed):
                    self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)
        return self.model

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
        msa_max_depth: int,
        include_pae_matrix: bool,
        device: str,
        verbose: bool,
        seed: int | None,
    ) -> dict[str, Any]:
        """Fold one complex and return CIF + metrics."""
        self.ensure_loaded(device, verbose, seed)

        # Deterministic seed forwarded to the upstream sampler.
        effective_seed = seed if seed is not None else get_random_int()
        set_torch_seed(effective_seed)

        upstream_input = _build_upstream_input(chains, msas, msas_paired, unpaired_msas)
        self._msa_rows_sent = {
            str(chain["id"]): max(len((msas or {}).get(str(idx), [])), len((unpaired_msas or {}).get(str(idx), [])))
            for idx, chain in enumerate(chains)
            if chain.get("entity_type", "protein") == "protein"
        }

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
                        msa_max_depth=msa_max_depth,
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
# Inversion gradients
# ============================================================================
#: Folding settings the reference recipe uses for a gradient step rather than a scoring pass.
_NUM_LOOPS = 1
_NUM_SAMPLING_STEPS = 1
#: A chain occupies the amino-acid columns of a 33-token vocabulary, so a 20-wide relaxed state
#: is padded on both sides to line up.
_TOKEN_PAD = (2, 11)
#: Residue-type indices 0 and 1 are padding and a gap, so the amino acids start at 2.
_FIRST_AMINO_ACID_INDEX = 2


def _folding_vocabulary() -> str:
    """One-letter amino acids in the folding trunk's column order.

    Returns:
        str: Twenty one-letter amino acids in trunk column order.
    """
    from esm.models.esmfold2.constants import PROTEIN_3TO1, RES_TYPE_TO_CCD

    start = _FIRST_AMINO_ACID_INDEX
    return "".join(PROTEIN_3TO1[RES_TYPE_TO_CCD[index]] for index in range(start, start + 20))


def _move_language_model(device: str) -> None:
    """Move the cached ESM C to ``device`` if it is loaded somewhere else."""
    global _language_model, _language_model_device
    from standalone_helpers import move_model_to_device

    if _language_model is not None and _language_model_device != device:
        _language_model = move_model_to_device(_language_model, _language_model_device, device)
        _language_model_device = device


def _load_language_model(device: str, seed: int) -> Any:
    """Load ESM C for the pseudo-perplexity term, caching it for the life of the worker."""
    global _language_model, _language_model_device
    from esm.models.esmc import EsmcForMaskedLM
    from esm.models.esmfold2.layers import _seed_context as seed_context

    if _language_model is None:
        logger.info("Loading ESM C for the pseudo-perplexity term")
        with seed_context(seed):
            _language_model = EsmcForMaskedLM.from_pretrained("biohub/esmc-300m").to(device)
        _language_model_device = device
    else:
        _move_language_model(device)
    return _language_model


def _trunk_to_lm_matrix(device: torch.device) -> torch.Tensor:
    """Permutation mapping folding-trunk amino-acid columns onto ESM C's."""
    from esm.models.esmc import EsmcTokenizer

    lm_vocabulary = sorted(EsmcTokenizer().vocab.items(), key=lambda item: item[1])
    lm_amino_acids = [token for token, _ in lm_vocabulary[4:24]]
    matrix = torch.zeros(20, 20, device=device)
    for trunk_index, amino_acid in enumerate(_folding_vocabulary()):
        matrix[trunk_index, lm_amino_acids.index(amino_acid)] = 1
    return matrix


def _relax(source: torch.Tensor, order: list[int], data: dict[str, Any]) -> torch.Tensor:
    """Turn one chain's caller-ordered logits into its relaxed state in trunk column order."""
    reordered = source[:, order]
    temperature = data["temperature"]
    relaxed = reordered if temperature is None else torch.softmax(reordered / temperature, dim=-1)
    soft, hard = data["soft"], data["hard"]
    if soft != 1.0:
        relaxed = soft * relaxed + (1 - soft) * reordered
    if hard != 0.0:
        relaxed = hard * objective.straight_through(relaxed) + (1 - hard) * relaxed
    return relaxed


class _Complex:
    """One prepared complex: its chain spans, its relaxed state, and what to differentiate.

    Attributes:
        spans (dict[str, tuple[int, int]]): Chain id to its half-open rows in the complex.
        sources (list[torch.Tensor]): Leaf tensors, one per designable chain, in fold order.
        designable (list[str]): Designable chain ids, in fold order.
        sequences (dict[str, str]): Decoded sequence per chain, as folded.
        res_type_soft (torch.Tensor): The complex's relaxed residue types, shape ``(1, L, 33)``.
        relaxed (dict[str, torch.Tensor]): Per-chain relaxed state, shape ``(1, len, 20)``.
    """

    def __init__(self, data: dict[str, Any], vocabulary: str) -> None:
        """Build the complex from the worker payload.

        Args:
            data (dict[str, Any]): Worker payload carrying ``chains`` and relaxation settings.
            vocabulary (str): Folding-trunk column order.
        """
        import torch.nn.functional as F

        device = data["device"]
        order = [data["alphabet"].index(residue) for residue in vocabulary]
        self.spans: dict[str, tuple[int, int]] = {}
        self.sources: list[torch.Tensor] = []
        self.designable: list[str] = []
        self.sequences: dict[str, str] = {}
        self.relaxed: dict[str, torch.Tensor] = {}
        blocks: list[torch.Tensor] = []
        cursor = 0

        for chain in data["chains"]:
            chain_id = chain["chain_id"]
            if chain["logits"] is not None:
                source = torch.tensor(chain["logits"], dtype=torch.float32, device=device, requires_grad=True)
                relaxed = _relax(source, order, data).unsqueeze(0)
                self.sources.append(source)
                self.designable.append(chain_id)
                self.relaxed[chain_id] = relaxed
                self.sequences[chain_id] = "".join(vocabulary[index] for index in relaxed[0].argmax(-1).tolist())
                length = relaxed.shape[1]
            else:
                sequence = chain["sequence"]
                relaxed = torch.zeros(1, len(sequence), 20, device=device)
                for position, residue in enumerate(sequence):
                    relaxed[0, position, vocabulary.index(residue)] = 1
                self.relaxed[chain_id] = relaxed
                self.sequences[chain_id] = sequence
                length = len(sequence)
            blocks.append(F.pad(relaxed, _TOKEN_PAD, mode="constant", value=0))
            self.spans[chain_id] = (cursor, cursor + length)
            cursor += length

        self.res_type_soft = torch.cat(blocks, dim=1)
        self.length = cursor

    def chain_rows(self) -> dict[str, list[int]]:
        """Where each designable chain's rows sit in the concatenated gradient."""
        rows: dict[str, list[int]] = {}
        cursor = 0
        for chain_id in self.designable:
            length = self.relaxed[chain_id].shape[1]
            rows[chain_id] = [cursor, cursor + length]
            cursor += length
        return rows


def _fold(model: Any, prepared: _Complex, data: dict[str, Any]) -> torch.Tensor:
    """Fold the complex and return its distogram logits.

    Args:
        model (Any): The loaded folding model.
        prepared (_Complex): The assembled complex.
        data (dict[str, Any]): Worker payload, read for device and seed.

    Returns:
        torch.Tensor: Distogram logits, shape ``(1, length, length, bins)``.
    """
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput, load_ccd, prepare_esmfold2_input
    from esm.models.esmfold2.layers import _seed_context as seed_context

    device, seed = data["device"], data["seed"]
    load_ccd()
    sequences = [
        ProteinInput(id=str(index), sequence=prepared.sequences[chain["chain_id"]], msa=None)
        for index, chain in enumerate(data["chains"])
    ]
    features, _ = prepare_esmfold2_input(StructurePredictionInput(sequences=sequences), seed=seed)
    # Distogram conditioning is unimplemented upstream and the forward pass rejects it.
    features.pop("disto_cond", None)
    features.pop("disto_cond_mask", None)
    inputs = {key: value.unsqueeze(0).to(device) for key, value in features.items()}
    # The relaxed state enters here, which is what keeps every designable chain differentiable.
    inputs["res_type_soft"] = prepared.res_type_soft

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=_NUM_SAMPLING_STEPS,
            num_loops=_NUM_LOOPS,
            seed=seed,
        )
    return output["distogram_logits"]


def _epitope_mask(data: dict[str, Any], prepared: _Complex) -> torch.Tensor:
    """Boolean mask over complex positions, true at the requested target hotspots."""
    target_id = data["target_chain_id"]
    start, _ = prepared.spans[target_id]
    mask = torch.zeros(1, prepared.length, dtype=torch.bool, device=prepared.res_type_soft.device)
    for hotspot in data.get("target_hotspot_ids") or []:
        mask[0, start + int(hotspot[1:]) - 1] = True
    return mask


def _quantity(
    name: str,
    chain_id: str,
    distogram: torch.Tensor,
    midpoints: torch.Tensor,
    prepared: _Complex,
    data: dict[str, Any],
) -> torch.Tensor:
    """Evaluate one requested quantity over the chain it names.

    Args:
        name (str): Quantity id.
        chain_id (str): Chain the quantity is evaluated over.
        distogram (torch.Tensor): Distogram logits for the folded complex.
        midpoints (torch.Tensor): Bin midpoint distances.
        prepared (_Complex): The assembled complex, for its spans and relaxed state.
        data (dict[str, Any]): Worker payload.

    Returns:
        torch.Tensor: A scalar for this candidate.

    Raises:
        ValueError: If the quantity is unknown, or an interface quantity is asked for without a
            target chain.
    """
    span = prepared.spans[chain_id]
    if name == "intra_contact":
        return objective.intra_contact_loss(distogram, span, midpoints).squeeze(0)
    if name == "globularity":
        return objective.globularity_loss(distogram, span, midpoints).squeeze(0)
    if name == "plm_nll":
        return _pseudo_perplexity(prepared.relaxed[chain_id], data).squeeze(0)
    if name in ("inter_contact", "epitope"):
        target_id = data.get("target_chain_id")
        if target_id is None:
            raise ValueError(f"quantity {name!r} needs a target chain; it is offered only by the binder tool")
        if name == "inter_contact":
            return objective.inter_contact_loss(distogram, prepared.spans[target_id], span, midpoints).squeeze(0)
        return objective.epitope_loss(
            distogram, _epitope_mask(data, prepared), span, midpoints, data["epitope_contact_distance"]
        ).squeeze(0)
    raise ValueError(f"esmfold2: unknown quantity {name!r}")


def _pseudo_perplexity(relaxed: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    """Score one chain under ESM C, loading the language model only when it is asked for."""
    from esm.models.esmc import EsmcTokenizer

    device, seed = data["device"], data["seed"]
    masked_lm = _load_language_model(device, seed)
    tokenizer = EsmcTokenizer()
    scored = torch.ones(relaxed.shape[0], relaxed.shape[1], dtype=torch.bool, device=relaxed.device)
    generator = torch.Generator(device=relaxed.device).manual_seed(seed)
    return objective.pseudo_perplexity_nll(
        masked_lm,
        relaxed,
        scored,
        trunk_to_lm=_trunk_to_lm_matrix(relaxed.device),
        cls_id=tokenizer.cls_token_id,
        eos_id=tokenizer.eos_token_id,
        mask_id=masked_lm.config.mask_token_id,
        vocab_size=masked_lm.config.vocab_size,
        amino_acid_offset=4,
        passes=data["plm_passes"],
        batch_size=data["plm_batch_size"],
        generator=generator,
    )


def complex_gradient(data: dict[str, Any]) -> dict[str, Any]:
    """Fold one complex and differentiate each requested quantity against its designable chains.

    Args:
        data (dict[str, Any]): Worker payload.

    Returns:
        dict[str, Any]: ``terms``, ``chain_rows`` and ``metrics``.
    """
    vocabulary = _folding_vocabulary()
    prepared = _Complex(data, vocabulary)
    folder = _wrapper(data["model_checkpoint"]).ensure_loaded(data["device"], seed=data["seed"])
    distogram = _fold(folder, prepared, data)
    midpoints = objective.bin_midpoints(distogram.shape[-1], distogram.device).to(distogram.dtype)

    requested = list(data["gradient_terms"])
    values = [
        (entry, _quantity(entry["id"], entry["chain_id"], distogram, midpoints, prepared, data)) for entry in requested
    ]

    terms = []
    for position, (entry, value) in enumerate(values):
        gradient = None
        if data["compute_gradient"]:
            # One backward per quantity over the shared forward graph, retained until the last.
            # Chains a quantity does not touch get a zero block rather than being dropped, so
            # every gradient covers the same rows and the terms stay composable.
            parts = torch.autograd.grad(
                value,
                prepared.sources,
                retain_graph=position < len(values) - 1,
                allow_unused=True,
                materialize_grads=True,
            )
            gradient = serialize_output(torch.cat(parts, dim=0))
        terms.append(
            {"id": entry["id"], "chain_id": entry["chain_id"], "value": float(value.detach()), "gradient": gradient}
        )

    return {
        "terms": terms,
        "chain_rows": prepared.chain_rows(),
        "metrics": {
            "complex_length": prepared.length,
            "designable_length": sum(prepared.relaxed[name].shape[1] for name in prepared.designable),
        },
    }


# ============================================================================
# Dispatch
# ============================================================================
#: The folding checkpoint, shared by the prediction and the gradient operations so one set of
#: weights serves both. reload_on_change on model_checkpoint restarts the worker when the
#: caller asks for the other variant, but the swap here is defensive.
_model: ESMFold2Model | None = None
#: Loaded only when the pseudo-perplexity term is requested, with the device it sits on.
_language_model: Any = None
_language_model_device: str | None = None


def _wrapper(model_checkpoint: str) -> ESMFold2Model:
    """Return the cached wrapper for model_checkpoint, replacing it if the caller switched."""
    global _model
    if _model is None or _model.model_checkpoint != model_checkpoint:
        _model = ESMFold2Model(model_checkpoint=model_checkpoint)
    return _model


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution.

    Args:
        input_dict (dict[str, Any]): Worker payload carrying an operation key.

    Returns:
        dict[str, Any]: The operation's reply.

    Raises:
        ValueError: If the operation is not one this worker offers.
    """
    operation = input_dict["operation"]
    if operation == "predict":
        return _wrapper(input_dict["model_checkpoint"])(
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
            msa_max_depth=input_dict["msa_max_depth"],
            include_pae_matrix=input_dict["include_pae_matrix"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            seed=input_dict.get("seed"),
        )
    if operation == "complex_gradient":
        try:
            return complex_gradient(input_dict)
        except RuntimeError as error:
            if is_cuda_oom(error):
                release_cuda_memory()
            raise
    raise ValueError(f"esmfold2: unknown operation {operation!r}; valid: ['predict', 'complex_gradient']")


def to_device(device: str) -> dict[str, Any]:
    """Move every cached model to the specified device (called by DeviceManager)."""
    moved = []
    if _model is not None and _model._loaded:
        _model.to_device(device)
        moved.append("esmfold2")
    if _language_model is not None:
        _move_language_model(device)
        moved.append("esmc")
    if not moved:
        return {"success": True, "device": device, "note": "no model loaded yet"}
    return {"success": True, "device": device, "moved": moved}


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
        # Non-finite values fail here rather than becoming NaN in the caller's JSON.
        json.dump(serialize_output(result), f, allow_nan=False)
