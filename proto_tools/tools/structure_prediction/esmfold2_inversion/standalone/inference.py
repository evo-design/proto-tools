"""ESMFold2 inversion worker: folds a protein complex and differentiates per-chain objectives."""

import json
import logging
import sys
from typing import Any

import objective  # type: ignore[import-not-found]  # sibling module, resolved inside the tool env
import torch
from standalone_helpers import get_logger, is_cuda_oom, release_cuda_memory, serialize_output

logger = get_logger(__name__)

logging.getLogger("esm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

#: Checkpoints are cached per worker process, keyed by name.
_folders: dict[str, Any] = {}
_language_model: Any = None

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


def _load_folder(model_name: str, device: str, seed: int) -> Any:
    """Load an inversion checkpoint, caching it for the life of the worker.

    Loading happens inside the seed context because it can initialize runtime-only modules the
    checkpoint does not carry; leaving that unseeded makes a run irreproducible for reasons that
    never show up in the forward pass.

    Args:
        model_name (str): Inversion checkpoint name.
        device (str): Device to place the model on.
        seed (int): Seed scoping the load.

    Returns:
        Any: The loaded experimental folding model.
    """
    from esm.models.esmfold2.layers import _seed_context as seed_context

    if model_name not in _folders:
        from esm.models.esmfold2.experimental import EsmFold2ExperimentalModel

        logger.info("Loading ESMFold2 inversion checkpoint %s", model_name)
        with seed_context(seed):
            _folders[model_name] = EsmFold2ExperimentalModel.from_pretrained(f"biohub/{model_name}").to(device)
    return _folders[model_name]


def _load_language_model(device: str, seed: int) -> Any:
    """Load ESM C for the pseudo-perplexity term, caching it for the life of the worker."""
    global _language_model
    from esm.models.esmc import EsmcForMaskedLM
    from esm.models.esmfold2.layers import _seed_context as seed_context

    if _language_model is None:
        logger.info("Loading ESM C for the pseudo-perplexity term")
        with seed_context(seed):
            _language_model = EsmcForMaskedLM.from_pretrained("biohub/esmc-300m").to(device)
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
        model (Any): Loaded experimental folding model.
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
    raise ValueError(f"esmfold2_inversion: unknown quantity {name!r}")


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
    folder = _load_folder(data["model_name"], data["device"], data["seed"])
    distogram = _fold(folder, prepared, data)
    midpoints = objective.bin_midpoints(distogram.device).to(distogram.dtype)

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


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run one ESMFold2 inversion operation.

    Args:
        input_dict (dict[str, Any]): Worker payload carrying an ``operation`` key.

    Returns:
        dict[str, Any]: The operation's reply.

    Raises:
        ValueError: If the operation is not one this worker offers.
    """
    operation = input_dict.get("operation")
    if operation != "complex_gradient":
        raise ValueError(f"esmfold2_inversion: unknown operation {operation!r}; valid: ['complex_gradient']")
    try:
        return complex_gradient(input_dict)
    except RuntimeError as error:
        if is_cuda_oom(error):
            release_cuda_memory()
        raise


def get_memory_stats() -> dict[str, Any]:
    """Report the worker's GPU memory use."""
    from standalone_helpers import get_pytorch_memory_stats

    return dict(get_pytorch_memory_stats("cuda"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esmfold2_inversion: usage: python inference.py <input_json_path> <output_json_path>")
    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    result = dispatch(payload)
    with open(sys.argv[2], "w") as handle:
        # Non-finite values fail here rather than becoming NaN in the caller's JSON.
        json.dump(serialize_output(result), handle, allow_nan=False)
