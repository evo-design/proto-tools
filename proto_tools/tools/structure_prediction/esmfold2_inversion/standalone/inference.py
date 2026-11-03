"""ESMFold2 inversion worker: folds a binder/target complex and differentiates its objectives."""

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

#: Folding settings the reference recipe uses for a gradient step, as opposed to a scoring pass.
_NUM_LOOPS = 1
_NUM_SAMPLING_STEPS = 1
#: The designed chain occupies the trailing columns of a 33-token vocabulary whose amino acids
#: start at index 2, so a 20-wide relaxed state is padded on both sides to line up.
_TOKEN_PAD = (2, 11)


#: Residue-type indices 0 and 1 are padding and a gap, so the amino acids start at 2.
_FIRST_AMINO_ACID_INDEX = 2


def _folding_vocabulary() -> str:
    """One-letter amino acids in the folding trunk's column order.

    The trunk's residue-type table is keyed by index and spelled in three-letter codes; the
    twenty amino acids occupy the entries after padding and the gap token.

    Returns:
        str: Twenty one-letter amino acids in trunk column order.
    """
    from esm.models.esmfold2.constants import PROTEIN_3TO1, RES_TYPE_TO_CCD

    start = _FIRST_AMINO_ACID_INDEX
    return "".join(PROTEIN_3TO1[RES_TYPE_TO_CCD[index]] for index in range(start, start + 20))


def _load_folder(model_name: str, device: str, seed: int) -> Any:
    """Load an inversion checkpoint, caching it for the life of the worker.

    Loading happens inside the seed context because it can initialize runtime-only modules that
    the checkpoint does not carry; leaving that unseeded makes a run irreproducible for reasons
    that never show up in the forward pass.

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


def _fold(model: Any, target_sequence: str, design: torch.Tensor, device: str, seed: int) -> torch.Tensor:
    """Fold the target/binder complex and return its distogram logits.

    Args:
        model (Any): Loaded experimental folding model.
        target_sequence (str): Fixed target chain.
        design (torch.Tensor): Relaxed binder, shape ``(1, length, 20)``, trunk column order.
        device (str): Device the complex is assembled on.
        seed (int): Seed scoping the forward pass.

    Returns:
        torch.Tensor: Distogram logits, shape ``(1, complex_length, complex_length, bins)``.
    """
    import torch.nn.functional as F
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput, load_ccd, prepare_esmfold2_input
    from esm.models.esmfold2.layers import _seed_context as seed_context

    load_ccd()
    vocabulary = _folding_vocabulary()
    padded = F.pad(design, _TOKEN_PAD, mode="constant", value=0)
    decoded = "".join(vocabulary[index] for index in design[0].argmax(-1).tolist())

    prepared = StructurePredictionInput(
        sequences=[
            ProteinInput(id="0", sequence=target_sequence, msa=None),
            ProteinInput(id="1", sequence=decoded, msa=None),
        ]
    )
    features, _ = prepare_esmfold2_input(prepared, seed=seed)
    # Distogram conditioning is unimplemented upstream and the forward pass rejects it.
    features.pop("disto_cond", None)
    features.pop("disto_cond_mask", None)
    inputs = {key: value.unsqueeze(0).to(device) for key, value in features.items()}

    target_one_hot = torch.zeros(1, len(target_sequence), padded.shape[-1], device=device, dtype=padded.dtype)
    for position, residue in enumerate(target_sequence):
        target_one_hot[0, position, _TOKEN_PAD[0] + vocabulary.index(residue)] = 1
    inputs["res_type_soft"] = torch.cat((target_one_hot, padded), dim=1)

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=_NUM_SAMPLING_STEPS,
            num_loops=_NUM_LOOPS,
            seed=seed,
        )
    return output["distogram_logits"]


def _relaxed_design(data: dict[str, Any], vocabulary: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the relaxed binder in trunk column order, and the tensor to differentiate against.

    The gradient is taken with respect to the caller's own pre-softmax, pre-permutation matrix,
    so the matrix handed back is already in the column order the caller asked in.

    Args:
        data (dict[str, Any]): Worker payload.
        vocabulary (str): Folding-trunk column order.
        device (str): Device to build on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The relaxed design ``(1, length, 20)`` and the leaf
            tensor its gradient is taken against.
    """
    source = torch.tensor(data["logits"], dtype=torch.float32, device=device, requires_grad=True)
    order = [data["alphabet"].index(residue) for residue in vocabulary]
    reordered = source[:, order]
    temperature = data["temperature"]
    relaxed = reordered if temperature is None else torch.softmax(reordered / temperature, dim=-1)
    soft, hard = data["soft"], data["hard"]
    if soft != 1.0:
        relaxed = soft * relaxed + (1 - soft) * reordered
    if hard != 0.0:
        relaxed = hard * objective.straight_through(relaxed) + (1 - hard) * relaxed
    return relaxed.unsqueeze(0), source


def structure_gradient(data: dict[str, Any]) -> dict[str, Any]:
    """Fold one complex and differentiate each requested quantity against the relaxed binder.

    Args:
        data (dict[str, Any]): Worker payload.

    Returns:
        dict[str, Any]: ``terms`` (one entry per requested quantity) and ``metrics``.

    Raises:
        ValueError: If an unknown quantity is requested.
    """
    device, seed = data["device"], data["seed"]
    vocabulary = _folding_vocabulary()
    design, source = _relaxed_design(data, vocabulary, device)
    binder_length = design.shape[1]
    target_sequence = data["target_sequence"]

    requested = list(data["gradient_terms"])
    unknown = [name for name in requested if name not in _QUANTITIES]
    if unknown:
        raise ValueError(f"esmfold2_inversion: unknown quantities {sorted(unknown)}")

    folder = _load_folder(data["model_name"], device, seed)
    distogram = _fold(folder, target_sequence, design, device, seed)
    midpoints = objective.bin_midpoints(distogram.device).to(distogram.dtype)

    hotspots = data.get("target_hotspot_ids") or []
    epitope_mask = torch.zeros(1, distogram.shape[1], dtype=torch.bool, device=distogram.device)
    for hotspot in hotspots:
        epitope_mask[0, int(hotspot[1:]) - 1] = True

    values: dict[str, torch.Tensor] = {}
    for name in requested:
        values[name] = _QUANTITIES[name](
            distogram=distogram,
            binder_length=binder_length,
            midpoints=midpoints,
            epitope_mask=epitope_mask,
            contact_distance=data["epitope_contact_distance"],
            design=design,
            data=data,
            device=device,
            seed=seed,
        ).squeeze(0)

    terms = []
    for position, name in enumerate(requested):
        gradient = None
        if data["compute_gradient"]:
            # One backward per quantity over the shared forward graph, retained until the last.
            gradient = torch.autograd.grad(
                values[name], source, retain_graph=position < len(requested) - 1, allow_unused=False
            )[0]
        terms.append(
            {
                "id": name,
                "value": float(values[name].detach()),
                "gradient": serialize_output(gradient) if gradient is not None else None,
            }
        )

    return {
        "terms": terms,
        "metrics": {
            "binder_length": binder_length,
            "target_length": len(target_sequence),
        },
    }


def _structure_quantity(fn: Any) -> Any:
    """Adapt a distogram-only objective to the uniform quantity signature."""

    def call(*, distogram: torch.Tensor, binder_length: int, midpoints: torch.Tensor, **_: Any) -> torch.Tensor:
        return fn(distogram, binder_length, midpoints)

    return call


def _epitope_quantity(
    *,
    distogram: torch.Tensor,
    binder_length: int,
    midpoints: torch.Tensor,
    epitope_mask: torch.Tensor,
    contact_distance: float,
    **_: Any,
) -> torch.Tensor:
    return objective.epitope_loss(distogram, epitope_mask, binder_length, midpoints, contact_distance)


def _plm_quantity(*, design: torch.Tensor, data: dict[str, Any], device: str, seed: int, **_: Any) -> torch.Tensor:
    """Score the designed chain under ESM C, loading the language model only if asked for."""
    from esm.models.esmc import EsmcTokenizer

    masked_lm = _load_language_model(device, seed)
    tokenizer = EsmcTokenizer()
    scored = torch.ones(design.shape[0], design.shape[1], dtype=torch.bool, device=design.device)
    generator = torch.Generator(device=design.device).manual_seed(seed)
    return objective.pseudo_perplexity_nll(
        masked_lm,
        design,
        scored,
        trunk_to_lm=_trunk_to_lm_matrix(design.device),
        cls_id=tokenizer.cls_token_id,
        eos_id=tokenizer.eos_token_id,
        mask_id=masked_lm.config.mask_token_id,
        vocab_size=masked_lm.config.vocab_size,
        amino_acid_offset=4,
        passes=data["plm_passes"],
        batch_size=data["plm_batch_size"],
        generator=generator,
    )


#: Every quantity this worker can differentiate, by the id the wrapper resolves to a term class.
_QUANTITIES: dict[str, Any] = {
    "intra_contact": _structure_quantity(objective.intra_contact_loss),
    "inter_contact": _structure_quantity(objective.inter_contact_loss),
    "globularity": _structure_quantity(objective.globularity_loss),
    "epitope": _epitope_quantity,
    "plm_nll": _plm_quantity,
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
    if operation != "structure_gradient":
        raise ValueError(f"esmfold2_inversion: unknown operation {operation!r}; valid: ['structure_gradient']")
    try:
        return structure_gradient(input_dict)
    except RuntimeError as error:
        if is_cuda_oom(error):
            release_cuda_memory()
        raise


def to_device(device: str) -> dict[str, Any]:
    """Move every cached model to ``device``."""
    from standalone_helpers import move_model_to_device

    for model in list(_folders.values()) + ([_language_model] if _language_model is not None else []):
        move_model_to_device(model, device)
    return {"success": True, "device": device}


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
