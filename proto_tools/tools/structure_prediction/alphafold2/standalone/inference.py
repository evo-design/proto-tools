"""AlphaFold2 standalone inference — prediction and design-protocol gradients.

Runs in an isolated venv with JAX and upstream sokrypton/ColabDesign.

Extension loss callbacks (rg, NC, i_ptm, beta_strand, fw_i_con) are registered when a
requested gradient term needs one.

Usage:
    python inference.py <input_json_path> <output_json_path>
"""

import copy
import gc
import json
import logging
import os
import sys
from collections.abc import Collection, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from standalone_helpers import (
    AMINO_ACIDS_LIST,
    count_a3m_rows,
    get_jax_memory_stats,
    get_logger,
    report_msa_rows_loaded,
    resolve_jax_device,
    serialize_output,
)
from standalone_helpers.relaxation import resolve_temperature
from standalone_helpers.vocabulary import gather_indices

logger = get_logger(__name__)

# AlphaFold2's own residue order: 3-letter codes alphabetically, not proto's one-letter sort.
ALPHAFOLD2_AMINO_ACIDS: list[str] = list("ARNDCQEGHILKMFPSTWYV")

# Proto columns into AF2 order, and the gradient back; only the argument order differs.
PROTO_TO_AF2 = gather_indices(ALPHAFOLD2_AMINO_ACIDS, AMINO_ACIDS_LIST, label="alphafold2")
AF2_TO_PROTO = gather_indices(AMINO_ACIDS_LIST, ALPHAFOLD2_AMINO_ACIDS, label="alphafold2")

# Configure stderr handler so debug output is captured by persistent worker
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

_params_dir: str | None = None


def _relaxation_opt(*, temperature: float | None, soft: float, hard: float) -> dict[str, float]:
    """Build the ColabDesign ``opt`` entries that drive ``soft_seq``.

    ``alpha`` scales the logits before the softmax, so ColabDesign's default of 2.0 would make
    ``temperature`` mean half what the shared gradient contract says and would stop
    ``from_probabilities`` reconstructing the supplied distribution. It is pinned to 1.0, which
    costs no expressiveness: ``alpha=a`` with temperature ``T`` is ``alpha=1`` at ``T / a``.

    Probability mode is expressed as ``soft=0``, which selects ``soft_seq``'s raw-input branch;
    ``temp`` then feeds only the unused softmax branch and is pinned to keep it finite.

    Args:
        temperature (float | None): Positive softmax temperature, or ``None`` for probability mode.
        soft (float): Blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Straight-through coefficient.

    Returns:
        dict[str, float]: ``temp``, ``soft``, ``hard`` and ``alpha`` for ``af_model.opt``.
    """
    if temperature is None:
        return {"temp": 1.0, "soft": 0.0, "hard": 0.0, "alpha": 1.0}
    return {"temp": temperature, "soft": soft, "hard": hard, "alpha": 1.0}


def _resolve_params_dir() -> str:
    """Locate the AlphaFold2 parameters directory."""
    global _params_dir
    if _params_dir is not None:
        return _params_dir

    from standalone_helpers import resolve_weights_dir

    weights_dir = resolve_weights_dir("alphafold2")
    if weights_dir:
        params_dir = Path(weights_dir) / "params"
    else:
        venv_path = os.environ.get("TOOL_VENV_PATH")
        if not venv_path:
            raise RuntimeError("alphafold2: TOOL_VENV_PATH not set — must be run via ToolInstance")
        params_dir = Path(venv_path) / "data" / "params"

    if not params_dir.exists() or not any(params_dir.glob("*.npz")):
        raise RuntimeError(f"alphafold2: parameters not found at {params_dir}; run setup.sh to download")

    _params_dir = str(params_dir)
    return _params_dir


def _metric_value(value: Any) -> float | None:
    """Serialize an aux value into one representative scalar metric."""
    import numpy as np

    serialized = serialize_output(value)
    if isinstance(serialized, list):
        array = np.asarray(serialized, dtype=np.float32)
        if array.size == 0:
            return None
        return float(np.mean(array))
    return float(serialized)


def _extract_metrics(aux: dict[str, Any] | None, include_pae_matrix: bool = False) -> dict[str, Any]:
    """Convert ColabDesign aux outputs into simple scalar metrics."""
    if not aux:
        return {}

    metrics: dict[str, Any] = {}
    for src_key, dst_key in (("plddt", "avg_plddt"), ("ptm", "ptm"), ("i_ptm", "iptm"), ("pae", "avg_pae")):
        value = aux.get(src_key)
        if value is not None:
            scalar = _metric_value(value)
            if scalar is not None:
                metrics[dst_key] = scalar

    # Terms carry the confidence itself, so copying the raw loss here would duplicate the name.
    for key, value in aux.get("losses", {}).items():
        if key in _NATIVE_LOSSES or key in _INVERTED_LOSSES or key == "pae":
            continue
        scalar = _metric_value(value)
        if scalar is not None:
            metrics[key] = scalar

    if include_pae_matrix:
        metrics["pae"] = serialize_output(aux["pae"])

    return metrics


# ============================================================================
# Extension loss callbacks
# ============================================================================


def _extension_loss_fns(designed_len: int, term_options: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build the extension loss functions, each over the trailing ``designed_len`` residues.

    The designed segment is always the trailing rows (the binder, or the whole chain for the
    single-chain protocols), so every extension applies under all three protocols.

    Args:
        designed_len (int): Number of trailing residues being designed.
        term_options (Mapping[str, Mapping[str, Any]]): Per-term selectors, by term id.

    Returns:
        dict[str, Any]: One loss callback per extension term; ColabDesign passes arguments by name.
    """
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants
    from colabdesign.af.loss import get_con_loss, get_dgram_bins, get_ptm

    def designed_ca(outputs: Any) -> Any:
        """CA coordinates of the designed residues."""
        return outputs["structure_module"]["final_atom_positions"][-designed_len:, residue_constants.atom_order["CA"]]

    def rg(outputs: Any) -> dict[str, Any]:
        """Excess radius of gyration, as in BindCraft and Germinal."""
        ca = designed_ca(outputs)
        radius = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        return {"rg": jax.nn.elu(radius - 2.38 * designed_len**0.365)}

    def termini(outputs: Any) -> dict[str, Any]:
        """Terminal CA distance beyond 7 angstroms, as in BindCraft and Germinal."""
        ca = designed_ca(outputs)
        return {"NC": jax.nn.relu(jax.nn.elu(jnp.linalg.norm(ca[0] - ca[-1]) - 7.0))}

    def i_ptm(inputs: Any, outputs: Any) -> dict[str, Any]:
        """Interface pTM, as ``1 - ipTM`` like ColabDesign's other confidences."""
        return {"i_ptm": 1 - get_ptm(inputs, outputs, interface=True)}

    def beta_strand(outputs: Any) -> dict[str, Any]:
        """Mean of the worst third of ``-log P(9.75 <= d <= 11.5)`` over designed i,i+3 pairs."""
        dgram = outputs["distogram"]["logits"]
        bins = get_dgram_bins(outputs)
        in_range = jnp.logical_and(bins >= 9.75, bins <= 11.5)
        pair_loss = -jnp.log((in_range * jax.nn.softmax(dgram) + 1e-8).sum(-1))
        worst = jnp.sort(jnp.diagonal(pair_loss[-designed_len:, -designed_len:], 3))[::-1]
        return {"beta_strand": worst[: max(designed_len // 3, 1)].mean()}

    def fw_i_con(inputs: Any, outputs: Any, opt: Any) -> dict[str, Any]:
        """Contact loss from the target to the framework residues, as in Germinal."""
        length = inputs["seq_mask"].shape[0]
        positions = term_options["fw_i_con"]["framework_positions"]
        target = jnp.zeros(length).at[: length - designed_len].set(1)
        framework = jnp.zeros(length).at[jnp.asarray(positions) + length - designed_len - 1].set(1)
        return {"fw_i_con": get_con_loss(inputs, outputs, opt["i_con"], mask_1d=target, mask_1b=framework)}

    return {"rg": rg, "NC": termini, "i_ptm": i_ptm, "beta_strand": beta_strand, "fw_i_con": fw_i_con}


#: Quantities ColabDesign computes itself; an extension must never shadow one.
_NATIVE_LOSSES = frozenset({"plddt", "pae", "i_pae", "con", "i_con", "exp_res", "helix", "dgram_cce", "fape", "rmsd"})

#: Quantities ColabDesign reports as ``1 - x``; reported back as the confidence they describe.
_INVERTED_LOSSES = frozenset({"plddt", "exp_res", "i_ptm"})


def _configure_extension_losses(
    af_model: Any, requested: Collection[str], designed_len: int, term_options: Mapping[str, Mapping[str, Any]]
) -> None:
    """Register the extension loss callbacks the requested terms need.

    Args:
        af_model (Any): The ColabDesign design model.
        requested (Collection[str]): Requested term ids.
        designed_len (int): Number of trailing residues being designed.
        term_options (Mapping[str, Mapping[str, Any]]): Per-term selectors, by term id.

    Raises:
        ValueError: If a framework position lies outside the designed residues.
    """
    positions = term_options.get("fw_i_con", {}).get("framework_positions", [])
    if any(position > designed_len for position in positions):
        raise ValueError(f"alphafold2: framework_positions {positions} exceed the {designed_len} designed residues")
    extensions = _extension_loss_fns(designed_len, term_options)
    for name in set(requested) - _NATIVE_LOSSES:
        af_model._callbacks["model"]["loss"].append(extensions[name])


def _per_term_gradients(af_model: Any, term_names: list[str]) -> Any:
    """Make the backprop pass yield one gradient per loss term instead of one fused gradient.

    ColabDesign sums its weighted terms into a single scalar and differentiates that, so
    ``aux["grad"]`` answers "how does the whole objective move" and cannot be taken apart
    afterwards. The individual terms are right there though: ``_model["fn"]`` is the same
    function ``grad_fn`` differentiates, exposed undifferentiated, and it returns the per-term
    scalars in ``aux["losses"]`` one line before they are summed.

    So this linearizes that function once and pushes a one-hot cotangent through it per
    requested term: one forward pass and one cheap backward each, which is JAX's equivalent of
    retaining a graph across several backwards. The per-term gradients are stashed on
    ``aux["per_term_grad"]``; ColabDesign's own ``aux["grad"]`` is zeroed, because a raw-term
    tool never reports a fused gradient and reproducing it would cost a backward per
    unrequested term as well.

    Only the backprop pass is replaced, so recycling, model selection and PDB export behave
    exactly as before.

    Args:
        af_model (Any): The ColabDesign design model.
        term_names (list[str]): Loss-dict keys to differentiate.

    Returns:
        Any: A zero-argument callable that restores the original method.

    Raises:
        RuntimeError: If this ColabDesign pin exposes no undifferentiated loss function, or the
            recycle mode backprops somewhere other than the final pass.
    """
    import jax
    import jax.numpy as jnp

    if "fn" not in af_model._model:
        raise RuntimeError(
            "alphafold2: ColabDesign exposes no _model['fn']; per-term gradients need the "
            "undifferentiated loss function and this pin does not provide it"
        )
    mode = af_model._args.get("recycle_mode")
    if mode != "last":
        raise RuntimeError(
            f"alphafold2: recycle_mode={mode!r} does not carry the gradient on the final recycle "
            '("first" and "sample" backprop through another pass, "average" through all of them), '
            "so per-term gradients could not be attributed to one recycle"
        )

    original = af_model._single
    loss_fn = af_model._model["fn"]

    def single_with_per_term(model_params: Any, backprop: bool = True) -> Any:
        """Stand in for ColabDesign's single pass, adding per-term gradients on the backprop one."""
        if not backprop:
            return original(model_params, backprop=False)

        # Some ColabDesign forks bind opt only here, so the sync has to be repeated.
        af_model._inputs["opt"] = af_model.opt

        # One key draw, so every term is differentiated through the same MSA subsample.
        rest = (model_params, af_model._inputs, af_model.key())

        def losses_of(params: Any) -> Any:
            """Expose the per-term loss dict as the differentiated output."""
            loss, aux = loss_fn(params, *rest)
            return aux["losses"], (loss, aux)

        losses, pull_back, (loss, aux) = jax.vjp(losses_of, af_model._params, has_aux=True)

        missing = [name for name in term_names if name not in losses]
        if missing:
            raise RuntimeError(
                f"alphafold2: requested terms {missing} are not among this configuration's "
                f"losses; available: {sorted(losses)}"
            )

        zeros = jax.tree_util.tree_map(jnp.zeros_like, losses)
        per_term = {}
        for name in term_names:
            cotangent = dict(zeros)
            cotangent[name] = jnp.ones_like(losses[name])
            (per_term[name],) = pull_back(cotangent)

        aux.update(
            {
                "loss": loss,
                "grad": jax.tree_util.tree_map(jnp.zeros_like, af_model._params),
                "per_term_grad": per_term,
            }
        )
        return aux

    af_model._single = single_with_per_term

    def restore() -> None:
        """Put ColabDesign's own single-pass method back."""
        af_model._single = original

    return restore


# AlphaFold-Multimer MSA layout: row 0 = concatenated queries; paired rows span all columns, unpaired rows are block-diagonal; ColabDesign gap char = 21.
_AA_GAP = 21


def _build_combined_multimer_msa(
    *,
    per_chain_msas_a3m: list[str | None],
    chains: list[str],
    seq_lengths: list[int],
    is_paired: bool,
    unpaired_per_chain_msas_a3m: list[str | None] | None = None,
) -> tuple[Any, Any]:
    """Stitch per-chain A3Ms into one combined MSA of shape (N, L_total) for the multimer model.

    Layout:
      * row 0: concatenated chain queries
      * if ``is_paired``: paired rows — row k = ``[chain_0_row_k | chain_1_row_k | ...]``
        (each chain's MSA is taxonomy-row-aligned so row k is the same species across chains),
        then, when ``unpaired_per_chain_msas_a3m`` is supplied, each chain's deep unpaired
        rows are appended block-diagonally below the paired block for per-chain depth.
      * else: block-diagonal rows — for each chain i and each non-query row in chain i's
        MSA, place that row in chain i's column band and gap-pad the other chains.

    Chains without an MSA (``per_chain_msas_a3m[i] is None``) contribute their raw query
    residues to row 0 (encoded via ColabDesign's ``aa2num``) and contribute no extra rows.
    """
    import numpy as np
    from colabdesign.shared.parsers import aa2num, parse_a3m

    num_chains = len(chains)

    def _parse(per_chain_a3m: list[str | None]) -> list[tuple[Any, Any] | None]:
        parsed: list[tuple[Any, Any] | None] = []
        for i, a3m in enumerate(per_chain_a3m):
            if a3m is None:
                parsed.append(None)
                continue
            msa_i, del_i = parse_a3m(a3m_string=a3m)
            if msa_i.ndim == 1:
                msa_i = msa_i[None]
                del_i = del_i[None]
            if msa_i.shape[1] != seq_lengths[i]:
                raise ValueError(
                    f"alphafold2: chain {i} MSA length {msa_i.shape[1]} does not match "
                    f"expected chain length {seq_lengths[i]}"
                )
            parsed.append((msa_i, del_i))
        return parsed

    parsed = _parse(per_chain_msas_a3m)

    # Row 0, the concatenated query: row 0 of each chain's MSA, or the aa-encoded bare chain sequence when it has no MSA.
    query_segments = []
    query_del_segments = []
    for i, p in enumerate(parsed):
        if p is not None:
            query_segments.append(np.asarray(p[0][0], dtype=np.int32))
            query_del_segments.append(np.asarray(p[1][0], dtype=np.int32))
        else:
            query_segments.append(np.asarray(aa2num(chains[i]), dtype=np.int32))
            query_del_segments.append(np.zeros(seq_lengths[i], dtype=np.int32))

    combined_rows = [np.concatenate(query_segments)]
    combined_dels = [np.concatenate(query_del_segments)]

    def _append_block_diagonal(parsed_list: list[tuple[Any, Any] | None]) -> None:
        # Each chain's non-query rows go in its own column band; other chains are gap-padded.
        for i, p in enumerate(parsed_list):
            if p is None:
                continue
            msa_i, del_i = p
            for k in range(1, msa_i.shape[0]):
                row_segments, del_segments = [], []
                for j in range(num_chains):
                    if j == i:
                        row_segments.append(np.asarray(msa_i[k], dtype=np.int32))
                        del_segments.append(np.asarray(del_i[k], dtype=np.int32))
                    else:
                        row_segments.append(np.full(seq_lengths[j], _AA_GAP, dtype=np.int32))
                        del_segments.append(np.zeros(seq_lengths[j], dtype=np.int32))
                combined_rows.append(np.concatenate(row_segments))
                combined_dels.append(np.concatenate(del_segments))

    if is_paired:
        # Paired block: row k spans all chains (each chain's MSA is taxonomy-row-aligned).
        n_paired = next((p[0].shape[0] for p in parsed if p is not None), 0)
        for k in range(1, n_paired):
            row_segments, del_segments = [], []
            for i in range(num_chains):
                p_i = parsed[i]
                if p_i is not None:
                    row_segments.append(np.asarray(p_i[0][k], dtype=np.int32))
                    del_segments.append(np.asarray(p_i[1][k], dtype=np.int32))
                else:
                    row_segments.append(np.full(seq_lengths[i], _AA_GAP, dtype=np.int32))
                    del_segments.append(np.zeros(seq_lengths[i], dtype=np.int32))
            combined_rows.append(np.concatenate(row_segments))
            combined_dels.append(np.concatenate(del_segments))
        # Unpaired depth block: each chain's deep unpaired rows, block-diagonal under the paired block.
        if unpaired_per_chain_msas_a3m is not None:
            _append_block_diagonal(_parse(unpaired_per_chain_msas_a3m))
    else:
        _append_block_diagonal(parsed)

    return np.stack(combined_rows, axis=0), np.stack(combined_dels, axis=0)


def _log_loaded_msa_rows(
    af_model: Any,
    complex_data: dict[str, Any],
    msa_a3m_content: str | None,
    per_chain_msas_a3m: list[str | None] | None,
    unpaired_per_chain_msas_a3m: list[str | None] | None,
) -> None:
    """Log the MSA rows ColabDesign holds per chain against the rows sent, keyed by chain ID."""
    import numpy as np

    msa = np.asarray(af_model._inputs["msa"])
    chains, lengths = complex_data["chains"], complex_data["seq_lengths"]
    keys = complex_data.get("chain_ids") or [str(i) for i in range(len(chains))]
    if per_chain_msas_a3m is None:  # monomer / homo-oligomer: one alignment tiled across copies
        loaded = dict.fromkeys(keys, int((msa != _AA_GAP).any(axis=1).sum()))
        sent = dict.fromkeys(keys, count_a3m_rows(msa_a3m_content))
    else:
        offsets = np.cumsum([0, *lengths])
        loaded = {
            key: int((msa[:, offsets[i] : offsets[i + 1]] != _AA_GAP).any(axis=1).sum()) for i, key in enumerate(keys)
        }
        unpaired = unpaired_per_chain_msas_a3m or [None] * len(chains)
        sent = {
            key: max(count_a3m_rows(per_chain_msas_a3m[i]), count_a3m_rows(unpaired[i])) for i, key in enumerate(keys)
        }
    report_msa_rows_loaded("alphafold2", loaded, sent)


class AlphaFold2Model:
    """Persistent ColabDesign-backed AlphaFold2 worker."""

    def __init__(self) -> None:
        """Initialize persistent worker state."""
        self._loaded = False
        self.device = "cuda"
        self.params_dir: str | None = None
        self._jax: Any | None = None
        self._jnp: Any | None = None
        self._mk_af_model: Any | None = None
        self._models: dict[tuple[str, bool, bool, str], Any] = {}
        self._model_defaults: dict[tuple[str, bool, bool, str], dict[str, Any]] = {}

    def load(self, device: str, verbose: bool = False) -> None:
        """Load ColabDesign entrypoints and resolve parameter locations.

        Args:
            device: JAX device string ("cuda", "cpu", "cuda:0", etc.).
            verbose: Log loading details.
        """
        if self._loaded:
            return

        # Device env (CUDA_VISIBLE_DEVICES / JAX_PLATFORMS) is owned by the worker spawn (pin_visible_devices).
        import jax
        import jax.numpy as jnp
        from colabdesign import mk_afdesign_model as mk_af_model

        self._jax = jax
        self._jnp = jnp
        self._mk_af_model = mk_af_model
        self.params_dir = _resolve_params_dir()
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("AlphaFold2 ColabDesign runtime loaded on %s", device)

    def _ensure_loaded(self, device: str, verbose: bool = False) -> None:
        if not self._loaded:
            self.load(device, verbose=verbose)
        elif self.device != device:
            self.to_device(device, verbose=verbose)

    def _get_model(
        self,
        *,
        protocol: str = "hallucination",
        use_multimer: bool,
        use_msa: bool,
        recycle_mode: str = "last",
        verbose: bool = False,
    ) -> Any:
        """Return a cached ColabDesign model, creating one if needed."""
        key = (protocol, use_multimer, use_msa, recycle_mode)
        model = self._models.get(key)
        if model is not None:
            return model

        model_kwargs: dict[str, Any] = {
            "protocol": protocol,
            "use_multimer": use_multimer,
            "data_dir": self.params_dir,
            "recycle_mode": recycle_mode,
        }
        if use_msa:
            model_kwargs["optimize_seq"] = False
            model_kwargs["num_msa"] = 512
            model_kwargs["num_extra_msa"] = 1024

        if verbose:
            logger.info("Creating AlphaFold2 model: protocol=%s multimer=%s msa=%s", protocol, use_multimer, use_msa)

        jax = self._jax
        mk_af_model = self._mk_af_model
        assert jax is not None and mk_af_model is not None
        device_ctx = (
            jax.default_device(resolve_jax_device(self.device)) if hasattr(jax, "default_device") else nullcontext()
        )
        with device_ctx:
            model = mk_af_model(**model_kwargs)

        self._models[key] = model
        self._model_defaults[key] = {
            "callbacks": copy.deepcopy(model._callbacks),
            "weights": copy.deepcopy(model.opt.get("weights", {})),
        }
        return model

    def _restore_model_defaults(self, key: tuple[str, bool, bool, str], model: Any) -> None:
        """Restore callbacks and opt weights to baseline before request-specific setup."""
        defaults = self._model_defaults.get(key)
        if defaults is None:
            return
        model._callbacks = copy.deepcopy(defaults["callbacks"])
        if "weights" in defaults:
            model.opt["weights"] = copy.deepcopy(defaults["weights"])

    # -------------------------------------------------------------------------
    # Prediction (unchanged from stock ColabDesign)
    # -------------------------------------------------------------------------

    @staticmethod
    def _prep_prediction_inputs(
        af_model: Any,
        complex_data: dict[str, Any],
        msa_a3m_content: str | None,
        per_chain_msas_a3m: list[str | None] | None = None,
        is_paired: bool = False,
        unpaired_per_chain_msas_a3m: list[str | None] | None = None,
    ) -> None:
        """Prepare ColabDesign inputs for a prediction request.

        Single-chain / homo-oligomer paths use ``msa_a3m_content`` (one A3M); ColabDesign
        tiles it across copies for homo-oligomers. Heteromultimers use ``per_chain_msas_a3m``
        (one A3M per chain, parallel to ``chains``) to build the combined block-diagonal
        MSA the multimer architecture expects. For paired heteromultimers,
        ``unpaired_per_chain_msas_a3m`` carries each chain's deep unpaired MSA appended as
        block-diagonal depth below the paired block.
        """
        chains = complex_data["chains"]
        seq_lengths = complex_data["seq_lengths"]
        num_chains = complex_data["num_chains"]

        is_homooligomer = num_chains > 1 and len(set(chains)) == 1
        if is_homooligomer:
            af_model.prep_inputs(length=seq_lengths[0], copies=num_chains)
        elif num_chains > 1:
            af_model.prep_inputs(length=seq_lengths)
        else:
            af_model.prep_inputs(length=seq_lengths[0])

        # Heteromultimer with per-chain MSAs → build combined block-diagonal MSA.
        if num_chains > 1 and not is_homooligomer and per_chain_msas_a3m and any(per_chain_msas_a3m):
            msa, deletion_matrix = _build_combined_multimer_msa(
                per_chain_msas_a3m=per_chain_msas_a3m,
                chains=chains,
                seq_lengths=seq_lengths,
                is_paired=is_paired,
                unpaired_per_chain_msas_a3m=unpaired_per_chain_msas_a3m,
            )
            af_model.set_msa(msa, deletion_matrix)
            return

        # Single-chain or homo-oligomer path (ColabDesign expands for copies).
        if msa_a3m_content:
            from colabdesign.shared.parsers import parse_a3m

            msa, deletion_matrix = parse_a3m(a3m_string=msa_a3m_content)
            af_model.set_msa(msa, deletion_matrix)
        elif is_homooligomer:
            af_model.set_seq(seq=chains[0])
        else:
            af_model.set_seq(seq="".join(chains))

    def predict(
        self,
        *,
        complex_data: dict[str, Any],
        num_recycles: int = 3,
        model_num: int = 1,
        num_ensemble_models: int = 1,
        seed: int | None = None,
        msa_a3m_content: str | None = None,
        per_chain_msas_a3m: list[str | None] | None = None,
        is_paired: bool = False,
        unpaired_per_chain_msas_a3m: list[str | None] | None = None,
        device: str = "cuda",
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run AlphaFold2 prediction on one complex."""
        self._ensure_loaded(device, verbose=verbose)

        use_multimer = complex_data["num_chains"] > 1
        use_msa = msa_a3m_content is not None or bool(per_chain_msas_a3m and any(per_chain_msas_a3m))
        af_model = self._get_model(use_multimer=use_multimer, use_msa=use_msa, verbose=verbose)

        self._prep_prediction_inputs(
            af_model,
            complex_data,
            msa_a3m_content,
            per_chain_msas_a3m=per_chain_msas_a3m,
            is_paired=is_paired,
            unpaired_per_chain_msas_a3m=unpaired_per_chain_msas_a3m,
        )
        if use_msa:
            _log_loaded_msa_rows(
                af_model, complex_data, msa_a3m_content, per_chain_msas_a3m, unpaired_per_chain_msas_a3m
            )

        predict_kwargs: dict[str, Any] = {
            "num_recycles": num_recycles,
            "seed": seed,
            "verbose": verbose,
        }
        if num_ensemble_models > 1:
            predict_kwargs["num_models"] = num_ensemble_models
        else:
            predict_kwargs["models"] = [model_num - 1]

        from standalone_helpers import oom_guard

        with oom_guard("alphafold2", hint="Shorten the sequence or lower num_recycles."):
            af_model.predict(**predict_kwargs)

        aux = getattr(af_model, "aux", {})
        metrics = _extract_metrics(aux, include_pae_matrix=include_pae_matrix)
        return {
            "pdb": af_model.save_pdb(),
            "avg_plddt": metrics["avg_plddt"],
            "ptm": metrics["ptm"],
            "iptm": metrics.get("iptm"),
            "avg_pae": metrics["avg_pae"],
            "pae": metrics.get("pae"),
        }

    # -------------------------------------------------------------------------
    # Binder gradient computation (base or Germinal backend)
    # -------------------------------------------------------------------------

    def _run_gradient(
        self,
        af_model: Any,
        *,
        logits: list[list[float]],
        temperature: float | None,
        soft: float,
        hard: float,
        num_recycles: int,
        model_num: int,
        sample_models: bool,
        gradient_terms: list[str] | None,
        seed: int | None,
        backprop: bool = True,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run one design-protocol AF2 pass and return loss, metrics, Structure, and optionally gradient.

        ``soft_seq()`` applies the relaxation and any persistent bias internally; see
        :func:`_relaxation_opt` for the options proto pins. When ``backprop=True``, JAX autograd
        chain-rules through the full expression and the returned gradient is exact
        ``∂loss/∂logits``.
        ``temperature=None`` consumes ``logits`` as probabilities instead, so that
        gradient is probability-space; see the ``opt`` assignment below.
        When ``backprop=False``, the forward pass runs through ``model["fn"]``
        (no grad_fn) and ``gradient`` is ``None`` in the returned dict.
        """
        import numpy as np

        jnp = self._jnp
        assert jnp is not None

        opt = af_model.opt
        opt["num_recycles"] = num_recycles
        opt.update(_relaxation_opt(temperature=temperature, soft=soft, hard=hard))
        if seed is not None:
            af_model.set_seed(seed)

        # Inject logits at binder positions (last binder_len rows), reordering to AF2 AA order.
        af_logits = jnp.asarray(logits, dtype=jnp.float32)[:, PROTO_TO_AF2]
        binder_len = len(logits)
        full_len = int(af_model._params["seq"].shape[1])

        full_seq = np.array(af_model._params["seq"], dtype=np.float32)
        full_seq[0, full_len - binder_len :] = np.asarray(af_logits)
        af_model._params["seq"] = full_seq

        # run() populates af_model.aux["all"], which save_pdb() requires.
        # num_recycles is read from opt, already set above.
        model_nums = [int(np.random.choice(5))] if sample_models else [model_num - 1]
        af_model._args["clear_prev"] = True
        from standalone_helpers import oom_guard

        restore = _per_term_gradients(af_model, gradient_terms) if (backprop and gradient_terms) else None
        try:
            with oom_guard("alphafold2", hint="Shorten the binder/target or lower num_recycles."):
                af_model.run(backprop=backprop, model_nums=model_nums)
            aux = af_model.aux
        finally:
            if restore is not None:
                restore()

        def _binder_rows(seq_grad: Any) -> list[list[float]]:
            """Reorder one gradient into proto column order and keep the binder's rows."""
            # Shape (batch, L, alphabet) expected; fail loudly if ColabDesign's layout shifts.
            assert seq_grad.ndim == 3, f"unexpected grad shape {seq_grad.shape}"
            rows: list[list[float]] = serialize_output(seq_grad[0][:, AF2_TO_PROTO][full_len - binder_len :])
            return rows

        terms: list[dict[str, Any]] = []
        for name in gradient_terms or []:
            value = _metric_value(aux["losses"][name])
            per_term = aux.get("per_term_grad", {}).get(name) if backprop else None
            rows = _binder_rows(per_term["seq"]) if per_term is not None else None
            if name in _INVERTED_LOSSES:
                # ColabDesign reports 1 - confidence; negate the gradient with the value.
                value = None if value is None else 1.0 - value
                rows = None if rows is None else [[-entry for entry in row] for row in rows]
            terms.append({"id": name, "value": value, "gradient": rows})

        return {
            "terms": terms,
            "loss": float(serialize_output(aux["loss"])),
            "metrics": _extract_metrics(aux, include_pae_matrix=include_pae_matrix),
            "vocab": AMINO_ACIDS_LIST,
            "pdb": af_model.save_pdb(get_best=False),
        }

    def compute_design_gradient(
        self,
        *,
        protocol: str,
        prep_kwargs: dict[str, Any],
        logits: list[list[float]],
        gradient_terms: list[str],
        term_options: Mapping[str, Mapping[str, Any]] | None = None,
        temperature: float | None = 1.0,
        soft: float = 1.0,
        hard: float = 0.0,
        num_recycles: int = 3,
        recycle_mode: str = "last",
        model_num: int = 1,
        sample_models: bool = False,
        use_multimer: bool = False,
        seed: int | None = None,
        compute_gradient: bool = True,
        device: str = "cuda",
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Differentiate one ColabDesign protocol's quantities with respect to relaxed logits.

        The three design tools differ only in which protocol they build and what they hand to
        ``prep_inputs``; everything after that — relaxation, per-term differentiation, structure
        and metric extraction — is shared. The designed segment is always the trailing
        ``len(logits)`` rows, which is the binder for the binder protocol and the whole chain for
        the others, so one gradient slice serves all three.

        Args:
            protocol (str): ColabDesign protocol: ``"binder"``, ``"fixbb"`` or ``"hallucination"``.
            prep_kwargs (dict[str, Any]): Protocol-specific arguments for ``prep_inputs``.
            logits (list[list[float]]): Relaxed ``(L, 20)`` state for the designed chain.
            gradient_terms (list[str]): Loss-dict keys to differentiate, one gradient each.
            term_options (Mapping[str, Mapping[str, Any]] | None): Per-term selectors, by term id.
            temperature (float | None): Softmax temperature, or ``None`` for probabilities.
            soft (float): Relaxation blend.
            hard (float): Straight-through coefficient.
            num_recycles (int): Structure-module recycling iterations.
            recycle_mode (str): ColabDesign recycle mode.
            model_num (int): Which AF2 parameter set to run.
            sample_models (bool): Whether to sample the parameter set at random.
            use_multimer (bool): Use multimer parameters.
            seed (int | None): Seed for the model's RNG.
            compute_gradient (bool): Run the backward passes.
            device (str): Device to run on.
            verbose (bool): Verbose model loading.
            include_pae_matrix (bool): Attach the full PAE matrix to the metrics.

        Returns:
            dict[str, Any]: One entry per requested term with its raw value and gradient, plus
                metrics, the predicted structure and the vocabulary.
        """
        self._ensure_loaded(device, verbose=verbose)
        # MSA mode drops params["seq"], the parameter the relaxed state is injected into.
        key = (protocol, use_multimer, False, recycle_mode)
        af_model = self._get_model(
            protocol=protocol,
            use_multimer=use_multimer,
            use_msa=False,
            recycle_mode=recycle_mode,
            verbose=verbose,
        )
        # The model is cached across requests, so a previous call's options must be cleared.
        self._restore_model_defaults(key, af_model)
        af_model.prep_inputs(**prep_kwargs)

        # prep_inputs parses the PDB, so only it knows how many residues were resolved.
        designed_len = int(af_model._binder_len if protocol == "binder" else af_model._len)
        if len(logits) != designed_len:
            raise ValueError(
                f"alphafold2: this input supplies {len(logits)} residues but the {protocol} protocol "
                f"designs {designed_len}; the relaxed state must cover exactly the designed residues"
            )

        # An extension loss only reaches aux["losses"] once its callback is installed.
        _configure_extension_losses(af_model, gradient_terms, designed_len, term_options or {})

        return self._run_gradient(
            af_model,
            logits=logits,
            temperature=temperature,
            soft=soft,
            hard=hard,
            num_recycles=num_recycles,
            model_num=model_num,
            sample_models=sample_models,
            gradient_terms=gradient_terms,
            seed=seed,
            backprop=compute_gradient,
            include_pae_matrix=include_pae_matrix,
        )

    def to_device(self, device: str, verbose: bool = False) -> None:
        """Switch target device for future model cache entries."""
        # If load() isolated us via CUDA_VISIBLE_DEVICES, any cuda:N maps to the sole visible cuda:0.
        if os.environ.get("CUDA_VISIBLE_DEVICES") and device.startswith("cuda") and ":" in device:
            device = "cuda:0"
        if self.device == device:
            return

        if verbose:
            logger.info("Switching AlphaFold2 worker device from %s to %s", self.device, device)

        self._models.clear()
        self._model_defaults.clear()
        self.device = device
        if self._jax is not None:
            self._jax.clear_caches()
        gc.collect()


_model: AlphaFold2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = AlphaFold2Model()

    operation = input_dict["operation"]
    if operation == "predict":
        return _model.predict(
            complex_data=input_dict["complex_data"],
            num_recycles=input_dict["num_recycles"],
            model_num=input_dict["model_num"],
            num_ensemble_models=input_dict["num_ensemble_models"],
            seed=input_dict["seed"],
            msa_a3m_content=input_dict.get("msa_a3m_content"),
            per_chain_msas_a3m=input_dict.get("per_chain_msas_a3m"),
            is_paired=input_dict.get("is_paired", False),
            unpaired_per_chain_msas_a3m=input_dict.get("unpaired_per_chain_msas_a3m"),
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
        )
    if operation == "compute_design_gradient":
        results = [
            _model.compute_design_gradient(
                protocol=input_dict["protocol"],
                prep_kwargs=item["prep_kwargs"],
                logits=item["logits"],
                gradient_terms=input_dict["gradient_terms"],
                term_options=input_dict.get("term_options"),
                temperature=resolve_temperature(input_dict["temperature"]),
                soft=input_dict["soft"],
                hard=input_dict["hard"],
                num_recycles=input_dict.get("num_recycles", 3),
                recycle_mode=input_dict.get("recycle_mode", "last"),
                model_num=input_dict.get("model_num", 1),
                sample_models=input_dict.get("sample_models", False),
                use_multimer=input_dict.get("use_multimer", False),
                seed=input_dict["seed"],
                compute_gradient=input_dict.get("compute_gradient", True),
                device=input_dict["device"],
                verbose=input_dict["verbose"],
                include_pae_matrix=input_dict.get("include_pae_matrix", False),
            )
            for item in input_dict["items"]
        ]
        return {
            "results": [{key: result[key] for key in ("terms", "metrics", "pdb")} for result in results],
            "vocab": results[0]["vocab"],
        }

    raise ValueError(f"alphafold2: unknown operation {operation!r}; valid: ['predict', 'compute_design_gradient']")


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    global _model
    device_index = 0
    if _model is not None and ":" in _model.device:
        device_index = int(_model.device.split(":", 1)[1])
    return get_jax_memory_stats(device_index=device_index)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("alphafold2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
