"""AlphaFold2 standalone inference — prediction and binder-design gradients.

Runs in an isolated venv with JAX and ColabDesign. Supports two backends:

- ``"base"``: upstream sokrypton/ColabDesign (pip-installed, used for prediction
  and generic gradient computation).
- ``"germinal"``: SantiagoMille/germinal ColabDesign fork (installed to
  data/colabdesign_germinal, loaded via sys.path swap). Adds alpha=2.0 logit
  scaling, persistent bias_redesign, framework contact penalties in
  ``_loss_binder``, and ``soft_seq()`` with full JAX autograd chain rule.
  Antibody-language-model gradients remain external to this tool.

Extension loss callbacks (rg, i_ptm, helix, beta_strand, NC) are ported from
``germinal/design/design.py`` and registered when their weight is > 0 in the
``loss_weights`` dict. These are JAX functions that run inside
``jax.value_and_grad`` and only activate with ``backend="germinal"``.

Usage:
    python inference.py <input_json_path> <output_json_path>
"""

import copy
import gc
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from standalone_helpers import AMINO_ACIDS_LIST, get_jax_memory_stats, resolve_jax_device, serialize_output

logger = logging.getLogger(__name__)

# AlphaFold restypes order (3-letter codes sorted alphabetically: ALA, ARG, ASN, ...).
# Differs from AMINO_ACIDS_LIST which is sorted by 1-letter code (A, C, D, ...).
AF2_RESTYPES = list("ARNDCQEGHILKMFPSTWYV")
PROTO_TO_AF2 = [AMINO_ACIDS_LIST.index(aa) for aa in AF2_RESTYPES]
AF2_TO_PROTO = [AF2_RESTYPES.index(aa) for aa in AMINO_ACIDS_LIST]

# Configure stderr handler so debug output is captured by persistent worker
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

_params_dir: str | None = None


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
            raise RuntimeError("TOOL_VENV_PATH not set. AlphaFold2 must be run via ToolInstance.")
        params_dir = Path(venv_path) / "data" / "params"

    if not params_dir.exists() or not any(params_dir.glob("*.npz")):
        raise RuntimeError(
            f"AlphaFold2 parameters not found at {params_dir}. "
            "Run the standalone setup.sh script to download parameters."
        )

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

    metrics["pae"] = serialize_output(aux["pae"]) if include_pae_matrix else None

    for key, value in aux.get("losses", {}).items():
        scalar = _metric_value(value)
        if scalar is not None:
            metrics[key] = scalar

    return metrics


# ---------------------------------------------------------------------------
# Germinal loss callbacks (ported from germinal/germinal/design/design.py)
#
# These are JAX functions appended to af_model._callbacks["model"]["loss"].
# They run inside jax.value_and_grad and must stay in the standalone venv.
# ---------------------------------------------------------------------------


def _add_rg_loss(af_model: Any, weight: float = 0.1) -> None:
    """Radius-of-gyration loss (germinal/design.py:467)."""
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants

    def loss_fn(_inputs: Any, outputs: Any) -> dict[str, Any]:
        ca = outputs["structure_module"]["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-af_model._binder_len :]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        return {"rg": jax.nn.elu(rg - rg_th)}

    af_model._callbacks["model"]["loss"].append(loss_fn)
    af_model.opt["weights"]["rg"] = weight


def _add_i_ptm_loss(af_model: Any, weight: float = 0.1) -> None:
    """Interface pTM loss (germinal/design.py:497)."""
    from colabdesign.af.loss import get_ptm

    def loss_fn(inputs: Any, outputs: Any) -> dict[str, Any]:
        return {"i_ptm": 1 - get_ptm(inputs, outputs, interface=True)}

    af_model._callbacks["model"]["loss"].append(loss_fn)
    af_model.opt["weights"]["i_ptm"] = weight


def _add_helix_loss(af_model: Any, weight: float = 0.1) -> None:
    """Alpha-helix loss — i,i+3 contacts in 2.0-6.2 A (germinal/design.py:522)."""
    import jax
    import jax.numpy as jnp
    from colabdesign.af.loss import get_dgram_bins

    def loss_fn(inputs: Any, outputs: Any) -> dict[str, Any]:
        offset = (
            inputs["offset"]
            if "offset" in inputs
            else (lambda idx: idx[:, None] - idx[None, :])(inputs["residue_index"].flatten())
        )
        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        bins = jnp.logical_or(dgram_bins > 6.2, dgram_bins < 2.0)
        x = -jnp.log((bins * jax.nn.softmax(dgram) + 1e-8).sum(-1))

        if "pos" in af_model.opt:
            mask_1d = jnp.zeros(af_model._target_len + af_model._binder_len)
            mask_1d = mask_1d.at[jnp.asarray(af_model.opt["pos"])].set(1)
        else:
            mask_1d = jnp.concatenate([jnp.zeros(af_model._target_len), jnp.ones(af_model._binder_len)])
        mask_2d = jnp.outer(mask_1d, mask_1d)
        mask = jnp.where(mask_2d, offset == 3, 0)
        return {"helix": jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)}

    af_model._callbacks["model"]["loss"].append(loss_fn)
    af_model.opt["weights"]["helix"] = weight


def _add_beta_strand_loss(af_model: Any, weight: float = 0.1) -> None:
    """Beta-strand loss — 9.75-11.5 A pairs, top-k (germinal/design.py:602)."""
    import jax
    import jax.numpy as jnp
    from colabdesign.af.loss import get_dgram_bins

    total_len = af_model._target_len + af_model._binder_len
    all_pos = jnp.asarray(af_model.opt.get("pos", []))
    valid_pos = all_pos[(all_pos > 0) & (all_pos < total_len - 1)]

    def loss_fn(_inputs: Any, outputs: Any) -> dict[str, Any]:
        if valid_pos.size == 0:
            return {"beta_strand": jnp.array(0.0)}

        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        bins = jnp.logical_or(dgram_bins > 11.5, dgram_bins < 9.75)
        x = -jnp.log((bins * jax.nn.softmax(dgram) + 1e-8).sum(-1))

        mask_1d = jnp.zeros(total_len).at[valid_pos - 1].set(1).at[valid_pos + 1].set(1)
        mask_2d = jnp.outer(mask_1d, mask_1d)
        loss_array = jnp.sort(jnp.diagonal(mask_2d * x, 3), descending=True)
        top_k_mask = jnp.arange(loss_array.size) < (valid_pos.size // 3)
        return {"beta_strand": jnp.where(top_k_mask, loss_array, 0).sum(-1) / (top_k_mask.sum() + 1e-8)}

    af_model._callbacks["model"]["loss"].append(loss_fn)
    af_model.opt["weights"]["beta_strand"] = weight


def _add_termini_distance_loss(af_model: Any, weight: float = 0.1, threshold: float = 7.0) -> None:
    """N-C termini distance loss (germinal/design.py:781)."""
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants

    def loss_fn(_inputs: Any, outputs: Any) -> dict[str, Any]:
        ca = outputs["structure_module"]["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-af_model._binder_len :]
        dist = jnp.linalg.norm(ca[0] - ca[-1])
        return {"NC": jax.nn.relu(jax.nn.elu(dist - threshold))}

    af_model._callbacks["model"]["loss"].append(loss_fn)
    af_model.opt["weights"]["NC"] = weight


def _configure_germinal_losses(af_model: Any, loss_weights: dict[str, float]) -> None:
    """Register Germinal loss callbacks based on the provided weight dict.

    Stock ColabDesign losses (plddt, i_plddt, pae, i_pae, con, i_con, etc.)
    are configured via ``af_model.set_weights()``. This function handles the
    Germinal-specific extension losses that require explicit callbacks.
    """
    if loss_weights.get("rg", 0.0) > 0:
        _add_rg_loss(af_model, loss_weights["rg"])
    if loss_weights.get("i_ptm", 0.0) > 0:
        _add_i_ptm_loss(af_model, loss_weights["i_ptm"])
    if loss_weights.get("helix", 0.0) > 0:
        _add_helix_loss(af_model, loss_weights["helix"])
    if loss_weights.get("beta_strand", 0.0) > 0:
        _add_beta_strand_loss(af_model, loss_weights["beta_strand"])
    if loss_weights.get("NC", 0.0) > 0:
        _add_termini_distance_loss(af_model, loss_weights["NC"])


class AlphaFold2Model:
    """Persistent ColabDesign-backed AlphaFold2 worker."""

    def __init__(self) -> None:
        """Initialize persistent worker state."""
        self._loaded = False
        self._backend: str = "base"
        self.device = "cuda"
        self.params_dir: str | None = None
        self._jax: Any | None = None
        self._jnp: Any | None = None
        self._mk_af_model: Any | None = None
        self._models: dict[tuple[str, bool, bool, str, str], Any] = {}
        self._model_defaults: dict[tuple[str, bool, bool, str, str], dict[str, Any]] = {}

    def load(self, device: str, backend: str = "base", verbose: bool = False) -> None:
        """Load ColabDesign entrypoints and resolve parameter locations.

        Args:
            device: JAX device string ("cuda", "cpu", "cuda:0", etc.).
            backend: Which ColabDesign package to use:
                ``"base"`` — upstream sokrypton/ColabDesign (default for prediction).
                ``"germinal"`` — Germinal fork with alpha=2.0, bias, and framework contacts.
            verbose: Log loading details.
        """
        if self._loaded and self._backend == backend:
            return
        if self._loaded and self._backend != backend:
            # Backend switch: clear cached models and JAX JIT cache, then reload
            self._models.clear()
            self._model_defaults.clear()
            if self._jax is not None:
                self._jax.clear_caches()
            self._loaded = False

        # Configure JAX device hints before the first import
        if device == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
        elif device.startswith("cuda") and ":" in device:
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]

        germinal_dir = os.path.join(
            os.environ.get("TOOL_VENV_PATH", os.environ.get("VIRTUAL_ENV", "")),
            "data",
            "colabdesign_germinal",
        )
        if backend == "germinal":
            if not os.path.isdir(germinal_dir):
                raise RuntimeError(f"Germinal ColabDesign not found at {germinal_dir}. Run setup.sh.")
            if germinal_dir not in sys.path:
                sys.path.insert(0, germinal_dir)
                self._purge_colabdesign_modules()
            logger.info("Using Germinal ColabDesign backend from %s", germinal_dir)
        elif germinal_dir in sys.path:
            sys.path.remove(germinal_dir)
            self._purge_colabdesign_modules()

        import jax
        import jax.numpy as jnp
        from colabdesign import mk_afdesign_model as mk_af_model

        self._jax = jax
        self._jnp = jnp
        self._mk_af_model = mk_af_model
        self.params_dir = _resolve_params_dir()
        self.device = device
        self._backend = backend
        self._loaded = True

        if verbose:
            logger.info("AlphaFold2 ColabDesign runtime loaded on %s (backend=%s)", device, backend)

    @staticmethod
    def _purge_colabdesign_modules() -> None:
        """Remove cached colabdesign modules so the next import picks up the correct backend."""
        for mod_name in list(sys.modules):
            if mod_name == "colabdesign" or mod_name.startswith("colabdesign."):
                del sys.modules[mod_name]

    def _ensure_loaded(self, device: str, backend: str = "base", verbose: bool = False) -> None:
        if not self._loaded or self._backend != backend:
            self.load(device, backend=backend, verbose=verbose)
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
        key = (protocol, use_multimer, use_msa, recycle_mode, self._backend)
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

    def _restore_model_defaults(self, key: tuple[str, bool, bool, str, str], model: Any) -> None:
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
    ) -> None:
        """Prepare ColabDesign inputs for a prediction request."""
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
        device: str = "cuda",
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run AlphaFold2 prediction on one complex."""
        self._ensure_loaded(device, backend="base", verbose=verbose)

        use_multimer = complex_data["num_chains"] > 1
        use_msa = msa_a3m_content is not None
        af_model = self._get_model(use_multimer=use_multimer, use_msa=use_msa, verbose=verbose)

        self._prep_prediction_inputs(af_model, complex_data, msa_a3m_content)

        predict_kwargs: dict[str, Any] = {
            "num_recycles": num_recycles,
            "seed": seed,
            "verbose": verbose,
        }
        if num_ensemble_models > 1:
            predict_kwargs["num_models"] = num_ensemble_models
        else:
            predict_kwargs["models"] = [model_num - 1]

        af_model.predict(**predict_kwargs)

        aux = getattr(af_model, "aux", {})
        metrics = _extract_metrics(aux, include_pae_matrix=include_pae_matrix)
        return {
            "pdb": af_model.save_pdb(),
            "avg_plddt": metrics["avg_plddt"],
            "ptm": metrics["ptm"],
            "iptm": metrics.get("iptm"),
            "avg_pae": metrics["avg_pae"],
            "pae": metrics["pae"],
        }

    # -------------------------------------------------------------------------
    # Binder gradient computation (base or Germinal backend)
    # -------------------------------------------------------------------------

    def _run_gradient(
        self,
        af_model: Any,
        *,
        logits: list[list[float]],
        temperature: float,
        soft: float,
        hard: float,
        num_recycles: int,
        model_num: int,
        sample_models: bool,
        loss_weights: dict[str, float] | None,
        seed: int | None,
        backprop: bool = True,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run one binder-design AF2 pass and return loss, metrics, Structure, and optionally gradient.

        Germinal's ``soft_seq()`` handles alpha=2.0 scaling and persistent bias
        internally. When ``backprop=True``, JAX autograd chain-rules through the
        full expression and the returned gradient is exact ``∂loss/∂logits``.
        When ``backprop=False``, the forward pass runs through ``model["fn"]``
        (no grad_fn) and ``gradient`` is ``None`` in the returned dict.
        """
        import numpy as np

        jnp = self._jnp
        assert jnp is not None

        opt = af_model.opt
        opt["num_recycles"] = num_recycles
        opt["temp"] = temperature
        opt["soft"] = soft
        opt["hard"] = hard
        if seed is not None:
            af_model.set_seed(seed)
        if loss_weights:
            af_model.set_weights(**loss_weights)

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
        af_model.run(backprop=backprop, model_nums=model_nums)
        aux = af_model.aux

        gradient: list[list[float]] | None = None
        if backprop:
            # Extract binder gradient, reordering back to proto canonical order.
            # Shape (batch, L, alphabet) expected; fail loudly if ColabDesign's layout shifts.
            assert aux["grad"]["seq"].ndim == 3, f"unexpected grad shape {aux['grad']['seq'].shape}"
            full_grad = aux["grad"]["seq"][0][:, AF2_TO_PROTO]
            seq_grad = full_grad[full_len - binder_len :]
            gradient = serialize_output(seq_grad)

        return {
            "gradient": gradient,
            "loss": float(serialize_output(aux["loss"])),
            "metrics": _extract_metrics(aux, include_pae_matrix=include_pae_matrix),
            "vocab": AMINO_ACIDS_LIST,
            "pdb": af_model.save_pdb(get_best=False),
        }

    def compute_binder_gradient(
        self,
        *,
        logits: list[list[float]],
        temperature: float = 1.0,
        soft: float = 1.0,
        hard: float = 0.0,
        target_pdb: str,
        target_chain: str = "A",
        target_hotspot: str | None = None,
        binder_chain: str,
        design_positions: list[int] | None = None,
        bias_redesign: float | None = None,
        omit_aas: str | None = None,
        num_recycles: int = 3,
        recycle_mode: str = "last",
        model_num: int = 1,
        sample_models: bool = False,
        starting_binder_seq: str | None = None,
        loss_weights: dict[str, float] | None = None,
        intra_contact_num: int = 2,
        intra_contact_cutoff: float = 14.0,
        inter_contact_num: int = 10,
        inter_contact_cutoff: float = 20.0,
        framework_contact_offset: float = 1.0,
        seed: int | None = None,
        backend: str = "base",
        compute_gradient: bool = True,
        device: str = "cuda",
        verbose: bool = False,
        include_pae_matrix: bool = False,
    ) -> dict[str, Any]:
        """Run AF2 binder design against a frozen target (forward, optionally backward).

        When ``backend="germinal"``, uses Germinal's ColabDesign fork for the full setup:
        - ``prep_inputs()`` handles position resolution, bias initialization,
          template masking, and framework contact configuration.
        - ``soft_seq()`` applies alpha=2.0 scaling and persistent bias.
        - Extension losses (rg, i_ptm, helix, beta_strand, NC) are registered
          as callbacks based on ``loss_weights``.
        """
        if target_pdb is None:
            raise ValueError("target_pdb is required for binder gradient computation.")

        self._ensure_loaded(device, backend=backend, verbose=verbose)
        key = ("binder", True, False, recycle_mode, self._backend)
        af_model = self._get_model(
            protocol="binder",
            use_multimer=True,
            use_msa=False,
            recycle_mode=recycle_mode,
            verbose=verbose,
        )
        self._restore_model_defaults(key, af_model)

        # Germinal's prep_inputs handles position resolution, bias
        # initialization, template setup.
        prep_kwargs: dict[str, Any] = {
            "pdb_filename": target_pdb,
            "target_chain": target_chain,
            "binder_len": len(logits),
            "binder_chain": binder_chain,
            "rm_target_seq": True,
            "rm_target_sc": False,
            "rm_binder": False,
            "rm_binder_seq": True,
            "rm_binder_sc": True,
            "rm_template_ic": True,
        }
        if target_hotspot:
            prep_kwargs["hotspot"] = target_hotspot
        if omit_aas:
            prep_kwargs["rm_aa"] = omit_aas

        # Germinal-specific prep_inputs kwargs (bias_redesign, pos,
        # starting_binder_seq) are only supported by the Germinal ColabDesign fork.
        if backend == "germinal":
            prep_kwargs["pos"] = design_positions if design_positions is not None else list(range(len(logits)))
            if bias_redesign is not None:
                prep_kwargs["bias_redesign"] = bias_redesign
            # Germinal's prep_inputs bootstraps an antibody LM before _prep_model();
            # pass hidden defaults so the fork initializes cleanly (we only use run()).
            prep_kwargs["ablm_model"] = "ablang"
            prep_kwargs["ablm_temp"] = 0.6
            prep_kwargs["lens"] = {"fw": [0, 0, 0, 0], "cdrs": [0, 0, 0]}
            if starting_binder_seq is not None:
                if len(starting_binder_seq) != len(logits):
                    raise ValueError(f"starting_binder_seq len {len(starting_binder_seq)} != logits len {len(logits)}")
                prep_kwargs["starting_binder_seq"] = starting_binder_seq

        af_model.prep_inputs(**prep_kwargs)

        if backend == "germinal":
            # Contact parameter configuration (germinal/design.py:191-208).
            af_model.opt["con"].update(
                {"num": intra_contact_num, "cutoff": intra_contact_cutoff, "binary": False, "seqsep": 9}
            )
            af_model.opt["i_con"].update(
                {
                    "num": inter_contact_num,
                    "cutoff": inter_contact_cutoff,
                    "binary": False,
                    "framework_contact_loss": True,
                    "framework_contact_offset": framework_contact_offset,
                }
            )
            # Register Germinal extension loss callbacks.
            _configure_germinal_losses(af_model, loss_weights or {})

        return self._run_gradient(
            af_model,
            logits=logits,
            temperature=temperature,
            soft=soft,
            hard=hard,
            num_recycles=num_recycles,
            model_num=model_num,
            sample_models=sample_models,
            loss_weights=loss_weights,
            seed=seed,
            backprop=compute_gradient,
            include_pae_matrix=include_pae_matrix,
        )

    def to_device(self, device: str, verbose: bool = False) -> None:
        """Switch target device for future model cache entries."""
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
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
        )
    if operation == "compute_gradient":
        return _model.compute_binder_gradient(
            logits=input_dict["logits"],
            temperature=input_dict["temperature"],
            soft=input_dict.get("soft", 1.0),
            hard=input_dict.get("hard", 0.0),
            target_pdb=input_dict["target_pdb"],
            target_chain=input_dict["target_chain"],
            target_hotspot=input_dict.get("target_hotspot"),
            binder_chain=input_dict["binder_chain"],
            design_positions=input_dict.get("design_positions"),
            bias_redesign=input_dict.get("bias_redesign"),
            omit_aas=input_dict.get("omit_aas"),
            num_recycles=input_dict["num_recycles"],
            recycle_mode=input_dict.get("recycle_mode", "last"),
            model_num=input_dict["model_num"],
            sample_models=input_dict.get("sample_models", False),
            starting_binder_seq=input_dict.get("starting_binder_seq"),
            loss_weights=input_dict.get("loss_weights"),
            intra_contact_num=input_dict.get("intra_contact_num", 2),
            intra_contact_cutoff=input_dict.get("intra_contact_cutoff", 14.0),
            inter_contact_num=input_dict.get("inter_contact_num", 10),
            inter_contact_cutoff=input_dict.get("inter_contact_cutoff", 20.0),
            framework_contact_offset=input_dict.get("framework_contact_offset", 1.0),
            seed=input_dict["seed"],
            backend=input_dict.get("backend", "base"),
            compute_gradient=input_dict.get("compute_gradient", True),
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
        )
    raise ValueError(f"Unknown operation: {operation}")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    global _model
    device_index = 0
    if _model is not None and ":" in _model.device:
        device_index = int(_model.device.split(":", 1)[1])
    return get_jax_memory_stats(device_index=device_index)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
