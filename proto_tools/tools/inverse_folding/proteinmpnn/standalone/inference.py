"""ProteinMPNN standalone inference implementation for venv execution."""

import json
import os
import sys
from typing import Any

import numpy as np

# JAX memory settings - prevent preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from standalone_helpers import get_logger, log_likelihood_metrics
from standalone_helpers.relaxation import relaxed_sequence_jax, resolve_temperature
from standalone_helpers.vocabulary import AMINO_ACIDS_LIST, gather_indices

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 1.0
CANONICAL_VOCAB: list[str] = AMINO_ACIDS_LIST

PROTEINMPNN_AMINO_ACIDS_WITH_X: list[str] = list("ARNDCQEGHILKMFPSTWYVX")
PROTEINMPNN_AMINO_ACIDS: list[str] = PROTEINMPNN_AMINO_ACIDS_WITH_X[:20]
CANONICAL_TO_PROTEINMPNN_INDICES: list[int] = gather_indices(
    PROTEINMPNN_AMINO_ACIDS, CANONICAL_VOCAB, label="proteinmpnn"
)
PROTEINMPNN_TO_CANONICAL_INDICES: list[int] = gather_indices(
    CANONICAL_VOCAB, PROTEINMPNN_AMINO_ACIDS_WITH_X, label="proteinmpnn"
)

# Maps model_choice to ColabDesign's (model_name, weights) parameters.
# v_48_002/010/020/030 are the same architecture trained at different noise levels;
# v_48_020 is ColabDesign's default and is exposed via the "proteinmpnn" alias.
_MODEL_CONFIG: dict[str, tuple[str, str]] = {
    "proteinmpnn": ("v_48_020", "original"),
    "v_48_002": ("v_48_002", "original"),
    "v_48_010": ("v_48_010", "original"),
    "v_48_030": ("v_48_030", "original"),
    "abmpnn": ("abmpnn", "original"),
    "soluble": ("v_48_020", "soluble"),
}


def _effective_score_length(inputs: dict[str, Any]) -> float:
    mask = np.asarray(inputs["mask"], dtype=np.float32).copy()
    if "fix_pos" in inputs:
        mask[np.asarray(inputs["fix_pos"], dtype=int)] = 0.0
    return float(mask.sum())


class ProteinMPNNModel:
    """ProteinMPNN model for structure-conditioned protein sequence design."""

    def __init__(self) -> None:
        """Initialize ProteinMPNNModel."""
        self._loaded = False
        self._model_choice: str | None = None
        self._backbone_noise: float | None = None
        self.device: str | None = None
        self.params: Any = None
        self.model: Any = None

    def sample(
        self,
        pdb_path: str,
        chain_ids: list[str],
        batch_size: int,
        temperature: float | None = DEFAULT_TEMPERATURE,
        fixed_positions: dict[str, list[int]] | None = None,
        excluded_amino_acids: list[str] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        model_choice: str = "proteinmpnn",
        verbose: bool = False,
        return_logits: bool = False,
        backbone_noise: float = 0.0,
        homooligomer: bool = False,
    ) -> dict[str, Any]:
        """Sample protein sequences from the ProteinMPNN model.

        Args:
            pdb_path: Path to PDB file containing the structure.
            chain_ids: List of chain IDs to design
            batch_size: Number of sequences to generate
            temperature: Sampling temperature
            fixed_positions: Dict mapping chain IDs to fixed residue positions
            excluded_amino_acids: List of amino acids to exclude
            seed: Random seed
            device: Device to run on ('cuda' or 'cpu')
            model_choice: Model weights ('proteinmpnn', 'abmpnn', or 'soluble')
            verbose: Whether to print status messages
            return_logits: Whether to include logits in the output
            backbone_noise: Gaussian noise (A) added to backbone coordinates before each forward pass.
            homooligomer: Tie corresponding positions across equal-length designed chains.

        Returns:
            Dictionary with keys: seq, score, seqid, and optionally logits
        """
        from standalone_helpers import set_jax_seed

        key = set_jax_seed(seed)
        if key is None:
            raise ValueError("proteinmpnn: sample requires an explicit int seed (jax.random.PRNGKey rejects None)")

        self._ensure_loaded(device, model_choice, verbose, backbone_noise=backbone_noise)

        fix_pos = (
            ",".join(f"{chain}{idx}" for chain, positions in fixed_positions.items() for idx in positions)
            if fixed_positions is not None
            else None
        )

        # Load the PDB file
        self.model.prep_inputs(
            pdb_path,
            fix_pos=fix_pos,
            chain=",".join(chain_ids),
            homooligomer=homooligomer,
            rm_aa=",".join(excluded_amino_acids) if excluded_amino_acids else None,
        )

        # Sample sequences
        sequences = self.model.sample_parallel(
            batch=batch_size,
            temperature=temperature,
            key=key,
        )

        self.unload()
        return {
            "seq": sequences["seq"],
            "score": sequences["score"],
            "seqid": sequences["seqid"],
            "logits": sequences["logits"][..., PROTEINMPNN_TO_CANONICAL_INDICES] if return_logits else None,
        }

    def score(
        self,
        pdb_path: str,
        chain_ids: list[str],
        sequence: str,
        fixed_positions: dict[str, list[int]] | None = None,
        seed: int | None = None,
        device: str = "cuda",
        model_choice: str = "proteinmpnn",
        verbose: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Score a protein sequence against a structure.

        Args:
            pdb_path: Path to PDB file containing the structure.
            chain_ids: List of chain IDs
            sequence: Sequence to score
            fixed_positions: Dict mapping chain IDs to fixed positions
            seed: Random seed (required — jax.random.PRNGKey rejects None)
            device: Device to run on
            model_choice: Model weights ('proteinmpnn', 'abmpnn', or 'soluble')
            verbose: Whether to print status messages
            return_logits: Whether to include logits in the output

        Returns:
            Dictionary with keys: metrics (log_likelihood, avg_log_likelihood, perplexity),
            and optionally logits.
        """
        from standalone_helpers import set_jax_seed

        key = set_jax_seed(seed)
        if key is None:
            raise ValueError("proteinmpnn: score requires an explicit int seed (jax.random.PRNGKey rejects None)")

        self._ensure_loaded(device, model_choice, verbose, backbone_noise=0.0)

        fix_pos = (
            ",".join(f"{chain}{idx}" for chain, positions in fixed_positions.items() for idx in positions)
            if fixed_positions is not None
            else None
        )

        # Prepare input
        self.model.prep_inputs(
            pdb_path,
            fix_pos=fix_pos,
            chain=",".join(chain_ids),
        )

        parsed_len = int(self.model._inputs["S"].shape[0])
        if len(sequence) != parsed_len:
            raise ValueError(f"Sequence length {len(sequence)} does not match structure ({parsed_len} residues).")

        # Score the sequence (model returns "score" (negative avg log likelihood), "logits")
        output = self.model.score(
            seq=sequence,
            key=key,
        )

        neg_avg_ll = float(output["score"])
        effective_length = _effective_score_length(self.model._inputs)
        if effective_length == 0:
            raise ValueError("proteinmpnn: no residues available to score")

        self.unload()
        return {
            "logits": output["logits"][..., PROTEINMPNN_TO_CANONICAL_INDICES] if return_logits else None,
            "metrics": log_likelihood_metrics(-neg_avg_ll, effective_length),
            "vocab": CANONICAL_VOCAB,
        }

    def compute_gradient(
        self,
        pdb_path: str,
        chain_ids: list[str],
        logits_list: list[list[float]],
        *,
        temperature: float | None = None,
        soft: float = 1.0,
        hard: float = 0.0,
        fixed_positions: dict[str, list[int]] | None = None,
        seed: int | None = None,
        seed_index: int = 0,
        device: str = "cuda",
        model_choice: str = "proteinmpnn",
        verbose: bool = False,
        backprop: bool = True,
    ) -> dict[str, Any]:
        """Compute ProteinMPNN mean-NLL gradient for relaxed sequence logits.

        The public gradient contract uses canonical amino-acid order
        ``ACDEFGHIKLMNPQRSTVWY``. ColabDesign's ProteinMPNN wrapper scores in
        AlphaFold order internally, so this method maps both the relaxed context
        and returned gradient at the boundary.

        ``_score`` draws its autoregressive decoding order from the key it is given, so every
        candidate needs its own: sharing one key scores a whole batch in a single order, and
        identical candidates then come back identical. ``seed_index`` folds the candidate's
        position into the key, leaving index 0 on the base key so a lone candidate and the first
        of a batch agree.
        """
        if not logits_list:
            raise ValueError("proteinmpnn: compute_gradient requires at least one residue")
        if any(len(row) != len(CANONICAL_VOCAB) for row in logits_list):
            raise ValueError(f"proteinmpnn: compute_gradient expects L x {len(CANONICAL_VOCAB)} logits")

        import jax
        import jax.numpy as jnp
        from colabdesign.shared.utils import copy_dict
        from standalone_helpers import set_jax_seed

        key = set_jax_seed(seed)
        if key is None:
            raise ValueError(
                "proteinmpnn: compute_gradient requires an explicit int seed (jax.random.PRNGKey rejects None)"
            )
        if seed_index:
            key = jax.random.fold_in(key, seed_index)

        self._ensure_loaded(device, model_choice, verbose, backbone_noise=0.0)

        fix_pos = (
            ",".join(f"{chain}{idx}" for chain, positions in fixed_positions.items() for idx in positions)
            if fixed_positions is not None
            else None
        )
        self.model.prep_inputs(
            pdb_path,
            fix_pos=fix_pos,
            chain=",".join(chain_ids),
        )

        parsed_len = int(self.model._inputs["S"].shape[0])
        if len(logits_list) != parsed_len:
            raise ValueError(f"Logits length {len(logits_list)} does not match structure ({parsed_len} residues).")

        base_inputs = copy_dict(self.model._inputs)
        base_inputs.pop("S", None)
        raw_logits = jnp.asarray(logits_list, dtype=jnp.float32)
        canonical_to_af = jnp.asarray(CANONICAL_TO_PROTEINMPNN_INDICES, dtype=jnp.int32)

        def _loss_fn(logits: Any) -> Any:
            context_canonical = relaxed_sequence_jax(logits, temperature=temperature, soft=soft, hard=hard)
            # NLL labels are the decoded sequence, not the blend; softmax is monotonic, so argmax(logits) suffices.
            labels_canonical = jax.nn.one_hot(jnp.argmax(logits, axis=-1), len(CANONICAL_VOCAB))

            context_af = context_canonical[:, canonical_to_af]
            labels_af = labels_canonical[:, canonical_to_af]

            output = self.model._score(**base_inputs, key=key, S=context_af)
            log_probs = jax.nn.log_softmax(output["logits"], axis=-1)[..., :20]
            per_position_nll = -(labels_af * log_probs).sum(axis=-1)

            mask = jnp.asarray(base_inputs["mask"], dtype=per_position_nll.dtype)
            if "fix_pos" in base_inputs:
                mask = mask.at[jnp.asarray(base_inputs["fix_pos"])].set(0.0)
            return (per_position_nll * mask).sum() / (mask.sum() + 1e-8)

        if backprop:
            loss_value, gradient = jax.value_and_grad(_loss_fn)(raw_logits)
            gradient_value: list[list[float]] | None = np.asarray(gradient).tolist()
        else:
            loss_value = _loss_fn(raw_logits)
            gradient_value = None

        mean_nll = float(loss_value)
        effective_length = _effective_score_length(base_inputs)

        self.unload()
        return {
            "gradient": gradient_value,
            "loss": mean_nll,
            "metrics": {
                **log_likelihood_metrics(-mean_nll, effective_length),
                "sequence_length": parsed_len,
                "effective_sequence_length": effective_length,
                "model_choice": model_choice,
                "objective": "autoregressive_nll",
            },
            "vocab": CANONICAL_VOCAB,
        }

    def _ensure_loaded(
        self,
        device: str,
        model_choice: str,
        verbose: bool,
        *,
        backbone_noise: float,
    ) -> None:
        if not self._loaded or self._model_choice != model_choice or self._backbone_noise != backbone_noise:
            self.load(device, model_choice, verbose, backbone_noise=backbone_noise)
        elif self.device != device:
            self.to_device(device)

    def load(
        self,
        device: str,
        model_choice: str = "proteinmpnn",
        verbose: bool = False,
        backbone_noise: float = 0.0,
    ) -> None:
        """Load ProteinMPNN model to device.

        Args:
            device: Device to load the model on.
            model_choice: Model weights ('proteinmpnn', 'abmpnn', or 'soluble').
            verbose: Whether to print status messages.
            backbone_noise: Gaussian noise (A) baked into ColabDesign's model config.
        """
        self.verbose = verbose
        model_name, weights = _MODEL_CONFIG.get(model_choice, ("v_48_020", "original"))

        if self.verbose:
            logger.info(f"Loading {model_choice} (model_name={model_name}, weights={weights}) on {device}")

        # Lazy import ProteinMPNN from ColabDesign
        from colabdesign.mpnn import mk_mpnn_model

        # Load the Flax module (params land on CPU by default)
        self.model = mk_mpnn_model(model_name=model_name, weights=weights, backbone_noise=backbone_noise)
        self.params = self.model._model.params
        self.device = "cpu"
        self._model_choice = model_choice
        self._backbone_noise = backbone_noise

        # Move the model parameters to the selected device
        self.to_device(device)
        self._loaded = True

        if self.verbose:
            logger.info(f"{model_choice} model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move the model params to the selected device via move_model_to_device."""
        from standalone_helpers import move_model_to_device

        if self.model is None:
            raise ValueError("proteinmpnn: cannot move unloaded model to device — call load() first")
        if self.device == device:
            return

        if self.verbose:
            logger.info(f"Moving ProteinMPNN to {device}")

        # params is a dict pytree; move_model_to_device handles via device_put
        self.params = move_model_to_device(self.params, self.device, device)
        self.model._model.params = self.params
        self.device = device

    def unload(self) -> None:
        """Move ProteinMPNN params back to CPU and free GPU HBM."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            return

        if self.verbose:
            logger.info("Unloading ProteinMPNN to CPU")

        self.params = move_model_to_device(self.params, self.device, "cpu")
        self.model._model.params = self.params
        self.device = "cpu"


# ============================================================================
# Dispatch
# ============================================================================
_model: ProteinMPNNModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ProteinMPNNModel()

    operation = input_dict["operation"]
    # The gradient operation carries one backbone per item; sample and score take a single one.
    pdb_path = input_dict.get("pdb_path", "")
    model_choice = input_dict["model_choice"]
    if operation == "sample":
        return _model.sample(
            pdb_path=pdb_path,
            chain_ids=input_dict["chain_ids"],
            batch_size=input_dict["batch_size"],
            temperature=input_dict["temperature"],
            fixed_positions=input_dict.get("fixed_positions"),
            excluded_amino_acids=input_dict.get("excluded_amino_acids"),
            seed=input_dict["seed"],
            device=input_dict["device"],
            model_choice=model_choice,
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
            backbone_noise=input_dict["backbone_noise"],
            homooligomer=input_dict["homooligomer"],
        )
    if operation == "score":
        return _model.score(
            pdb_path=pdb_path,
            chain_ids=input_dict["chain_ids"],
            sequence=input_dict["sequence"],
            fixed_positions=input_dict.get("fixed_positions"),
            seed=input_dict["seed"],
            device=input_dict["device"],
            model_choice=model_choice,
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
        )
    if operation == "compute_gradient":
        # One chunk of independent candidates: the weights load once and prep_inputs re-runs per item.
        results = [
            _model.compute_gradient(
                pdb_path=item["pdb_path"],
                chain_ids=item["chain_ids"],
                logits_list=item["logits"],
                temperature=resolve_temperature(input_dict["temperature"]),
                soft=float(input_dict["soft"]),
                hard=float(input_dict["hard"]),
                fixed_positions=item.get("fixed_positions"),
                seed=input_dict["seed"],
                seed_index=index,
                device=input_dict["device"],
                model_choice=model_choice,
                verbose=input_dict["verbose"],
                backprop=input_dict.get("compute_gradient", True),
            )
            for index, item in enumerate(input_dict["items"])
        ]
        return {
            "results": [{key: result[key] for key in ("gradient", "loss", "metrics")} for result in results],
            "vocab": results[0]["vocab"],
        }
    raise ValueError(f"proteinmpnn: unknown operation {operation!r}; valid: ['sample', 'score', 'compute_gradient']")


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_jax_memory_stats

    return get_jax_memory_stats(device_index=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("proteinmpnn: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    from standalone_helpers import serialize_output

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
