"""proto_tools/tools/structure_prediction/esmfold2/esmfold2.py.

All-atom biomolecular complex structure prediction using ESMFold2 from Biohub.
"""

from logging import getLogger
from typing import Any, ClassVar, Literal

from pydantic import model_validator

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    normalize_output_chain_ids,
    resolve_chain_ids,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ESMFold2Input(StructurePredictionInput):
    """Input object for ESMFold2 all-atom complex structure prediction.

    Inherits from ``StructurePredictionInput``. Supports proteins, DNA, RNA, and
    small-molecule ligands (CCD codes or SMILES) in arbitrary combinations,
    optionally with chain-level modifications.

    Attributes:
        complexes (list[Complex]): List of biomolecular complexes to fold.
            Inherited from ``StructurePredictionInput``.
        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one
            entry per complex. Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by
            chain index); ``paired=True`` marks rows taxonomy-aligned across chains. Populated by ``Config.preprocess()`` or supplied
            directly. Only consumed when ``Config.model_checkpoint == "esmfold2"``.
            Default: ``None``.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class ESMFold2Metrics(Metrics):
    """Per-structure metrics emitted by ESMFold2 prediction.

    Metrics documented in ``metric_spec``:
        plddt (float): Mean per-residue predicted LDDT confidence. Always present.
        ptm (float): Predicted Template Modeling score. Always present.
        iptm (float): Interface predicted TM-score. Depends on complex composition
            (multi-chain only).
        avg_pae (float): Mean of the per-token PAE matrix in angstroms.
            Always present.
        pae (list[list[float]]): Full per-token PAE matrix in angstroms.
            Present when ``include_pae_matrix=True``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {
            "availability": "depends on complex composition",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "avg_pae": {"availability": "always", "type": "float", "min": 0.0, "max": 32.0, "better_values_are": "lower"},
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 32.0,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "plddt"


# Output:
class ESMFold2Output(StructurePredictionOutput):
    """ESMFold2 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying an
            :class:`ESMFold2Metrics` instance on ``.metrics``.
    """


# Config:
class ESMFold2Config(MSAStructurePredictionConfig):
    """Configuration for ESMFold2 structure prediction.

    Two checkpoints are exposed:

    - ``esmfold2-fast`` (default): inference-optimized single-sequence variant.
    - ``esmfold2``: MSA-capable larger variant; pass ``use_msa=True`` (or
      supply pre-computed ``msas`` on the input) to enable MSA conditioning.

    Inherits from ``MSAStructurePredictionConfig``. Combining
    ``model_checkpoint='esmfold2-fast'`` with ``use_msa=True`` raises in the
    validator since the fast variant is single-sequence.

    Attributes:
        model_checkpoint (Literal["esmfold2", "esmfold2-fast"]): Which ESMFold2
            variant to load. Default ``"esmfold2-fast"``.
        num_loops (int): Iterative refinement loops through the model. Higher =
            more accurate but slower. Default 3.
        num_sampling_steps (int): Diffusion sampling steps for the structure
            module. Higher = more refined but slower. Default 50.
        diffusion_samples (int): Independent diffusion samples per complex; the
            highest-pLDDT sample is returned. Higher = better quality but slower.
            Default 1.
        step_scale (float | None): Diffusion step size override (typical range
            1.0 to 2.0). Lower values produce more sample diversity. ``None``
            uses the upstream sampler default. Default ``None``.
        noise_scale (float | None): Diffusion noise scale override. ``None``
            uses the upstream sampler default. Default ``None``.
        max_inference_sigma (float | None): Maximum sigma value for the
            diffusion sampler. ``None`` uses the upstream default (256.0).
            Default ``None``.
        early_exit (bool): Exit refinement loops early when convergence is
            detected. Default ``False``.
        use_msa (bool): Whether to generate MSAs for protein chains via
            MMseqs2 homology search. Only valid with ``model_checkpoint='esmfold2'``.
            Supplied MSAs are always used and override ``use_msa=False``.
            Default ``False``.
        pair_heterocomplex_msas (bool): Whether heterocomplex protein chains
            should use taxonomy-paired MSA generation. Inherited. Default:
            ``True``.
        msa_search_config (Mmseqs2HomologySearchConfig | None): Configuration for
            MMseqs2 homology search (MSA generation). Only used when ``use_msa=True``.
            Inherited. Default: ``None``.
        device (str): Device to run the model on. Default ``"cuda"``. Inherited.
        include_pae_matrix (bool): Attach the full per-token PAE matrix to
            metrics (``avg_pae`` is always emitted). Default ``False``. Inherited.
        timeout (int | None): Maximum execution time in seconds. ``None`` waits
            indefinitely. Default 1200.
    """

    model_checkpoint: Literal["esmfold2", "esmfold2-fast"] = ConfigField(
        title="Model Checkpoint",
        default="esmfold2-fast",
        description="'esmfold2' is the larger MSA-capable variant; 'esmfold2-fast' is single-sequence and faster.",
        reload_on_change=True,
    )
    num_loops: int = ConfigField(
        title="Number of Refinement Loops",
        default=3,
        ge=1,
        description="Iterative refinement loops through the model. Higher = more accurate but slower.",
    )
    num_sampling_steps: int = ConfigField(
        title="Number of Sampling Steps",
        default=50,
        ge=1,
        description="Diffusion sampling steps for the structure module. Higher = more refined but slower.",
    )
    diffusion_samples: int = ConfigField(
        title="Number of Diffusion Samples",
        default=1,
        ge=1,
        description="Independent diffusion samples per complex; the highest-pLDDT sample is returned.",
    )
    step_scale: float | None = ConfigField(
        title="Step Scale",
        default=None,
        gt=0.0,
        description="Diffusion step size override (typical 1.0-2.0). None uses upstream default; lower = more diversity.",
    )
    noise_scale: float | None = ConfigField(
        title="Noise Scale",
        default=None,
        ge=0.0,
        description="Diffusion noise scale override. None uses the upstream sampler default.",
    )
    max_inference_sigma: float | None = ConfigField(
        title="Max Inference Sigma",
        default=None,
        gt=0.0,
        description="Maximum sigma value for the diffusion sampler. None uses the upstream default (256.0).",
    )
    early_exit: bool = ConfigField(
        title="Early Exit",
        default=False,
        description="Exit refinement loops early when convergence is detected.",
    )
    use_msa: bool = ConfigField(
        title="Use MSA",
        default=False,
        description="Generate MSAs via MMseqs2 for protein chains. Only valid with model_checkpoint='esmfold2'.",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds.",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def _validate_msa_compatibility(self) -> "ESMFold2Config":
        """Forbid MSA conditioning with the single-sequence ESMFold2-Fast checkpoint."""
        if self.model_checkpoint == "esmfold2-fast" and self.use_msa:
            raise ValueError(
                "model_checkpoint='esmfold2-fast' does not support MSA conditioning. "
                "Either set model_checkpoint='esmfold2' or use_msa=False."
            )
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for parametrized infra tests and examples."""
    return ESMFold2Input(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="esmfold2-prediction",
    label="ESMFold2 Structure Prediction",
    category="structure_prediction",
    input_class=ESMFold2Input,
    config_class=ESMFold2Config,
    output_class=ESMFold2Output,
    metrics_class=ESMFold2Metrics,
    description="All-atom biomolecular complex structure prediction using ESMFold2 from Biohub.",
    uses_gpu=True,
    device_count="1",
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
)
def run_esmfold2(inputs: ESMFold2Input, config: ESMFold2Config, instance: Any = None) -> ESMFold2Output:
    """Predict all-atom 3D structures of biomolecular complexes using ESMFold2.

    Uses ESMFold2 (Biohub) to fold proteins, DNA, RNA, small-molecule ligands,
    and their complexes. Runs locally via GPU in an isolated subprocess that
    shares the ``biohub_esm`` environment with ESM3 and ESM C.

    The default checkpoint ``esmfold2-fast`` is MSA-free and inference-optimized;
    switch to ``esmfold2`` and set ``use_msa=True`` (or supply pre-computed
    ``msas`` on the input) to use the larger MSA-conditioned variant for
    challenging targets.

    Args:
        inputs (ESMFold2Input): Validated input containing one or more
            biomolecular complexes to fold.
        config (ESMFold2Config): Validated ESMFold2 configuration including
            checkpoint selection, refinement loops, sampling steps, and MSA
            settings.
        instance (Any): Optional ``ToolInstance`` for persistent subprocess
            execution.

    Returns:
        ESMFold2Output: Output containing one ``Structure`` per input complex,
            each carrying an :class:`ESMFold2Metrics` instance on ``.metrics``.

    See Also:
        - ESMFold2 paper (Biohub, 2026): https://biohub.ai/papers/esm_protein.pdf
        - ESMFold2 model card: https://huggingface.co/biohub/ESMFold2
        - ESMFold2-Fast model card: https://huggingface.co/biohub/ESMFold2-Fast
        - upstream package: https://github.com/Biohub/esm

    Example:
        >>> inputs = ESMFold2Input(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
        >>> config = ESMFold2Config(num_loops=3, num_sampling_steps=50)
        >>> result = run_esmfold2(inputs, config)
        >>> print(f"plddt: {result.structures[0].metrics.plddt:.3f}")
    """
    base_seed = config.seed if config.seed is not None else config.get_random_int()
    results = [
        _run_esmfold2_on_complex(
            config=config,
            sp_complex=comp,
            complex_msas=inputs.msas[dispatch_idx] if inputs.msas else None,
            instance=instance,
            seed=base_seed + dispatch_idx,
        )
        for dispatch_idx, comp in enumerate(
            progress_bar(
                inputs.complexes,
                desc="Folding structures (ESMFold2)",
                unit="complex",
                total=len(inputs.complexes),
            )
        )
    ]
    return ESMFold2Output(structures=results)


def _run_esmfold2_on_complex(
    config: ESMFold2Config,
    sp_complex: Any,
    complex_msas: Any = None,
    instance: Any = None,
    seed: int | None = None,
) -> Structure:
    """Fold a single complex via the standalone ESMFold2 worker.

    Args:
        config (ESMFold2Config): Tool configuration.
        sp_complex (Any): The ``Complex`` to fold.
        complex_msas (Any): Pre-computed ``ComplexMSAs`` for this complex,
            keyed by chain index. When ``paired``, rows are row-aligned across
            chains by taxonomy and ``key=<row_idx>`` headers are synthesized to
            engage upstream pair-aware code. Ignored when
            ``config.model_checkpoint == 'esmfold2-fast'``.
        instance (Any): Optional ``ToolInstance`` for persistent subprocess
            execution.
        seed (int | None): Per-complex RNG seed forwarded to the worker.

    Returns:
        Structure: Predicted structure with ``ESMFold2Metrics`` attached.
    """
    from proto_tools.tools.structure_prediction.shared_data_models import unwrap_complex_msas

    if seed is None:
        seed = config.seed
    if config.verbose:
        logger.info("Using local GPU for ESMFold2 structure prediction...")

    # Serialize the complex into a JSON-safe payload for the worker.
    chain_ids = resolve_chain_ids(sp_complex.chains)
    chains_payload: list[dict[str, Any]] = [
        _chain_to_payload(chain, chain_id) for chain, chain_id in zip(sp_complex.chains, chain_ids, strict=True)
    ]

    per_chain_msas, unpaired_per_chain, is_paired = unwrap_complex_msas(complex_msas)

    # MSAs are only meaningful for the MSA-capable checkpoint.
    msa_payload: dict[str, list[str]] | None = None
    unpaired_payload: dict[str, list[str]] | None = None
    if config.model_checkpoint == "esmfold2" and per_chain_msas:
        msa_payload = _serialize_msas_for_worker(per_chain_msas)
        # Deep per-chain unpaired MSAs ride alongside the paired rows for block-diagonal depth.
        if unpaired_per_chain:
            unpaired_payload = _serialize_msas_for_worker(unpaired_per_chain)
    elif config.model_checkpoint == "esmfold2-fast" and per_chain_msas:
        logger.warning("ESMFold2-Fast is single-sequence; ignoring supplied MSAs.")

    input_data: dict[str, Any] = {
        "operation": "predict",
        "chains": chains_payload,
        "msas": msa_payload,
        "msas_paired": is_paired,
        "unpaired_msas": unpaired_payload,
        "model_checkpoint": config.model_checkpoint,
        "num_loops": config.num_loops,
        "num_sampling_steps": config.num_sampling_steps,
        "diffusion_samples": config.diffusion_samples,
        "step_scale": config.step_scale,
        "noise_scale": config.noise_scale,
        "max_inference_sigma": config.max_inference_sigma,
        "early_exit": config.early_exit,
        "include_pae_matrix": config.include_pae_matrix,
        "device": config.device,
        "verbose": config.verbose,
        "seed": seed,
    }

    output_data = ToolInstance.dispatch(
        "esmfold2",
        input_data,
        instance=instance,
        config=config,
    )

    cif_output = output_data["structure_cif_output"]
    formatted_metrics = output_data["metrics"]

    metrics_dict: dict[str, Any] = {
        "plddt": float(formatted_metrics["plddt"]),
        "ptm": float(formatted_metrics["ptm"]),
        "avg_pae": float(formatted_metrics["avg_pae"]),
    }
    if "iptm" in formatted_metrics and formatted_metrics["iptm"] is not None:
        metrics_dict["iptm"] = float(formatted_metrics["iptm"])
    if "pae" in formatted_metrics and formatted_metrics["pae"] is not None:
        metrics_dict["pae"] = formatted_metrics["pae"]

    structure = Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=ESMFold2Metrics(**metrics_dict),
        source="esmfold2-prediction",
    )
    return normalize_output_chain_ids(structure, sp_complex.chains)


def _chain_to_payload(chain: Any, chain_id: str) -> dict[str, Any]:
    """Serialize a Complex chain (Chain or Fragment) into a JSON-safe dict for the worker.

    ``chain_id`` is the resolved chain label (see ``resolve_chain_ids``) the
    worker uses for the upstream input and that downstream selection keys on.
    """
    from proto_tools.entities.ligands import Fragment

    if isinstance(chain, Fragment):
        # Prefer CCD when available; fall back to SMILES.
        payload: dict[str, Any] = {"entity_type": "ligand", "id": chain_id}
        if chain.ccd_code:
            payload["ccd_code"] = chain.ccd_code
        else:
            payload["smiles"] = chain.smiles
        return payload
    payload = {
        "id": chain_id,
        "entity_type": chain.entity_type,
        "sequence": chain.sequence,
    }
    # Optional chain-level modifications (1-indexed positions + CCD residue codes).
    mods = getattr(chain, "modifications", None)
    if mods:
        payload["modifications"] = [{"position": m.position, "ccd_code": m.modification_code} for m in mods]
    return payload


def _serialize_msas_for_worker(complex_msas: dict[int, Any]) -> dict[str, list[str]]:
    """Convert per-chain MSAs into row-ordered sequence lists keyed by chain index (as str)."""
    from proto_tools.utils import extract_msa_sequences

    serialized: dict[str, list[str]] = {}
    for chain_idx, msa in complex_msas.items():
        aligned, _ids = extract_msa_sequences(msa, query_index=0)
        serialized[str(chain_idx)] = list(aligned)
    return serialized
