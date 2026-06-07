"""proto_tools/tools/structure_prediction/rf3/rf3_prediction.py.

All-atom structure prediction with RoseTTAFold3 (RF3).
"""

import os
import tempfile
from logging import getLogger
from typing import Any, ClassVar

from pydantic import model_validator

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.rf3.helpers import build_chain_a3m_paths, complex_to_rf3_json
from proto_tools.tools.structure_prediction.shared_data_models import (
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    normalize_output_chain_ids,
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
class RF3Input(StructurePredictionInput):
    """Input object for RoseTTAFold3 structure prediction.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[Complex]): List of complexes to predict structures for.
            Inherited from ``StructurePredictionInput``. Each complex can contain
            multiple chains of proteins, DNA, RNA, and/or ligands.
        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one entry per complex.
            Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by chain index);
            ``paired=True`` marks rows taxonomy-aligned across chains. Populated
            by preprocess() or supplied directly. Default: None.

    Note:
        RF3 supports entity types: ``"protein"``, ``"dna"``, ``"rna"``, and
        ``"ligand"``. Chain modifications (PTMs) are not currently supported by
        this wrapper; upstream RF3 accepts them via inline ``"MTG(PTM)..."``
        syntax, which the proto-tools serializer does not yet emit.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False


# Metrics:
class RF3Metrics(Metrics):
    """Per-structure metrics emitted by RF3 prediction.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Overall predicted LDDT in [0, 1]. Always present.
        ptm (float): Predicted TM-score in [0, 1]. Always present.
        iptm (float): Interface predicted TM-score in [0, 1]. Present only for
            multi-chain inputs.
        avg_pae (float): Overall predicted aligned error in Å. Always present.
        pde (float): Overall predicted distance error in Å (RF3-specific).
            Always present.
        ranking_score (float): RF3 composite ranking score
            (``0.8*iptm + 0.2*ptm - 100*has_clash``); used to select the best
            sample. Always present.
        chain_ptm (list[float]): Per-chain mean pLDDT in ``[0, 1]`` (note: upstream
            uses the ``chain_ptm`` key for pLDDT despite the name; see comment in
            ``standalone/inference.py``). Always present.
        chain_pair_pae (list[list[float]]): Upper-triangular n-by-n chain-pair PAE
            matrix in Å. Always present.
        chain_pair_pae_min (list[list[float]]): Per-pair minimum-PAE aggregate
            in Å. Always present.
        chain_pair_pde (list[list[float]]): Upper-triangular n-by-n chain-pair PDE
            matrix in Å. Always present.
        chain_pair_pde_min (list[list[float]]): Per-pair minimum-PDE aggregate
            in Å. Always present.
        has_clash (bool): True if the predicted structure contains atom clashes.
            Always present.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {
            "availability": "multi-chain input only",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "avg_pae": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 32.0,
            "unit": "Å",
            "better_values_are": "lower",
        },
        "pde": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 32.0,
            "unit": "Å",
            "better_values_are": "lower",
        },
        "ranking_score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "higher",
        },
        "chain_ptm": {
            "availability": "always",
            "type": "list[float]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "chain_pair_pae": {
            "availability": "always (empty list for single-chain inputs)",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 32.0,
            "unit": "Å",
            "better_values_are": "lower",
        },
        "chain_pair_pae_min": {
            "availability": "always (empty list for single-chain inputs)",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 32.0,
            "unit": "Å",
            "better_values_are": "lower",
        },
        "chain_pair_pde": {
            "availability": "always (empty list for single-chain inputs)",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 32.0,
            "unit": "Å",
            "better_values_are": "lower",
        },
        "chain_pair_pde_min": {
            "availability": "always (empty list for single-chain inputs)",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 32.0,
            "unit": "Å",
            "better_values_are": "lower",
        },
        "has_clash": {
            "availability": "always",
            "type": "bool",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "ranking_score"


# Output:
class RF3Output(StructurePredictionOutput):
    """RF3 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying a
            :class:`RF3Metrics` instance on ``.metrics``.
    """


# Config:
class RF3Config(MSAStructurePredictionConfig):
    """Configuration object for RoseTTAFold3 structure prediction.

    Inherits from ``MSAStructurePredictionConfig``. The inherited
    ``include_pae_matrix`` flag is rejected: RF3 emits only chain-pair PAE
    aggregates and an ``avg_pae`` scalar, never a per-token LxL matrix.

    RF3's template/conformer-conditioning knobs and ``add_missing_atoms`` act on
    input atomic coordinates; this wrapper sends only sequences, SMILES, and CCD
    codes, so they would be no-ops and are not exposed.

    Attributes:
        n_recycles (int): Iterative refinement passes through the network.
            Higher = more accurate but slower. Default 10 (upstream default).
        diffusion_batch_size (int): Independent diffusion samples drawn per
            complex; the best by ``ranking_score`` is returned. Default 5.
        num_steps (int): Denoising steps in the diffusion process. Default 50.
        cyclic_chains (list[str]): Chain IDs (e.g. ``["A"]``) to mark as
            cyclic. Default ``[]``.
        use_msa (bool): Generate MSAs for protein chains via MMseqs2 homology
            search. Inherited from ``MSAStructurePredictionConfig``. Default True.
        msa_search_config (Mmseqs2HomologySearchConfig | None): Inherited.
            Default ``None``.
        pair_heterocomplex_msas (bool): Use taxonomy-paired MSA generation for
            heterocomplex protein chains. Inherited. Default ``True``.
        include_pae_matrix (bool): Inherited. **Must remain False** for RF3
            (no per-token PAE matrix is emitted).
        device (str): ``"cuda"`` or ``"cpu"``. Inherited. Default ``"cuda"``.
        verbose (int): Verbosity level (0=quiet, 1=info, 2=debug, 3=raw subprocess
            stderr). Inherited from ``BaseConfig``. Default ``0``.
        seed (int | None): Inherited. Default ``None``.
        timeout (int | None): Maximum execution time in seconds. RF3 is
            heavier than Boltz2; the default is set accordingly. Default 1800.
    """

    n_recycles: int = ConfigField(
        title="Number of Recycles",
        default=10,
        ge=1,
        description="Iterative refinement passes through the network. Higher = more accurate but slower.",
    )
    diffusion_batch_size: int = ConfigField(
        title="Diffusion Batch Size",
        default=5,
        ge=1,
        description="Independent diffusion samples per complex; the best by ranking_score is kept.",
    )
    num_steps: int = ConfigField(
        title="Number of Diffusion Steps",
        default=50,
        ge=1,
        description="Denoising steps in the diffusion process. Higher = more refined but slower.",
    )
    cyclic_chains: list[str] = ConfigField(
        title="Cyclic Chains",
        default_factory=list,
        description="Chain IDs (e.g. ['A']) to mark as cyclic.",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds. RF3 is heavier than Boltz2; default is set accordingly.",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def reject_pae_matrix(self) -> "RF3Config":
        """RF3 emits no per-token PAE matrix; reject any non-default ``include_pae_matrix``."""
        if self.include_pae_matrix:
            raise ValueError(
                "'include_pae_matrix' is not supported by RF3 - upstream emits only chain-pair PAE "
                "aggregates and an avg_pae scalar, never a per-token LxL matrix."
            )
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return RF3Input(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="rf3-prediction",
    label="RoseTTAFold3 Structure Prediction",
    category="structure_prediction",
    input_class=RF3Input,
    config_class=RF3Config,
    output_class=RF3Output,
    metrics_class=RF3Metrics,
    description="All-atom structure prediction with explicit chirality (RoseTTAFold3)",
    uses_gpu=True,
    device_count="1-2",
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
)
def run_rf3_prediction(inputs: RF3Input, config: RF3Config, instance: Any = None) -> RF3Output:
    """Predict 3D structures with RoseTTAFold3.

    RF3 is a diffusion-based all-atom predictor from the Baker/DiMaio labs (IPD,
    UW) supporting proteins, DNA, RNA, and small-molecule ligands. It runs via
    local GPU execution in an isolated micromamba environment.

    Args:
        inputs (RF3Input): Validated input containing one or more complexes to
            predict structures for.
        config (RF3Config): Validated RF3 configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RF3Output: ``structures`` is a list of :class:`Structure` instances,
            one per input complex, each carrying an :class:`RF3Metrics`.

    See Also:
        - RF3 Foundry: https://github.com/RosettaCommons/foundry/tree/production/models/rf3
        - RF3 preprint: https://www.biorxiv.org/content/10.1101/2025.08.14.670328v2

    Example:
        >>> result = run_rf3_prediction(RF3Input(complexes=["MKTL"]))
        >>> print(f"ranking_score: {result.structures[0].metrics.ranking_score:.2f}")

    Note:
        - RF3 processes complexes sequentially.
        - ``stochastic=True``: ``config.seed`` is advanced per-complex so duplicate
          inputs in one batch get distinct seeds.
        - The inherited ``include_pae_matrix`` flag is rejected; RF3 emits only
          chain-pair PAE aggregates and an ``avg_pae`` scalar.
    """
    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advance the seed per complex so duplicate inputs get distinct seeds.
    results = [
        run_rf3_prediction_on_complex(
            config=config,
            sp_complex=comp,
            complex_msas=inputs.msas[dispatch_idx] if inputs.msas else None,
            instance=instance,
            seed=base_seed + dispatch_idx,
        )
        for dispatch_idx, comp in enumerate(
            progress_bar(inputs.complexes, desc="Folding structures (RF3)", unit="complex", total=len(inputs.complexes))
        )
    ]
    return RF3Output(structures=results)


def run_rf3_prediction_on_complex(
    config: RF3Config,
    sp_complex: Any,
    complex_msas: Any = None,
    instance: Any = None,
    seed: int | None = None,
) -> Structure:
    """Run RF3 structure prediction on a single complex.

    Args:
        config (RF3Config): RF3 configuration.
        sp_complex (Any): Complex instance containing chain information.
        complex_msas (Any): Pre-computed ``ComplexMSAs`` for this complex, keyed
            by chain index. When ``paired``, row N of each chain's MSA pairs
            with row N of the other chains.
        instance (Any): Optional ToolInstance for persistent execution.
        seed (int | None): Per-complex seed to pass to the standalone. When
            ``None``, falls back to ``config.seed``.

    Returns:
        Structure: Predicted structure with RF3 metrics on ``.metrics``.
    """
    if seed is None:
        seed = config.seed
    if config.verbose:
        logger.info("Using local GPU for RF3 structure prediction...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "rf3_output")
        os.makedirs(output_dir)

        # Honor MSAs whenever present: auto-generated when use_msa=True, or supplied by the caller (always respected regardless of use_msa).
        chain_msa_paths: dict[str, str] | None = None
        if complex_msas is not None:
            chain_msa_paths = build_chain_a3m_paths(sp_complex, complex_msas, temp_dir, verbose=config.verbose)

        json_payload = complex_to_rf3_json(
            sp_complex.chains,
            name="complex",
            chain_msa_paths=chain_msa_paths,
        )

        input_json_path = os.path.join(temp_dir, "rf3_input.json")
        with open(input_json_path, "w") as f:
            f.write(json_payload)

        input_data = {
            "operation": "predict",
            "input_json_path": str(input_json_path),
            "output_dir": str(output_dir),
            "device": config.device,
            "n_recycles": config.n_recycles,
            "diffusion_batch_size": config.diffusion_batch_size,
            "num_steps": config.num_steps,
            "cyclic_chains": list(config.cyclic_chains),
            "seed": seed,
            "verbose": config.verbose,
        }

        output_data = ToolInstance.dispatch(
            "rf3",
            input_data,
            instance=instance,
            config=config,
        )

        cif_output = output_data["structure_cif_output"]
        raw_metrics = output_data["metrics"]

    # All metrics are always populated except iptm (multi-chain only).
    metrics_kwargs: dict[str, Any] = {
        "avg_plddt": float(raw_metrics["avg_plddt"]),
        "ptm": float(raw_metrics["ptm"]),
        "avg_pae": float(raw_metrics["avg_pae"]),
        "pde": float(raw_metrics["pde"]),
        "ranking_score": float(raw_metrics["ranking_score"]),
        "chain_ptm": raw_metrics["chain_ptm"],
        "chain_pair_pae": raw_metrics["chain_pair_pae"],
        "chain_pair_pae_min": raw_metrics["chain_pair_pae_min"],
        "chain_pair_pde": raw_metrics["chain_pair_pde"],
        "chain_pair_pde_min": raw_metrics["chain_pair_pde_min"],
        "has_clash": bool(raw_metrics["has_clash"]),
    }
    if "iptm" in raw_metrics:
        metrics_kwargs["iptm"] = float(raw_metrics["iptm"])

    structure = Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=RF3Metrics(**metrics_kwargs),
        source="rf3-prediction",
    )
    return normalize_output_chain_ids(structure, sp_complex.chains)
