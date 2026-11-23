"""proto_tools/tools/structure_prediction/opendde/opendde.py.

All-atom biomolecular structure prediction using OpenDDE.

Example:
    >>> from proto_tools.tools.structure_prediction.opendde import run_opendde, OpenDDEConfig, OpenDDEInput
    >>> inputs = OpenDDEInput(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
    >>> result = run_opendde(inputs, OpenDDEConfig())
    >>> print(f"pLDDT: {result.structures[0].metrics['avg_plddt']:.1f}")
"""

import json
import os
import tempfile
from logging import getLogger
from typing import Any, ClassVar, cast

from proto_tools.entities.structures import BFactorType, MSASubsampling, Structure
from proto_tools.tools.structure_prediction.opendde.helpers import (
    build_chain_msa_paths,
    complex_to_opendde_json,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
    ComplexMSAs,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    align_with_input,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = getLogger(__name__)

# MSA rows OpenDDE keeps from its reshuffled pool on every pass (data_configs["msa"]["msa_depth"]).
OPENDDE_MSA_DEPTH = 1280

# Bundled model names auto-downloaded by setup.sh; other model_checkpoint values are user paths.
_BUNDLED_MODELS: frozenset[str] = frozenset({"opendde_v1", "opendde_abag"})


# ============================================================================
# Data Models
# ============================================================================
# Input:
class OpenDDEInput(StructurePredictionInput):
    """Input object for OpenDDE structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins,
    DNA, RNA, and ligands using OpenDDE, an all-atom co-folding model.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[Complex]): List of complexes to predict structures for.
            Inherited from ``StructurePredictionInput``. Each complex can contain
            one or more chains of proteins, DNA, RNA, or ligands.

        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one entry per complex.
            Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by chain index).
            Populated by ``preprocess()`` or supplied directly. Default: None.

    Note:
        OpenDDE supports entity types: ``"protein"``, ``"dna"``, ``"rna"``, and
        ``"ligand"``. Entity types are automatically inferred if not explicitly
        provided. Chain modifications are supported; their CCD codes are emitted
        with the ``CCD_`` prefix OpenDDE's input schema requires.
    """

    # OpenDDE supports all standard entity types except glycan
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class OpenDDEMetrics(Metrics):
    """Per-structure metrics emitted by OpenDDE prediction.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Mean predicted LDDT score (0-100). Always present.
        ptm (float): Predicted TM-score (0-1). Always present.
        iptm (float): Interface predicted TM-score (0-1); 0.0 for single-chain inputs.
        gpde (float): Global predicted distance error (lower is better). Always present.
        ranking_score (float): OpenDDE ranking score used to select the best sample. Always present.
        has_clash (bool): Whether the predicted structure has a steric clash. Always present.
        avg_pae (float): Mean of the per-token PAE matrix in Å. Present when include_pae_matrix=True.
        pae (list[list[float]]): Full per-token PAE matrix in Å. Present when include_pae_matrix=True.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "better_values_are": "higher",
        },
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "gpde": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "ranking_score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "higher",
        },
        "has_clash": {
            "availability": "always",
            "type": "bool",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
        "avg_pae": {
            "availability": "when include_pae_matrix=True",
            "type": "float",
            "min": 0.0,
            "max": 32.0,
            "better_values_are": "lower",
        },
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 32.0,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "avg_plddt"


# Output:
class OpenDDEOutput(StructurePredictionOutput):
    """OpenDDE prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, one per input complex,
            each carrying an :class:`OpenDDEMetrics` instance on ``.metrics``.
    """


# Config:
class OpenDDEConfig(MSAStructurePredictionConfig):
    """Configuration object for OpenDDE structure prediction.

    This class defines configuration parameters for running OpenDDE, an all-atom
    co-folding model supporting proteins, DNA, RNA, and ligands. Fields map to the
    ``opendde pred`` CLI (``--sample``, ``--step``, ``--cycle``, ``--use_template``,
    ``--use_rna_msa``, ``-n``).

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        model_checkpoint (str): Which weights to fold with — a bundled model name
            (``"opendde_v1"`` general-purpose or ``"opendde_abag"`` antibody-antigen,
            both auto-downloaded into ``PROTO_MODEL_CACHE`` on first inference) or a
            path to a custom ``.pt`` checkpoint. Default: ``"opendde_v1"``.

        num_samples (int): Independent diffusion samples per complex (``--sample``);
            the best by ranking score is returned. Default: 1.

        num_steps (int): Diffusion denoising steps (``--step``). Higher = more
            refined but slower. Default: 200.

        num_cycles (int): Recycling iterations (``--cycle``). Higher = more accurate
            but slower. Default: 10.

        use_template (bool): Enable OpenDDE's template pipeline (``--use_template``).
            proto_tools does not generate templates, so this relies on OpenDDE's own
            search, which needs the ``search_database/`` assets ``setup.sh`` does not
            download. Default: False.

        use_rna_msa (bool): Enable OpenDDE's RNA MSA pipeline (``--use_rna_msa``).
            Also requires the ``search_database/`` assets. Default: False.

        use_msa (bool): Auto-generate protein MSAs via MMseqs2 homology search;
            supplied MSAs always override this. Inherited. Default: True.

        pair_heterocomplex_msas (bool): Taxonomy-pair heterocomplex protein MSAs.
            Inherited. Default: True.

        msa_search_config (Mmseqs2HomologySearchConfig | None): MMseqs2 search config;
            only used when ``use_msa=True``. Inherited. Default: None.

        include_pae_matrix (bool): Inherited. When True, the per-token PAE matrix (``pae``)
            plus a derived ``avg_pae`` scalar are attached to the metrics; it reads OpenDDE's
            full confidence data, so it forces ``atom_confidence`` on. Off by default — the
            matrix is O(n_token^2) to serialize. Default: False.

        atom_confidence (bool): Run OpenDDE with ``--need_atom_confidence`` so the structure's
            B-factors carry per-atom pLDDT (0-100) and ``per_residue_plddt`` works, like other
            predictors (about +2% time on a 1,010-token complex). Set False to skip writing the
            full confidence output; B-factors are then all 0 and tagged ``UNKNOWN`` unless
            ``include_pae_matrix`` is set. Default: True.

        device (str): Device to run on (``"cuda"``, ``"cpu"``). Inherited. Default: ``"cuda"``.

        timeout (int | None): Max execution time in seconds; ``None`` waits indefinitely. Default: 1200.
    """

    model_checkpoint: str = ConfigField(
        title="Model / Checkpoint",
        default="opendde_v1",
        description="Bundled model name ('opendde_v1' or 'opendde_abag') or a path to a custom .pt checkpoint.",
    )
    num_samples: int = ConfigField(
        title="Number of Samples",
        default=1,
        ge=1,
        description="Independent diffusion samples per complex (--sample); best by ranking score is kept.",
    )
    num_steps: int = ConfigField(
        title="Number of Diffusion Steps",
        default=200,
        ge=1,
        description="Diffusion denoising steps (--step). Higher = more refined but slower.",
    )
    num_cycles: int = ConfigField(
        title="Number of Recycling Cycles",
        default=10,
        ge=1,
        description="Recycling iterations (--cycle). Higher = more accurate but slower.",
    )
    use_template: bool = ConfigField(
        title="Use Template",
        default=False,
        description="Enable OpenDDE's template search pipeline (--use_template).",
    )
    use_rna_msa: bool = ConfigField(
        title="Use RNA MSA",
        default=False,
        description="Enable OpenDDE's RNA MSA pipeline (--use_rna_msa).",
    )
    atom_confidence: bool = ConfigField(
        title="Per-Atom Confidence",
        default=True,
        description="Write per-atom pLDDT into B-factors (--need_atom_confidence); include_pae_matrix forces it.",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds.",
        include_in_key=False,
    )

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:  # noqa: ARG002 - registry contract
        """A local checkpoint file exists only on this machine, so no remote worker can read it."""
        if self.model_checkpoint not in _BUNDLED_MODELS:
            return (
                "model_checkpoint points to a local file. "
                f"Use a bundled model name ({', '.join(sorted(_BUNDLED_MODELS))}), or run locally with "
                "device='cuda' or device='cpu'."
            )
        return None

    @classmethod
    def minimal(cls, **kwargs: Any) -> "OpenDDEConfig":
        """One sample over 20 diffusion steps and one recycle (defaults: 1, 200, 10).

        The device, seed and placement tests only need the model to load and run, so one
        sample through a short diffusion schedule tests the same thing as the full one.

        Args:
            **kwargs (Any): Field values passed to the config constructor; they take precedence.

        Returns:
            OpenDDEConfig: Config with cheap-mode sampling applied.
        """
        kwargs.setdefault("num_samples", 1)
        kwargs.setdefault("num_steps", 20)
        kwargs.setdefault("num_cycles", 1)
        return cast("OpenDDEConfig", super().minimal(**kwargs))


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return OpenDDEInput(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="opendde-prediction",
    label="OpenDDE Structure Prediction",
    category="structure_prediction",
    input_class=OpenDDEInput,
    config_class=OpenDDEConfig,
    output_class=OpenDDEOutput,
    metrics_class=OpenDDEMetrics,
    description="All-atom biomolecular structure prediction using OpenDDE",
    uses_gpu=True,
    # The worker holds OpenDDE's runner in-process, and OpenDDE resolves "cuda" to cuda:0 of the
    # visible GPUs, so the worker must see only the GPU(s) it was leased.
    pin_visible_devices=True,
    device_count="1-2",
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    max_chunk_size=1,
    cacheable=True,
    stochastic=True,
)
def run_opendde(inputs: OpenDDEInput, config: OpenDDEConfig, instance: Any = None) -> OpenDDEOutput:
    """Predict all-atom 3D structures using the OpenDDE co-folding model.

    Runs OpenDDE via local GPU execution in an isolated environment. Each complex
    is folded independently; the best of ``num_samples`` diffusion samples (by
    ranking score) is returned per complex. MSA generation is delegated to the
    MMseqs2 homology-search tool and supplied to OpenDDE as A3M files.

    Args:
        inputs (OpenDDEInput): Validated input with one or more complexes to fold.
        config (OpenDDEConfig): Validated OpenDDE configuration (checkpoint,
            sampling parameters, MSA settings, execution options).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        OpenDDEOutput: Output with one ``Structure`` per input complex, each
            carrying :class:`OpenDDEMetrics` (avg_plddt, ptm, iptm, gpde,
            ranking_score, has_clash) on ``.metrics``.

    See Also:
        - OpenDDE GitHub: https://github.com/aurekaresearch/OpenDDE
        - OpenDDE paper: https://arxiv.org/abs/2607.03787
        - OpenDDE weights: https://huggingface.co/aurekaresearch/OpenDDE

    Example:
        >>> inputs = OpenDDEInput(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
        >>> config = OpenDDEConfig(num_steps=200, num_cycles=10, verbose=True)
        >>> result = run_opendde(inputs, config)
        >>> print(f"pLDDT: {result.structures[0].metrics['avg_plddt']:.1f}")
    """
    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advance the seed per complex so duplicate inputs get distinct seeds.
    results = [
        run_opendde_on_complex(
            config=config,
            sp_complex=comp,
            complex_msas=inputs.msas[dispatch_idx] if inputs.msas else None,
            instance=instance,
            seed=base_seed + dispatch_idx,
            dispatch_idx=dispatch_idx,
        )
        for dispatch_idx, comp in enumerate(
            progress_bar(
                inputs.complexes, desc="Folding structures (OpenDDE)", unit="complex", total=len(inputs.complexes)
            )
        )
    ]
    return OpenDDEOutput(
        structures=results,
        metadata={
            "num_complexes": len(results),
            "total_chains": sum(s.num_chains for s in results),
        },
    )


def run_opendde_on_complex(
    config: OpenDDEConfig,
    sp_complex: Complex,
    complex_msas: ComplexMSAs | None,
    instance: Any,
    seed: int,
    dispatch_idx: int,
) -> Structure:
    """Fold a single complex with OpenDDE and return its best-ranked structure.

    Args:
        config (OpenDDEConfig): OpenDDE configuration.
        sp_complex (Complex): The complex to fold.
        complex_msas (ComplexMSAs | None): Pre-computed per-chain MSAs for this complex.
        instance (Any): Optional ToolInstance for persistent execution.
        seed (int): Per-complex ``modelSeeds`` value passed to OpenDDE.
        dispatch_idx (int): Position of this complex in the batch (used for the job name).

    Returns:
        Structure: The best-ranked predicted structure with OpenDDE metrics attached.
    """
    if config.verbose:
        logger.info("Using local GPU for OpenDDE structure prediction...")

    # Internal label for OpenDDE's ephemeral output files (temp dir is discarded; results returned via Output).
    job_name = f"opendde_{dispatch_idx}"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "opendde_output")
        os.makedirs(output_dir)

        # Use supplied/auto-generated MSAs when present (unpaired + paired MsaPath); else single-sequence.
        chain_msa_paths, chain_paired_msa_paths, msa_sent = build_chain_msa_paths(
            sp_complex, complex_msas, temp_dir, verbose=config.verbose
        )
        input_json = complex_to_opendde_json(
            sp_complex.chains,
            job_name,
            [seed],
            chain_msa_paths=chain_msa_paths,
            chain_paired_msa_paths=chain_paired_msa_paths,
        )

        input_json_path = os.path.join(temp_dir, f"{job_name}.json")
        with open(input_json_path, "w") as f:
            json.dump([input_json], f, indent=2)

        # --use_msa must be true whenever MSAs exist (OpenDDE skips its featurizer when false), and only
        # then: with it true, OpenDDE searches its own server unless every protein chain has a path.
        opendde_use_msa = bool(chain_msa_paths or chain_paired_msa_paths)

        input_data = {
            "operation": "predict",
            "input_json_path": input_json_path,
            "output_dir": output_dir,
            "job_name": job_name,
            "model_checkpoint": config.model_checkpoint,
            "num_samples": config.num_samples,
            "num_steps": config.num_steps,
            "num_cycles": config.num_cycles,
            "use_msa": opendde_use_msa,
            "use_template": config.use_template,
            "use_rna_msa": config.use_rna_msa,
            "seed": seed,
            "device": config.device,
            "verbose": config.verbose,
            "include_pae_matrix": config.include_pae_matrix,
            "atom_confidence": config.atom_confidence,
        }

        output_data = ToolInstance.dispatch("opendde", input_data, instance=instance, config=config)

    cif_output = output_data["structure_cif_output"]
    formatted_metrics = output_data["metrics"]

    metrics_dict: dict[str, Any] = {
        "avg_plddt": float(formatted_metrics["avg_plddt"]),
        "ptm": float(formatted_metrics["ptm"]),
        "iptm": float(formatted_metrics["iptm"]),
        "gpde": float(formatted_metrics["gpde"]),
        "ranking_score": float(formatted_metrics["ranking_score"]),
        "has_clash": bool(formatted_metrics["has_clash"]),
    }
    # PAE is only returned when include_pae_matrix asked for it.
    if "pae" in formatted_metrics:
        metrics_dict["avg_pae"] = float(formatted_metrics["avg_pae"])
        metrics_dict["pae"] = formatted_metrics["pae"]

    structure = Structure(
        structure=cif_output,
        # With full confidence data (atom_confidence, or include_pae_matrix) OpenDDE writes per-atom
        # pLDDT (0-100) as B-factors; without it every B-factor is 0, so they carry no pLDDT.
        b_factor_type=(
            BFactorType.PLDDT if config.atom_confidence or config.include_pae_matrix else BFactorType.UNKNOWN
        ),
        metrics=OpenDDEMetrics(**metrics_dict),
        source="opendde-prediction",
    )
    return align_with_input(
        structure,
        sp_complex.chains,
        complex_msas=complex_msas,
        report_msa=True,
        msa_sent=msa_sent,
        msa_subsampling=MSASubsampling(mode="stochastic", max_rows=OPENDDE_MSA_DEPTH),
    )
