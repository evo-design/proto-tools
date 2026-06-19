"""FreeBindCraft PyRosetta-free binder design (AF2 hallucination + ProteinMPNN + OpenMM relax + FreeSASA/sc-rs scoring)."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

_FREEBINDCRAFT_FIXTURE_PDB = (
    Path(__file__).resolve().parents[2] / "structure_prediction" / "alphafold2" / "example_input_fixture.pdb"
)

_USER_FACING_UPSTREAM_KEYS: frozenset[str] = frozenset(
    {
        "design_algorithm",
        "use_multimer_design",
        "omit_AAs",
        "force_reject_AA",
        "soft_iterations",
        "temporary_iterations",
        "hard_iterations",
        "greedy_iterations",
        "greedy_percentage",
        "weights_plddt",
        "weights_pae_intra",
        "weights_pae_inter",
        "weights_con_intra",
        "weights_con_inter",
        "weights_helicity",
        "weights_iptm",
        "weights_rg",
        "weights_termini_loss",
        "random_helicity",
        "use_i_ptm_loss",
        "use_rg_loss",
        "use_termini_distance_loss",
        "rm_template_seq_design",
        "rm_template_seq_predict",
        "predict_initial_guess",
        "predict_bigbang",
        "enable_mpnn",
        "mpnn_fix_interface",
        "num_seqs",
        "max_mpnn_sequences",
        "optimise_beta",
        "max_trajectories",
        "enable_rejection_check",
        "acceptance_rate",
        "start_monitoring",
    }
)

_HARDCODED_INTERNAL_SETTINGS: dict[str, Any] = {
    # File-output toggles whose artifacts proto-tools never returns, plus sample_models.
    "sample_models": True,
    "save_design_animations": False,
    "save_design_trajectory_plots": False,
    "save_trajectory_pickle": False,
    "save_mpnn_fasta": False,
    "zip_animations": False,
    "zip_plots": False,
    "remove_unrelaxed_trajectory": True,
    "remove_unrelaxed_complex": True,
    "remove_binder_monomer": True,
    # Dials constant across all BindCraft presets — pinned to upstream defaults, not user settings.
    "intra_contact_distance": 14.0,
    "inter_contact_distance": 20.0,
    "intra_contact_number": 2,
    "inter_contact_number": 2,
    "num_recycles_design": 1,
    "num_recycles_validation": 3,
    "optimise_beta_extra_soft": 0,
    "optimise_beta_extra_temp": 0,
    "optimise_beta_recycles_design": 3,
    "optimise_beta_recycles_valid": 3,
    "model_path": "v_48_020",
    "mpnn_weights": "soluble",
    "sampling_temp": 0.1,
    "backbone_noise": 0.0,
    "rm_template_sc_design": False,
    "rm_template_sc_predict": False,
}


# ============================================================================
# Metrics
# ============================================================================


class FreeBindCraftMetrics(Metrics):
    """Metrics for one accepted FreeBindCraft binder design.

    Mirrors the PyRosetta-free subset of upstream's ``final_design_stats.csv``
    ``Average_*`` columns. FreeBindCraft replaces PyRosetta scoring with
    open-source tools (OpenMM relaxation, FreeSASA/Biopython surface analysis,
    the sc-rs shape-complementarity binary), so only the metrics it computes for
    real are exposed. PyRosetta-only quantities — interface energies (``dG``,
    ``dG/dSASA``, ``Binder_Energy_Score``), H-bond counts, and ``PackStat`` — are emitted upstream as placeholders to satisfy default
    filters and are intentionally **not** surfaced here. Geometric clash counts
    (Biopython) and ipSAE (PAE-derived) are real and are surfaced.

    Attributes:
        metric_spec (ClassVar[dict[str, MetricSpec]]): Per-metric metadata
            (type, range, unit). Used by the validation helper in
            ``tests/tool_infra_tests/_metric_helpers.py``.
        primary_metric (str | None): Headline metric name; defaults to ``"avg_iptm"``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        # AlphaFold2 confidence
        "avg_plddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_ptm": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_iptm": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_pae": {"type": "float", "min": 0.0, "unit": "Å", "better_values_are": "lower"},
        "avg_ipae": {"type": "float", "min": 0.0, "unit": "Å", "better_values_are": "lower"},
        "avg_ipsae": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_iplddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_ss_plddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_binder_plddt": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "unit": "fraction",
            "better_values_are": "higher",
        },
        "avg_binder_ptm": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction", "better_values_are": "higher"},
        "avg_binder_pae": {"type": "float", "min": 0.0, "unit": "Å", "better_values_are": "lower"},
        # Surface / interface geometry (FreeSASA + Biopython + sc-rs)
        "dSASA": {"type": "float", "min": 0.0, "unit": "Å^2", "better_values_are": "context-dependent"},
        "interface_sasa_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "interface_hydrophobicity": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "surface_hydrophobicity": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "unit": "fraction",
            "better_values_are": "lower",
        },
        "shape_complementarity": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "unit": "fraction",
            "better_values_are": "higher",
        },
        "n_interface_residues": {
            "type": "float",
            "min": 0.0,
            "unit": "count",
            "better_values_are": "context-dependent",
        },
        # Secondary structure (percentages, 0-100)
        "binder_helix_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "binder_betasheet_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "binder_loop_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "interface_helix_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "interface_betasheet_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        "interface_loop_pct": {
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "percent",
            "better_values_are": "context-dependent",
        },
        # RMSDs
        "hotspot_rmsd": {"type": "float", "min": 0.0, "unit": "Å", "better_values_are": "lower"},
        "target_rmsd": {"type": "float", "min": 0.0, "unit": "Å", "better_values_are": "lower"},
        "binder_rmsd": {"type": "float", "min": 0.0, "unit": "Å", "better_values_are": "lower"},
        # Clashes (geometric, Biopython — pre- and post-OpenMM relaxation)
        "unrelaxed_clashes": {"type": "float", "min": 0.0, "unit": "count", "better_values_are": "lower"},
        "relaxed_clashes": {"type": "float", "min": 0.0, "unit": "count", "better_values_are": "lower"},
    }
    primary_metric: str | None = Field(
        default="avg_iptm",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


# ============================================================================
# Single accepted design
# ============================================================================


class FreeBindCraftDesign(BaseModel):
    """One accepted binder design from a FreeBindCraft pipeline run.

    Attributes:
        design_name (str): Unique design identifier emitted by upstream
            (e.g. ``"binder_l60_s12345_mpnn3"``).
        binder_sequence (str): Designed binder amino-acid sequence (1-letter codes).
        structure (Structure): OpenMM-relaxed target+binder complex; B-factors are
            pLDDT on the 0-100 PDB scale (``b_factor_type=PLDDT``).
        metrics (FreeBindCraftMetrics): Per-design averaged metrics that the
            filter check evaluates against.
        seed (int): Random seed of the trajectory that produced this design.
        interface_aas (dict[str, int]): Amino-acid composition at the binder-target interface.
        interface_residues (list[int]): 1-indexed binder residue positions at the interface.
    """

    model_config = ConfigDict(extra="forbid")

    design_name: str = Field(
        title="Design Name",
        description="Unique design identifier emitted by FreeBindCraft.",
    )
    binder_sequence: str = Field(
        title="Binder Sequence",
        description="Designed binder amino-acid sequence (1-letter codes).",
    )
    structure: Structure = Field(
        title="Design Structure",
        description="OpenMM-relaxed target+binder complex with per-residue pLDDT in B-factors.",
    )
    metrics: FreeBindCraftMetrics = Field(
        title="Design Metrics",
        description="Per-design averaged metrics evaluated by the filter check.",
    )
    seed: int = Field(
        ge=0,
        title="Seed",
        description="Random seed of the trajectory that produced this design.",
    )
    interface_aas: dict[str, int] = Field(
        default_factory=dict,
        title="Interface AA Composition",
        description="Amino-acid composition at the binder-target interface.",
    )
    interface_residues: list[int] = Field(
        default_factory=list,
        title="Interface Residues",
        description="1-indexed binder residue positions at the interface.",
    )


# ============================================================================
# Input
# ============================================================================


class FreeBindCraftInput(BaseToolInput):
    """Target specification for one FreeBindCraft binder-design run.

    Attributes:
        target_pdb (Structure): Target structure. Accepts a file path, raw
            PDB/CIF content string, ``Structure`` object, or a dict in the
            shape produced by ``Structure.model_dump(mode='json')``.
        target_chain (str): Chain ID(s) of the frozen target (comma-separated for
            multi-chain). Maps to BindCraft's ``chains``.
        target_hotspot_residues (str | None): Comma-separated 1-indexed residue
            positions on the target that the binder must contact. Supports ranges
            (e.g. ``"1-10,56,78"``). ``None`` or empty = unrestricted.
        binder_lengths (tuple[int, int]): ``(min, max)`` binder length range.
            Maps to BindCraft's ``lengths``.
        binder_name (str): Project identifier — used as a prefix in output filenames.
        number_of_final_designs (int): Target accepted-design count. The pipeline
            stops after reaching this count or after ``max_trajectories`` attempts
            (whichever comes first).
    """

    target_pdb: Structure = InputField(
        title="Target PDB",
        description="Target structure.",
    )
    target_chain: str = InputField(
        default="A",
        title="Target Chain",
        description="Chain ID(s) of the frozen target (comma-separated for multi-chain).",
    )
    target_hotspot_residues: str | None = InputField(
        default=None,
        title="Target Hotspot Residues",
        description="Comma-separated 1-indexed residue positions on the target (e.g. '1-10,56,78').",
    )
    binder_lengths: tuple[int, int] = InputField(
        default=(65, 150),
        title="Binder Lengths",
        description="(min, max) binder length range. Upstream default: (65, 150).",
    )
    binder_name: str = InputField(
        default="binder",
        title="Binder Name",
        description="Filename prefix for accepted designs (e.g. 'binder' yields 'binder_l60_s12345_mpnn3').",
    )
    number_of_final_designs: int = InputField(
        default=100,
        ge=1,
        title="Number of Final Designs",
        description="Target accepted-design count. Set to 1 for one-off sampling. Upstream default: 100.",
    )

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        """Ensure ``binder_lengths`` is ``(min, max)`` with ``min <= max`` and both positive."""
        lo, hi = self.binder_lengths
        if lo <= 0 or hi <= 0:
            raise ValueError(f"binder_lengths must be positive integers, got ({lo}, {hi})")
        if lo > hi:
            raise ValueError(f"binder_lengths must be (min, max) with min <= max, got ({lo}, {hi})")
        return self


# ============================================================================
# Config
# ============================================================================


class FreeBindCraftConfig(BaseConfig):
    """User-facing FreeBindCraft settings.

    Defaults match upstream BindCraft's pinned ``default_4stage_multimer.json``
    (FreeBindCraft is a drop-in fork, so the hallucination/MPNN settings are
    identical); ``filter_overrides`` merges per-metric thresholds on top of the
    default-filters snapshot at dispatch.

    Attributes:
        design_algorithm (Literal["2stage", "3stage", "4stage", "greedy", "mcmc"]):
            Hallucination algorithm. Drives which iteration-count fields below
            are actually consumed (see each field's description). Upstream
            default: ``"4stage"``.
        use_multimer_design (bool): Use AF2 multimer parameters during hallucination.
            Every upstream preset uses multimer.
        omit_AAs (str): Amino acids to ban during design (no separator). Upstream default: ``"C"``.
        force_reject_AA (bool): Reject any design containing ``omit_AAs``.
        soft_iterations (int): Soft-stage iterations. Used by 2stage/3stage/4stage.
        temporary_iterations (int): Temporary-stage iterations. Used by 3stage/4stage.
        hard_iterations (int): Hard-stage iterations. Used by 3stage/4stage.
        greedy_iterations (int): Greedy/MCMC iterations. Used by 2stage/4stage/greedy/mcmc.
        greedy_percentage (float): Greedy/MCMC mutation rate as % of binder length.
        weights_plddt (float): pLDDT loss weight.
        weights_pae_intra (float): Intra-chain PAE loss weight.
        weights_pae_inter (float): Inter-chain (interface) PAE loss weight.
        weights_con_intra (float): Intra-chain contact loss weight.
        weights_con_inter (float): Inter-chain (interface) contact loss weight.
        weights_helicity (float): Helicity bias weight (negative discourages helices).
        weights_iptm (float): Interface pTM loss weight (only used when ``use_i_ptm_loss=True``).
        weights_rg (float): Radius-of-gyration loss weight (only used when ``use_rg_loss=True``).
        weights_termini_loss (float): N-/C-termini distance loss weight (only used when ``use_termini_distance_loss=True``).
        random_helicity (bool): Randomize the sign of ``weights_helicity`` per trajectory.
        use_i_ptm_loss (bool): Enable interface pTM loss.
        use_rg_loss (bool): Enable radius-of-gyration loss.
        use_termini_distance_loss (bool): Enable termini-distance loss.
        rm_template_seq_design (bool): Mask target template sequence during hallucination.
        rm_template_seq_predict (bool): Mask target template sequence during validation.
        predict_initial_guess (bool): Use the trajectory structure as AF2's initial guess.
        predict_bigbang (bool): Use AF2's "Big Bang" recycle initialisation.
        enable_mpnn (bool): Run ProteinMPNN sequence refinement after each accepted trajectory.
            When False, ``num_seqs`` / ``max_mpnn_sequences`` are inert.
        mpnn_fix_interface (bool): Fix interface residues during MPNN redesign.
        num_seqs (int): Number of MPNN sequences to sample per trajectory.
        max_mpnn_sequences (int): Max MPNN sequences to validate per trajectory.
        optimise_beta (bool): 4stage-only — increase recycles + iterations
            mid-trajectory when the soft-stage output is beta-heavy.
        max_trajectories (int | bool): Max hallucination trajectories before stopping.
            ``False`` (upstream default) = unlimited; positive int = cap.
        enable_rejection_check (bool): Enable rolling acceptance-rate monitoring
            (stops the run if it stalls).
        acceptance_rate (float): Minimum design acceptance rate to keep running.
        start_monitoring (int): Trajectory count before acceptance-rate monitoring starts.
        filter_overrides (dict[str, Any]): Per-metric threshold overrides merged on top
            of the upstream default filters at dispatch time. Keys are upstream metric
            names (e.g. ``"Average_pLDDT"``); values are upstream filter dicts
            (e.g. ``{"threshold": 0.85, "higher": True}``).
        timeout (int | None): Maximum execution time in seconds. ``None`` (default) waits indefinitely.
    """

    design_algorithm: Literal["2stage", "3stage", "4stage", "greedy", "mcmc"] = ConfigField(
        title="Design Algorithm",
        default="4stage",
        description="Hallucination algorithm; '4stage' runs soft, temporary, hard, then greedy stages.",
    )
    use_multimer_design: bool = ConfigField(
        title="Use Multimer Design",
        default=True,
        description="Use AF2 multimer params during hallucination. Every upstream preset uses multimer.",
        reload_on_change=True,
    )
    omit_AAs: str = ConfigField(
        title="Omit Amino Acids",
        default="C",
        description="Amino acids to ban during design (no separator), e.g. 'C' or 'CW'.",
    )
    force_reject_AA: bool = ConfigField(
        title="Force Reject AA",
        default=False,
        description="Drop any MPNN sequence containing residues from omit_AAs (hard reject, not a soft penalty).",
    )

    soft_iterations: int = ConfigField(
        title="Soft Iterations",
        default=75,
        ge=0,
        description="Soft-stage hallucination iterations. Used by 2stage / 3stage / 4stage.",
    )
    temporary_iterations: int = ConfigField(
        title="Temporary Iterations",
        default=45,
        ge=0,
        description="Temporary-stage hallucination iterations. Used by 3stage / 4stage.",
    )
    hard_iterations: int = ConfigField(
        title="Hard Iterations",
        default=5,
        ge=0,
        description="Hard-stage hallucination iterations. Used by 3stage / 4stage.",
    )
    greedy_iterations: int = ConfigField(
        title="Greedy Iterations",
        default=15,
        ge=0,
        description="Greedy-stage iteration count. Used by 2stage / 4stage / greedy / mcmc (3stage doesn't use it).",
    )
    greedy_percentage: float = ConfigField(
        title="Greedy Percentage",
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Mutation rate for greedy/MCMC tries (% of binder length). Used by 2stage / 4stage / greedy / mcmc.",
    )

    weights_plddt: float = ConfigField(
        title="pLDDT Loss Weight",
        default=0.1,
        description="Loss weight on AF2 pLDDT (higher = push for more confident-folding designs).",
    )
    weights_pae_intra: float = ConfigField(
        title="Intra-Chain PAE Weight",
        default=0.4,
        description="Loss weight on within-binder PAE (higher = push for cleaner internal geometry).",
    )
    weights_pae_inter: float = ConfigField(
        title="Inter-Chain PAE Weight",
        default=0.1,
        description="Loss weight on the binder-target interface PAE; higher values push for more confident pairing.",
    )
    weights_con_intra: float = ConfigField(
        title="Intra-Chain Contact Weight",
        default=1.0,
        description="Loss weight on within-binder C-alpha contacts (encourages a compact fold).",
    )
    weights_con_inter: float = ConfigField(
        title="Inter-Chain Contact Weight",
        default=1.0,
        description="Loss weight on the binder-target interface contacts; encourages docking.",
    )
    weights_helicity: float = ConfigField(
        title="Helicity Weight",
        default=-0.3,
        description="Helicity bias weight (negative discourages helices, positive encourages them).",
    )
    weights_iptm: float = ConfigField(
        title="Interface pTM Weight",
        default=0.05,
        description="Loss weight on AF2 interface pTM (higher = push for higher iPTM).",
    )
    weights_rg: float = ConfigField(
        title="Radius-of-Gyration Weight",
        default=0.3,
        description="Loss weight on binder radius of gyration (higher = compress toward a tighter binder).",
    )
    weights_termini_loss: float = ConfigField(
        title="Termini Distance Weight",
        default=0.1,
        description="Loss weight pulling binder N- and C-termini together (for cyclisable backbones).",
    )

    random_helicity: bool = ConfigField(
        title="Random Helicity",
        default=False,
        description="Randomise the sign of weights_helicity per trajectory.",
    )
    use_i_ptm_loss: bool = ConfigField(
        title="Use Interface pTM Loss",
        default=True,
        description="Enable interface pTM loss term (weights_iptm). Pushes for higher iPTM during hallucination.",
    )
    use_rg_loss: bool = ConfigField(
        title="Use Radius-of-Gyration Loss",
        default=True,
        description="Enable radius-of-gyration loss.",
    )
    use_termini_distance_loss: bool = ConfigField(
        title="Use Termini Distance Loss",
        default=False,
        description="Enable N-/C-termini distance loss.",
    )

    rm_template_seq_design: bool = ConfigField(
        title="Mask Template Seq (Design)",
        default=False,
        description="Hide target sequence from AF2 template during hallucination (geometry-only target).",
    )
    rm_template_seq_predict: bool = ConfigField(
        title="Mask Template Seq (Predict)",
        default=False,
        description="Hide target sequence from AF2 template during validation (geometry-only target).",
    )
    predict_initial_guess: bool = ConfigField(
        title="Predict Initial Guess",
        default=False,
        description="Seed AF2 validation with the hallucinated complex coords (helps hard targets).",
    )
    predict_bigbang: bool = ConfigField(
        title="Predict Big Bang",
        default=False,
        description="Seed AF2 validation with the hallucinated atom positions (alternative to initial_guess).",
    )

    enable_mpnn: bool = ConfigField(
        title="Enable MPNN",
        default=True,
        description="Refine each hallucinated binder with ProteinMPNN before AF2 re-validation.",
    )
    mpnn_fix_interface: bool = ConfigField(
        title="MPNN Fix Interface",
        default=True,
        description="Fix interface residues during MPNN redesign.",
    )
    num_seqs: int = ConfigField(
        title="MPNN Samples per Traj",
        default=20,
        ge=1,
        description="MPNN sequences sampled per accepted trajectory before filtering.",
    )
    max_mpnn_sequences: int = ConfigField(
        title="MPNN Validated per Traj",
        default=2,
        ge=1,
        description="Top-scoring MPNN sequences (out of num_seqs) carried forward to AF2 validation.",
    )
    optimise_beta: bool = ConfigField(
        title="Optimise Beta Designs",
        default=True,
        description="When a trajectory looks β-heavy (>15% sheet), bump iterations + recycles to help it converge.",
    )

    max_trajectories: int | bool = ConfigField(
        title="Max Trajectories",
        default=False,
        description="Max hallucination trajectories before stopping. False = unlimited; positive int = cap.",
        include_in_key=False,
    )
    enable_rejection_check: bool = ConfigField(
        title="Enable Rejection Check",
        default=True,
        description="Enable rolling acceptance-rate monitoring (stops the run if it stalls).",
        include_in_key=False,
    )
    acceptance_rate: float = ConfigField(
        title="Acceptance Rate",
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Minimum design acceptance rate to keep running.",
        include_in_key=False,
    )
    start_monitoring: int = ConfigField(
        title="Monitor Start (Trajectories)",
        default=600,
        ge=0,
        description="Trajectory count before acceptance-rate monitoring starts.",
        include_in_key=False,
    )

    filter_overrides: dict[str, Any] = ConfigField(
        title="Filter Overrides",
        default_factory=dict,
        description="Per-metric threshold overrides merged on top of upstream default_filters.json.",
    )

    @model_validator(mode="after")
    def _validate_max_trajectories(self) -> Self:
        """``max_trajectories`` must be ``False`` (unlimited) or a positive int (``True`` coerces to 1)."""
        mt = self.max_trajectories
        if mt is False:
            return self
        if mt is True:  # int|bool field: collapse the nonsensical bool to its numeric meaning
            self.max_trajectories = 1
            return self
        if mt < 1:
            raise ValueError(f"max_trajectories must be False or a positive int, got {mt!r}.")
        return self

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the tool on.",
        include_in_key=False,
    )
    # FreeBindCraft campaigns run hours-to-days; truncating mid-run wastes compute (default: no cap).
    timeout: int | None = ConfigField(
        title="Timeout",
        default=None,
        ge=1,
        description="Maximum execution time in seconds. None (default) waits indefinitely.",
        include_in_key=False,
    )

    @property
    def gpus_per_instance(self) -> int:
        """FreeBindCraft uses one GPU per design run (sequential trajectories)."""
        return 1

    @classmethod
    def minimal(cls, **kwargs: Any) -> "FreeBindCraftConfig":
        """Cheapest path through the FreeBindCraft pipeline for smoke testing.

        Caps trajectories at 1 and slashes per-trajectory iteration counts
        relative to the upstream-matched production defaults. Used by the
        parametrized env-report and smoke-test infrastructure; production
        callers never invoke this. Acceptance is best-effort — the env-report
        only verifies execution end-to-end.

        Args:
            **kwargs (Any): Field values passed to the config constructor.
                Take precedence over the cheap-mode defaults set here.

        Returns:
            FreeBindCraftConfig: Config with cheap-mode iteration counts applied.
        """
        kwargs.setdefault("max_trajectories", 1)
        kwargs.setdefault("soft_iterations", 10)
        kwargs.setdefault("temporary_iterations", 5)
        kwargs.setdefault("hard_iterations", 2)
        kwargs.setdefault("greedy_iterations", 2)
        kwargs.setdefault("num_seqs", 2)
        kwargs.setdefault("max_mpnn_sequences", 1)
        return super().minimal(**kwargs)  # type: ignore[return-value]


# ============================================================================
# Output
# ============================================================================


class FreeBindCraftOutput(BaseToolOutput):
    """Output from a FreeBindCraft binder-design pipeline run.

    Attributes:
        designs (list[FreeBindCraftDesign]): Accepted binder designs (length is at most
            ``FreeBindCraftInput.number_of_final_designs``).
        n_trajectories_run (int): Total trajectories attempted before stopping
            (success or hitting ``max_trajectories``).
        n_designs_accepted (int): Designs that passed all filters (equals
            ``len(designs)``).
    """

    designs: list[FreeBindCraftDesign] = Field(
        default_factory=list,
        title="Designs",
        description="Accepted binder designs.",
    )
    n_trajectories_run: int = Field(
        default=0,
        ge=0,
        title="Trajectories Run",
        description="Total trajectories attempted before stopping.",
    )
    n_designs_accepted: int = Field(
        default=0,
        ge=0,
        title="Designs Accepted",
        description="Designs that passed all filters (equals len(designs)).",
    )

    def __len__(self) -> int:
        """Number of accepted designs."""
        return len(self.designs)

    def __getitem__(self, index: int) -> FreeBindCraftDesign:
        """Get an accepted design by index."""
        return self.designs[index]

    def __iter__(self) -> Iterator[FreeBindCraftDesign]:  # type: ignore[override]
        """Iterate over accepted designs."""
        return iter(self.designs)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        """Default export format — directory of PDBs + a stats.json summary."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write designs to disk.

        - ``"pdb"``: directory containing one PDB per accepted design plus a
          ``stats.json`` summary of metrics.
        - ``"json"``: single JSON file with the full Pydantic dump (PDB strings inline).

        Args:
            export_path (str | Path): Output path (directory for ``pdb``, file for ``json``).
            file_format (str): One of ``"pdb"`` or ``"json"``.
        """
        if file_format not in self.output_format_options:
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path)
        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for design in self.designs:
                design.structure.write_pdb(path / f"{design.design_name}.pdb")
            stats_rows = [
                {
                    "design_name": d.design_name,
                    "binder_sequence": d.binder_sequence,
                    "seed": d.seed,
                    **dict(d.metrics.items()),
                }
                for d in self.designs
            ]
            (path / "stats.json").write_text(json.dumps(stats_rows, indent=2))
        else:
            path.write_text(self.model_dump_json(indent=2))


# ============================================================================
# Tool registration
# ============================================================================


def example_input() -> FreeBindCraftInput:
    """Minimal valid input — small target with a single hotspot for one-off sampling."""
    return FreeBindCraftInput(
        target_pdb=Structure.from_file(_FREEBINDCRAFT_FIXTURE_PDB),
        target_chain="A",
        target_hotspot_residues="56",
        binder_lengths=(60, 70),
        binder_name="example",
        number_of_final_designs=1,
    )


@tool(
    key="freebindcraft-design",
    label="FreeBindCraft Binder Design",
    category="binder_design",
    input_class=FreeBindCraftInput,
    config_class=FreeBindCraftConfig,
    output_class=FreeBindCraftOutput,
    metrics_class=FreeBindCraftMetrics,
    description=(
        "PyRosetta-free de novo binder design: AlphaFold2 hallucination + ProteinMPNN refinement "
        "+ AlphaFold2 validation + OpenMM relaxation and FreeSASA/sc-rs interface scoring. "
        "Returns accepted binders against a target."
    ),
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    stochastic=True,
)
def run_freebindcraft_design(
    inputs: FreeBindCraftInput,
    config: FreeBindCraftConfig,
    instance: Any = None,
) -> FreeBindCraftOutput:
    """Run the PyRosetta-free FreeBindCraft binder-design pipeline against a target.

    Args:
        inputs (FreeBindCraftInput): Target + binder spec.
        config (FreeBindCraftConfig): Advanced settings + ``filter_overrides``.
        instance (Any): Optional ``ToolInstance`` for persistent execution.

    Returns:
        FreeBindCraftOutput: Accepted binder designs and run-level counters.
    """
    logger.debug("Dispatching FreeBindCraft design pipeline (target_chain=%s)", inputs.target_chain)

    # Materialize the target Structure to a tempfile for the standalone to read.
    with inputs.target_pdb.temp_file() as target_pdb_path:
        payload: dict[str, Any] = {
            "operation": "design",
            "target_pdb": str(target_pdb_path),
            "target_chain": inputs.target_chain,
            "target_hotspot_residues": inputs.target_hotspot_residues,
            "binder_lengths": list(inputs.binder_lengths),
            "binder_name": inputs.binder_name,
            "number_of_final_designs": inputs.number_of_final_designs,
            "advanced_settings": {
                **{key: getattr(config, key) for key in _USER_FACING_UPSTREAM_KEYS},
                **_HARDCODED_INTERNAL_SETTINGS,
            },
            "filter_overrides": config.filter_overrides,
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        }

        result = ToolInstance.dispatch("freebindcraft", payload, instance=instance, config=config)

    designs: list[FreeBindCraftDesign] = []
    for raw in result["designs"]:
        metrics = FreeBindCraftMetrics(**raw["metrics"])
        designs.append(
            FreeBindCraftDesign(
                design_name=raw["design_name"],
                binder_sequence=raw["binder_sequence"],
                structure=Structure(
                    structure=raw["pdb"],
                    b_factor_type=BFactorType.PLDDT,
                    metrics=metrics,
                    source="freebindcraft-design",
                ),
                metrics=metrics,
                seed=raw["seed"],
                interface_aas=raw.get("interface_aas", {}),
                interface_residues=raw.get("interface_residues", []),
            )
        )

    return FreeBindCraftOutput(
        designs=designs,
        n_trajectories_run=result["n_trajectories_run"],
        n_designs_accepted=result["n_designs_accepted"],
    )
